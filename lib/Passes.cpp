//===- Passes.cpp - QEMU conversion passes implementation -----------------===//
//
// Implementation of three passes for LLHD to QEMU conversion:
// 1. ClockSignalDetectionPass - Detect and mark clock signals
// 2. DrvClassificationPass - Classify drv operations
// 3. ClockDrvRemovalPass - Remove clock-related and UNCHANGED drvs
//
//===----------------------------------------------------------------------===//

#include "ClockAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

// TableGen 生成的 Pass 声明
#define GEN_PASS_DECL_CLOCKSIGNALDETECTION
#define GEN_PASS_DEF_CLOCKSIGNALDETECTION
#define GEN_PASS_DECL_DRVCLASSIFICATION
#define GEN_PASS_DEF_DRVCLASSIFICATION
#define GEN_PASS_DECL_CLOCKDRVREMOVAL
#define GEN_PASS_DEF_CLOCKDRVREMOVAL
#include "Passes.h.inc"

using namespace mlir;
using namespace circt;
using namespace clock_analysis;

namespace {

//===----------------------------------------------------------------------===//
// Pass 1: ClockSignalDetectionPass
//===----------------------------------------------------------------------===//

struct ClockSignalDetectionPass
    : public ::impl::ClockSignalDetectionBase<ClockSignalDetectionPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    llvm::outs() << "========================================\n";
    llvm::outs() << "Pass 1: Clock Signal Detection\n";
    llvm::outs() << "========================================\n\n";

    int clockSignalsFound = 0;
    int clockTriggeredProcesses = 0;

    mod.walk([&](hw::HWModuleOp hwMod) {
      llvm::outs() << "Module: @" << hwMod.getName() << "\n";

      // 收集所有信号
      llvm::DenseMap<Value, llhd::SignalOp> signals;
      hwMod.walk([&](llhd::SignalOp sigOp) {
        signals[sigOp.getResult()] = sigOp;
      });

      // 对每个 process 分析时钟信号
      hwMod.walk([&](llhd::ProcessOp proc) {
        bool isClockTriggered = false;
        Value clockSignal = nullptr;

        // 检查 wait 的观察信号
        proc.walk([&](llhd::WaitOp waitOp) {
          for (Value observed : waitOp.getObserved()) {
            if (auto prbOp = observed.getDefiningOp<llhd::PrbOp>()) {
              Value sig = prbOp.getSignal();
              std::string sigName = getSignalName(sig);

              // 第一级检测
              bool passedLevel1 = isClockSignalByUsagePattern(sig, proc);

              // 第二级检测
              bool passedLevel2 = false;
              if (passedLevel1) {
                passedLevel2 = isClockByTriggerEffect(sig, proc);
                if (!passedLevel2) {
                  // 调试：为什么第二级检测失败
                  llvm::outs() << "  [DEBUG] " << sigName
                               << " passed level 1 but failed level 2 (has state modification)\n";
                }
              }

              // 两级检测
              if (passedLevel1 && passedLevel2) {
                // 标记信号为时钟
                if (auto sigOp = sig.getDefiningOp<llhd::SignalOp>()) {
                  if (!sigOp->hasAttr("qemu.is_clock")) {
                    sigOp->setAttr("qemu.is_clock", UnitAttr::get(ctx));
                    clockSignalsFound++;
                    llvm::outs() << "  [CLOCK] Signal: " << sigName << "\n";
                  }
                }
                isClockTriggered = true;
                clockSignal = sig;
              }
            }
          }
        });

        // 标记 process 为时钟触发
        if (isClockTriggered) {
          proc->setAttr("qemu.clock_triggered", UnitAttr::get(ctx));
          clockTriggeredProcesses++;
          llvm::outs() << "  [CLOCK-TRIGGERED] Process at " << proc.getLoc() << "\n";
        }
      });
    });

    llvm::outs() << "\n----------------------------------------\n";
    llvm::outs() << "Summary:\n";
    llvm::outs() << "  Clock signals found: " << clockSignalsFound << "\n";
    llvm::outs() << "  Clock-triggered processes: " << clockTriggeredProcesses << "\n";
    llvm::outs() << "========================================\n\n";
  }
};

//===----------------------------------------------------------------------===//
// Pass 2: DrvClassificationPass
//===----------------------------------------------------------------------===//

struct DrvClassificationPass
    : public ::impl::DrvClassificationBase<DrvClassificationPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    llvm::outs() << "========================================\n";
    llvm::outs() << "Pass 2: Drv Classification\n";
    llvm::outs() << "========================================\n\n";

    int unchangedCount = 0;
    int accumulateCount = 0;
    int loopIterCount = 0;
    int complexCount = 0;

    mod.walk([&](hw::HWModuleOp hwMod) {
      llvm::outs() << "Module: @" << hwMod.getName() << "\n";

      hwMod.walk([&](llhd::ProcessOp proc) {
        // 收集 wait blocks
        llvm::DenseSet<Block*> waitBlocks;
        proc.walk([&](llhd::WaitOp wait) {
          waitBlocks.insert(wait->getBlock());
        });

        // 分类每个 drv
        proc.walk([&](llhd::DrvOp drv) {
          DrvClassResult result = classifyDrv(drv, waitBlocks);

          // 添加分类属性
          StringAttr classAttr = StringAttr::get(ctx, drvClassToString(result.classification));
          drv->setAttr("qemu.drv_class", classAttr);

          // 对于 ACCUMULATE，添加步进值
          if (result.classification == DrvClass::ACCUMULATE) {
            drv->setAttr("qemu.step", IntegerAttr::get(
                IntegerType::get(ctx, 32), result.stepValue));
          }

          // 统计
          std::string sigName = getSignalName(drv.getSignal());
          switch (result.classification) {
            case DrvClass::UNCHANGED:
              unchangedCount++;
              llvm::outs() << "  [UNCHANGED] " << sigName << "\n";
              break;
            case DrvClass::ACCUMULATE:
              accumulateCount++;
              llvm::outs() << "  [ACCUMULATE] " << sigName
                           << " (step=" << result.stepValue << ")\n";
              break;
            case DrvClass::LOOP_ITER:
              loopIterCount++;
              llvm::outs() << "  [LOOP_ITER] " << sigName << "\n";
              break;
            case DrvClass::COMPLEX:
              complexCount++;
              llvm::outs() << "  [COMPLEX] " << sigName << "\n";
              break;
          }
        });
      });

      // 处理 process 外部的 drv（组合逻辑）
      hwMod.walk([&](llhd::DrvOp drv) {
        Operation *parent = drv->getParentOp();
        while (parent && !isa<llhd::ProcessOp>(parent) &&
               !isa<hw::HWModuleOp>(parent)) {
          parent = parent->getParentOp();
        }
        if (isa<llhd::ProcessOp>(parent))
          return;  // 已处理

        // process 外部的 drv 默认为 UNCHANGED（组合逻辑）
        if (!drv->hasAttr("qemu.drv_class")) {
          drv->setAttr("qemu.drv_class",
                       StringAttr::get(ctx, "UNCHANGED"));
          unchangedCount++;
          std::string sigName = getSignalName(drv.getSignal());
          llvm::outs() << "  [UNCHANGED] (comb) " << sigName << "\n";
        }
      });
    });

    llvm::outs() << "\n----------------------------------------\n";
    llvm::outs() << "Summary:\n";
    llvm::outs() << "  UNCHANGED:  " << unchangedCount << "\n";
    llvm::outs() << "  ACCUMULATE: " << accumulateCount << "\n";
    llvm::outs() << "  LOOP_ITER:  " << loopIterCount << "\n";
    llvm::outs() << "  COMPLEX:    " << complexCount << "\n";
    llvm::outs() << "========================================\n\n";
  }
};

//===----------------------------------------------------------------------===//
// Pass 3: ClockDrvRemovalPass
// 完整删除时钟信号相关的 LLHD 拓扑链：
// 1. 收集时钟信号（qemu.is_clock 属性）
// 2. 从 wait observed 列表中移除时钟
// 3. 重写 CFG：将时钟边沿检测的 cond_br 改为无条件 br
// 4. DCE：删除死代码（prb、组合逻辑）
// 5. 删除时钟的端口连接 drv
// 6. 删除时钟的 sig 定义
//===----------------------------------------------------------------------===//

struct ClockDrvRemovalPass
    : public ::impl::ClockDrvRemovalBase<ClockDrvRemovalPass> {

  // 检查操作是否只被时钟相关操作使用
  bool isOnlyUsedByClockOps(Operation *op,
                            llvm::DenseSet<Operation*> &opsToRemove) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (opsToRemove.count(user) == 0) {
          return false;
        }
      }
    }
    return true;
  }

  // 检查条件是否涉及时钟信号的边沿检测
  bool isClockEdgeCondition(Value cond, llvm::DenseSet<Value> &clockSignals) {
    Operation *defOp = cond.getDefiningOp();
    if (!defOp)
      return false;

    // 检查 AND 操作（边沿检测）
    if (auto andOp = dyn_cast<comb::AndOp>(defOp)) {
      for (Value operand : andOp.getOperands()) {
        // 直接 prb
        if (auto prbOp = operand.getDefiningOp<llhd::PrbOp>()) {
          if (clockSignals.count(prbOp.getSignal())) {
            return true;
          }
        }
        // 取反的 prb (xor prb, true)
        if (auto xorOp = operand.getDefiningOp<comb::XorOp>()) {
          for (Value xorOperand : xorOp.getOperands()) {
            if (auto prbOp = xorOperand.getDefiningOp<llhd::PrbOp>()) {
              if (clockSignals.count(prbOp.getSignal())) {
                return true;
              }
            }
          }
        }
      }
    }

    // 检查 OR 操作（复合边沿条件）
    if (auto orOp = dyn_cast<comb::OrOp>(defOp)) {
      for (Value operand : orOp.getOperands()) {
        if (isClockEdgeCondition(operand, clockSignals)) {
          return true;
        }
      }
    }

    return false;
  }

  // 收集条件表达式中涉及的所有操作
  void collectConditionOps(Value cond, llvm::DenseSet<Operation*> &ops) {
    Operation *defOp = cond.getDefiningOp();
    if (!defOp || ops.count(defOp))
      return;

    ops.insert(defOp);

    for (Value operand : defOp->getOperands()) {
      collectConditionOps(operand, ops);
    }
  }

  // 运行 DCE：删除时钟相关的死代码
  // 策略：迭代删除没有使用者的操作
  void runLocalDCE(hw::HWModuleOp hwMod, llvm::DenseSet<Value> &clockSignals,
                   int &removedOps) {
    // 多轮迭代删除死代码
    bool changed = true;
    int iterations = 0;
    const int maxIterations = 20;  // 防止无限循环

    while (changed && iterations < maxIterations) {
      changed = false;
      iterations++;

      // 先收集所有操作到一个列表中（使用 post-order 确保先访问子操作）
      llvm::SmallVector<Operation*, 64> allOps;
      hwMod.walk<WalkOrder::PostOrder>([&](Operation *op) {
        allOps.push_back(op);
      });

      // 检查并删除死代码
      for (Operation *op : allOps) {
        // 跳过有副作用的操作
        if (isa<llhd::DrvOp, llhd::WaitOp, llhd::SignalOp>(op))
          continue;
        // 跳过 terminator
        if (op->hasTrait<OpTrait::IsTerminator>())
          continue;
        // 跳过 hw.module 自身
        if (isa<hw::HWModuleOp>(op))
          continue;
        // 跳过 llhd.process
        if (isa<llhd::ProcessOp>(op))
          continue;

        // 检查结果是否都没有使用者
        bool hasLiveUsers = false;
        for (Value result : op->getResults()) {
          if (!result.use_empty()) {
            hasLiveUsers = true;
            break;
          }
        }

        if (!hasLiveUsers) {
          op->erase();
          removedOps++;
          changed = true;
        }
      }
    }
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    llvm::outs() << "========================================\n";
    llvm::outs() << "Pass 3: Clock Signal Removal (Complete)\n";
    llvm::outs() << "========================================\n\n";

    int removedFromWait = 0;
    int rewrittenBranches = 0;
    int removedDCE = 0;
    int removedDrvs = 0;
    int removedSignals = 0;

    //=== Step 1: 收集所有时钟信号 ===//
    llvm::DenseSet<Value> clockSignals;
    mod.walk([&](llhd::SignalOp sigOp) {
      if (sigOp->hasAttr("qemu.is_clock")) {
        clockSignals.insert(sigOp.getResult());
        llvm::outs() << "[STEP 1] Clock signal: "
                     << getSignalName(sigOp.getResult()) << "\n";
      }
    });

    if (clockSignals.empty()) {
      llvm::outs() << "No clock signals found. Nothing to remove.\n";
      return;
    }

    mod.walk([&](hw::HWModuleOp hwMod) {
      llvm::outs() << "\nModule: @" << hwMod.getName() << "\n";

      //=== Step 2: 从 wait observed 列表中移除时钟 ===//
      hwMod.walk([&](llhd::WaitOp waitOp) {
        // 收集需要移除的 observed 索引
        llvm::SmallVector<unsigned, 4> indicesToRemove;
        auto observed = waitOp.getObserved();

        for (unsigned i = 0; i < observed.size(); ++i) {
          Value obs = observed[i];
          if (auto prbOp = obs.getDefiningOp<llhd::PrbOp>()) {
            if (clockSignals.count(prbOp.getSignal())) {
              std::string sigName = getSignalName(prbOp.getSignal());
              llvm::outs() << "[STEP 2] Remove from wait: " << sigName << "\n";
              indicesToRemove.push_back(i);
              removedFromWait++;
            }
          }
        }

        // 逆序移除，避免索引变化
        auto mutableObserved = waitOp.getObservedMutable();
        for (auto it = indicesToRemove.rbegin(); it != indicesToRemove.rend(); ++it) {
          mutableObserved.erase(*it);
        }
      });

      //=== Step 3: 重写 CFG（将时钟边沿检测的 cond_br 改为无条件 br）===//
      llvm::SmallVector<cf::CondBranchOp, 16> branchesToRewrite;

      hwMod.walk([&](cf::CondBranchOp condBr) {
        Value cond = condBr.getCondition();
        if (isClockEdgeCondition(cond, clockSignals)) {
          branchesToRewrite.push_back(condBr);
        }
      });

      for (cf::CondBranchOp condBr : branchesToRewrite) {
        // 确定目标 block：选择非 wait block 的分支
        // 通常 true 分支是触发路径（包含 drv），false 分支是等待路径（包含 wait）
        Block *targetBlock = nullptr;
        llvm::SmallVector<Value, 4> targetArgs;

        // 检查 false 分支是否包含 wait
        Block *falseBlock = condBr.getFalseDest();
        bool falseHasWait = false;
        for (Operation &op : *falseBlock) {
          if (isa<llhd::WaitOp>(op)) {
            falseHasWait = true;
            break;
          }
        }

        if (falseHasWait) {
          // false 分支是 wait，跳转到 true 分支
          targetBlock = condBr.getTrueDest();
          for (Value arg : condBr.getTrueDestOperands()) {
            targetArgs.push_back(arg);
          }
          llvm::outs() << "[STEP 3] Rewrite cond_br -> br (to true branch)\n";
        } else {
          // 检查 true 分支
          Block *trueBlock = condBr.getTrueDest();
          bool trueHasWait = false;
          for (Operation &op : *trueBlock) {
            if (isa<llhd::WaitOp>(op)) {
              trueHasWait = true;
              break;
            }
          }

          if (trueHasWait) {
            targetBlock = condBr.getFalseDest();
            for (Value arg : condBr.getFalseDestOperands()) {
              targetArgs.push_back(arg);
            }
            llvm::outs() << "[STEP 3] Rewrite cond_br -> br (to false branch)\n";
          } else {
            // 两个分支都没有 wait，跳转到 true 分支（触发路径）
            targetBlock = condBr.getTrueDest();
            for (Value arg : condBr.getTrueDestOperands()) {
              targetArgs.push_back(arg);
            }
            llvm::outs() << "[STEP 3] Rewrite cond_br -> br (default to true)\n";
          }
        }

        // 创建无条件 br
        OpBuilder builder(condBr);
        builder.create<cf::BranchOp>(condBr.getLoc(), targetBlock, targetArgs);
        condBr.erase();
        rewrittenBranches++;
      }

      //=== Step 4: DCE - 删除死代码 ===//
      llvm::outs() << "[STEP 4] Running DCE...\n";
      runLocalDCE(hwMod, clockSignals, removedDCE);

      //=== Step 5: 删除时钟的端口连接 drv ===//
      llvm::SmallVector<llhd::DrvOp, 16> drvsToRemove;
      hwMod.walk([&](llhd::DrvOp drv) {
        if (clockSignals.count(drv.getSignal())) {
          drvsToRemove.push_back(drv);
        }
      });

      for (llhd::DrvOp drv : drvsToRemove) {
        std::string sigName = getSignalName(drv.getSignal());
        llvm::outs() << "[STEP 5] Remove drv: " << sigName << "\n";
        drv.erase();
        removedDrvs++;
      }
    });

    //=== Step 6: 删除时钟的 sig 定义 ===//
    llvm::SmallVector<llhd::SignalOp, 8> sigsToRemove;
    mod.walk([&](llhd::SignalOp sigOp) {
      if (sigOp->hasAttr("qemu.is_clock")) {
        if (sigOp.getResult().use_empty()) {
          sigsToRemove.push_back(sigOp);
        } else {
          std::string sigName = getSignalName(sigOp.getResult());
          llvm::outs() << "[STEP 6] Cannot remove sig (has users): "
                       << sigName << "\n";
        }
      }
    });

    for (llhd::SignalOp sigOp : sigsToRemove) {
      std::string sigName = getSignalName(sigOp.getResult());
      llvm::outs() << "[STEP 6] Remove sig: " << sigName << "\n";
      sigOp.erase();
      removedSignals++;
    }

    llvm::outs() << "\n----------------------------------------\n";
    llvm::outs() << "Summary:\n";
    llvm::outs() << "  Removed from wait:   " << removedFromWait << "\n";
    llvm::outs() << "  Rewritten branches:  " << rewrittenBranches << "\n";
    llvm::outs() << "  DCE removed ops:     " << removedDCE << "\n";
    llvm::outs() << "  Removed drvs:        " << removedDrvs << "\n";
    llvm::outs() << "  Removed signals:     " << removedSignals << "\n";
    llvm::outs() << "========================================\n\n";
  }
};

} // namespace
