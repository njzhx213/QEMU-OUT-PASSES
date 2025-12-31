//===- Passes.cpp - QEMU conversion passes implementation -----------------===//
//
// Implementation of three passes for LLHD to QEMU conversion:
// 1. ClockSignalDetectionPass - Detect and mark clock signals (two-level)
// 2. DrvClassificationPass - Classify drv operations
// 3. ClockDrvRemovalPass - Remove filterable clock topology
//    (aligned with old framework SignalTracing.h)
//
//===----------------------------------------------------------------------===//

#include "ClockAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
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
// 完整删除时钟信号相关的 LLHD 拓扑链（对齐旧框架 SignalTracing.h）：
// 1. 收集时钟信号（qemu.is_clock 属性）
// 2. 对每个 clock-triggered process：
//    a. 收集 waitBlocks（旧框架同款）
//    b. 从 trigger OR 中剔除 clock edge term（最小改写，不误伤 enable guard）
//    c. 从 wait observed 中移除 clock probe（防止空 wait）
// 3. DCE：删除死代码（需要 isMemoryEffectFree）
// 4. 删除时钟的端口连接 drv（仅 BlockArgument 驱动值）
// 5. 删除时钟的 sig 定义
//===----------------------------------------------------------------------===//

struct ClockDrvRemovalPass
    : public ::impl::ClockDrvRemovalBase<ClockDrvRemovalPass> {

  //=== Step 0: 统一的 base signal 工具（对齐 ClockAnalysis.h 的 unwrap 思路）===//

  /// 获取信号的 base signal（穿透 SigExtract）
  static Value getBaseSignal(Value sig) {
    while (auto ex = sig.getDefiningOp<llhd::SigExtractOp>())
      sig = ex.getInput();
    return sig;
  }

  /// 检查信号是否是时钟信号（使用 base signal 比较）
  static bool isClockBaseSignal(Value sigOrExtract,
                                const llvm::DenseSet<Value> &clockSignals) {
    return clockSignals.count(getBaseSignal(sigOrExtract));
  }

  //=== Step 3.1: 拍平 OR 操作 ===//

  /// 将 OR 操作递归拍平为 terms 列表
  static void flattenOr(Value v, llvm::SmallVectorImpl<Value> &terms) {
    if (auto orOp = v.getDefiningOp<comb::OrOp>()) {
      for (auto opnd : orOp.getOperands())
        flattenOr(opnd, terms);
      return;
    }
    terms.push_back(v);
  }

  //=== Step 3.2: 严格的边沿检测 term 识别 ===//

  /// 边沿检测 term 结构：包含 base signal 信息
  struct EdgeTermInfo {
    bool isEdgeTerm = false;
    Value baseSignal;  // 边沿检测的信号
  };

  /// 检查一个 term 是否是边沿检测模式，返回其 base signal
  /// 边沿检测模式：comb.and(prb(sig), xor(prb(sig), true))
  /// 必须同时有 direct prb 和 inverted prb，且指向同一 base signal
  EdgeTermInfo analyzeEdgeTerm(Value term) {
    EdgeTermInfo info;

    auto andOp = term.getDefiningOp<comb::AndOp>();
    if (!andOp)
      return info;

    Value directPrbSignal;
    Value invertedPrbSignal;

    for (Value operand : andOp.getOperands()) {
      // 检查直接的 prb
      if (auto prbOp = operand.getDefiningOp<llhd::PrbOp>()) {
        directPrbSignal = getBaseSignal(prbOp.getSignal());
      }
      // 检查 xor(prb signal, true) - 取反
      if (auto xorOp = operand.getDefiningOp<comb::XorOp>()) {
        bool hasConstTrue = false;
        Value prbSig;
        for (Value xorOperand : xorOp.getOperands()) {
          if (auto constOp = xorOperand.getDefiningOp<hw::ConstantOp>()) {
            if (constOp.getValue().isAllOnes()) {
              hasConstTrue = true;
            }
          }
          if (auto prbOp = xorOperand.getDefiningOp<llhd::PrbOp>()) {
            prbSig = getBaseSignal(prbOp.getSignal());
          }
        }
        if (hasConstTrue && prbSig) {
          invertedPrbSignal = prbSig;
        }
      }
    }

    // 边沿检测必须同时有直接 prb 和取反 prb，且指向同一 base signal
    if (directPrbSignal && invertedPrbSignal &&
        directPrbSignal == invertedPrbSignal) {
      info.isEdgeTerm = true;
      info.baseSignal = directPrbSignal;
    }

    return info;
  }

  //=== Step 3.3: 从 trigger 中剔除 clock edge terms 并重建条件 ===//

  /// 重建 OR 操作（从 terms 列表）
  Value rebuildOrChain(OpBuilder &builder, Location loc,
                       ArrayRef<Value> terms) {
    if (terms.empty())
      return nullptr;
    if (terms.size() == 1)
      return terms[0];

    // 递归构建二叉 OR 树
    Value result = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) {
      result = builder.create<comb::OrOp>(loc, result, terms[i]);
    }
    return result;
  }

  //=== Step 4: 安全的 DCE（使用 MLIR MemoryEffect 接口 + 显式白名单）===//

  /// 检查操作是否没有内存副作用（对齐旧框架边界）
  /// 使用 MLIR 的 MemoryEffectOpInterface 优先，加上显式白名单
  static bool isOpMemoryEffectFree(Operation *op) {
    // 显式排除有副作用的 LLHD 操作（不可删除）
    if (isa<llhd::DrvOp, llhd::WaitOp, llhd::SignalOp>(op))
      return false;
    // 跳过 terminator
    if (op->hasTrait<OpTrait::IsTerminator>())
      return false;
    // 跳过容器操作
    if (isa<hw::HWModuleOp, llhd::ProcessOp>(op))
      return false;

    // 显式白名单：这些操作可以安全删除
    // llhd.sig.extract - 信号切片，纯函数，必须支持以解除 clock sig 的引用
    if (isa<llhd::SigExtractOp>(op))
      return true;
    // llhd.prb 是只读操作，可以安全删除
    if (isa<llhd::PrbOp>(op))
      return true;
    // llhd.constant_time 是纯常量
    if (isa<llhd::ConstantTimeOp>(op))
      return true;
    // hw::ConstantOp
    if (isa<hw::ConstantOp>(op))
      return true;
    // comb 操作都是纯函数，无副作用
    if (op->getDialect() &&
        op->getDialect()->getNamespace() == "comb")
      return true;

    // 使用 MLIR MemoryEffect 接口作为兜底
    // 注意：某些 LLHD 操作可能没有正确实现此接口，所以显式白名单优先
    if (mlir::isMemoryEffectFree(op))
      return true;

    return false;
  }

  /// 运行安全的 DCE：只删除无副作用且无使用者的操作
  void runSafeDCE(hw::HWModuleOp hwMod, int &removedOps) {
    bool changed = true;
    int iterations = 0;
    const int maxIterations = 20;

    while (changed && iterations < maxIterations) {
      changed = false;
      iterations++;

      // 使用 post-order 收集所有操作
      llvm::SmallVector<Operation*, 64> allOps;
      hwMod.walk<WalkOrder::PostOrder>([&](Operation *op) {
        allOps.push_back(op);
      });

      for (Operation *op : allOps) {
        // 跳过不能删除的操作
        if (!isOpMemoryEffectFree(op))
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

    llvm::outs() << "========================================\n";
    llvm::outs() << "Pass 3: Clock Signal Removal (Fixed)\n";
    llvm::outs() << "(Aligned with old framework SignalTracing.h)\n";
    llvm::outs() << "========================================\n\n";

    int removedFromWait = 0;
    int rewrittenBranches = 0;
    int removedClockTerms = 0;
    int removedDCE = 0;
    int removedDrvs = 0;
    int removedSignals = 0;
    int removedProcesses = 0;

    //=== Step 1: 收集所有时钟信号（使用 base signal）===//
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

      // 收集需要删除的 process（纯时钟触发的 process）
      // 使用 SmallPtrSet 去重，避免 double erase
      llvm::SmallPtrSet<Operation*, 8> processesToRemove;

      //=== 对每个 clock-triggered process 进行处理 ===//
      hwMod.walk([&](llhd::ProcessOp proc) {
        // 只处理标记为 clock_triggered 的 process
        if (!proc->hasAttr("qemu.clock_triggered"))
          return;

        llvm::outs() << "  Processing clock-triggered process at "
                     << proc.getLoc() << "\n";

        //=== Step 2a: 收集 waitBlocks（旧框架同款）===//
        llvm::SmallPtrSet<Block*, 8> waitBlocks;
        proc.walk([&](llhd::WaitOp waitOp) {
          waitBlocks.insert(waitOp->getBlock());
        });

        bool processModified = false;
        bool allConditionsRemoved = true;  // 是否所有触发条件都被移除

        //=== Step 2b: 对触发门 cond_br 做最小化改写 ===//
        llvm::SmallVector<cf::CondBranchOp, 8> condBrsToProcess;
        proc.walk([&](cf::CondBranchOp condBr) {
          condBrsToProcess.push_back(condBr);
        });

        for (cf::CondBranchOp condBr : condBrsToProcess) {
          Value cond = condBr.getCondition();

          // 使用 waitBlocks 判断 idle/body 分支
          Block *trueDest = condBr.getTrueDest();
          Block *falseDest = condBr.getFalseDest();

          bool trueIsWait = waitBlocks.count(trueDest) > 0;
          bool falseIsWait = waitBlocks.count(falseDest) > 0;

          // 不是标准触发门（两边都在或都不在 waitBlocks）：不处理
          if (trueIsWait == falseIsWait) {
            allConditionsRemoved = false;
            continue;
          }

          Block *idleBlock = trueIsWait ? trueDest : falseDest;
          Block *bodyBlock = trueIsWait ? falseDest : trueDest;
          OperandRange idleArgs = trueIsWait ? condBr.getTrueDestOperands()
                                             : condBr.getFalseDestOperands();
          OperandRange bodyArgs = trueIsWait ? condBr.getFalseDestOperands()
                                             : condBr.getTrueDestOperands();

          // 拍平 OR 条件
          llvm::SmallVector<Value, 4> terms;
          flattenOr(cond, terms);

          // 分析每个 term，剔除 clock edge terms
          llvm::SmallVector<Value, 4> remainingTerms;
          bool removedAnyClockTerm = false;

          for (Value term : terms) {
            EdgeTermInfo info = analyzeEdgeTerm(term);
            if (info.isEdgeTerm && isClockBaseSignal(info.baseSignal, clockSignals)) {
              // 这是 clock edge term，剔除
              llvm::outs() << "    [STEP 2b] Remove clock edge term for: "
                           << getSignalName(info.baseSignal) << "\n";
              removedAnyClockTerm = true;
              removedClockTerms++;
            } else {
              // 保留（例如 reset edge term 或 enable guard）
              remainingTerms.push_back(term);
            }
          }

          if (!removedAnyClockTerm) {
            // 没有移除任何 clock term，不改写（避免误伤 enable gating）
            allConditionsRemoved = false;
            continue;
          }

          processModified = true;

          if (remainingTerms.empty()) {
            // 所有 terms 都被移除：这个触发门只由 clock 触发
            // 去掉 clock 后永不触发，跳转回 idle/wait 分支
            llvm::outs() << "    [STEP 2b] All terms removed, jump to idle\n";
            OpBuilder builder(condBr);
            llvm::SmallVector<Value, 4> idleArgsVec(idleArgs.begin(), idleArgs.end());
            builder.create<cf::BranchOp>(condBr.getLoc(), idleBlock, idleArgsVec);
            condBr.erase();
            rewrittenBranches++;
          } else {
            // 重建条件（只保留非 clock terms）
            allConditionsRemoved = false;
            OpBuilder builder(condBr);
            Value newCond = rebuildOrChain(builder, condBr.getLoc(), remainingTerms);

            // 创建新的 cond_br
            llvm::SmallVector<Value, 4> trueArgsVec, falseArgsVec;
            if (trueIsWait) {
              trueArgsVec.assign(idleArgs.begin(), idleArgs.end());
              falseArgsVec.assign(bodyArgs.begin(), bodyArgs.end());
            } else {
              trueArgsVec.assign(bodyArgs.begin(), bodyArgs.end());
              falseArgsVec.assign(idleArgs.begin(), idleArgs.end());
            }

            builder.create<cf::CondBranchOp>(
                condBr.getLoc(), newCond,
                trueDest, trueArgsVec,
                falseDest, falseArgsVec);
            condBr.erase();
            rewrittenBranches++;
            llvm::outs() << "    [STEP 2b] Rebuilt trigger condition\n";
          }
        }

        //=== Step 2c: 从 wait observed 中移除 clock probe ===//
        if (processModified) {
          // 检查是否已经被标记为要删除的 process
          if (processesToRemove.contains(proc.getOperation()))
            return;  // 已标记删除，跳过

          proc.walk([&](llhd::WaitOp waitOp) -> WalkResult {
            // 如果 process 已被标记删除，中断 walk
            if (processesToRemove.contains(proc.getOperation()))
              return WalkResult::interrupt();

            llvm::SmallVector<unsigned, 4> indicesToRemove;
            auto observed = waitOp.getObserved();
            unsigned nonClockCount = 0;

            for (unsigned i = 0; i < observed.size(); ++i) {
              Value obs = observed[i];
              if (auto prbOp = obs.getDefiningOp<llhd::PrbOp>()) {
                if (isClockBaseSignal(prbOp.getSignal(), clockSignals)) {
                  indicesToRemove.push_back(i);
                } else {
                  nonClockCount++;
                }
              } else {
                nonClockCount++;
              }
            }

            // 防止空 wait：如果移除后 observed 为空
            if (nonClockCount == 0 && !indicesToRemove.empty()) {
              // 这个 process 只由 clock 触发
              if (allConditionsRemoved) {
                // 触发条件也全被移除了：可以删除整个 process
                llvm::outs() << "    [STEP 2c] Process is pure clock-only, mark for removal\n";
                processesToRemove.insert(proc.getOperation());
                return WalkResult::interrupt();  // 中断 walk，不继续处理此 process
              } else {
                // 保留至少一个 clock observed（不产生空 wait）
                llvm::outs() << "    [STEP 2c] Keep one clock observed to avoid empty wait\n";
                indicesToRemove.pop_back();
              }
            }

            // 逆序移除
            auto mutableObserved = waitOp.getObservedMutable();
            for (auto it = indicesToRemove.rbegin(); it != indicesToRemove.rend(); ++it) {
              std::string sigName = "unknown";
              if (auto prbOp = observed[*it].getDefiningOp<llhd::PrbOp>()) {
                sigName = getSignalName(prbOp.getSignal());
              }
              llvm::outs() << "    [STEP 2c] Remove from wait: " << sigName << "\n";
              mutableObserved.erase(*it);
              removedFromWait++;
            }
            return WalkResult::advance();
          });
        }
      });

      // 删除纯时钟触发的 process（使用 SmallPtrSet 已去重）
      for (Operation *op : processesToRemove) {
        llvm::outs() << "  [REMOVE PROCESS] Pure clock-only process\n";
        op->erase();
        removedProcesses++;
      }

      //=== Step 3: DCE - 安全删除死代码 ===//
      llvm::outs() << "  [STEP 3] Running safe DCE...\n";
      runSafeDCE(hwMod, removedDCE);

      //=== Step 4: 删除时钟的端口连接 drv（仅 BlockArgument 驱动值）===//
      llvm::SmallVector<llhd::DrvOp, 16> drvsToRemove;
      hwMod.walk([&](llhd::DrvOp drv) {
        if (isClockBaseSignal(drv.getSignal(), clockSignals)) {
          // 按旧框架边界：只删除端口连接 drv（驱动值是 BlockArgument）
          if (isa<BlockArgument>(drv.getValue())) {
            drvsToRemove.push_back(drv);
          } else {
            std::string sigName = getSignalName(drv.getSignal());
            llvm::outs() << "  [STEP 4] Keep logic drv (not port connection): "
                         << sigName << "\n";
          }
        }
      });

      for (llhd::DrvOp drv : drvsToRemove) {
        std::string sigName = getSignalName(drv.getSignal());
        llvm::outs() << "  [STEP 4] Remove port drv: " << sigName << "\n";
        drv.erase();
        removedDrvs++;
      }
    });

    //=== Step 5: 删除时钟的 sig 定义（base signal 维度）===//
    llvm::SmallVector<llhd::SignalOp, 8> sigsToRemove;
    mod.walk([&](llhd::SignalOp sigOp) {
      if (sigOp->hasAttr("qemu.is_clock")) {
        if (sigOp.getResult().use_empty()) {
          sigsToRemove.push_back(sigOp);
        } else {
          std::string sigName = getSignalName(sigOp.getResult());
          llvm::outs() << "[STEP 5] Cannot remove sig (has users): "
                       << sigName << "\n";
        }
      }
    });

    for (llhd::SignalOp sigOp : sigsToRemove) {
      std::string sigName = getSignalName(sigOp.getResult());
      llvm::outs() << "[STEP 5] Remove sig: " << sigName << "\n";
      sigOp.erase();
      removedSignals++;
    }

    llvm::outs() << "\n----------------------------------------\n";
    llvm::outs() << "Summary:\n";
    llvm::outs() << "  Removed clock terms:   " << removedClockTerms << "\n";
    llvm::outs() << "  Rewritten branches:    " << rewrittenBranches << "\n";
    llvm::outs() << "  Removed from wait:     " << removedFromWait << "\n";
    llvm::outs() << "  DCE removed ops:       " << removedDCE << "\n";
    llvm::outs() << "  Removed port drvs:     " << removedDrvs << "\n";
    llvm::outs() << "  Removed signals:       " << removedSignals << "\n";
    llvm::outs() << "  Removed processes:     " << removedProcesses << "\n";
    llvm::outs() << "========================================\n\n";
  }
};

} // namespace
