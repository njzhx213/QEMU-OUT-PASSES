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
// 删除时钟信号相关的操作：
// 1. 删除时钟信号的 llhd.sig 定义
// 2. 删除时钟信号的端口连接 drv
// 3. 删除时钟信号的 prb 操作
//===----------------------------------------------------------------------===//

struct ClockDrvRemovalPass
    : public ::impl::ClockDrvRemovalBase<ClockDrvRemovalPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    llvm::outs() << "========================================\n";
    llvm::outs() << "Pass 3: Clock Signal Removal\n";
    llvm::outs() << "(Removes clock signals and their connections)\n";
    llvm::outs() << "========================================\n\n";

    int removedSignals = 0;
    int removedDrvs = 0;
    int removedPrbs = 0;

    // 收集所有标记为时钟的信号
    llvm::DenseSet<Value> clockSignals;
    mod.walk([&](llhd::SignalOp sigOp) {
      if (sigOp->hasAttr("qemu.is_clock")) {
        clockSignals.insert(sigOp.getResult());
        llvm::outs() << "  [CLOCK SIGNAL] " << getSignalName(sigOp.getResult()) << "\n";
      }
    });

    // 收集要删除的操作
    llvm::SmallVector<Operation*, 32> opsToRemove;

    mod.walk([&](hw::HWModuleOp hwMod) {
      llvm::outs() << "\nModule: @" << hwMod.getName() << "\n";

      // 收集要删除的 drv（时钟信号的端口连接）
      hwMod.walk([&](llhd::DrvOp drv) {
        if (clockSignals.count(drv.getSignal())) {
          std::string sigName = getSignalName(drv.getSignal());
          llvm::outs() << "  [REMOVE DRV] " << sigName << "\n";
          opsToRemove.push_back(drv);
          removedDrvs++;
        }
      });

      // 收集要删除的 prb（时钟信号的读取）
      // 注意：需要先替换使用者，或者确保没有使用者
      hwMod.walk([&](llhd::PrbOp prb) {
        if (clockSignals.count(prb.getSignal())) {
          std::string sigName = getSignalName(prb.getSignal());
          // 检查是否有使用者
          if (prb.getResult().use_empty()) {
            llvm::outs() << "  [REMOVE PRB] " << sigName << "\n";
            opsToRemove.push_back(prb);
            removedPrbs++;
          } else {
            llvm::outs() << "  [KEEP PRB] " << sigName << " (has users)\n";
          }
        }
      });
    });

    // 收集要删除的 signal（时钟信号定义）
    mod.walk([&](llhd::SignalOp sigOp) {
      if (sigOp->hasAttr("qemu.is_clock")) {
        // 检查是否还有使用者
        bool canRemove = true;
        for (Operation *user : sigOp.getResult().getUsers()) {
          if (std::find(opsToRemove.begin(), opsToRemove.end(), user) == opsToRemove.end()) {
            canRemove = false;
            break;
          }
        }
        if (canRemove) {
          std::string sigName = getSignalName(sigOp.getResult());
          llvm::outs() << "  [REMOVE SIG] " << sigName << "\n";
          opsToRemove.push_back(sigOp);
          removedSignals++;
        } else {
          std::string sigName = getSignalName(sigOp.getResult());
          llvm::outs() << "  [KEEP SIG] " << sigName << " (has other users)\n";
        }
      }
    });

    // 执行删除（逆序，先删除使用者）
    for (auto it = opsToRemove.rbegin(); it != opsToRemove.rend(); ++it) {
      (*it)->erase();
    }

    llvm::outs() << "\n----------------------------------------\n";
    llvm::outs() << "Summary:\n";
    llvm::outs() << "  Removed signals: " << removedSignals << "\n";
    llvm::outs() << "  Removed drvs:    " << removedDrvs << "\n";
    llvm::outs() << "  Removed prbs:    " << removedPrbs << "\n";
    llvm::outs() << "========================================\n\n";
  }
};

} // namespace
