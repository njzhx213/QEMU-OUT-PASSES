//===- ClockAnalysis.h - Clock signal analysis utilities ------------------===//
//
// Utilities for detecting and analyzing clock signals in LLHD IR.
// Extracted from SignalTracing.h for use in QEMU conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef QEMU_PASSES_CLOCK_ANALYSIS_H
#define QEMU_PASSES_CLOCK_ANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

namespace clock_analysis {

//===----------------------------------------------------------------------===//
// Base Signal 工具（统一穿透 SigExtract）
//===----------------------------------------------------------------------===//

/// 获取信号的 base signal（穿透 SigExtract）
/// 这是统一的 base-signal 规则，Pass1/Pass3 都应使用
inline mlir::Value getBaseSignal(mlir::Value sig) {
  while (auto ex = sig.getDefiningOp<circt::llhd::SigExtractOp>())
    sig = ex.getInput();
  return sig;
}

/// 比较两个信号是否是同一个 base signal
inline bool isSameBaseSignal(mlir::Value sig1, mlir::Value sig2) {
  return getBaseSignal(sig1) == getBaseSignal(sig2);
}

//===----------------------------------------------------------------------===//
// 信号名称获取
//===----------------------------------------------------------------------===//

/// 获取信号名称（穿透 SigExtract）
inline std::string getSignalName(mlir::Value signal) {
  // 先获取 base signal
  signal = getBaseSignal(signal);
  if (auto sigOp = signal.getDefiningOp<circt::llhd::SignalOp>()) {
    if (auto name = sigOp.getName()) {
      return name->str();
    }
  }
  return "unnamed";
}

/// 获取信号位宽
inline int getSignalBitWidth(mlir::Value signal) {
  mlir::Type sigType = signal.getType();
  if (auto hwInOut = mlir::dyn_cast<circt::hw::InOutType>(sigType)) {
    mlir::Type inner = hwInOut.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(inner)) {
      return intType.getWidth();
    }
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(sigType)) {
    return intType.getWidth();
  }
  return 32;
}

//===----------------------------------------------------------------------===//
// 时钟信号检测 - 第一级：结构特征
//===----------------------------------------------------------------------===//

/// 检查信号是否满足时钟的结构特征：
/// 1. 单比特 (i1)
/// 2. 在 wait 敏感列表中
/// 3. 无逻辑驱动（只有端口连接）
///
/// 使用 base-signal 比较，统一穿透 SigExtract
inline bool isClockSignalByUsagePattern(mlir::Value signal,
                                         circt::llhd::ProcessOp processOp) {
  if (!processOp)
    return false;

  // 获取 base signal 进行比较
  mlir::Value baseSignal = getBaseSignal(signal);
  auto sigOp = baseSignal.getDefiningOp<circt::llhd::SignalOp>();
  if (!sigOp)
    return false;

  // 1. 必须是单比特
  mlir::Type sigType = sigOp.getType();
  if (auto inoutType = mlir::dyn_cast<circt::hw::InOutType>(sigType)) {
    mlir::Type elementType = inoutType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
      if (intType.getWidth() != 1) {
        return false;  // 多比特信号不是时钟
      }
    }
  }

  // 2. 检查是否在 wait 敏感列表中（使用 base-signal 比较）
  bool isInWaitSensitivity = false;
  processOp.walk([&](circt::llhd::WaitOp waitOp) {
    for (mlir::Value observed : waitOp.getObserved()) {
      if (auto prbOp = observed.getDefiningOp<circt::llhd::PrbOp>()) {
        // 使用 base-signal 比较
        if (isSameBaseSignal(prbOp.getSignal(), baseSignal)) {
          isInWaitSensitivity = true;
          return mlir::WalkResult::interrupt();
        }
      }
    }
    return mlir::WalkResult::advance();
  });

  if (!isInWaitSensitivity)
    return false;

  // 3. 检查是否有逻辑驱动（非端口连接的 drv）
  // 使用 base-signal 比较
  bool hasLogicDrv = false;
  if (auto parentOp = processOp->getParentOfType<circt::hw::HWModuleOp>()) {
    parentOp.walk([&](circt::llhd::DrvOp drvOp) {
      // 使用 base-signal 比较
      if (isSameBaseSignal(drvOp.getSignal(), baseSignal)) {
        mlir::Value drvValue = drvOp.getValue();
        // 如果驱动值不是 BlockArgument（输入端口），则是逻辑驱动
        if (!mlir::isa<mlir::BlockArgument>(drvValue)) {
          hasLogicDrv = true;
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
  }

  return !hasLogicDrv;
}

//===----------------------------------------------------------------------===//
// 时钟信号检测 - 第二级：触发效果分析
//===----------------------------------------------------------------------===//

/// 检查 drv 操作是否是 hold 模式（reg = prb reg）
inline bool isDrvHoldPattern(circt::llhd::DrvOp drvOp) {
  mlir::Value target = drvOp.getSignal();
  mlir::Value value = drvOp.getValue();

  // 处理 sig.extract 的情况
  if (auto sigExtract = target.getDefiningOp<circt::llhd::SigExtractOp>()) {
    target = sigExtract.getInput();
  }

  if (auto prbOp = value.getDefiningOp<circt::llhd::PrbOp>()) {
    mlir::Value prbTarget = prbOp.getSignal();
    // 处理 prb 的目标也是 sig.extract 的情况
    if (auto prbExtract = prbTarget.getDefiningOp<circt::llhd::SigExtractOp>()) {
      prbTarget = prbExtract.getInput();
    }
    if (prbTarget == target) {
      return true;
    }
  }
  return false;
}

/// 触发分支效果分析结果
struct TriggerBranchEffect {
  bool hasAnyDrv = false;
  bool allDrvsAreHold = true;
  bool hasStateModification = false;
  unsigned holdCount = 0;
  unsigned modifyCount = 0;

  bool canFilter() const {
    return hasAnyDrv && allDrvsAreHold && !hasStateModification;
  }
};

/// 信号追溯结果（与原 SignalTracing.h 保持一致）
struct TracedSignal {
  mlir::Value signal;           // 原始 llhd.sig
  std::string name;             // 信号名
  bool isInverted = false;      // 是否被取反

  bool isValid() const { return signal != nullptr && !name.empty(); }
};

/// 追踪一个 Value 的源头，穿透 XOR(取反)、PRB(探测) 等操作
/// 与原 SignalTracing.h 中的 traceToSignal 保持一致
inline TracedSignal traceToSignal(mlir::Value val) {
  TracedSignal result;
  result.isInverted = false;

  // 1. 穿透取反逻辑 (XOR true)
  while (auto xorOp = val.getDefiningOp<circt::comb::XorOp>()) {
    if (xorOp.getNumOperands() != 2) break;

    mlir::Value lhs = xorOp.getOperand(0);
    mlir::Value rhs = xorOp.getOperand(1);

    // 检查 rhs 是否为常数 1 (true)
    if (auto constOp = rhs.getDefiningOp<circt::hw::ConstantOp>()) {
      if (constOp.getValue().isAllOnes()) {
        val = lhs;
        result.isInverted = !result.isInverted;
        continue;
      }
    }
    // 检查 lhs 是否为常数 1 (true)
    if (auto constOp = lhs.getDefiningOp<circt::hw::ConstantOp>()) {
      if (constOp.getValue().isAllOnes()) {
        val = rhs;
        result.isInverted = !result.isInverted;
        continue;
      }
    }
    break;
  }

  // 2. 穿透 PRB (探测)
  if (auto prbOp = val.getDefiningOp<circt::llhd::PrbOp>()) {
    mlir::Value sig = prbOp.getSignal();
    result.signal = sig;
    result.name = getSignalName(sig);
  }

  return result;
}

/// 收集分支中的 drv 效果
inline void collectBranchDrvEffects(
    mlir::Block *block,
    TriggerBranchEffect &effect,
    llvm::SmallPtrSetImpl<mlir::Block*> &visited,
    llvm::SmallPtrSetImpl<mlir::Block*> &waitBlocks,
    unsigned maxDepth = 10) {

  if (!block || !visited.insert(block).second)
    return;
  if (visited.size() > maxDepth)
    return;
  if (waitBlocks.count(block))
    return;

  for (mlir::Operation &op : *block) {
    if (auto drvOp = mlir::dyn_cast<circt::llhd::DrvOp>(&op)) {
      effect.hasAnyDrv = true;
      if (isDrvHoldPattern(drvOp)) {
        effect.holdCount++;
      } else {
        effect.allDrvsAreHold = false;
        effect.hasStateModification = true;
        effect.modifyCount++;
      }
    }
  }

  if (auto *terminator = block->getTerminator()) {
    if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
      collectBranchDrvEffects(br.getDest(), effect, visited, waitBlocks, maxDepth);
    }
    if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
      collectBranchDrvEffects(condBr.getTrueDest(), effect, visited, waitBlocks, maxDepth);
      collectBranchDrvEffects(condBr.getFalseDest(), effect, visited, waitBlocks, maxDepth);
    }
  }
}

/// 检查 AND 操作是否是边沿检测模式
/// 边沿检测模式：and(!old_sig, new_sig) 或 and(old_sig, !new_sig)
/// 关键特征：同一个信号的两次 prb（旧值和新值）参与运算
/// 使用 base-signal 比较
inline bool isEdgeDetectionPattern(circt::comb::AndOp andOp, mlir::Value signal) {
  bool hasDirectPrb = false;   // 有直接的 prb signal
  bool hasInvertedPrb = false; // 有 xor(prb signal, true)
  mlir::Value baseSignal = getBaseSignal(signal);

  for (mlir::Value operand : andOp.getOperands()) {
    // 检查直接的 prb
    if (auto prbOp = operand.getDefiningOp<circt::llhd::PrbOp>()) {
      if (isSameBaseSignal(prbOp.getSignal(), baseSignal)) {
        hasDirectPrb = true;
      }
    }
    // 检查 xor(prb signal, true) - 取反
    if (auto xorOp = operand.getDefiningOp<circt::comb::XorOp>()) {
      bool hasConstTrue = false;
      bool hasPrbToSignal = false;
      for (mlir::Value xorOperand : xorOp.getOperands()) {
        if (auto constOp = xorOperand.getDefiningOp<circt::hw::ConstantOp>()) {
          if (constOp.getValue().isAllOnes()) {
            hasConstTrue = true;
          }
        }
        if (auto prbOp = xorOperand.getDefiningOp<circt::llhd::PrbOp>()) {
          if (isSameBaseSignal(prbOp.getSignal(), baseSignal)) {
            hasPrbToSignal = true;
          }
        }
      }
      if (hasConstTrue && hasPrbToSignal) {
        hasInvertedPrb = true;
      }
    }
  }

  // 边沿检测必须同时有直接 prb 和取反 prb（代表新旧值）
  return hasDirectPrb && hasInvertedPrb;
}

/// 检查信号是否在复合边沿条件中被用于边沿检测
/// 复合边沿条件模式：or(and(!old_sig, new_sig), and(...))
/// 关键：必须是真正的边沿检测（新旧值比较），不是简单的电平检测
inline bool isSignalInEdgeDetection(mlir::Value cond, mlir::Value signal) {
  mlir::Operation *defOp = cond.getDefiningOp();
  if (!defOp)
    return false;

  // 检查 AND 操作是否是边沿检测模式
  if (auto andOp = mlir::dyn_cast<circt::comb::AndOp>(defOp)) {
    if (isEdgeDetectionPattern(andOp, signal)) {
      return true;
    }
  }

  // 检查 OR 操作：递归检查子条件
  if (auto orOp = mlir::dyn_cast<circt::comb::OrOp>(defOp)) {
    for (mlir::Value operand : orOp.getOperands()) {
      if (isSignalInEdgeDetection(operand, signal))
        return true;
    }
  }

  return false;
}

/// 检查信号是否在 process 中被用作消歧检查
/// 消歧检查：cf.cond_br (prb signal) 或 cf.cond_br (xor (prb signal), true)
/// 使用 base-signal 比较
inline bool isSignalUsedForDisambiguation(mlir::Value signal,
                                           circt::llhd::ProcessOp processOp) {
  bool isUsed = false;
  mlir::Value baseSignal = getBaseSignal(signal);
  processOp.walk([&](mlir::cf::CondBranchOp condBr) {
    TracedSignal traced = traceToSignal(condBr.getCondition());
    if (traced.isValid() && isSameBaseSignal(traced.signal, baseSignal)) {
      isUsed = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return isUsed;
}

/// 分析信号的触发效果
///
/// 时钟信号检测逻辑：
/// 1. 如果有直接的边沿检测分支，分析其 drv 效果
/// 2. 如果在复合边沿条件中，检查是否用于消歧：
///    - 不用于消歧（纯边沿触发）→ 可过滤的时钟
///    - 用于消歧（控制行为）→ 不可过滤
/// 使用 base-signal 比较
inline TriggerBranchEffect analyzeTriggerBranchEffects(
    mlir::Value signal,
    circt::llhd::ProcessOp processOp) {

  TriggerBranchEffect effect;
  if (!signal || !processOp)
    return effect;

  // 获取 base signal
  mlir::Value baseSignal = getBaseSignal(signal);

  // 收集 wait blocks
  llvm::SmallPtrSet<mlir::Block*, 8> waitBlocks;
  processOp.walk([&](circt::llhd::WaitOp waitOp) {
    waitBlocks.insert(waitOp->getBlock());
  });

  std::string sigName = getSignalName(signal);
  bool foundDirectEdgeCheck = false;
  bool foundInCompositeEdge = false;

  // 查找检测该信号边沿的条件分支
  processOp.walk([&](mlir::cf::CondBranchOp condBr) {
    mlir::Value cond = condBr.getCondition();
    TracedSignal traced = traceToSignal(cond);

    // 情况1：直接的边沿检测（使用 base-signal 比较）
    if (traced.isValid() && isSameBaseSignal(traced.signal, baseSignal)) {
      foundDirectEdgeCheck = true;
      llvm::SmallPtrSet<mlir::Block*, 16> visitedTrue;
      llvm::SmallPtrSet<mlir::Block*, 16> visitedFalse;
      collectBranchDrvEffects(condBr.getTrueDest(), effect, visitedTrue, waitBlocks);
      collectBranchDrvEffects(condBr.getFalseDest(), effect, visitedFalse, waitBlocks);
      return;
    }

    // 情况2：检查是否在复合边沿条件中
    if (isSignalInEdgeDetection(cond, baseSignal)) {
      foundInCompositeEdge = true;
    }
  });

  // 如果在复合边沿条件中但没有直接检测，检查是否用于消歧
  if (!foundDirectEdgeCheck && foundInCompositeEdge) {
    // 检查信号是否用于消歧检查
    bool isDisambiguator = isSignalUsedForDisambiguation(signal, processOp);

    if (!isDisambiguator) {
      // 信号只用于边沿触发，不用于消歧 → 可过滤的时钟
      // 标记为"纯时钟"：hasAnyDrv=true, allDrvsAreHold=true
      effect.hasAnyDrv = true;
      effect.allDrvsAreHold = true;
      effect.hasStateModification = false;
      effect.holdCount = 1;  // 虚拟计数
    }
    // 如果用于消歧，effect 保持默认（hasAnyDrv=false）
  }

  return effect;
}

/// 基于触发效果判断是否是可过滤的时钟信号
/// 时钟信号特征：触发的所有 drv 操作都是 hold 模式
/// 复位/控制信号特征：触发的 drv 操作有状态修改
inline bool isClockByTriggerEffect(mlir::Value signal,
                                    circt::llhd::ProcessOp processOp) {
  TriggerBranchEffect effect = analyzeTriggerBranchEffects(signal, processOp);

  // 如果没有任何 drv 操作，可能是：
  // 1. 无法追溯到该信号的边沿检测分支（复合条件）
  // 2. 纯控制信号，不能简单过滤
  if (!effect.hasAnyDrv)
    return false;

  // 如果所有 drv 都是 hold 模式，则是时钟信号，可以过滤
  return effect.canFilter();
}

//===----------------------------------------------------------------------===//
// drv 分类
//===----------------------------------------------------------------------===//

/// drv 操作的分类
enum class DrvClass {
  UNCHANGED,   // 状态不变（hold 或不依赖自身）
  ACCUMULATE,  // 累加/累减（counter++）
  LOOP_ITER,   // 循环迭代器
  COMPLEX      // 复杂状态变化
};

inline llvm::StringRef drvClassToString(DrvClass cls) {
  switch (cls) {
    case DrvClass::UNCHANGED:  return "UNCHANGED";
    case DrvClass::ACCUMULATE: return "ACCUMULATE";
    case DrvClass::LOOP_ITER:  return "LOOP_ITER";
    case DrvClass::COMPLEX:    return "COMPLEX";
  }
  return "UNKNOWN";
}

/// 检查 value 是否依赖于 signal
inline bool checkDependsOnSignal(mlir::Value value, mlir::Value signal) {
  llvm::SmallVector<mlir::Value> worklist;
  llvm::DenseSet<mlir::Value> visited;
  worklist.push_back(value);

  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    if (visited.contains(v))
      continue;
    visited.insert(v);

    if (auto prb = v.getDefiningOp<circt::llhd::PrbOp>()) {
      mlir::Value prbSig = prb.getSignal();
      // 处理 sig.extract
      if (auto extract = prbSig.getDefiningOp<circt::llhd::SigExtractOp>()) {
        prbSig = extract.getInput();
      }
      mlir::Value targetSig = signal;
      if (auto extract = targetSig.getDefiningOp<circt::llhd::SigExtractOp>()) {
        targetSig = extract.getInput();
      }
      if (prbSig == targetSig)
        return true;
    }

    if (mlir::Operation *defOp = v.getDefiningOp()) {
      for (mlir::Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
  }
  return false;
}

/// 检查是否是累加模式（signal + constant）
inline int getAccumulateStep(circt::comb::AddOp addOp, mlir::Value signal) {
  bool hasSignal = false;
  int constVal = 0;

  // 获取原始信号（处理 sig.extract）
  mlir::Value targetSig = signal;
  if (auto extract = targetSig.getDefiningOp<circt::llhd::SigExtractOp>()) {
    targetSig = extract.getInput();
  }

  for (mlir::Value operand : addOp.getOperands()) {
    if (auto prb = operand.getDefiningOp<circt::llhd::PrbOp>()) {
      mlir::Value prbSig = prb.getSignal();
      if (auto extract = prbSig.getDefiningOp<circt::llhd::SigExtractOp>()) {
        prbSig = extract.getInput();
      }
      if (prbSig == targetSig) {
        hasSignal = true;
        continue;
      }
    }
    if (auto constOp = operand.getDefiningOp<circt::hw::ConstantOp>()) {
      constVal = constOp.getValue().getSExtValue();
    }
  }

  return hasSignal ? constVal : 0;
}

/// 检查是否是减法模式（signal - constant）
inline int getSubtractStep(circt::comb::SubOp subOp, mlir::Value signal) {
  mlir::Value lhs = subOp.getLhs();
  mlir::Value rhs = subOp.getRhs();

  mlir::Value targetSig = signal;
  if (auto extract = targetSig.getDefiningOp<circt::llhd::SigExtractOp>()) {
    targetSig = extract.getInput();
  }

  if (auto prb = lhs.getDefiningOp<circt::llhd::PrbOp>()) {
    mlir::Value prbSig = prb.getSignal();
    if (auto extract = prbSig.getDefiningOp<circt::llhd::SigExtractOp>()) {
      prbSig = extract.getInput();
    }
    if (prbSig == targetSig) {
      if (auto constOp = rhs.getDefiningOp<circt::hw::ConstantOp>()) {
        return constOp.getValue().getSExtValue();
      }
    }
  }
  return 0;
}

/// 检查是否是 for 循环迭代器
inline bool isLoopIterator(circt::llhd::DrvOp drv,
                           llvm::DenseSet<mlir::Block*> &waitBlocks) {
  mlir::Block *currentBlock = drv->getBlock();
  mlir::Operation *terminator = currentBlock->getTerminator();

  if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
    if (waitBlocks.contains(br.getDest())) {
      return false;  // 直接跳到 wait，是跨时钟周期的累积
    }
    // 简单检查：如果不直接跳到 wait，可能是循环
    // 更精确的检查需要分析循环结构
    return true;
  }

  if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
    if (waitBlocks.contains(condBr.getTrueDest()) ||
        waitBlocks.contains(condBr.getFalseDest())) {
      return false;
    }
  }

  return false;
}

/// 分类一个 drv 操作
struct DrvClassResult {
  DrvClass classification;
  int stepValue = 0;  // 仅对 ACCUMULATE 有效
};

inline DrvClassResult classifyDrv(circt::llhd::DrvOp drv,
                                   llvm::DenseSet<mlir::Block*> &waitBlocks) {
  DrvClassResult result;
  result.classification = DrvClass::COMPLEX;
  result.stepValue = 0;

  mlir::Value signal = drv.getSignal();
  mlir::Value value = drv.getValue();

  // 1. 检查是否是 hold 模式
  if (isDrvHoldPattern(drv)) {
    result.classification = DrvClass::UNCHANGED;
    return result;
  }

  // 2. 检查是否依赖自身
  bool dependsOnSelf = checkDependsOnSignal(value, signal);

  if (!dependsOnSelf) {
    // 不依赖自身 = 覆盖型赋值
    result.classification = DrvClass::UNCHANGED;
    return result;
  }

  // 3. 检查累加模式
  if (auto addOp = value.getDefiningOp<circt::comb::AddOp>()) {
    int step = getAccumulateStep(addOp, signal);
    if (step != 0) {
      if (isLoopIterator(drv, waitBlocks)) {
        result.classification = DrvClass::LOOP_ITER;
      } else {
        result.classification = DrvClass::ACCUMULATE;
        result.stepValue = step;
      }
      return result;
    }
  }

  // 4. 检查减法模式
  if (auto subOp = value.getDefiningOp<circt::comb::SubOp>()) {
    int step = getSubtractStep(subOp, signal);
    if (step != 0) {
      if (isLoopIterator(drv, waitBlocks)) {
        result.classification = DrvClass::LOOP_ITER;
      } else {
        result.classification = DrvClass::ACCUMULATE;
        result.stepValue = -step;  // 负数表示减法
      }
      return result;
    }
  }

  // 5. 其他情况 = COMPLEX
  result.classification = DrvClass::COMPLEX;
  return result;
}

} // namespace clock_analysis

#endif // QEMU_PASSES_CLOCK_ANALYSIS_H
