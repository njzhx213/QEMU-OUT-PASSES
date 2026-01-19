#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

// 只生成 per-pass 的声明/定义；不要手写 create/注册函数
#define GEN_PASS_DECL_DFFDEMO
#define GEN_PASS_DEF_DFFDEMO
#include "Passes.h.inc"

using namespace mlir;
using namespace circt;

namespace {

// Helper: unwrap llhd.sig.extract chains to get base signal
static Value unwrapSigExtract(Value sig) {
  while (auto ex = sig.getDefiningOp<llhd::SigExtractOp>()) {
    sig = ex.getInput();
  }
  return sig;
}

// Enhanced traceSignal: supports multi-layer NOT (xor, icmp) + SigExtract
// Returns (signalName, isInverted)
// - isInverted tracks odd/even NOT parity
// - Supports: comb.xor(x, allones), comb.icmp eq/ne with 0/1 constants
// - Limitation: seen map in caller doesn't handle complex inversion combinations
std::pair<StringRef, bool> traceSignal(Value val) {
  bool invertParity = false;
  Value cur = val;

  // Loop to peel multi-layer NOT operations (max depth 32 to prevent infinite loops)
  for (int depth = 0; depth < 32; ++depth) {
    // 1. Handle comb.xor(x, allones) or comb.xor(allones, x)
    if (auto xorOp = cur.getDefiningOp<comb::XorOp>()) {
      if (xorOp.getNumOperands() == 2) {
        Value lhs = xorOp.getOperand(0);
        Value rhs = xorOp.getOperand(1);

        // Check RHS is allones constant
        if (auto constOp = rhs.getDefiningOp<hw::ConstantOp>()) {
          if (constOp.getValue().isAllOnes()) {
            invertParity = !invertParity;
            cur = lhs;
            continue;
          }
        }

        // Check LHS is allones constant
        if (auto constOp = lhs.getDefiningOp<hw::ConstantOp>()) {
          if (constOp.getValue().isAllOnes()) {
            invertParity = !invertParity;
            cur = rhs;
            continue;
          }
        }
      }
    }

    // 3. Handle comb.icmp for i1 NOT patterns
    if (auto icmpOp = cur.getDefiningOp<comb::ICmpOp>()) {
      Value lhs = icmpOp.getLhs();
      Value rhs = icmpOp.getRhs();
      auto pred = icmpOp.getPredicate();

      // Try to match: cmp(x, const) or cmp(const, x)
      Value nonConst;
      APInt constVal;
      bool constOnLeft = false;

      if (auto constOp = rhs.getDefiningOp<hw::ConstantOp>()) {
        nonConst = lhs;
        constVal = constOp.getValue();
      } else if (auto constOp = lhs.getDefiningOp<hw::ConstantOp>()) {
        nonConst = rhs;
        constVal = constOp.getValue();
        constOnLeft = true;
      }

      if (nonConst) {
        // Only handle i1 comparisons with 0/1
        if (constVal.getBitWidth() == 1) {
          bool constIsZero = constVal.isZero();
          bool constIsOne = constVal.isOne();

          if (constIsZero || constIsOne) {
            // Normalize predicate if constant is on left
            auto effectivePred = pred;
            if (constOnLeft) {
              // Swap predicate: const cmp x -> x cmp_rev const
              if (pred == comb::ICmpPredicate::eq) effectivePred = comb::ICmpPredicate::eq;
              else if (pred == comb::ICmpPredicate::ne) effectivePred = comb::ICmpPredicate::ne;
              // For eq/ne, swap doesn't change semantics
            }

            // i1 semantics:
            // x == 0 -> !x  (invert)
            // x != 0 -> x   (no change)
            // x == 1 -> x   (no change)
            // x != 1 -> !x  (invert)
            if (effectivePred == comb::ICmpPredicate::eq && constIsZero) {
              invertParity = !invertParity;
              cur = nonConst;
              continue;
            } else if (effectivePred == comb::ICmpPredicate::ne && constIsZero) {
              // x != 0 is identity for i1
              cur = nonConst;
              continue;
            } else if (effectivePred == comb::ICmpPredicate::eq && constIsOne) {
              // x == 1 is identity for i1
              cur = nonConst;
              continue;
            } else if (effectivePred == comb::ICmpPredicate::ne && constIsOne) {
              invertParity = !invertParity;
              cur = nonConst;
              continue;
            }
          }
        }
      }
    }

    // No more NOT layers to peel
    break;
  }

  // Now try to extract signal name from llhd.prb
  if (auto prbOp = cur.getDefiningOp<llhd::PrbOp>()) {
    Value sig = prbOp.getSignal();

    // Unwrap SigExtract chains
    sig = unwrapSigExtract(sig);

    // Try to get name from llhd.sig (SignalOp) using type-based check
    if (auto sigOp = sig.getDefiningOp<llhd::SignalOp>()) {
      if (auto nameAttr = sigOp.getNameAttr()) {
        return {nameAttr.getValue(), invertParity};
      }
    }

    // Fallback: BlockArgument (module port) - try to get name from hw.module
    if (auto blockArg = dyn_cast<BlockArgument>(sig)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();

      // Try hw.module
      if (auto hwMod = dyn_cast<hw::HWModuleOp>(parentOp)) {
        unsigned idx = blockArg.getArgNumber();
        StringRef argName = hwMod.getArgName(idx);
        if (!argName.empty()) {
          return {argName, invertParity};
        }
      }
      // Note: Other module types (llhd.entity, etc.) not currently supported
    }
  }

  return {StringRef(), false};
}

struct DffDemoPass : public ::impl::DffDemoBase<DffDemoPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::errs() << "[Analysis] Starting APB Control Signal Inference...\n";

    // 收集路径信息
    struct PathInfo {
      SmallVector<std::pair<StringRef, bool>> controls;
      bool isWrite;
      comb::AndOp andOp;
    };
    SmallVector<PathInfo, 8> allPaths;

    mod.walk([&](comb::AndOp andOp) {
      SmallVector<std::pair<StringRef, bool>> controls;
      llvm::SmallDenseMap<StringRef, bool> seen;

      bool hasPwrite = false;
      bool pwriteVal = true;

      for (Value operand : andOp.getOperands()) {
        auto [name, isInv] = traceSignal(operand);
        if (name.empty())
          continue;
        if (seen.count(name))
          continue;

        bool requiredVal = !isInv;
        if (name == "pwrite") {
          hasPwrite = true;
          pwriteVal = requiredVal;
          seen.insert({name, requiredVal});
          continue;
        }

        controls.push_back({name, requiredVal});
        seen.insert({name, requiredVal});
      }

      if (!hasPwrite)
        return;

      llvm::errs() << "\n[Found Trigger Condition] at " << andOp.getLoc() << "\n";
      llvm::errs() << "  Path type: " << (pwriteVal ? "Write (pwrite=1)" : "Read (pwrite=0)") << "\n";
      llvm::errs() << "  To activate this path, set:\n";
      llvm::errs() << "    pwrite = " << (pwriteVal ? "1" : "0") << "\n";
      for (auto [n, v] : controls)
        llvm::errs() << "    " << n << " = " << (v ? "1" : "0") << "\n";

      // 保存路径信息
      PathInfo info;
      info.controls = controls;
      info.isWrite = pwriteVal;
      info.andOp = andOp;
      allPaths.push_back(info);

      if (!andOp.getResult().hasOneUse())
        return;

      Operation *user = *andOp.getResult().getUsers().begin();
      if (auto brOp = dyn_cast<cf::CondBranchOp>(user)) {
        llvm::errs() << "  -> Controls a branch (likely State Transition)\n";

        Block *trueBlock = brOp.getTrueDest();
        std::vector<Block *> worklist;
        worklist.push_back(trueBlock);
        llvm::SmallPtrSet<Block *, 8> visited;

        while (!worklist.empty()) {
          Block *bb = worklist.back();
          worklist.pop_back();
          if (!visited.insert(bb).second)
            continue;

          for (Operation &op : *bb) {
            if (auto drv = dyn_cast<llhd::DrvOp>(&op)) {
              Value target = drv.getSignal();
              Value value = drv.getValue();

              StringRef signalName = "(unnamed)";
              if (auto sigOp = target.getDefiningOp()) {
                if (auto nameAttr = sigOp->getAttrOfType<StringAttr>("name"))
                  signalName = nameAttr.getValue();
              }

              llvm::errs() << "     [Action] Drives Signal: " << signalName;
              if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
                if (constOp.getValue().isAllOnes())
                  llvm::errs() << " to TRUE (Active)\n";
                else if (constOp.getValue().isZero())
                  llvm::errs() << " to FALSE (Inactive)\n";
                else
                  llvm::errs() << " to Constant " << constOp.getValue() << "\n";
              } else {
                llvm::errs() << " to Dynamic Value\n";
              }
            }

            if (auto br = dyn_cast<cf::BranchOp>(op))
              worklist.push_back(br.getDest());
            if (auto condBr = dyn_cast<cf::CondBranchOp>(op)) {
              worklist.push_back(condBr.getTrueDest());
              worklist.push_back(condBr.getFalseDest());
            }
          }

          if (visited.size() > 5)
            break;
        }
      }
    });

    // === 优化部分 ===
    llvm::errs() << "\n[Optimization] Analyzing phase-agnostic signals...\n";

    // 统计每个信号在读/写路径中的取值
    llvm::DenseMap<StringRef, std::pair<int, int>> writeSignalVals;
    llvm::DenseMap<StringRef, std::pair<int, int>> readSignalVals;

    for (auto &path : allPaths) {
      auto &targetMap = path.isWrite ? writeSignalVals : readSignalVals;
      for (auto [name, val] : path.controls) {
        auto &counts = targetMap[name];
        if (val)
          counts.first++;
        else
          counts.second++;
      }
    }

    // 找出恒定信号
    llvm::DenseMap<StringRef, bool> writeConstSignals;
    llvm::DenseMap<StringRef, bool> readConstSignals;

    for (auto &[name, counts] : writeSignalVals) {
      if (counts.first > 0 && counts.second == 0)
        writeConstSignals[name] = true;
      else if (counts.first == 0 && counts.second > 0)
        writeConstSignals[name] = false;
    }

    for (auto &[name, counts] : readSignalVals) {
      if (counts.first > 0 && counts.second == 0)
        readConstSignals[name] = true;
      else if (counts.first == 0 && counts.second > 0)
        readConstSignals[name] = false;
    }

    if (!writeConstSignals.empty()) {
      llvm::errs() << "\n  [Write Path] Phase-agnostic signals:\n";
      for (auto &[name, val] : writeConstSignals)
        llvm::errs() << "    " << name << " = " << (val ? "1" : "0") << "\n";
    }

    if (!readConstSignals.empty()) {
      llvm::errs() << "\n  [Read Path] Phase-agnostic signals:\n";
      for (auto &[name, val] : readConstSignals)
        llvm::errs() << "    " << name << " = " << (val ? "1" : "0") << "\n";
    }

    // 执行替换
    int optimizedCount = 0;
    OpBuilder builder(mod);

    for (auto &path : allPaths) {
      auto &constMap = path.isWrite ? writeConstSignals : readConstSignals;
      if (constMap.empty())
        continue;

      builder.setInsertionPoint(path.andOp);
      SmallVector<Value, 4> newOperands;
      bool changed = false;

      for (Value operand : path.andOp.getOperands()) {
        auto [name, isInv] = traceSignal(operand);

        if (!name.empty() && constMap.count(name)) {
          bool constVal = constMap[name];
          bool finalVal = isInv ? !constVal : constVal;

          auto constOp = builder.create<hw::ConstantOp>(
            path.andOp.getLoc(),
            APInt(1, finalVal ? 1 : 0)
          );
          newOperands.push_back(constOp);
          changed = true;
          optimizedCount++;
        } else {
          newOperands.push_back(operand);
        }
      }

      if (changed) {
        auto newAnd = builder.create<comb::AndOp>(
          path.andOp.getLoc(),
          newOperands,
          false
        );
        path.andOp.getResult().replaceAllUsesWith(newAnd.getResult());
        path.andOp.erase();
      }
    }

    if (optimizedCount > 0)
      llvm::errs() << "\n[Optimization] ✓ Replaced " << optimizedCount
                   << " signal(s) with constants\n";
    else
      llvm::errs() << "\n[Optimization] No optimization needed\n";

    // === 死代码消除（DCE）部分 - 迭代执行 ===
    llvm::errs() << "\n[DCE] Removing dead code...\n";

    int totalRemovedProbes = 0;
    int totalRemovedXors = 0;
    int totalRemovedSignals = 0;
    int iteration = 0;
    bool madeProgress = true;

    while (madeProgress && iteration < 10) {
      madeProgress = false;
      iteration++;

      // 移除未使用的 XOR 操作
      SmallVector<Operation*, 8> deadXors;
      mod.walk([&](comb::XorOp xorOp) {
        if (xorOp.getResult().use_empty()) {
          deadXors.push_back(xorOp);
        }
      });

      for (Operation* op : deadXors) {
        op->erase();
        totalRemovedXors++;
        madeProgress = true;
      }

      // 移除未使用的 probe 操作
      SmallVector<Operation*, 8> deadProbes;
      mod.walk([&](llhd::PrbOp prbOp) {
        if (prbOp.getResult().use_empty()) {
          deadProbes.push_back(prbOp);
        }
      });

      for (Operation* op : deadProbes) {
        op->erase();
        totalRemovedProbes++;
        madeProgress = true;
      }

      // 移除未使用的信号定义
      SmallVector<Operation*, 8> deadSignals;
      mod.walk([&](Operation* op) {
        if (op->getName().getStringRef() == "llhd.sig") {
          if (op->getResult(0).use_empty()) {
            deadSignals.push_back(op);
          }
        }
      });

      for (Operation* op : deadSignals) {
        op->erase();
        totalRemovedSignals++;
        madeProgress = true;
      }
    }

    if (totalRemovedProbes > 0 || totalRemovedXors > 0 || totalRemovedSignals > 0) {
      llvm::errs() << "[DCE] ✓ Removed " << totalRemovedProbes << " dead probe(s), "
                   << totalRemovedXors << " dead XOR(s), and "
                   << totalRemovedSignals << " dead signal(s) in "
                   << iteration << " iteration(s)\n";
    } else {
      llvm::errs() << "[DCE] No dead code found\n";
    }
  }
};

} // namespace
