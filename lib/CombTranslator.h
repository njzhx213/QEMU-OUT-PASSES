#ifndef QEMU_PASSES_COMB_TRANSLATOR_H
#define QEMU_PASSES_COMB_TRANSLATOR_H

//===----------------------------------------------------------------------===//
// CombTranslator.h - Comb Dialect to C Expression Translator
//
// Ported from qemu-output/src/lib/CombTranslator.h
// This is a direct port to ensure behavioral consistency with the old framework.
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include <string>
#include <optional>

namespace comb_translator {

//===----------------------------------------------------------------------===//
// Comb Dialect to C Expression Translator
//
// Supported operations:
//   Arithmetic: add, sub, mul, divs, divu, mods, modu
//   Bitwise: and, or, xor, shl, shrs, shru
//   Compare: icmp (eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge)
//   Data: extract, concat, replicate, mux, reverse, parity
//
// Reference: https://circt.llvm.org/docs/Dialects/Comb/
//===----------------------------------------------------------------------===//

/// Translation result
struct TranslateResult {
  bool success;           // Whether translation succeeded
  std::string expr;       // C expression string
  std::string errorMsg;   // Error message on failure

  static TranslateResult ok(const std::string &e) {
    return {true, e, ""};
  }
  static TranslateResult fail(const std::string &msg) {
    return {false, "", msg};
  }
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get name from SSA value (for signals without name attribute)
inline std::string getSSANameFromValue(mlir::Value val) {
  if (auto opResult = mlir::dyn_cast<mlir::OpResult>(val)) {
    mlir::Operation *defOp = opResult.getOwner();
    if (auto nameAttr = defOp->getAttrOfType<mlir::StringAttr>("name")) {
      return nameAttr.getValue().str();
    }
    if (auto symName = defOp->getAttrOfType<mlir::StringAttr>("sym_name")) {
      return symName.getValue().str();
    }
    std::string str;
    llvm::raw_string_ostream os(str);
    val.print(os);
    os.flush();
    if (!str.empty() && str[0] == '%') {
      size_t end = str.find_first_of(" :");
      if (end != std::string::npos) {
        return str.substr(1, end - 1);
      }
      return str.substr(1);
    }
  }
  return "";
}

/// Bit extraction pattern detection result
struct BitExtractPattern {
  bool isPattern;           // Whether it matches bit extraction pattern
  std::string signalName;   // Extracted signal name
  bool usesBlockArgument;   // Whether index is BlockArgument

  static BitExtractPattern none() {
    return {false, "", false};
  }
  static BitExtractPattern match(const std::string &sig, bool usesArg) {
    return {true, sig, usesArg};
  }
};

/// Detect (signal >> index) & 1 pattern
inline BitExtractPattern detectBitExtractPattern(mlir::Value val) {
  // Pattern: comb.extract (signal >> blockArg) from 0 : (i32) -> i1
  if (auto extractOp = val.getDefiningOp<circt::comb::ExtractOp>()) {
    if (extractOp.getLowBit() == 0 &&
        extractOp.getType().getIntOrFloatBitWidth() == 1) {
      mlir::Value input = extractOp.getInput();

      if (auto shruOp = input.getDefiningOp<circt::comb::ShrUOp>()) {
        mlir::Value signal = shruOp.getLhs();
        mlir::Value index = shruOp.getRhs();

        if (auto prbOp = signal.getDefiningOp<circt::llhd::PrbOp>()) {
          mlir::Value sig = prbOp.getSignal();
          std::string sigName;
          if (auto sigOp = sig.getDefiningOp()) {
            if (auto nameAttr = sigOp->getAttrOfType<mlir::StringAttr>("name")) {
              sigName = nameAttr.getValue().str();
            } else {
              sigName = getSSANameFromValue(sig);
            }
          }

          bool usesBlockArg = mlir::isa<mlir::BlockArgument>(index);

          if (!usesBlockArg) {
            if (auto idxPrb = index.getDefiningOp<circt::llhd::PrbOp>()) {
              mlir::Value idxSig = idxPrb.getSignal();
              if (auto idxSigOp = idxSig.getDefiningOp()) {
                std::string idxName;
                if (auto nameAttr = idxSigOp->getAttrOfType<mlir::StringAttr>("name")) {
                  idxName = nameAttr.getValue().str();
                } else {
                  idxName = getSSANameFromValue(idxSig);
                }
                if (idxName.find("int_k") != std::string::npos ||
                    idxName.find("_k") != std::string::npos ||
                    idxName.find("_i") != std::string::npos) {
                  usesBlockArg = true;
                }
              }
            }
          }

          if (!sigName.empty()) {
            return BitExtractPattern::match(sigName, usesBlockArg);
          }
        }
      }
    }
  }
  return BitExtractPattern::none();
}

/// Sanitize signal name (replace . [] etc with _)
inline std::string sanitizeSignalName(const std::string &name) {
  std::string result = name;
  for (char &c : result) {
    if (c == '.' || c == '[' || c == ']' || c == '-' || c == ' ') {
      c = '_';
    }
  }
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0]))) {
    result = "_" + result;
  }
  return result;
}

/// Get signal name (penetrate llhd.prb)
inline std::string getSignalName(mlir::Value val) {
  // Direct llhd.prb
  if (auto prbOp = val.getDefiningOp<circt::llhd::PrbOp>()) {
    mlir::Value sig = prbOp.getSignal();
    if (auto sigOp = sig.getDefiningOp()) {
      if (auto nameAttr = sigOp->getAttrOfType<mlir::StringAttr>("name")) {
        return "s->" + sanitizeSignalName(nameAttr.getValue().str());
      }
      std::string ssaName = getSSANameFromValue(sig);
      if (!ssaName.empty()) {
        return "s->" + sanitizeSignalName(ssaName);
      }
    }
    return "s->unnamed";
  }

  // Constant
  if (auto constOp = val.getDefiningOp<circt::hw::ConstantOp>()) {
    llvm::APInt v = constOp.getValue();
    if (v.getBitWidth() <= 64) {
      return std::to_string(v.getZExtValue());
    }
    llvm::SmallString<64> str;
    v.toStringUnsigned(str, 16);
    return "0x" + str.str().str();
  }

  // BlockArgument
  if (mlir::isa<mlir::BlockArgument>(val)) {
    return "arg" + std::to_string(mlir::cast<mlir::BlockArgument>(val).getArgNumber());
  }

  return "";
}

/// Generate bit mask
inline std::string genMask(unsigned width) {
  if (width >= 64) {
    return "0xFFFFFFFFFFFFFFFFULL";
  }
  uint64_t mask = (1ULL << width) - 1;
  if (mask <= 0xFFFF) {
    return "0x" + llvm::utohexstr(mask);
  }
  return "0x" + llvm::utohexstr(mask) + "ULL";
}

//===----------------------------------------------------------------------===//
// Expression Translator (recursive)
//===----------------------------------------------------------------------===//

/// Recursively translate Value to C expression
inline TranslateResult translateValue(mlir::Value val, unsigned maxDepth = 10) {
  if (maxDepth == 0) {
    return TranslateResult::fail("max recursion depth exceeded");
  }

  // 1. Get name directly (signal or constant)
  std::string name = getSignalName(val);
  if (!name.empty()) {
    return TranslateResult::ok(name);
  }

  // 2. Get defining operation
  mlir::Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    return TranslateResult::fail("no defining op");
  }

  // ========== Arithmetic operations ==========

  // comb.add
  if (auto addOp = mlir::dyn_cast<circt::comb::AddOp>(defOp)) {
    std::string result;
    for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
      auto sub = translateValue(addOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " + ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.sub
  if (auto subOp = mlir::dyn_cast<circt::comb::SubOp>(defOp)) {
    auto lhs = translateValue(subOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(subOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") - (" + rhs.expr + ")");
  }

  // comb.mul
  if (auto mulOp = mlir::dyn_cast<circt::comb::MulOp>(defOp)) {
    std::string result;
    for (unsigned i = 0; i < mulOp.getNumOperands(); ++i) {
      auto sub = translateValue(mulOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " * ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.divu
  if (auto divuOp = mlir::dyn_cast<circt::comb::DivUOp>(defOp)) {
    auto lhs = translateValue(divuOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(divuOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") / (" + rhs.expr + ")");
  }

  // comb.divs
  if (auto divsOp = mlir::dyn_cast<circt::comb::DivSOp>(defOp)) {
    auto lhs = translateValue(divsOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(divsOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("((int64_t)(" + lhs.expr + ")) / ((int64_t)(" + rhs.expr + "))");
  }

  // comb.modu
  if (auto moduOp = mlir::dyn_cast<circt::comb::ModUOp>(defOp)) {
    auto lhs = translateValue(moduOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(moduOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") % (" + rhs.expr + ")");
  }

  // comb.mods
  if (auto modsOp = mlir::dyn_cast<circt::comb::ModSOp>(defOp)) {
    auto lhs = translateValue(modsOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(modsOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("((int64_t)(" + lhs.expr + ")) % ((int64_t)(" + rhs.expr + "))");
  }

  // ========== Bitwise operations ==========

  // comb.and
  if (auto andOp = mlir::dyn_cast<circt::comb::AndOp>(defOp)) {
    std::string result;
    for (unsigned i = 0; i < andOp.getNumOperands(); ++i) {
      auto sub = translateValue(andOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " & ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.or
  if (auto orOp = mlir::dyn_cast<circt::comb::OrOp>(defOp)) {
    std::string result;
    for (unsigned i = 0; i < orOp.getNumOperands(); ++i) {
      auto sub = translateValue(orOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " | ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.xor
  if (auto xorOp = mlir::dyn_cast<circt::comb::XorOp>(defOp)) {
    // Special case: XOR 1 = NOT
    if (xorOp.getNumOperands() == 2) {
      if (auto constOp = xorOp.getOperand(1).getDefiningOp<circt::hw::ConstantOp>()) {
        if (constOp.getValue().isAllOnes()) {
          auto inner = translateValue(xorOp.getOperand(0), maxDepth - 1);
          if (!inner.success) return inner;
          return TranslateResult::ok("~(" + inner.expr + ")");
        }
      }
    }
    std::string result;
    for (unsigned i = 0; i < xorOp.getNumOperands(); ++i) {
      auto sub = translateValue(xorOp.getOperand(i), maxDepth - 1);
      if (!sub.success) return sub;
      if (i > 0) result += " ^ ";
      result += "(" + sub.expr + ")";
    }
    return TranslateResult::ok(result);
  }

  // comb.shl
  if (auto shlOp = mlir::dyn_cast<circt::comb::ShlOp>(defOp)) {
    auto lhs = translateValue(shlOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(shlOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") << (" + rhs.expr + ")");
  }

  // comb.shru
  if (auto shruOp = mlir::dyn_cast<circt::comb::ShrUOp>(defOp)) {
    auto lhs = translateValue(shruOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(shruOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("(" + lhs.expr + ") >> (" + rhs.expr + ")");
  }

  // comb.shrs
  if (auto shrsOp = mlir::dyn_cast<circt::comb::ShrSOp>(defOp)) {
    auto lhs = translateValue(shrsOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(shrsOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;
    return TranslateResult::ok("((int64_t)(" + lhs.expr + ")) >> (" + rhs.expr + ")");
  }

  // ========== Data operations ==========

  // comb.extract
  if (auto extractOp = mlir::dyn_cast<circt::comb::ExtractOp>(defOp)) {
    auto input = translateValue(extractOp.getInput(), maxDepth - 1);
    if (!input.success) return input;

    unsigned lowBit = extractOp.getLowBit();
    unsigned width = extractOp.getType().getIntOrFloatBitWidth();

    if (lowBit == 0 && width == 1) {
      return TranslateResult::ok("((" + input.expr + ") & 1)");
    } else if (lowBit == 0) {
      return TranslateResult::ok("((" + input.expr + ") & " + genMask(width) + ")");
    } else {
      return TranslateResult::ok("(((" + input.expr + ") >> " +
                                  std::to_string(lowBit) + ") & " + genMask(width) + ")");
    }
  }

  // comb.concat
  if (auto concatOp = mlir::dyn_cast<circt::comb::ConcatOp>(defOp)) {
    if (concatOp.getNumOperands() == 0) {
      return TranslateResult::fail("empty concat");
    }

    std::string result;
    unsigned accumulatedWidth = 0;

    for (int i = concatOp.getNumOperands() - 1; i >= 0; --i) {
      mlir::Value operand = concatOp.getOperand(i);
      auto sub = translateValue(operand, maxDepth - 1);
      if (!sub.success) return sub;

      unsigned opWidth = operand.getType().getIntOrFloatBitWidth();

      if (result.empty()) {
        result = "(" + sub.expr + ")";
      } else {
        result = "((" + sub.expr + ") << " + std::to_string(accumulatedWidth) + ") | " + result;
      }
      accumulatedWidth += opWidth;
    }
    return TranslateResult::ok("(" + result + ")");
  }

  // comb.replicate
  if (auto repOp = mlir::dyn_cast<circt::comb::ReplicateOp>(defOp)) {
    auto input = translateValue(repOp.getInput(), maxDepth - 1);
    if (!input.success) return input;

    unsigned resultWidth = repOp.getType().getIntOrFloatBitWidth();
    std::string allOnes = genMask(resultWidth);

    return TranslateResult::ok("((" + input.expr + ") ? " + allOnes + " : 0)");
  }

  // comb.mux
  if (auto muxOp = mlir::dyn_cast<circt::comb::MuxOp>(defOp)) {
    auto truePattern = detectBitExtractPattern(muxOp.getTrueValue());
    auto falsePattern = detectBitExtractPattern(muxOp.getFalseValue());

    if (truePattern.isPattern && falsePattern.isPattern &&
        truePattern.usesBlockArgument && falsePattern.usesBlockArgument) {
      auto cond = translateValue(muxOp.getCond(), maxDepth - 1);
      if (!cond.success) return cond;
      return TranslateResult::ok("((" + cond.expr + ") ? (s->" +
                                  sanitizeSignalName(truePattern.signalName) + ") : (s->" +
                                  sanitizeSignalName(falsePattern.signalName) + "))");
    }

    auto cond = translateValue(muxOp.getCond(), maxDepth - 1);
    auto trueVal = translateValue(muxOp.getTrueValue(), maxDepth - 1);
    auto falseVal = translateValue(muxOp.getFalseValue(), maxDepth - 1);
    if (!cond.success) return cond;
    if (!trueVal.success) return trueVal;
    if (!falseVal.success) return falseVal;

    return TranslateResult::ok("((" + cond.expr + ") ? (" + trueVal.expr + ") : (" + falseVal.expr + "))");
  }

  // comb.reverse
  if (auto revOp = mlir::dyn_cast<circt::comb::ReverseOp>(defOp)) {
    auto input = translateValue(revOp.getInput(), maxDepth - 1);
    if (!input.success) return input;
    unsigned width = revOp.getType().getIntOrFloatBitWidth();
    return TranslateResult::ok("bit_reverse" + std::to_string(width) + "(" + input.expr + ")");
  }

  // comb.parity
  if (auto parOp = mlir::dyn_cast<circt::comb::ParityOp>(defOp)) {
    auto input = translateValue(parOp.getInput(), maxDepth - 1);
    if (!input.success) return input;
    return TranslateResult::ok("(__builtin_parityll(" + input.expr + "))");
  }

  // ========== Compare operations ==========

  // comb.icmp
  if (auto icmpOp = mlir::dyn_cast<circt::comb::ICmpOp>(defOp)) {
    auto lhs = translateValue(icmpOp.getLhs(), maxDepth - 1);
    auto rhs = translateValue(icmpOp.getRhs(), maxDepth - 1);
    if (!lhs.success) return lhs;
    if (!rhs.success) return rhs;

    std::string op;
    bool isSigned = false;
    switch (icmpOp.getPredicate()) {
      case circt::comb::ICmpPredicate::eq:  op = "=="; break;
      case circt::comb::ICmpPredicate::ne:  op = "!="; break;
      case circt::comb::ICmpPredicate::slt: op = "<";  isSigned = true; break;
      case circt::comb::ICmpPredicate::sle: op = "<="; isSigned = true; break;
      case circt::comb::ICmpPredicate::sgt: op = ">";  isSigned = true; break;
      case circt::comb::ICmpPredicate::sge: op = ">="; isSigned = true; break;
      case circt::comb::ICmpPredicate::ult: op = "<";  break;
      case circt::comb::ICmpPredicate::ule: op = "<="; break;
      case circt::comb::ICmpPredicate::ugt: op = ">";  break;
      case circt::comb::ICmpPredicate::uge: op = ">="; break;
      default:
        return TranslateResult::fail("unknown icmp predicate");
    }

    if (isSigned) {
      return TranslateResult::ok("(((int64_t)(" + lhs.expr + ")) " + op +
                                  " ((int64_t)(" + rhs.expr + ")))");
    }
    return TranslateResult::ok("((" + lhs.expr + ") " + op + " (" + rhs.expr + "))");
  }

  // Unsupported operation
  return TranslateResult::fail("unsupported op: " + defOp->getName().getStringRef().str());
}

/// Translate llhd.drv value to C expression
inline TranslateResult translateDrvValue(circt::llhd::DrvOp drv) {
  return translateValue(drv.getValue());
}

} // namespace comb_translator

#endif // QEMU_PASSES_COMB_TRANSLATOR_H
