//===----------------------------------------------------------------------===//
// CombLogicExtractPass.cpp - Combinational Logic Extraction Pass
//
// This pass extracts combinational logic (llhd.drv outside of llhd.process)
// and translates it to C expressions for QEMU update_state().
//
// Design aligns with qemu-output/src/lib/ClkAnalysisResult.cpp
// Key change: qemu.comb_logic is now per hw.module, not on ModuleOp
//===----------------------------------------------------------------------===//

#include "CombTranslator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include <vector>

// TableGen-generated pass declaration and definition
#define GEN_PASS_DECL_COMBLOGICEXTRACT
#define GEN_PASS_DEF_COMBLOGICEXTRACT
#include "Passes.h.inc"

using namespace mlir;
using namespace circt;

namespace {

//===----------------------------------------------------------------------===//
// Data Structures (aligned with old framework)
//===----------------------------------------------------------------------===//

/// Combinational logic assignment (for update_state)
struct CombinationalAssignment {
  std::string targetSignal;   // Target signal name (sanitized)
  std::string expression;     // C expression string
  unsigned bitWidth;          // Signal bit width
};

/// Per-module combinational logic collection
struct ModuleCombLogic {
  std::string moduleName;
  std::vector<CombinationalAssignment> assignments;
};

/// Action types for expression generation
enum class ActionType {
  ASSIGN_CONST,     // target = constant
  ASSIGN_SIGNAL,    // target = source_signal
  ACCUMULATE,       // target = target +/- step
  COMPUTE,          // target = complex_expression
  UNKNOWN
};

/// Event action (simplified from old framework)
struct EventAction {
  ActionType type = ActionType::UNKNOWN;
  std::string targetSignal;
  std::string expression;
  std::string sourceSignal;
  int64_t constValue = 0;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get signal name from llhd.sig or llhd.sig.extract
static std::string getSignalNameFromValue(Value signal) {
  // Handle llhd.sig.extract
  if (auto sigExtract = signal.getDefiningOp<llhd::SigExtractOp>()) {
    Value baseSig = sigExtract.getInput();
    if (auto sigOp = baseSig.getDefiningOp<llhd::SignalOp>()) {
      if (auto name = sigOp.getName()) {
        return name->str();
      }
    }
    return "unnamed";
  }

  // Regular signal
  if (auto sigOp = signal.getDefiningOp<llhd::SignalOp>()) {
    if (auto name = sigOp.getName()) {
      return name->str();
    }
    // Fallback to SSA name
    std::string str;
    llvm::raw_string_ostream os(str);
    signal.print(os);
    os.flush();
    if (!str.empty() && str[0] == '%') {
      size_t end = str.find_first_of(" :");
      if (end != std::string::npos) {
        return str.substr(1, end - 1);
      }
      return str.substr(1);
    }
  }
  return "unnamed";
}

/// Get signal bit width
static unsigned getSignalBitWidth(Value signal) {
  Type ty = signal.getType();
  if (auto inoutTy = dyn_cast<hw::InOutType>(ty)) {
    ty = inoutTy.getElementType();
  }
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    return intTy.getWidth();
  }
  return 0;
}

/// Try to generate action from drv operation (aligned with old framework)
static EventAction tryGenerateAction(llhd::DrvOp drv) {
  EventAction action;
  action.targetSignal = getSignalNameFromValue(drv.getSignal());

  Value value = drv.getValue();

  // Check if constant assignment
  if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
    action.type = ActionType::ASSIGN_CONST;
    action.constValue = constOp.getValue().getSExtValue();
    return action;
  }

  // Check if simple signal assignment
  if (auto prb = value.getDefiningOp<llhd::PrbOp>()) {
    action.type = ActionType::ASSIGN_SIGNAL;
    action.sourceSignal = getSignalNameFromValue(prb.getSignal());
    return action;
  }

  // Check if accumulate (signal = signal + const)
  if (auto addOp = value.getDefiningOp<comb::AddOp>()) {
    for (Value operand : addOp.getOperands()) {
      if (auto prb = operand.getDefiningOp<llhd::PrbOp>()) {
        std::string prbSig = getSignalNameFromValue(prb.getSignal());
        if (prbSig == action.targetSignal) {
          for (Value op2 : addOp.getOperands()) {
            if (auto constOp = op2.getDefiningOp<hw::ConstantOp>()) {
              action.type = ActionType::ACCUMULATE;
              action.constValue = constOp.getValue().getSExtValue();
              return action;
            }
          }
        }
      }
    }
  }

  // Check if subtract (signal = signal - const)
  if (auto subOp = value.getDefiningOp<comb::SubOp>()) {
    Value lhs = subOp.getLhs();
    Value rhs = subOp.getRhs();
    if (auto prb = lhs.getDefiningOp<llhd::PrbOp>()) {
      std::string prbSig = getSignalNameFromValue(prb.getSignal());
      if (prbSig == action.targetSignal) {
        if (auto constOp = rhs.getDefiningOp<hw::ConstantOp>()) {
          action.type = ActionType::ACCUMULATE;
          action.constValue = -constOp.getValue().getSExtValue();
          return action;
        }
      }
    }
  }

  // Use CombTranslator for complex expressions
  auto translateResult = comb_translator::translateValue(value);
  if (translateResult.success) {
    action.type = ActionType::COMPUTE;
    action.expression = action.targetSignal + " = " + translateResult.expr;
    return action;
  }

  // Truly complex expression (cannot generate)
  action.type = ActionType::COMPUTE;
  action.expression = "/* complex expression: " + translateResult.errorMsg + " */";
  return action;
}

/// Check if action is complex (cannot generate code)
static bool isComplexAction(const EventAction &action) {
  return action.type == ActionType::COMPUTE &&
         action.expression.find("/* complex expression") != std::string::npos;
}

/// Escape string for JSON output
static std::string escapeJsonString(const std::string &s) {
  std::string result;
  for (char c : s) {
    switch (c) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default: result += c; break;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// CombLogicExtractPass Implementation
//===----------------------------------------------------------------------===//

struct CombLogicExtractPass
    : public ::impl::CombLogicExtractBase<CombLogicExtractPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    // Per-module collection for JSON output
    std::vector<ModuleCombLogic> allModules;
    unsigned totalAssignments = 0;

    llvm::errs() << "\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "Combinational Logic Extraction Pass\n";
    llvm::errs() << "(Per-module scope - aligned with old framework)\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "\n";

    // Process each hw.module separately
    mod.walk([&](hw::HWModuleOp hwMod) {
      llvm::errs() << "Module: @" << hwMod.getName() << "\n";

      // Per-module data structures
      ModuleCombLogic moduleComb;
      moduleComb.moduleName = hwMod.getName().str();

      llvm::StringSet<> modulePortNames;
      llvm::StringSet<> processedSignals;
      SmallVector<Attribute> combAttrs;

      // Collect module port names for filtering
      for (auto port : hwMod.getPortList()) {
        modulePortNames.insert(port.getName());
      }

      // Walk all drv operations in this module
      hwMod.walk([&](llhd::DrvOp drv) {
        // Check if inside ProcessOp (sequential logic - skip)
        Operation *parent = drv->getParentOp();
        while (parent && !isa<llhd::ProcessOp>(parent) && !isa<hw::HWModuleOp>(parent)) {
          parent = parent->getParentOp();
        }
        if (isa<llhd::ProcessOp>(parent)) {
          return;  // Inside process - sequential logic, skip
        }

        // This is combinational logic (outside process)
        Value signal = drv.getSignal();
        std::string sigName = getSignalNameFromValue(signal);

        // Skip if already processed (within this module)
        if (processedSignals.contains(sigName)) {
          return;
        }
        processedSignals.insert(sigName);

        // Skip if module port
        if (modulePortNames.contains(sigName)) {
          llvm::errs() << "  [SKIP] " << sigName << " (module port)\n";
          return;
        }

        // Generate action
        EventAction action = tryGenerateAction(drv);

        // Skip if complex
        if (isComplexAction(action)) {
          llvm::errs() << "  [COMPLEX] " << sigName << "\n";
          llvm::errs() << "    -> " << action.expression << "\n";
          return;
        }

        // Generate expression
        std::string expr;
        switch (action.type) {
          case ActionType::ASSIGN_CONST:
            expr = std::to_string(action.constValue);
            break;
          case ActionType::ASSIGN_SIGNAL:
            expr = "s->" + comb_translator::sanitizeSignalName(action.sourceSignal);
            break;
          case ActionType::COMPUTE:
            // Extract expression part from "target = expr"
            if (action.expression.find(" = ") != std::string::npos) {
              expr = action.expression.substr(action.expression.find(" = ") + 3);
            } else {
              expr = action.expression;
            }
            break;
          case ActionType::ACCUMULATE:
            // For accumulate, generate s->sig + step
            if (action.constValue >= 0) {
              expr = "s->" + comb_translator::sanitizeSignalName(sigName) +
                     " + " + std::to_string(action.constValue);
            } else {
              expr = "s->" + comb_translator::sanitizeSignalName(sigName) +
                     " - " + std::to_string(-action.constValue);
            }
            break;
          default:
            return;
        }

        if (!expr.empty()) {
          // Add to per-module assignment list
          CombinationalAssignment combAssign;
          combAssign.targetSignal = sigName;
          combAssign.expression = expr;
          combAssign.bitWidth = getSignalBitWidth(signal);
          moduleComb.assignments.push_back(combAssign);

          // Build attribute for this entry
          std::string sanitizedTarget = comb_translator::sanitizeSignalName(sigName);
          SmallVector<NamedAttribute> fields;
          fields.push_back(NamedAttribute(
              StringAttr::get(ctx, "target"),
              StringAttr::get(ctx, sanitizedTarget)));
          fields.push_back(NamedAttribute(
              StringAttr::get(ctx, "expression"),
              StringAttr::get(ctx, expr)));
          fields.push_back(NamedAttribute(
              StringAttr::get(ctx, "bitWidth"),
              IntegerAttr::get(IntegerType::get(ctx, 32), combAssign.bitWidth)));
          combAttrs.push_back(DictionaryAttr::get(ctx, fields));

          llvm::errs() << "  [COMB] " << sigName << " = " << expr << "\n";
        }
      });

      // Set attribute on this hw.module (not on ModuleOp!)
      if (!combAttrs.empty()) {
        hwMod->setAttr("qemu.comb_logic", ArrayAttr::get(ctx, combAttrs));
        llvm::errs() << "  -> Added qemu.comb_logic with " << combAttrs.size() << " entries\n";
      }

      // Collect for JSON output
      if (!moduleComb.assignments.empty()) {
        totalAssignments += moduleComb.assignments.size();
        allModules.push_back(std::move(moduleComb));
      }
    });

    // Summary
    llvm::errs() << "\n";
    llvm::errs() << "----------------------------------------\n";
    llvm::errs() << "Summary:\n";
    llvm::errs() << "  Modules with comb logic: " << allModules.size() << "\n";
    llvm::errs() << "  Total assignments: " << totalAssignments << "\n";
    llvm::errs() << "----------------------------------------\n";

    // Output: Save as JSON to results folder (per-module grouped format)
    if (!allModules.empty()) {
      // Ensure results directory exists
      llvm::sys::fs::create_directories("results");

      std::string resultsPath = "results/comb_logic_extract.json";
      std::error_code EC;
      llvm::raw_fd_ostream outFile(resultsPath, EC);

      if (!EC) {
        outFile << "{\n";
        outFile << "  \"modules\": [\n";

        for (size_t m = 0; m < allModules.size(); ++m) {
          const auto &modComb = allModules[m];
          outFile << "    {\n";
          outFile << "      \"module\": \"" << escapeJsonString(modComb.moduleName) << "\",\n";
          outFile << "      \"combinational_logic\": [\n";

          for (size_t i = 0; i < modComb.assignments.size(); ++i) {
            const auto &assign = modComb.assignments[i];
            std::string sanitizedTarget = comb_translator::sanitizeSignalName(assign.targetSignal);
            outFile << "        {\n";
            outFile << "          \"target\": \"" << escapeJsonString(sanitizedTarget) << "\",\n";
            outFile << "          \"expression\": \"" << escapeJsonString(assign.expression) << "\",\n";
            outFile << "          \"bitWidth\": " << assign.bitWidth << ",\n";
            outFile << "          \"c_code\": \"s->" << escapeJsonString(sanitizedTarget)
                    << " = " << escapeJsonString(assign.expression) << ";\"\n";
            outFile << "        }";
            if (i < modComb.assignments.size() - 1) {
              outFile << ",";
            }
            outFile << "\n";
          }

          outFile << "      ]\n";
          outFile << "    }";
          if (m < allModules.size() - 1) {
            outFile << ",";
          }
          outFile << "\n";
        }

        outFile << "  ]\n";
        outFile << "}\n";
        llvm::errs() << "\nJSON output written to: " << resultsPath << "\n";
      } else {
        llvm::errs() << "\nWarning: Could not write JSON to " << resultsPath
                     << ": " << EC.message() << "\n";
      }
    }

    // NOTE: We do NOT set any attribute on the top-level ModuleOp
    // Each hw.module now has its own qemu.comb_logic attribute
  }
};

} // anonymous namespace
