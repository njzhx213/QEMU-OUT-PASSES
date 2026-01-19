//===----------------------------------------------------------------------===//
// QEMUEmitCPass.cpp - QEMU C Code Emission Pass
//
// Final pass in the LLHD-to-QEMU conversion pipeline.
// Reads qemu.* metadata from the IR and emits QEMU-compatible C code
// using the existing QEMUCodeGen framework from qemu-output.
//
// Design:
// - Generates ONE QEMU device with flattened state struct
// - Uses hierarchical naming: <ModuleName>__<signalName>
// - Rewrites comb_logic expressions to match sanitized field names
// - All signal names are sanitized with collision detection
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

// Include the existing QEMUCodeGen from qemu-output
#include "QEMUCodeGen.h"
#include "ClkAnalysisResult.h"

#include <map>
#include <set>
#include <regex>

// TableGen-generated pass declaration and definition
#define GEN_PASS_DECL_QEMUEMITC
#define GEN_PASS_DEF_QEMUEMITC
#include "Passes.h.inc"

using namespace mlir;
using namespace circt;

namespace {

//===----------------------------------------------------------------------===//
// Helper: Name Qualification and Sanitization
//===----------------------------------------------------------------------===//

/// Build qualified name with module prefix
static std::string qualifyName(llvm::StringRef moduleName, llvm::StringRef signalName) {
  return moduleName.str() + "__" + signalName.str();
}

/// Basic sanitization (same logic as QEMUCodeGen::sanitizeName)
static std::string basicSanitize(llvm::StringRef name) {
  std::string result = name.str();
  for (char &c : result) {
    if (c == '.' || c == '[' || c == ']' || c == '-' || c == ' ' || c == ':') {
      c = '_';
    }
  }
  if (!result.empty() && std::isdigit(static_cast<unsigned char>(result[0]))) {
    result = "_" + result;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Signal Registry - Tracks all signals with unique sanitized names
//===----------------------------------------------------------------------===//

class SignalRegistry {
public:
  struct SignalInfo {
    std::string qualifiedName;   // e.g., "gpio_ctrl__int_level"
    std::string sanitizedName;   // e.g., "gpio_ctrl__int_level" (after sanitize)
    unsigned bitWidth;
    bool isPort;
    bool isTopLevel;             // true if from top module (no prefix needed)
  };

  /// Register a signal and get its unique sanitized name
  std::string registerSignal(llvm::StringRef qualifiedName, unsigned bitWidth,
                             bool isPort = false, bool isTopLevel = false) {
    // Check if already registered
    auto it = qualifiedToSanitized_.find(qualifiedName.str());
    if (it != qualifiedToSanitized_.end()) {
      return it->second;
    }

    // Sanitize the name
    std::string sanitized = basicSanitize(qualifiedName);

    // Handle collisions
    std::string unique = sanitized;
    unsigned counter = 2;
    while (usedSanitized_.count(unique)) {
      unique = sanitized + "_" + std::to_string(counter++);
    }

    // Record mappings
    qualifiedToSanitized_[qualifiedName.str()] = unique;
    usedSanitized_.insert(unique);

    // Store signal info
    SignalInfo info;
    info.qualifiedName = qualifiedName.str();
    info.sanitizedName = unique;
    info.bitWidth = bitWidth;
    info.isPort = isPort;
    info.isTopLevel = isTopLevel;
    signals_[unique] = info;

    return unique;
  }

  /// Lookup sanitized name for a qualified name
  std::string lookup(llvm::StringRef qualifiedName) const {
    auto it = qualifiedToSanitized_.find(qualifiedName.str());
    if (it != qualifiedToSanitized_.end()) {
      return it->second;
    }
    // Fallback: just sanitize
    return basicSanitize(qualifiedName);
  }

  /// Check if a sanitized name exists
  bool hasSanitized(llvm::StringRef sanitized) const {
    return usedSanitized_.count(sanitized.str()) > 0;
  }

  /// Get all registered signals
  const std::map<std::string, SignalInfo>& getSignals() const {
    return signals_;
  }

  /// Get the qualified->sanitized mapping
  const std::map<std::string, std::string>& getMapping() const {
    return qualifiedToSanitized_;
  }

private:
  std::map<std::string, std::string> qualifiedToSanitized_;
  llvm::StringSet<> usedSanitized_;
  std::map<std::string, SignalInfo> signals_;
};

//===----------------------------------------------------------------------===//
// Expression Rewriter - Rewrites s->field references in expressions
//===----------------------------------------------------------------------===//

class ExpressionRewriter {
public:
  ExpressionRewriter(const SignalRegistry &registry) : registry_(registry) {}

  /// Rewrite all s->field references in an expression
  /// Uses the registry's qualified->sanitized mapping
  std::string rewrite(llvm::StringRef expression, llvm::StringRef modulePrefix) const {
    std::string result = expression.str();

    // Pattern: s->identifier
    // We need to replace identifier with the sanitized qualified name
    std::regex pattern(R"(s->([A-Za-z_][A-Za-z0-9_]*))");
    std::string output;
    std::sregex_iterator it(result.begin(), result.end(), pattern);
    std::sregex_iterator end;

    size_t lastPos = 0;
    while (it != end) {
      std::smatch match = *it;
      // Append text before the match
      output += result.substr(lastPos, match.position() - lastPos);

      // Get the field name
      std::string fieldName = match[1].str();

      // Try to find the qualified name
      // First, try with module prefix
      std::string qualifiedWithPrefix = modulePrefix.str() + fieldName;
      std::string sanitized = registry_.lookup(qualifiedWithPrefix);

      // If not found with prefix, try without (for top-level ports)
      if (!registry_.hasSanitized(sanitized)) {
        sanitized = registry_.lookup(fieldName);
      }

      // If still not found, use basic sanitization
      if (!registry_.hasSanitized(sanitized)) {
        sanitized = basicSanitize(fieldName);
      }

      // Append the rewritten reference
      output += "s->" + sanitized;

      lastPos = match.position() + match.length();
      ++it;
    }

    // Append remaining text
    output += result.substr(lastPos);

    return output;
  }

private:
  const SignalRegistry &registry_;
};

//===----------------------------------------------------------------------===//
// QEMUEmitCPass Implementation
//===----------------------------------------------------------------------===//

struct QEMUEmitCPass
    : public ::impl::QEMUEmitCBase<QEMUEmitCPass> {
  using QEMUEmitCBase::QEMUEmitCBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    llvm::errs() << "\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "QEMU C Code Emission Pass\n";
    llvm::errs() << "========================================\n";

    // Step 1: Find the public top module
    hw::HWModuleOp topModule = nullptr;
    mod.walk([&](hw::HWModuleOp hwMod) {
      if (!hwMod.isPrivate()) {
        if (!topModule) {
          topModule = hwMod;
        }
      }
    });

    if (!topModule) {
      llvm::errs() << "Error: No public hw.module found\n";
      return signalPassFailure();
    }

    std::string devName = deviceName.empty() ? topModule.getName().str() : std::string(deviceName);
    llvm::errs() << "Device name: " << devName << "\n";
    llvm::errs() << "Top module: @" << topModule.getName() << "\n";

    // Step 2: Create signal registry and generator
    SignalRegistry registry;
    qemu_codegen::QEMUDeviceGenerator gen(devName);

    // Step 3: Collect signals from ALL modules
    llvm::errs() << "\nCollecting signals from all modules...\n";

    unsigned totalSignals = 0;
    unsigned totalCombLogic = 0;

    // First pass: collect all signals with qualified names
    mod.walk([&](hw::HWModuleOp hwMod) {
      std::string moduleName = hwMod.getName().str();
      std::string modulePrefix = moduleName + "__";
      bool isTopLevel = (hwMod == topModule);

      llvm::errs() << "  Module: @" << moduleName;
      if (isTopLevel) llvm::errs() << " (top)";
      llvm::errs() << "\n";

      // Collect ports
      for (auto port : hwMod.getPortList()) {
        std::string portName = port.getName().str();
        unsigned bitWidth = 32;
        if (auto intTy = dyn_cast<IntegerType>(port.type)) {
          bitWidth = intTy.getWidth();
        }

        // For top-level module, don't add prefix to ports
        std::string qualifiedName = isTopLevel ? portName : (modulePrefix + portName);

        // Register the signal
        std::string sanitized = registry.registerSignal(qualifiedName, bitWidth,
                                                        /*isPort=*/true,
                                                        /*isTopLevel=*/isTopLevel);
        totalSignals++;
      }

      // Collect llhd.sig operations
      hwMod.walk([&](llhd::SignalOp sigOp) {
        std::string sigName;
        if (auto nameAttr = sigOp.getName()) {
          sigName = nameAttr->str();
        } else {
          return; // Skip unnamed signals
        }

        unsigned bitWidth = 32;
        Type ty = sigOp.getType();
        if (auto inoutTy = dyn_cast<hw::InOutType>(ty)) {
          ty = inoutTy.getElementType();
        }
        if (auto intTy = dyn_cast<IntegerType>(ty)) {
          bitWidth = intTy.getWidth();
        }

        std::string qualifiedName = modulePrefix + sigName;
        registry.registerSignal(qualifiedName, bitWidth, /*isPort=*/false, /*isTopLevel=*/false);
        totalSignals++;
      });

      // Collect comb_logic targets (to ensure they exist in struct)
      if (auto combAttr = hwMod->getAttrOfType<ArrayAttr>("qemu.comb_logic")) {
        for (Attribute elem : combAttr) {
          auto dictAttr = dyn_cast<DictionaryAttr>(elem);
          if (!dictAttr) continue;

          std::string target;
          unsigned bitWidth = 32;

          if (auto targetAttr = dictAttr.getAs<StringAttr>("target")) {
            target = targetAttr.getValue().str();
          }
          if (auto widthAttr = dictAttr.getAs<IntegerAttr>("bitWidth")) {
            bitWidth = widthAttr.getInt();
          }

          if (!target.empty()) {
            std::string qualifiedTarget = modulePrefix + target;
            registry.registerSignal(qualifiedTarget, bitWidth, /*isPort=*/false, /*isTopLevel=*/false);
          }

          // Also parse expression to find referenced fields
          if (auto exprAttr = dictAttr.getAs<StringAttr>("expression")) {
            std::string expr = exprAttr.getValue().str();
            // Extract s->field references
            std::regex pattern(R"(s->([A-Za-z_][A-Za-z0-9_]*))");
            std::sregex_iterator it(expr.begin(), expr.end(), pattern);
            std::sregex_iterator end;
            while (it != end) {
              std::string fieldName = (*it)[1].str();
              std::string qualifiedField = modulePrefix + fieldName;
              // Register with default width (will be updated if we see the actual signal)
              registry.registerSignal(qualifiedField, 32, /*isPort=*/false, /*isTopLevel=*/false);
              ++it;
            }
          }
        }
      }
    });

    llvm::errs() << "Total signals registered: " << totalSignals << "\n";

    // Step 4: Add all signals to generator
    llvm::errs() << "\nAdding signals to generator...\n";
    for (const auto &pair : registry.getSignals()) {
      const auto &info = pair.second;
      gen.addSimpleReg(info.sanitizedName, info.bitWidth);
    }

    // Step 5: Collect and rewrite combinational logic
    llvm::errs() << "\nProcessing combinational logic...\n";
    std::vector<clk_analysis::CombinationalAssignment> combLogic;
    ExpressionRewriter rewriter(registry);

    mod.walk([&](hw::HWModuleOp hwMod) {
      std::string moduleName = hwMod.getName().str();
      std::string modulePrefix = moduleName + "__";

      auto combAttr = hwMod->getAttrOfType<ArrayAttr>("qemu.comb_logic");
      if (!combAttr) return;

      for (Attribute elem : combAttr) {
        auto dictAttr = dyn_cast<DictionaryAttr>(elem);
        if (!dictAttr) continue;

        std::string target;
        std::string expression;
        unsigned bitWidth = 32;

        if (auto targetAttr = dictAttr.getAs<StringAttr>("target")) {
          target = targetAttr.getValue().str();
        }
        if (auto exprAttr = dictAttr.getAs<StringAttr>("expression")) {
          expression = exprAttr.getValue().str();
        }
        if (auto widthAttr = dictAttr.getAs<IntegerAttr>("bitWidth")) {
          bitWidth = widthAttr.getInt();
        }

        if (target.empty() || expression.empty()) continue;

        // Qualify and rewrite
        std::string qualifiedTarget = modulePrefix + target;
        std::string sanitizedTarget = registry.lookup(qualifiedTarget);
        std::string rewrittenExpr = rewriter.rewrite(expression, modulePrefix);

        clk_analysis::CombinationalAssignment assign;
        assign.targetSignal = sanitizedTarget;
        assign.expression = rewrittenExpr;
        assign.bitWidth = bitWidth;
        combLogic.push_back(assign);

        llvm::errs() << "  [" << moduleName << "] " << target << " -> "
                     << sanitizedTarget << "\n";
        llvm::errs() << "    expr: " << rewrittenExpr << "\n";

        totalCombLogic++;
      }
    });

    gen.setCombinationalLogic(combLogic);

    // Step 6: Detect GPIO inputs
    for (auto port : topModule.getPortList()) {
      std::string portName = port.getName().str();
      if (portName.find("gpio") != std::string::npos && port.isInput()) {
        unsigned bitWidth = 32;
        if (auto intTy = dyn_cast<IntegerType>(port.type)) {
          bitWidth = intTy.getWidth();
        }
        gen.addGPIOInputSignal(portName, bitWidth);
      }
    }

    // Step 7: Generate output files
    std::string outDir = outputDir.empty() ? "qemu-passes/generated" : std::string(outputDir);
    llvm::sys::fs::create_directories(outDir);

    std::string headerPath = outDir + "/" + devName + ".h";
    std::string sourcePath = outDir + "/" + devName + ".c";

    llvm::errs() << "\nGenerating output files...\n";

    // Generate header
    {
      std::error_code EC;
      llvm::raw_fd_ostream headerFile(headerPath, EC);
      if (EC) {
        llvm::errs() << "Error: Cannot create " << headerPath << ": "
                     << EC.message() << "\n";
        return signalPassFailure();
      }
      gen.generateHeader(headerFile);
      llvm::errs() << "  Header: " << headerPath << "\n";
    }

    // Generate source
    {
      std::error_code EC;
      llvm::raw_fd_ostream sourceFile(sourcePath, EC);
      if (EC) {
        llvm::errs() << "Error: Cannot create " << sourcePath << ": "
                     << EC.message() << "\n";
        return signalPassFailure();
      }
      gen.generateSource(sourceFile);
      llvm::errs() << "  Source: " << sourcePath << "\n";
    }

    llvm::errs() << "\n========================================\n";
    llvm::errs() << "Summary:\n";
    llvm::errs() << "  Output directory: " << outDir << "\n";
    llvm::errs() << "  Total fields in struct: " << registry.getSignals().size() << "\n";
    llvm::errs() << "  Combinational assignments: " << totalCombLogic << "\n";
    llvm::errs() << "========================================\n";
  }
};

} // anonymous namespace
