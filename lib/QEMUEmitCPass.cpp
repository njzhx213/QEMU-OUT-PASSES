//===----------------------------------------------------------------------===//
// QEMUEmitCPass.cpp - QEMU C Code Emission Pass
//
// Final pass in the LLHD-to-QEMU conversion pipeline.
// Reads qemu.* metadata from the IR and emits QEMU-compatible C code
// using the existing QEMUCodeGen framework from qemu-output.
//
// Design (aligned with original qemu-output architecture):
// - Uses FLAT naming (no hierarchical prefix)
// - Only APB-mapped registers appear in MMIO read/write
// - All signals exist in state struct, but MMIO only exposes APB registers
// - update_state() handles only combinational logic
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
// Helper: Name Sanitization (FLAT - no hierarchy)
//===----------------------------------------------------------------------===//

/// Sanitize signal name to valid C identifier (no module prefix)
static std::string sanitizeName(llvm::StringRef name) {
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
// Flat Signal Registry - FLAT naming (aligned with qemu-output architecture)
//===----------------------------------------------------------------------===//

class FlatSignalRegistry {
public:
  struct SignalInfo {
    std::string originalName;
    std::string sanitizedName;
    unsigned bitWidth;
  };

  /// Register a signal with flat name (no module prefix)
  std::string registerSignal(llvm::StringRef signalName, unsigned bitWidth) {
    std::string key = signalName.str();
    auto it = nameToSanitized_.find(key);
    if (it != nameToSanitized_.end()) {
      return it->second;
    }

    std::string sanitized = sanitizeName(signalName);

    // Handle collisions
    std::string unique = sanitized;
    unsigned counter = 2;
    while (usedNames_.count(unique)) {
      unique = sanitized + "_" + std::to_string(counter++);
    }

    nameToSanitized_[key] = unique;
    usedNames_.insert(unique);

    SignalInfo info;
    info.originalName = key;
    info.sanitizedName = unique;
    info.bitWidth = bitWidth;
    signals_[unique] = info;

    return unique;
  }

  std::string lookup(llvm::StringRef name) const {
    auto it = nameToSanitized_.find(name.str());
    if (it != nameToSanitized_.end()) {
      return it->second;
    }
    return sanitizeName(name);
  }

  const std::map<std::string, SignalInfo>& getSignals() const {
    return signals_;
  }

private:
  std::map<std::string, std::string> nameToSanitized_;
  llvm::StringSet<> usedNames_;
  std::map<std::string, SignalInfo> signals_;
};

//===----------------------------------------------------------------------===//
// APB Mapping Extraction
//===----------------------------------------------------------------------===//

/// Extract APB register mappings from IR attributes or use default GPIO layout
static std::vector<clk_analysis::APBRegisterMapping>
extractAPBMappings(hw::HWModuleOp topModule, ModuleOp mod) {
  std::vector<clk_analysis::APBRegisterMapping> mappings;

  // Try to get from IR attribute
  if (auto apbAttr = topModule->getAttrOfType<ArrayAttr>("qemu.apb_mappings")) {
    for (Attribute elem : apbAttr) {
      auto dictAttr = dyn_cast<DictionaryAttr>(elem);
      if (!dictAttr) continue;

      clk_analysis::APBRegisterMapping mapping;

      if (auto addrAttr = dictAttr.getAs<IntegerAttr>("address")) {
        mapping.address = addrAttr.getInt();
      }
      if (auto nameAttr = dictAttr.getAs<StringAttr>("register")) {
        mapping.registerName = nameAttr.getValue().str();
      }
      if (auto widthAttr = dictAttr.getAs<IntegerAttr>("bitWidth")) {
        mapping.bitWidth = widthAttr.getInt();
      }
      if (auto writableAttr = dictAttr.getAs<BoolAttr>("writable")) {
        mapping.isWritable = writableAttr.getValue();
      }
      if (auto readableAttr = dictAttr.getAs<BoolAttr>("readable")) {
        mapping.isReadable = readableAttr.getValue();
      }
      if (auto w1cAttr = dictAttr.getAs<BoolAttr>("isW1C")) {
        mapping.isW1C = w1cAttr.getValue();
      }

      mappings.push_back(mapping);
    }
  }

  // If no IR attribute, use standard GPIO register layout (from original gpio_top.c)
  if (mappings.empty()) {
    llvm::errs() << "  [WARNING] No APB mappings found in IR, using standard GPIO layout\n";

    // Standard GPIO APB register offsets (from original qemu-output/gpio0/gpio_top.c)
    struct { uint32_t addr; const char* name; int width; bool writable; bool readable; } stdRegs[] = {
      {0x00, "gpio_sw_data", 32, true, true},
      {0x04, "gpio_sw_dir", 32, true, true},
      {0x30, "gpio_int_en", 32, true, true},
      {0x34, "gpio_int_mask", 32, true, true},
      {0x38, "gpio_int_type", 32, true, true},
      {0x3c, "gpio_int_pol", 32, true, true},
      {0x40, "gpio_int_status", 32, false, true},  // Read-only
      {0x44, "gpio_raw_int_status", 32, false, true},  // Read-only
      {0x48, "gpio_debounce", 32, true, true},
      {0x50, "gpio_ext_data", 32, false, true},  // Read-only
      {0x60, "gpio_int_level_sync", 8, true, true},
    };

    for (const auto& reg : stdRegs) {
      clk_analysis::APBRegisterMapping mapping;
      mapping.address = reg.addr;
      mapping.registerName = reg.name;
      mapping.bitWidth = reg.width;
      mapping.isWritable = reg.writable;
      mapping.isReadable = reg.readable;
      mapping.isW1C = false;
      mappings.push_back(mapping);
    }
  }

  return mappings;
}

/// Detect address conflicts in APB mappings
static std::vector<clk_analysis::AddressConflict>
detectAddressConflicts(const std::vector<clk_analysis::APBRegisterMapping>& mappings) {
  std::vector<clk_analysis::AddressConflict> conflicts;

  std::map<uint32_t, std::vector<std::string>> addrToRegs;
  for (const auto& m : mappings) {
    addrToRegs[m.address].push_back(m.registerName);
  }

  for (const auto& pair : addrToRegs) {
    if (pair.second.size() > 1) {
      clk_analysis::AddressConflict conflict;
      conflict.address = pair.first;
      conflict.registerNames = pair.second;
      conflicts.push_back(conflict);
    }
  }

  return conflicts;
}

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
    llvm::errs() << "(Aligned with original qemu-output architecture)\n";
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

    // Step 2: Create generator and registry (FLAT naming)
    FlatSignalRegistry registry;
    qemu_codegen::QEMUDeviceGenerator gen(devName);

    // Step 3: Extract APB mappings (this determines MMIO registers)
    llvm::errs() << "\nExtracting APB register mappings...\n";
    auto apbMappings = extractAPBMappings(topModule, mod);
    auto conflicts = detectAddressConflicts(apbMappings);

    llvm::errs() << "  Found " << apbMappings.size() << " APB register mappings:\n";
    for (const auto& m : apbMappings) {
      llvm::errs() << "    0x" << llvm::format_hex_no_prefix(m.address, 2)
                   << ": " << m.registerName << " (" << m.bitWidth << "-bit, "
                   << (m.isReadable && m.isWritable ? "RW" :
                       (m.isReadable ? "RO" : "WO")) << ")\n";
    }

    if (!conflicts.empty()) {
      llvm::errs() << "  Address conflicts detected:\n";
      for (const auto& c : conflicts) {
        llvm::errs() << "    0x" << llvm::format_hex_no_prefix(c.address, 2) << ": ";
        for (size_t i = 0; i < c.registerNames.size(); i++) {
          if (i > 0) llvm::errs() << ", ";
          llvm::errs() << c.registerNames[i];
        }
        llvm::errs() << "\n";
      }
    }

    // Set APB mappings and conflicts on generator
    gen.setAPBMappings(apbMappings);
    gen.setAddressConflicts(conflicts);

    // Step 4: Collect ALL signals (flat naming) for struct
    llvm::errs() << "\nCollecting signals (flat naming)...\n";
    unsigned totalSignals = 0;

    // Collect signals from all modules (for the state struct)
    mod.walk([&](hw::HWModuleOp hwMod) {
      // Ports
      for (auto port : hwMod.getPortList()) {
        std::string portName = port.getName().str();
        unsigned bitWidth = 32;
        if (auto intTy = dyn_cast<IntegerType>(port.type)) {
          bitWidth = intTy.getWidth();
        }
        registry.registerSignal(portName, bitWidth);
        totalSignals++;
      }

      // LLHD signals
      hwMod.walk([&](llhd::SignalOp sigOp) {
        std::string sigName;
        if (auto nameAttr = sigOp.getName()) {
          sigName = nameAttr->str();
        } else {
          return;
        }

        unsigned bitWidth = 32;
        Type ty = sigOp.getType();
        if (auto inoutTy = dyn_cast<hw::InOutType>(ty)) {
          ty = inoutTy.getElementType();
        }
        if (auto intTy = dyn_cast<IntegerType>(ty)) {
          bitWidth = intTy.getWidth();
        }

        registry.registerSignal(sigName, bitWidth);
        totalSignals++;
      });

      // Signals from comb_logic
      if (auto combAttr = hwMod->getAttrOfType<ArrayAttr>("qemu.comb_logic")) {
        for (Attribute elem : combAttr) {
          auto dictAttr = dyn_cast<DictionaryAttr>(elem);
          if (!dictAttr) continue;

          if (auto targetAttr = dictAttr.getAs<StringAttr>("target")) {
            unsigned bitWidth = 32;
            if (auto widthAttr = dictAttr.getAs<IntegerAttr>("bitWidth")) {
              bitWidth = widthAttr.getInt();
            }
            registry.registerSignal(targetAttr.getValue(), bitWidth);
          }

          // Extract referenced signals from expression
          if (auto exprAttr = dictAttr.getAs<StringAttr>("expression")) {
            std::string expr = exprAttr.getValue().str();
            std::regex pattern(R"(s->([A-Za-z_][A-Za-z0-9_]*))");
            std::sregex_iterator it(expr.begin(), expr.end(), pattern);
            std::sregex_iterator end;
            while (it != end) {
              std::string fieldName = (*it)[1].str();
              registry.registerSignal(fieldName, 32);
              ++it;
            }
          }
        }
      }
    });

    llvm::errs() << "  Total signals: " << totalSignals << "\n";

    // Step 5: Add all signals to generator (for state struct)
    for (const auto& pair : registry.getSignals()) {
      const auto& info = pair.second;
      gen.addSimpleReg(info.sanitizedName, info.bitWidth);
    }

    // Step 6: Process combinational logic
    llvm::errs() << "\nProcessing combinational logic...\n";
    std::vector<clk_analysis::CombinationalAssignment> combLogic;
    unsigned combCount = 0;

    mod.walk([&](hw::HWModuleOp hwMod) {
      auto combAttr = hwMod->getAttrOfType<ArrayAttr>("qemu.comb_logic");
      if (!combAttr) return;

      for (Attribute elem : combAttr) {
        auto dictAttr = dyn_cast<DictionaryAttr>(elem);
        if (!dictAttr) continue;

        std::string target, expression;
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

        clk_analysis::CombinationalAssignment assign;
        assign.targetSignal = registry.lookup(target);
        assign.expression = expression;  // Expression already has s-> format
        assign.bitWidth = bitWidth;
        combLogic.push_back(assign);
        combCount++;
      }
    });

    gen.setCombinationalLogic(combLogic);
    llvm::errs() << "  Combinational assignments: " << combCount << "\n";

    // Step 7: Detect GPIO inputs
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

    // Step 8: Generate output files
    std::string baseDir = outputDir.empty() ? "qemu-passes/output" : std::string(outputDir);
    std::string outDir = baseDir + "/" + devName;
    llvm::sys::fs::create_directories(outDir);

    std::string headerPath = outDir + "/" + devName + ".h";
    std::string sourcePath = outDir + "/" + devName + ".c";

    llvm::errs() << "\nGenerating output files...\n";
    llvm::errs() << "  Header: " << headerPath << "\n";
    llvm::errs() << "  Source: " << sourcePath << "\n";

    // Generate header
    {
      std::error_code EC;
      llvm::raw_fd_ostream headerFile(headerPath, EC);
      if (EC) {
        llvm::errs() << "Error: Cannot create " << headerPath << ": " << EC.message() << "\n";
        return signalPassFailure();
      }
      gen.generateHeader(headerFile);
    }

    // Generate source
    {
      std::error_code EC;
      llvm::raw_fd_ostream sourceFile(sourcePath, EC);
      if (EC) {
        llvm::errs() << "Error: Cannot create " << sourcePath << ": " << EC.message() << "\n";
        return signalPassFailure();
      }
      gen.generateSource(sourceFile);
    }

    llvm::errs() << "\n========================================\n";
    llvm::errs() << "Summary:\n";
    llvm::errs() << "  Device: " << devName << "\n";
    llvm::errs() << "  State struct fields: " << registry.getSignals().size() << "\n";
    llvm::errs() << "  APB registers (MMIO): " << apbMappings.size() << "\n";
    llvm::errs() << "  Combinational logic: " << combCount << "\n";
    llvm::errs() << "  Output: " << outDir << "/\n";
    llvm::errs() << "========================================\n";
  }
};

} // anonymous namespace
