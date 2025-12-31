//===- tool_main.cpp - QEMU Passes Driver ---------------------------------===//
//
// Driver for QEMU conversion passes:
// 1. clock-signal-detection - Detect and mark clock signals (two-level)
// 2. drv-classification - Classify drv operations
// 3. clock-drv-removal - Remove filterable clock topology
//
// Usage:
//   qemu-passes input.mlir --clock-signal-detection
//   qemu-passes input.mlir --drv-classification
//   qemu-passes input.mlir --clock-drv-removal
//   qemu-passes input.mlir --all-passes -o output.mlir
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

// CIRCT Dialects
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"

// TableGen 生成的 Pass 声明
#define GEN_PASS_DECL_CLOCKSIGNALDETECTION
#define GEN_PASS_DECL_DRVCLASSIFICATION
#define GEN_PASS_DECL_CLOCKDRVREMOVAL
#include "Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Command line options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input .mlir>"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> outputFilename(
    "o",
    llvm::cl::desc("Output file (default: stdout)"),
    llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<bool> runClockDetection(
    "clock-signal-detection",
    llvm::cl::desc("Run clock signal detection pass"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> runDrvClassification(
    "drv-classification",
    llvm::cl::desc("Run drv classification pass"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> runClockDrvRemoval(
    "clock-drv-removal",
    llvm::cl::desc("Run clock drv removal pass"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> runAllPasses(
    "all-passes",
    llvm::cl::desc("Run all three passes in order"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> emitOutput(
    "emit-output",
    llvm::cl::desc("Emit the transformed MLIR output"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv,
      "QEMU Conversion Passes for LLHD IR\n\n"
      "Usage:\n"
      "  qemu-passes <input.mlir> --clock-signal-detection\n"
      "  qemu-passes <input.mlir> --drv-classification\n"
      "  qemu-passes <input.mlir> --clock-drv-removal\n"
      "  qemu-passes <input.mlir> --all-passes -o output.mlir\n"
  );

  // Create MLIR context and register dialects
  MLIRContext context;
  DialectRegistry registry;
  registry.insert<
    circt::hw::HWDialect,
    circt::sv::SVDialect,
    circt::comb::CombDialect,
    circt::seq::SeqDialect,
    circt::llhd::LLHDDialect,
    mlir::cf::ControlFlowDialect
  >();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Parse input file
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module) {
    llvm::WithColor::error() << "Failed to parse input MLIR: " << inputFilename << "\n";
    return 1;
  }

  // Create PassManager
  PassManager pm(&context);
  pm.enableVerifier(true);

  // Add passes based on command line options
  if (runAllPasses) {
    pm.addPass(createClockSignalDetection());
    pm.addPass(createDrvClassification());
    pm.addPass(createClockDrvRemoval());
  } else {
    if (runClockDetection) {
      pm.addPass(createClockSignalDetection());
    }
    if (runDrvClassification) {
      pm.addPass(createDrvClassification());
    }
    if (runClockDrvRemoval) {
      pm.addPass(createClockDrvRemoval());
    }
  }

  // Check if any pass was specified
  if (!runClockDetection && !runDrvClassification &&
      !runClockDrvRemoval && !runAllPasses) {
    llvm::outs() << "No pass specified. Available passes:\n";
    llvm::outs() << "  --clock-signal-detection  Detect and mark clock signals\n";
    llvm::outs() << "  --drv-classification      Classify drv operations\n";
    llvm::outs() << "  --clock-drv-removal       Remove clock-related drvs\n";
    llvm::outs() << "  --all-passes              Run all passes in order\n";
    llvm::outs() << "\nRunning all passes by default...\n\n";
    pm.addPass(createClockSignalDetection());
    pm.addPass(createDrvClassification());
    pm.addPass(createClockDrvRemoval());
  }

  // Run passes
  if (failed(pm.run(*module))) {
    llvm::WithColor::error() << "Pass execution failed\n";
    return 1;
  }

  // Output the result
  if (emitOutput || outputFilename != "-") {
    if (outputFilename != "-") {
      std::error_code ec;
      llvm::raw_fd_ostream os(outputFilename, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        llvm::WithColor::error() << "Could not open output file: " << ec.message() << "\n";
        return 1;
      }
      module->print(os);
      llvm::outs() << "\nOutput written to: " << outputFilename << "\n";
    } else {
      llvm::outs() << "\n========================================\n";
      llvm::outs() << "Transformed LLHD IR:\n";
      llvm::outs() << "========================================\n\n";
      module->print(llvm::outs());
    }
  }

  return 0;
}
