# QEMU Conversion Passes for LLHD IR

MLIR passes for converting LLHD IR to QEMU device C code.

## Pipeline Overview

```
LLHD IR → clock-detection → drv-classification → CSE → clock-removal → dff-demo → comb-extract → qemu-emit-c → C/H files
```

## Passes

### Pass 1: ClockSignalDetection
Detect and mark clock signals using two-level analysis.

- **Level 1 (Structural)**: 1-bit signal, in wait sensitivity list, no logic drv
- **Level 2 (Functional)**: Edge detection pattern, not used for disambiguation
- **Output**: `qemu.is_clock` on signals, `qemu.clock_triggered` on processes

### Pass 2: DrvClassification
Classify drv operations by their update patterns.

| Class | Description |
|-------|-------------|
| UNCHANGED | Hold pattern or simple overwrite |
| ACCUMULATE | Counter increment/decrement |
| LOOP_ITER | Loop iterator variable |
| COMPLEX | Other patterns |

- **Output**: `qemu.drv_class` attribute on llhd.drv operations

### Pass 3: ClockDrvRemoval
Remove filterable clock topology from IR.

- Remove clock edge terms from trigger OR conditions
- Remove clock probes from wait observed lists
- Run safe DCE with MLIR MemoryEffect interface
- Remove clock port connection drvs
- Remove dead clock signal definitions

### Pass 4: DffDemo
APB control signal inference and optimization.

- Detect APB bus patterns (psel, penable, pwrite)
- Infer phase-agnostic signals
- Replace redundant conditions with constants
- Run dead code elimination

### Pass 5: CombLogicExtract
Extract combinational logic for QEMU `update_state()`.

- Identify combinational (non-clocked) drv operations
- Generate C expressions from LLHD operations
- **Output**: `qemu.comb_logic` attribute on hw.module

### Pass 6: QEMUEmitC
Emit QEMU device C code from annotated IR.

- Generate flattened device state struct
- Hierarchical naming: `<ModuleName>__<signalName>`
- Expression rewriting for qualified field names
- **Output**: `<device>.c` and `<device>.h`

## Build

Requires CIRCT, MLIR, and LLVM.

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH="/path/to/circt/lib/cmake/circt;/path/to/mlir/lib/cmake/mlir;/path/to/llvm/lib/cmake/llvm"
make -j$(nproc)
```

## Usage

```bash
# Run individual passes
./qemu-passes input.mlir --clock-signal-detection
./qemu-passes input.mlir --drv-classification
./qemu-passes input.mlir --clock-drv-removal
./qemu-passes input.mlir --dff-demo
./qemu-passes input.mlir --comb-logic-extract
./qemu-passes input.mlir --qemu-emit-c --qemu-output-dir=./generated

# Run full pipeline (excluding qemu-emit-c)
./qemu-passes input.mlir --all-passes -o output.mlir

# Full pipeline with C code generation
./qemu-passes input.mlir --all-passes -o output.mlir
./qemu-passes output.mlir --qemu-emit-c --qemu-output-dir=./generated
```

## Directory Structure

```
qemu-passes/
├── CMakeLists.txt
├── tool_main.cpp              # Driver program
├── include/
│   └── Passes.td              # TableGen pass definitions
├── lib/
│   ├── Passes.cpp             # Clock detection, classification, removal
│   ├── DffDemoPass.cpp        # APB control signal inference
│   ├── CombLogicExtractPass.cpp  # Combinational logic extraction
│   ├── QEMUEmitCPass.cpp      # QEMU C code emission
│   ├── ClockAnalysis.h        # Clock analysis utilities
│   └── CombTranslator.h       # LLHD to C expression translator
└── test/
    └── *.mlir                 # Test files
```

## Generated Output

The `qemu-emit-c` pass generates QEMU-compatible device code:

**Header (`<device>.h`)**:
- Device state struct with all signals
- Type macros and declarations

**Source (`<device>.c`)**:
- `update_state()` - Combinational logic recalculation
- `<device>_read()` - MMIO read handler
- `<device>_write()` - MMIO write handler
- GPIO input callbacks
- Device initialization

## Dependencies

- CIRCT (with HW, Comb, Seq, LLHD dialects)
- MLIR
- LLVM
- qemu-output library (QEMUCodeGen.cpp/h)
