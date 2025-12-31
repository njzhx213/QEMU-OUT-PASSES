# QEMU Conversion Passes for LLHD IR

MLIR passes for converting LLHD IR to QEMU device code.

## Passes

1. **ClockSignalDetection** - Detect and mark clock signals
   - Level 1: Structural check (1-bit, in sensitivity list, no logic drv)
   - Level 2: Trigger effect check (edge detection pattern, not used for disambiguation)

2. **DrvClassification** - Classify drv operations
   - UNCHANGED: Hold pattern or overwrite
   - ACCUMULATE: Counter increment/decrement
   - LOOP_ITER: Loop iterator
   - COMPLEX: Other patterns

3. **ClockDrvRemoval** - Remove clock signals and their connections

## Build

```bash
mkdir build && cd build
cmake .. -G Ninja \
  -DCIRCT_DIR=/path/to/circt/build/lib/cmake/circt \
  -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm
ninja
```

## Usage

```bash
# Run clock signal detection
./qemu-passes input.mlir --clock-signal-detection

# Run all passes and emit output
./qemu-passes input.mlir --all-passes --emit-output

# Run all passes and save to file
./qemu-passes input.mlir --all-passes -o output.mlir
```

## Clock Detection Logic

A signal is detected as a filterable clock if:
1. **Structural**: 1-bit, in wait sensitivity list, only port connection (no logic drv)
2. **Functional**: Used in edge detection pattern `and(!old, new)`, NOT used for disambiguation

Example:
```
// clk: filterable (only edge triggering)
// rst: not filterable (used for disambiguation)
always @(posedge clk or posedge rst)
  if (rst) q <= 0;  // rst controls behavior
  else q <= d;      // clk path
```
