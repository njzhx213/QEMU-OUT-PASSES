include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Pass 1: 检测时钟信号
//===----------------------------------------------------------------------===//

def ClockSignalDetection : Pass<"clock-signal-detection", "mlir::ModuleOp"> {
  let summary = "Detect and mark clock signals in LLHD IR";
  let description = [{
    Analyzes LLHD signals using two-level detection:
    1. Structural: 1-bit, in wait sensitivity list, no logic drv
    2. Trigger effect: all triggered drvs are hold mode (reg = prb reg)

    Marks clock signals with `qemu.is_clock` attribute.
    Marks clock-triggered processes with `qemu.clock_triggered` attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Pass 2: 分类 drv 操作
//===----------------------------------------------------------------------===//

def DrvClassification : Pass<"drv-classification", "mlir::ModuleOp"> {
  let summary = "Classify drv operations in LLHD IR";
  let description = [{
    Classifies each llhd.drv operation into one of:
    - UNCHANGED: State unchanged (hold or not self-dependent)
    - ACCUMULATE: State accumulation (counter++/counter--)
    - LOOP_ITER: Loop iterator (single-cycle for loop)
    - COMPLEX: Complex state change

    Adds `qemu.drv_class` attribute to each drv operation.
    For ACCUMULATE, also adds `qemu.step` attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Pass 3: 删除时钟相关的 drv
//===----------------------------------------------------------------------===//

def ClockDrvRemoval : Pass<"clock-drv-removal", "mlir::ModuleOp"> {
  let summary = "Remove clock-related and UNCHANGED drv operations";
  let description = [{
    Removes drv operations that are:
    1. In clock-triggered processes (marked with `qemu.clock_triggered`)
    2. Classified as UNCHANGED (hold mode or simple overwrite)
    3. Classified as LOOP_ITER (loop iterators)

    Outputs a simplified LLHD IR suitable for QEMU conversion.

    Prerequisites:
    - Must run after `clock-signal-detection` pass
    - Must run after `drv-classification` pass
  }];
}
