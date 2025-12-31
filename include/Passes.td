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
// Pass 3: 删除时钟拓扑链（对齐旧框架 SignalTracing.h）
//===----------------------------------------------------------------------===//

def ClockDrvRemoval : Pass<"clock-drv-removal", "mlir::ModuleOp"> {
  let summary = "Remove filterable clock topology from LLHD IR";
  let description = [{
    Removes clock signal topology from clock-triggered processes:

    1. Remove clock edge terms from trigger OR conditions
       (minimal rewrite - preserves reset edge terms and enable guards)
    2. Remove clock probes from wait observed lists
       (prevents empty wait - keeps one if all would be removed)
    3. Run safe DCE to remove dead clock-related operations
       (uses MLIR MemoryEffectOpInterface + explicit whitelist)
    4. Remove clock port connection drvs (only BlockArgument values)
    5. Remove dead clock signal definitions (llhd.sig with qemu.is_clock)

    Key design decisions (aligned with old framework SignalTracing.h):
    - Uses waitBlocks concept for stable idle/body branch detection
    - Strict edge detection: requires both direct prb AND inverted prb
    - Only processes marked with `qemu.clock_triggered` are modified
    - Reset disambiguation branches are preserved

    Prerequisites:
    - Must run after `clock-signal-detection` pass
    - Optionally run after `drv-classification` pass
  }];
}
