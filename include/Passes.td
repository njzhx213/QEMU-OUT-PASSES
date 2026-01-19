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

//===----------------------------------------------------------------------===//
// Pass 4: APB 控制信号分析与优化
//===----------------------------------------------------------------------===//

def DffDemo : Pass<"dff-demo", "mlir::ModuleOp"> {
  let summary = "APB control signal inference and optimization";
  let description = [{
    Analyzes APB control signals in LLHD IR:

    1. Traces AND conditions to find pwrite-based trigger paths
    2. Classifies paths as read (pwrite=0) or write (pwrite=1)
    3. Identifies phase-agnostic signals (constant across all paths)
    4. Replaces redundant signal probes with constants
    5. Performs iterative DCE to remove dead probes, XORs, and signals

    Output:
    - Reports trigger conditions and their control signal requirements
    - Reports branch-controlled state transitions
    - Reports optimization statistics
  }];
}

//===----------------------------------------------------------------------===//
// Pass 5: Combinational Logic Extraction
//===----------------------------------------------------------------------===//

def CombLogicExtract : Pass<"comb-logic-extract", "mlir::ModuleOp"> {
  let summary = "Extract combinational logic for QEMU update_state()";
  let description = [{
    Extracts combinational logic (llhd.drv outside of llhd.process) and
    translates it to C expressions for QEMU update_state().

    Design aligns with qemu-output/src/lib/ClkAnalysisResult.cpp:
    1. Walk all llhd.drv in hw.module
    2. Filter out drv inside llhd.process (sequential logic)
    3. Filter out module ports
    4. Generate C expressions using CombTranslator

    Output:
    - JSON file with combinational assignments (target, expression, bitWidth)
    - Module attribute qemu.comb_logic with the same data
    - Console report of extracted assignments

    Prerequisites:
    - Should run after clock removal and APB optimization passes
    - Input IR should be simplified for best expression generation
  }];
}

//===----------------------------------------------------------------------===//
// Pass 6: QEMU C Code Emission
//===----------------------------------------------------------------------===//

def QEMUEmitC : Pass<"qemu-emit-c", "mlir::ModuleOp"> {
  let summary = "Emit QEMU device C code from annotated LLHD IR";
  let description = [{
    Final pass in the LLHD-to-QEMU conversion pipeline. Reads qemu.* metadata
    from the IR and emits QEMU-compatible C code using QEMUCodeGen.

    Input metadata consumed:
    - qemu.comb_logic on hw.module: combinational logic expressions
    - qemu.clock_triggered on llhd.process: sequential process markers
    - qemu.drv_class on llhd.drv: drive operation classifications

    Output:
    - <device>.c: QEMU device implementation
    - <device>.h: QEMU device header

    Design:
    - Generates ONE QEMU device for the public top hw.module
    - Flattens comb logic from all submodules into top device
    - update_state() emits only metadata-driven assignments (no hardcoded IRQ)
    - All signal names are sanitized with collision detection

    Prerequisites:
    - Must run after comb-logic-extract pass
    - Input IR should have all qemu.* annotations
  }];

  let options = [
    Option<"outputDir", "output-dir", "std::string", /*default=*/"\"qemu-output\"",
           "Output directory for generated C files">,
    Option<"deviceName", "device-name", "std::string", /*default=*/"\"\"",
           "Override device name (default: use top module name)">
  ];
}
