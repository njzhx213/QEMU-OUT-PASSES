// Test case: clock with enable gating
// Expected behavior: cond_br(and(prb(clk), en)) should NOT be treated as edge trigger
// The pass should NOT modify this because it's not an edge detection pattern
// (requires both direct prb AND inverted prb to same signal)

module {
  hw.module @clock_enable_test(in %clk : i1, in %en : i1) {
    %false = hw.constant false
    %true = hw.constant true
    %0 = llhd.constant_time <0ns, 0d, 1e>

    // Clock signal
    %clk_sig = llhd.sig name "clk" %false : i1
    llhd.drv %clk_sig, %clk after %0 : !hw.inout<i1>

    // Enable signal
    %en_sig = llhd.sig name "en" %false : i1
    llhd.drv %en_sig, %en after %0 : !hw.inout<i1>

    // Internal register
    %reg = llhd.sig name "reg" %false : i1

    // Process with enable gating (NOT edge detection)
    // This should NOT be modified by the pass because:
    // and(prb(clk), prb(en)) is NOT an edge detection pattern
    llhd.process {
      cf.br ^bb1
    ^bb1:
      %clk_prb = llhd.prb %clk_sig : !hw.inout<i1>
      %en_prb = llhd.prb %en_sig : !hw.inout<i1>
      llhd.wait (%clk_prb, %en_prb : i1, i1), ^bb2
    ^bb2:
      %clk_val = llhd.prb %clk_sig : !hw.inout<i1>
      %en_val = llhd.prb %en_sig : !hw.inout<i1>
      // Enable gating: and(clk, en) - NOT edge detection
      %gated = comb.and bin %clk_val, %en_val : i1
      cf.cond_br %gated, ^bb3, ^bb1
    ^bb3:
      %reg_val = llhd.prb %reg : !hw.inout<i1>
      %new_val = comb.xor %reg_val, %true : i1
      llhd.drv %reg, %new_val after %0 : !hw.inout<i1>
      cf.br ^bb1
    }
  }
}
