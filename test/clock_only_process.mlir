// Test case: clock-only process
// Expected behavior: Process should be removed entirely (pure clock trigger)
// or keep one clock in wait to avoid empty wait

module {
  hw.module @clock_only_test(in %clk : i1) {
    %false = hw.constant false
    %true = hw.constant true
    %0 = llhd.constant_time <0ns, 0d, 1e>

    // Clock signal
    %clk_sig = llhd.sig name "clk" %false : i1
    llhd.drv %clk_sig, %clk after %0 : !hw.inout<i1>

    // Internal register
    %reg = llhd.sig name "reg" %false : i1

    // Clock-only process: only waits on clock, no reset
    llhd.process {
      cf.br ^bb1
    ^bb1:
      %old_clk = llhd.prb %clk_sig : !hw.inout<i1>
      %clk_prb = llhd.prb %clk_sig : !hw.inout<i1>
      llhd.wait (%clk_prb : i1), ^bb2
    ^bb2:
      %new_clk = llhd.prb %clk_sig : !hw.inout<i1>
      %old_inv = comb.xor bin %old_clk, %true : i1
      %edge = comb.and bin %old_inv, %new_clk : i1
      cf.cond_br %edge, ^bb3, ^bb1
    ^bb3:
      // Hold pattern: reg = prb reg (should be UNCHANGED)
      %reg_val = llhd.prb %reg : !hw.inout<i1>
      llhd.drv %reg, %reg_val after %0 : !hw.inout<i1>
      cf.br ^bb1
    }
  }
}
