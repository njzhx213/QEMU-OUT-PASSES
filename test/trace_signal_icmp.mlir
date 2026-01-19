// RUN: qemu-passes %s --dff-demo
// Test: icmp-based NOT patterns for i1
// Intent: Verify traceSignal() correctly identifies NOT via comb.icmp eq/ne with 0/1

module {
  hw.module @test_icmp_not() {
    %c1_i1 = hw.constant true
    %c0_i1 = hw.constant false
    %time = llhd.constant_time <0ns, 0d, 1e>

    // Create signals
    %pwrite = llhd.sig name "pwrite" %c0_i1 : i1
    %psel = llhd.sig name "psel" %c0_i1 : i1
    %penable = llhd.sig name "penable" %c0_i1 : i1
    %prdy = llhd.sig name "prdy" %c0_i1 : i1

    llhd.process {
      // Read signals
      %prb_pwrite = llhd.prb %pwrite : !hw.inout<i1>
      %prb_psel = llhd.prb %psel : !hw.inout<i1>
      %prb_penable = llhd.prb %penable : !hw.inout<i1>
      %prb_prdy = llhd.prb %prdy : !hw.inout<i1>

      // Test 1: x == 0 -> !x (pwrite inverted -> need pwrite=0 to get true)
      %cmp1 = comb.icmp eq %prb_pwrite, %c0_i1 : i1

      // Test 2: x != 0 -> x (psel identity -> need psel=1)
      %cmp2 = comb.icmp ne %prb_psel, %c0_i1 : i1

      // Test 3: x == 1 -> x (penable identity -> need penable=1)
      %cmp3 = comb.icmp eq %prb_penable, %c1_i1 : i1

      // Test 4: x != 1 -> !x (prdy inverted -> need prdy=0)
      %cmp4 = comb.icmp ne %prb_prdy, %c1_i1 : i1

      // Should detect:
      // - pwrite: inverted (need 0)
      // - psel: not inverted (need 1)
      // - penable: not inverted (need 1)
      // - prdy: inverted (need 0)
      %and1 = comb.and %cmp1, %cmp2, %cmp3, %cmp4 : i1

      cf.cond_br %and1, ^bb1, ^bb2
    ^bb1:
      llhd.wait (%prb_pwrite, %prb_psel, %prb_penable, %prb_prdy : i1, i1, i1, i1), ^bb2
    ^bb2:
      llhd.halt
    }
  }
}
