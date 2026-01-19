// RUN: qemu-passes %s --dff-demo
// Test: multi-layer NOT peeling (comb.xor chains with odd/even parity)
// Intent: Verify traceSignal() correctly tracks inversion parity through multiple XOR layers

module {
  hw.module @test_multi_not() {
    %c1_i1 = hw.constant true
    %c0_i1 = hw.constant false
    %time = llhd.constant_time <0ns, 0d, 1e>

    // Create signals
    %pwrite = llhd.sig name "pwrite" %c0_i1 : i1
    %psel = llhd.sig name "psel" %c0_i1 : i1
    %enable = llhd.sig name "enable" %c0_i1 : i1

    llhd.process {
      // Read signals
      %prb_pwrite = llhd.prb %pwrite : !hw.inout<i1>
      %prb_psel = llhd.prb %psel : !hw.inout<i1>
      %prb_enable = llhd.prb %enable : !hw.inout<i1>

      // Test 1: Single NOT via XOR (pwrite inverted -> pwrite=0 needed)
      %not1 = comb.xor %prb_pwrite, %c1_i1 : i1

      // Test 2: Double NOT via XOR chain (should cancel out -> pwrite=1)
      %not2 = comb.xor %not1, %c1_i1 : i1

      // Test 3: Single NOT via XOR (enable inverted -> enable=0)
      %inv_enable = comb.xor %prb_enable, %c1_i1 : i1

      // Test 4: Triple NOT via XOR chain (odd count -> inverted -> enable=0)
      %inv_enable2 = comb.xor %inv_enable, %c1_i1 : i1
      %inv_enable3 = comb.xor %inv_enable2, %c1_i1 : i1

      // Test 5: Four NOTs via XOR chain (even count -> not inverted -> psel=1)
      %psel_not1 = comb.xor %prb_psel, %c1_i1 : i1
      %psel_not2 = comb.xor %psel_not1, %c1_i1 : i1
      %psel_not3 = comb.xor %psel_not2, %c1_i1 : i1
      %psel_not4 = comb.xor %psel_not3, %c1_i1 : i1

      // Create AND condition:
      // - not2 (double NOT pwrite) -> pwrite=1
      // - inv_enable3 (triple NOT enable) -> enable=0
      // - psel_not4 (quad NOT psel) -> psel=1
      %and1 = comb.and %not2, %inv_enable3, %psel_not4 : i1

      cf.cond_br %and1, ^bb1, ^bb2
    ^bb1:
      llhd.wait (%prb_pwrite, %prb_psel, %prb_enable : i1, i1, i1), ^bb2
    ^bb2:
      llhd.halt
    }
  }
}
