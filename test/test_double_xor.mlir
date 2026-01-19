// Test: Double XOR (should cancel out)
module {
  hw.module @test_double_xor() {
    %c1_i1 = hw.constant true
    %c0_i1 = hw.constant false
    %time = llhd.constant_time <0ns, 0d, 1e>

    %pwrite = llhd.sig name "pwrite" %c0_i1 : i1
    %psel = llhd.sig name "psel" %c0_i1 : i1

    llhd.process {
      %prb_pwrite = llhd.prb %pwrite : !hw.inout<i1>
      %prb_psel = llhd.prb %psel : !hw.inout<i1>

      // Double NOT: pwrite ^ 1 ^ 1 = pwrite (should detect pwrite=1)
      %inv1 = comb.xor %prb_pwrite, %c1_i1 : i1
      %inv2 = comb.xor %inv1, %c1_i1 : i1

      // Should detect: pwrite=1, psel=1 (double invert cancels)
      %and = comb.and %inv2, %prb_psel : i1

      cf.cond_br %and, ^bb1, ^bb2
    ^bb1:
      llhd.wait (%prb_pwrite, %prb_psel : i1, i1), ^bb2
    ^bb2:
      llhd.halt
    }
  }
}
