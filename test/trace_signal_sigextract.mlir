// RUN: qemu-passes %s --dff-demo
// Test: llhd.sig.extract unwrapping
// Intent: Verify traceSignal() correctly traces through sig.extract chains to base signal
// Requirement: Multiple extracts from same wide signal should be distinguishable in AND

module {
  hw.module @test_sigextract() {
    %c1_i1 = hw.constant true
    %c0_i1 = hw.constant false
    %c0_i8 = hw.constant 0 : i8
    %time = llhd.constant_time <0ns, 0d, 1e>

    // Index constants for sig.extract (must be SSA values, i3 for 8-bit signal)
    %idx0 = hw.constant 0 : i3
    %idx1 = hw.constant 1 : i3
    %idx7 = hw.constant 7 : i3

    // Create a wide control bus signal
    %ctrl_bus = llhd.sig name "ctrl_bus" %c0_i8 : i8

    // Create separate named signals for additional testing
    %pwrite = llhd.sig name "pwrite" %c0_i1 : i1
    %psel = llhd.sig name "psel" %c0_i1 : i1

    // Extract bit 0 from ctrl_bus (like a "write enable" bit)
    %bit0 = llhd.sig.extract %ctrl_bus from %idx0 : (!hw.inout<i8>) -> !hw.inout<i1>

    // Extract bit 1 from ctrl_bus (like a "read enable" bit)
    %bit1 = llhd.sig.extract %ctrl_bus from %idx1 : (!hw.inout<i8>) -> !hw.inout<i1>

    // Extract bit 7 from ctrl_bus (like a "valid" bit)
    %bit7 = llhd.sig.extract %ctrl_bus from %idx7 : (!hw.inout<i8>) -> !hw.inout<i1>

    llhd.process {
      // Probe extracted bits - all trace back to ctrl_bus
      %prb_bit0 = llhd.prb %bit0 : !hw.inout<i1>
      %prb_bit1 = llhd.prb %bit1 : !hw.inout<i1>
      %prb_bit7 = llhd.prb %bit7 : !hw.inout<i1>

      // Probe regular signals
      %prb_pwrite = llhd.prb %pwrite : !hw.inout<i1>
      %prb_psel = llhd.prb %psel : !hw.inout<i1>

      // Test: Apply NOT to one of the extracted bits
      %inv_bit0 = comb.xor %prb_bit0, %c1_i1 : i1

      // Test: Apply double NOT to another extracted bit (should cancel)
      %inv_bit1_1 = comb.xor %prb_bit1, %c1_i1 : i1
      %inv_bit1_2 = comb.xor %inv_bit1_1, %c1_i1 : i1

      // AND condition combining:
      // - inv_bit0: ctrl_bus bit0 inverted (need ctrl_bus[0]=0)
      // - inv_bit1_2: ctrl_bus bit1 double-inverted (need ctrl_bus[1]=1)
      // - prb_bit7: ctrl_bus bit7 direct (need ctrl_bus[7]=1)
      // - prb_pwrite: pwrite direct (need pwrite=1)
      // - prb_psel: psel direct (need psel=1)
      %and1 = comb.and %inv_bit0, %inv_bit1_2, %prb_bit7, %prb_pwrite, %prb_psel : i1

      cf.cond_br %and1, ^bb1, ^bb2
    ^bb1:
      llhd.wait (%prb_bit0, %prb_bit1, %prb_bit7, %prb_pwrite, %prb_psel : i1, i1, i1, i1, i1), ^bb2
    ^bb2:
      llhd.halt
    }
  }
}
