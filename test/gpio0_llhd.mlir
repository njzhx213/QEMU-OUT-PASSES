module {
  hw.module @gpio0(out gpio0_etb_trig : i32, in %gpio_ext_porta : i32, out gpio_intr_flag : i1, out gpio_intrclk_en : i1, out gpio_porta_ddr : i32, out gpio_porta_dr : i32, in %paddr : i32, in %pclk : i1, in %pclk_intr : i1, in %penable : i1, out prdata : i32, in %presetn : i1, in %psel : i1, in %pwdata : i32, in %pwrite : i1) {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %prdata = llhd.sig %c0_i32 : i32
    %1 = comb.extract %paddr from 0 : (i32) -> i7
    %x_gpio_top.gpio_etb_trig, %x_gpio_top.gpio_out_data, %x_gpio_top.gpio_out_en, %x_gpio_top.gpio_int_flag, %x_gpio_top.gpio_int_clk_en, %x_gpio_top.prdata = hw.instance "x_gpio_top" @gpio_top(pclk: %pclk: i1, pclk_int: %pclk_intr: i1, dbclk: %pclk: i1, dbclk_rstn: %presetn: i1, scan_mode: %false: i1, presetn: %presetn: i1, penable: %penable: i1, pwrite: %pwrite: i1, pwdata: %pwdata: i32, paddr: %1: i7, psel: %psel: i1, gpio_in_data: %gpio_ext_porta: i32) -> (gpio_etb_trig: i32, gpio_out_data: i32, gpio_out_en: i32, gpio_int_flag: i1, gpio_int_clk_en: i1, prdata: i32) {sv.namehint = "gpio0_prdata"}
    llhd.drv %prdata, %x_gpio_top.prdata after %0 : !hw.inout<i32>
    %2 = llhd.prb %prdata : !hw.inout<i32>
    hw.output %x_gpio_top.gpio_etb_trig, %x_gpio_top.gpio_int_flag, %x_gpio_top.gpio_int_clk_en, %x_gpio_top.gpio_out_en, %x_gpio_top.gpio_out_data, %2 : i32, i1, i1, i32, i32, i32
  }
  hw.module private @gpio_apbif(in %pclk : i1, in %presetn : i1, in %penable : i1, in %pwrite : i1, in %pwdata : i32, in %paddr : i7, in %psel : i1, in %gpio_ext_data : i32, in %gpio_int_status : i32, in %gpio_raw_int_status : i32, out gpio_sw_data : i32, out gpio_sw_dir : i32, out gpio_int_en : i32, out gpio_int_mask : i32, out gpio_int_type : i32, out gpio_int_pol : i32, out gpio_debounce : i32, out gpio_int_clr : i32, out gpio_int_level_sync : i1, out prdata : i32) {
    %c0_i27 = hw.constant 0 : i27
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i7 = hw.constant 0 : i7
    %c20_i32 = hw.constant 20 : i32
    %c24_i32 = hw.constant 24 : i32
    %c18_i32 = hw.constant 18 : i32
    %c17_i32 = hw.constant 17 : i32
    %c16_i32 = hw.constant 16 : i32
    %c15_i32 = hw.constant 15 : i32
    %c14_i32 = hw.constant 14 : i32
    %c13_i32 = hw.constant 13 : i32
    %c12_i32 = hw.constant 12 : i32
    %c2_i32 = hw.constant 2 : i32
    %c1_i32 = hw.constant 1 : i32
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %c-8_i5 = hw.constant -8 : i5
    %c15_i5 = hw.constant 15 : i5
    %c14_i5 = hw.constant 14 : i5
    %c13_i5 = hw.constant 13 : i5
    %c12_i5 = hw.constant 12 : i5
    %c1_i5 = hw.constant 1 : i5
    %c0_i5 = hw.constant 0 : i5
    %true = hw.constant true
    %pclk_0 = llhd.sig name "pclk" %false : i1
    %2 = llhd.prb %pclk_0 : !hw.inout<i1>
    %presetn_1 = llhd.sig name "presetn" %false : i1
    %3 = llhd.prb %presetn_1 : !hw.inout<i1>
    %penable_2 = llhd.sig name "penable" %false : i1
    %pwrite_3 = llhd.sig name "pwrite" %false : i1
    %pwdata_4 = llhd.sig name "pwdata" %c0_i32 : i32
    %paddr_5 = llhd.sig name "paddr" %c0_i7 : i7
    %psel_6 = llhd.sig name "psel" %false : i1
    %gpio_ext_data_7 = llhd.sig name "gpio_ext_data" %c0_i32 : i32
    %gpio_int_status_8 = llhd.sig name "gpio_int_status" %c0_i32 : i32
    %gpio_raw_int_status_9 = llhd.sig name "gpio_raw_int_status" %c0_i32 : i32
    %gpio_sw_data = llhd.sig %c0_i32 : i32
    %gpio_sw_dir = llhd.sig %c0_i32 : i32
    %gpio_int_en = llhd.sig %c0_i32 : i32
    %gpio_int_mask = llhd.sig %c0_i32 : i32
    %gpio_int_type = llhd.sig %c0_i32 : i32
    %gpio_int_pol = llhd.sig %c0_i32 : i32
    %gpio_debounce = llhd.sig %c0_i32 : i32
    %gpio_int_clr = llhd.sig %c0_i32 : i32
    %gpio_int_level_sync = llhd.sig %false : i1
    %prdata = llhd.sig %c0_i32 : i32
    %gpio_sw_data_wen = llhd.sig %false : i1
    %gpio_sw_dir_wen = llhd.sig %false : i1
    %gpio_int_en_wen = llhd.sig %false : i1
    %gpio_int_mask_wen = llhd.sig %false : i1
    %gpio_int_type_wen = llhd.sig %false : i1
    %gpio_int_pol_wen = llhd.sig %false : i1
    %gpio_debounce_wen = llhd.sig %false : i1
    %gpio_int_level_sync_wen = llhd.sig %false : i1
    %gpio_int_clr_wen = llhd.sig %false : i1
    %ri_gpio_sw_data = llhd.sig %c0_i32 : i32
    %ri_gpio_sw_dir = llhd.sig %c0_i32 : i32
    %ri_gpio_int_en = llhd.sig %c0_i32 : i32
    %ri_gpio_int_mask = llhd.sig %c0_i32 : i32
    %ri_gpio_int_type = llhd.sig %c0_i32 : i32
    %ri_gpio_int_pol = llhd.sig %c0_i32 : i32
    %ri_gpio_debounce = llhd.sig %c0_i32 : i32
    %ri_gpio_int_level_sync = llhd.sig %c0_i32 : i32
    %ri_gpio_ext_data = llhd.sig %c0_i32 : i32
    %ri_gpio_raw_int_status = llhd.sig %c0_i32 : i32
    %ri_gpio_int_status = llhd.sig %c0_i32 : i32
    %zero_value = llhd.sig %false : i1
    llhd.drv %zero_value, %false after %1 : !hw.inout<i1>
    %4 = llhd.prb %psel_6 : !hw.inout<i1>
    %5 = llhd.prb %penable_2 : !hw.inout<i1>
    %6 = llhd.prb %pwrite_3 : !hw.inout<i1>
    %7 = llhd.prb %paddr_5 : !hw.inout<i7>
    %8 = llhd.prb %gpio_sw_data_wen : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb6
      %43 = llhd.prb %psel_6 : !hw.inout<i1>
      %44 = llhd.prb %penable_2 : !hw.inout<i1>
      %45 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %46 = comb.and %43, %44, %45 : i1
      cf.cond_br %46, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %47 = llhd.prb %paddr_5 : !hw.inout<i7>
      %48 = comb.extract %47 from 2 : (i7) -> i5
      %49 = comb.icmp eq %48, %c0_i5 : i5
      cf.cond_br %49, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      llhd.drv %gpio_sw_data_wen, %true after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb4:  // pred: ^bb2
      llhd.drv %gpio_sw_data_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb1
      llhd.drv %gpio_sw_data_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
      llhd.wait (%4, %5, %6, %7, %8 : i1, i1, i1, i7, i1), ^bb1
    }
    %9 = llhd.prb %gpio_sw_dir_wen : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb6
      %43 = llhd.prb %psel_6 : !hw.inout<i1>
      %44 = llhd.prb %penable_2 : !hw.inout<i1>
      %45 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %46 = comb.and %43, %44, %45 : i1
      cf.cond_br %46, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %47 = llhd.prb %paddr_5 : !hw.inout<i7>
      %48 = comb.extract %47 from 2 : (i7) -> i5
      %49 = comb.icmp eq %48, %c1_i5 : i5
      cf.cond_br %49, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      llhd.drv %gpio_sw_dir_wen, %true after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb4:  // pred: ^bb2
      llhd.drv %gpio_sw_dir_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb1
      llhd.drv %gpio_sw_dir_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
      llhd.wait (%4, %5, %6, %7, %9 : i1, i1, i1, i7, i1), ^bb1
    }
    %10 = llhd.prb %gpio_int_en_wen : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb6
      %43 = llhd.prb %psel_6 : !hw.inout<i1>
      %44 = llhd.prb %penable_2 : !hw.inout<i1>
      %45 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %46 = comb.and %43, %44, %45 : i1
      cf.cond_br %46, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %47 = llhd.prb %paddr_5 : !hw.inout<i7>
      %48 = comb.extract %47 from 2 : (i7) -> i5
      %49 = comb.icmp eq %48, %c12_i5 : i5
      cf.cond_br %49, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      llhd.drv %gpio_int_en_wen, %true after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb4:  // pred: ^bb2
      llhd.drv %gpio_int_en_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb1
      llhd.drv %gpio_int_en_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
      llhd.wait (%4, %5, %6, %7, %10 : i1, i1, i1, i7, i1), ^bb1
    }
    %11 = llhd.prb %gpio_int_mask_wen : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb6
      %43 = llhd.prb %psel_6 : !hw.inout<i1>
      %44 = llhd.prb %penable_2 : !hw.inout<i1>
      %45 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %46 = comb.and %43, %44, %45 : i1
      cf.cond_br %46, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %47 = llhd.prb %paddr_5 : !hw.inout<i7>
      %48 = comb.extract %47 from 2 : (i7) -> i5
      %49 = comb.icmp eq %48, %c13_i5 : i5
      cf.cond_br %49, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      llhd.drv %gpio_int_mask_wen, %true after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb4:  // pred: ^bb2
      llhd.drv %gpio_int_mask_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb1
      llhd.drv %gpio_int_mask_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
      llhd.wait (%4, %5, %6, %7, %11 : i1, i1, i1, i7, i1), ^bb1
    }
    %12 = llhd.prb %gpio_int_type_wen : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb6
      %43 = llhd.prb %psel_6 : !hw.inout<i1>
      %44 = llhd.prb %penable_2 : !hw.inout<i1>
      %45 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %46 = comb.and %43, %44, %45 : i1
      cf.cond_br %46, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %47 = llhd.prb %paddr_5 : !hw.inout<i7>
      %48 = comb.extract %47 from 2 : (i7) -> i5
      %49 = comb.icmp eq %48, %c14_i5 : i5
      cf.cond_br %49, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      llhd.drv %gpio_int_type_wen, %true after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb4:  // pred: ^bb2
      llhd.drv %gpio_int_type_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb1
      llhd.drv %gpio_int_type_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
      llhd.wait (%4, %5, %6, %7, %12 : i1, i1, i1, i7, i1), ^bb1
    }
    %13 = llhd.prb %gpio_int_pol_wen : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb6
      %43 = llhd.prb %psel_6 : !hw.inout<i1>
      %44 = llhd.prb %penable_2 : !hw.inout<i1>
      %45 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %46 = comb.and %43, %44, %45 : i1
      cf.cond_br %46, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %47 = llhd.prb %paddr_5 : !hw.inout<i7>
      %48 = comb.extract %47 from 2 : (i7) -> i5
      %49 = comb.icmp eq %48, %c15_i5 : i5
      cf.cond_br %49, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      llhd.drv %gpio_int_pol_wen, %true after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb4:  // pred: ^bb2
      llhd.drv %gpio_int_pol_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb1
      llhd.drv %gpio_int_pol_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
      llhd.wait (%4, %5, %6, %7, %13 : i1, i1, i1, i7, i1), ^bb1
    }
    %14 = llhd.prb %gpio_int_level_sync_wen : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb6
      %43 = llhd.prb %psel_6 : !hw.inout<i1>
      %44 = llhd.prb %penable_2 : !hw.inout<i1>
      %45 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %46 = comb.and %43, %44, %45 : i1
      cf.cond_br %46, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %47 = llhd.prb %paddr_5 : !hw.inout<i7>
      %48 = comb.extract %47 from 2 : (i7) -> i5
      %49 = comb.icmp eq %48, %c-8_i5 : i5
      cf.cond_br %49, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      llhd.drv %gpio_int_level_sync_wen, %true after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb4:  // pred: ^bb2
      llhd.drv %gpio_int_level_sync_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb1
      llhd.drv %gpio_int_level_sync_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
      llhd.wait (%4, %5, %6, %7, %14 : i1, i1, i1, i7, i1), ^bb1
    }
    %15 = llhd.prb %gpio_int_clr_wen : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb6
      %43 = llhd.prb %psel_6 : !hw.inout<i1>
      %44 = llhd.prb %penable_2 : !hw.inout<i1>
      %45 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %46 = comb.and %43, %44, %45 : i1
      cf.cond_br %46, ^bb2, ^bb5
    ^bb2:  // pred: ^bb1
      %47 = llhd.prb %paddr_5 : !hw.inout<i7>
      %48 = comb.extract %47 from 2 : (i7) -> i5
      %49 = comb.icmp eq %48, %c-8_i5 : i5
      cf.cond_br %49, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      llhd.drv %gpio_int_clr_wen, %true after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb4:  // pred: ^bb2
      llhd.drv %gpio_int_clr_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb1
      llhd.drv %gpio_int_clr_wen, %false after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 3 preds: ^bb3, ^bb4, ^bb5
      llhd.wait (%4, %5, %6, %7, %15 : i1, i1, i1, i7, i1), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 5 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb6
      %43 = llhd.prb %pclk_0 : !hw.inout<i1>
      %44 = llhd.prb %presetn_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %45 = llhd.prb %pclk_0 : !hw.inout<i1>
      %46 = comb.xor bin %43, %true : i1
      %47 = comb.and bin %46, %45 : i1
      %48 = llhd.prb %presetn_1 : !hw.inout<i1>
      %49 = comb.xor bin %48, %true : i1
      %50 = comb.and bin %44, %49 : i1
      %51 = comb.or bin %47, %50 : i1
      cf.cond_br %51, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %52 = llhd.prb %presetn_1 : !hw.inout<i1>
      %53 = comb.xor %52, %true : i1
      cf.cond_br %53, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_int_en, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %54 = llhd.prb %gpio_int_en_wen : !hw.inout<i1>
      cf.cond_br %54, ^bb6, ^bb1
    ^bb6:  // pred: ^bb5
      %55 = llhd.prb %pwdata_4 : !hw.inout<i32>
      llhd.drv %gpio_int_en, %55 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    %16 = llhd.prb %zero_value : !hw.inout<i1>
    %17 = llhd.prb %ri_gpio_int_en : !hw.inout<i32>
    %18 = llhd.prb %gpio_int_en : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %43 = llhd.prb %zero_value : !hw.inout<i1>
      %44 = comb.replicate %43 : (i1) -> i32
      llhd.drv %ri_gpio_int_en, %44 after %1 : !hw.inout<i32>
      %45 = llhd.prb %gpio_int_en : !hw.inout<i32>
      llhd.drv %ri_gpio_int_en, %45 after %1 : !hw.inout<i32>
      llhd.wait (%16, %17, %18 : i1, i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 5 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb6
      %43 = llhd.prb %pclk_0 : !hw.inout<i1>
      %44 = llhd.prb %presetn_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %45 = llhd.prb %pclk_0 : !hw.inout<i1>
      %46 = comb.xor bin %43, %true : i1
      %47 = comb.and bin %46, %45 : i1
      %48 = llhd.prb %presetn_1 : !hw.inout<i1>
      %49 = comb.xor bin %48, %true : i1
      %50 = comb.and bin %44, %49 : i1
      %51 = comb.or bin %47, %50 : i1
      cf.cond_br %51, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %52 = llhd.prb %presetn_1 : !hw.inout<i1>
      %53 = comb.xor %52, %true : i1
      cf.cond_br %53, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_int_mask, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %54 = llhd.prb %gpio_int_mask_wen : !hw.inout<i1>
      cf.cond_br %54, ^bb6, ^bb1
    ^bb6:  // pred: ^bb5
      %55 = llhd.prb %pwdata_4 : !hw.inout<i32>
      llhd.drv %gpio_int_mask, %55 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    %19 = llhd.prb %ri_gpio_int_mask : !hw.inout<i32>
    %20 = llhd.prb %gpio_int_mask : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %43 = llhd.prb %zero_value : !hw.inout<i1>
      %44 = comb.replicate %43 : (i1) -> i32
      llhd.drv %ri_gpio_int_mask, %44 after %1 : !hw.inout<i32>
      %45 = llhd.prb %gpio_int_mask : !hw.inout<i32>
      llhd.drv %ri_gpio_int_mask, %45 after %1 : !hw.inout<i32>
      llhd.wait (%16, %19, %20 : i1, i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 5 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb6
      %43 = llhd.prb %pclk_0 : !hw.inout<i1>
      %44 = llhd.prb %presetn_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %45 = llhd.prb %pclk_0 : !hw.inout<i1>
      %46 = comb.xor bin %43, %true : i1
      %47 = comb.and bin %46, %45 : i1
      %48 = llhd.prb %presetn_1 : !hw.inout<i1>
      %49 = comb.xor bin %48, %true : i1
      %50 = comb.and bin %44, %49 : i1
      %51 = comb.or bin %47, %50 : i1
      cf.cond_br %51, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %52 = llhd.prb %presetn_1 : !hw.inout<i1>
      %53 = comb.xor %52, %true : i1
      cf.cond_br %53, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_int_type, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %54 = llhd.prb %gpio_int_type_wen : !hw.inout<i1>
      cf.cond_br %54, ^bb6, ^bb1
    ^bb6:  // pred: ^bb5
      %55 = llhd.prb %pwdata_4 : !hw.inout<i32>
      llhd.drv %gpio_int_type, %55 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    %21 = llhd.prb %ri_gpio_int_type : !hw.inout<i32>
    %22 = llhd.prb %gpio_int_type : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %43 = llhd.prb %zero_value : !hw.inout<i1>
      %44 = comb.replicate %43 : (i1) -> i32
      llhd.drv %ri_gpio_int_type, %44 after %1 : !hw.inout<i32>
      %45 = llhd.prb %gpio_int_type : !hw.inout<i32>
      llhd.drv %ri_gpio_int_type, %45 after %1 : !hw.inout<i32>
      llhd.wait (%16, %21, %22 : i1, i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 5 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb6
      %43 = llhd.prb %pclk_0 : !hw.inout<i1>
      %44 = llhd.prb %presetn_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %45 = llhd.prb %pclk_0 : !hw.inout<i1>
      %46 = comb.xor bin %43, %true : i1
      %47 = comb.and bin %46, %45 : i1
      %48 = llhd.prb %presetn_1 : !hw.inout<i1>
      %49 = comb.xor bin %48, %true : i1
      %50 = comb.and bin %44, %49 : i1
      %51 = comb.or bin %47, %50 : i1
      cf.cond_br %51, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %52 = llhd.prb %presetn_1 : !hw.inout<i1>
      %53 = comb.xor %52, %true : i1
      cf.cond_br %53, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_int_pol, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %54 = llhd.prb %gpio_int_pol_wen : !hw.inout<i1>
      cf.cond_br %54, ^bb6, ^bb1
    ^bb6:  // pred: ^bb5
      %55 = llhd.prb %pwdata_4 : !hw.inout<i32>
      llhd.drv %gpio_int_pol, %55 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    %23 = llhd.prb %ri_gpio_int_pol : !hw.inout<i32>
    %24 = llhd.prb %gpio_int_pol : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %43 = llhd.prb %zero_value : !hw.inout<i1>
      %44 = comb.replicate %43 : (i1) -> i32
      llhd.drv %ri_gpio_int_pol, %44 after %1 : !hw.inout<i32>
      %45 = llhd.prb %gpio_int_pol : !hw.inout<i32>
      llhd.drv %ri_gpio_int_pol, %45 after %1 : !hw.inout<i32>
      llhd.wait (%16, %23, %24 : i1, i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 5 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb6
      %43 = llhd.prb %pclk_0 : !hw.inout<i1>
      %44 = llhd.prb %presetn_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %45 = llhd.prb %pclk_0 : !hw.inout<i1>
      %46 = comb.xor bin %43, %true : i1
      %47 = comb.and bin %46, %45 : i1
      %48 = llhd.prb %presetn_1 : !hw.inout<i1>
      %49 = comb.xor bin %48, %true : i1
      %50 = comb.and bin %44, %49 : i1
      %51 = comb.or bin %47, %50 : i1
      cf.cond_br %51, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %52 = llhd.prb %presetn_1 : !hw.inout<i1>
      %53 = comb.xor %52, %true : i1
      cf.cond_br %53, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_int_level_sync, %false after %0 : !hw.inout<i1>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %54 = llhd.prb %gpio_int_level_sync_wen : !hw.inout<i1>
      cf.cond_br %54, ^bb6, ^bb1
    ^bb6:  // pred: ^bb5
      %55 = llhd.prb %pwdata_4 : !hw.inout<i32>
      %56 = comb.extract %55 from 0 : (i32) -> i1
      llhd.drv %gpio_int_level_sync, %56 after %0 : !hw.inout<i1>
      cf.br ^bb1
    }
    %25 = llhd.prb %pwdata_4 : !hw.inout<i32>
    %26 = llhd.prb %gpio_int_clr : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      %43 = llhd.prb %gpio_int_clr_wen : !hw.inout<i1>
      cf.cond_br %43, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %44 = llhd.prb %pwdata_4 : !hw.inout<i32>
      llhd.drv %gpio_int_clr, %44 after %1 : !hw.inout<i32>
      cf.br ^bb4
    ^bb3:  // pred: ^bb1
      llhd.drv %gpio_int_clr, %c0_i32 after %1 : !hw.inout<i32>
      cf.br ^bb4
    ^bb4:  // 2 preds: ^bb2, ^bb3
      llhd.wait (%15, %25, %26 : i1, i32, i32), ^bb1
    }
    %27 = llhd.prb %ri_gpio_int_level_sync : !hw.inout<i32>
    %28 = llhd.prb %gpio_int_level_sync : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %43 = llhd.prb %zero_value : !hw.inout<i1>
      %44 = comb.replicate %43 : (i1) -> i32
      llhd.drv %ri_gpio_int_level_sync, %44 after %1 : !hw.inout<i32>
      %45 = llhd.sig.extract %ri_gpio_int_level_sync from %c0_i5 : (!hw.inout<i32>) -> !hw.inout<i1>
      %46 = llhd.prb %gpio_int_level_sync : !hw.inout<i1>
      llhd.drv %45, %46 after %1 : !hw.inout<i1>
      llhd.wait (%16, %27, %28 : i1, i32, i1), ^bb1
    }
    %29 = llhd.prb %ri_gpio_int_status : !hw.inout<i32>
    %30 = llhd.prb %gpio_int_status_8 : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %43 = llhd.prb %zero_value : !hw.inout<i1>
      %44 = comb.replicate %43 : (i1) -> i32
      llhd.drv %ri_gpio_int_status, %44 after %1 : !hw.inout<i32>
      %45 = llhd.prb %gpio_int_status_8 : !hw.inout<i32>
      llhd.drv %ri_gpio_int_status, %45 after %1 : !hw.inout<i32>
      llhd.wait (%16, %29, %30 : i1, i32, i32), ^bb1
    }
    %31 = llhd.prb %ri_gpio_raw_int_status : !hw.inout<i32>
    %32 = llhd.prb %gpio_raw_int_status_9 : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %43 = llhd.prb %zero_value : !hw.inout<i1>
      %44 = comb.replicate %43 : (i1) -> i32
      llhd.drv %ri_gpio_raw_int_status, %44 after %1 : !hw.inout<i32>
      %45 = llhd.prb %gpio_raw_int_status_9 : !hw.inout<i32>
      llhd.drv %ri_gpio_raw_int_status, %45 after %1 : !hw.inout<i32>
      llhd.wait (%16, %31, %32 : i1, i32, i32), ^bb1
    }
    %33 = llhd.prb %gpio_debounce_wen : !hw.inout<i1>
    %34 = llhd.prb %gpio_debounce : !hw.inout<i32>
    %35 = llhd.prb %ri_gpio_debounce : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %43 = llhd.prb %zero_value : !hw.inout<i1>
      llhd.drv %gpio_debounce_wen, %43 after %1 : !hw.inout<i1>
      %44 = llhd.prb %zero_value : !hw.inout<i1>
      %45 = comb.replicate %44 : (i1) -> i32
      llhd.drv %gpio_debounce, %45 after %1 : !hw.inout<i32>
      %46 = llhd.prb %zero_value : !hw.inout<i1>
      %47 = comb.replicate %46 : (i1) -> i32
      llhd.drv %ri_gpio_debounce, %47 after %1 : !hw.inout<i32>
      llhd.wait (%16, %33, %34, %35 : i1, i1, i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 5 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb6
      %43 = llhd.prb %pclk_0 : !hw.inout<i1>
      %44 = llhd.prb %presetn_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %45 = llhd.prb %pclk_0 : !hw.inout<i1>
      %46 = comb.xor bin %43, %true : i1
      %47 = comb.and bin %46, %45 : i1
      %48 = llhd.prb %presetn_1 : !hw.inout<i1>
      %49 = comb.xor bin %48, %true : i1
      %50 = comb.and bin %44, %49 : i1
      %51 = comb.or bin %47, %50 : i1
      cf.cond_br %51, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %52 = llhd.prb %presetn_1 : !hw.inout<i1>
      %53 = comb.xor %52, %true : i1
      cf.cond_br %53, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_sw_data, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %54 = llhd.prb %gpio_sw_data_wen : !hw.inout<i1>
      cf.cond_br %54, ^bb6, ^bb1
    ^bb6:  // pred: ^bb5
      %55 = llhd.prb %pwdata_4 : !hw.inout<i32>
      llhd.drv %gpio_sw_data, %55 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    %36 = llhd.prb %ri_gpio_sw_data : !hw.inout<i32>
    %37 = llhd.prb %gpio_sw_data : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      llhd.drv %ri_gpio_sw_data, %c0_i32 after %1 : !hw.inout<i32>
      %43 = llhd.prb %gpio_sw_data : !hw.inout<i32>
      llhd.drv %ri_gpio_sw_data, %43 after %1 : !hw.inout<i32>
      llhd.wait (%36, %37 : i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 5 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb6
      %43 = llhd.prb %pclk_0 : !hw.inout<i1>
      %44 = llhd.prb %presetn_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %45 = llhd.prb %pclk_0 : !hw.inout<i1>
      %46 = comb.xor bin %43, %true : i1
      %47 = comb.and bin %46, %45 : i1
      %48 = llhd.prb %presetn_1 : !hw.inout<i1>
      %49 = comb.xor bin %48, %true : i1
      %50 = comb.and bin %44, %49 : i1
      %51 = comb.or bin %47, %50 : i1
      cf.cond_br %51, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %52 = llhd.prb %presetn_1 : !hw.inout<i1>
      %53 = comb.xor %52, %true : i1
      cf.cond_br %53, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_sw_dir, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %54 = llhd.prb %gpio_sw_dir_wen : !hw.inout<i1>
      cf.cond_br %54, ^bb6, ^bb1
    ^bb6:  // pred: ^bb5
      %55 = llhd.prb %pwdata_4 : !hw.inout<i32>
      llhd.drv %gpio_sw_dir, %55 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    %38 = llhd.prb %ri_gpio_sw_dir : !hw.inout<i32>
    %39 = llhd.prb %gpio_sw_dir : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      llhd.drv %ri_gpio_sw_dir, %c0_i32 after %1 : !hw.inout<i32>
      %43 = llhd.prb %gpio_sw_dir : !hw.inout<i32>
      llhd.drv %ri_gpio_sw_dir, %43 after %1 : !hw.inout<i32>
      llhd.wait (%38, %39 : i32, i32), ^bb1
    }
    %40 = llhd.prb %ri_gpio_ext_data : !hw.inout<i32>
    %41 = llhd.prb %gpio_ext_data_7 : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      llhd.drv %ri_gpio_ext_data, %c0_i32 after %1 : !hw.inout<i32>
      %43 = llhd.prb %gpio_ext_data_7 : !hw.inout<i32>
      llhd.drv %ri_gpio_ext_data, %43 after %1 : !hw.inout<i32>
      llhd.wait (%40, %41 : i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 17 preds: ^bb0, ^bb2, ^bb4, ^bb5, ^bb7, ^bb9, ^bb11, ^bb13, ^bb15, ^bb17, ^bb19, ^bb21, ^bb23, ^bb25, ^bb27, ^bb29, ^bb30
      %43 = llhd.prb %pclk_0 : !hw.inout<i1>
      %44 = llhd.prb %presetn_1 : !hw.inout<i1>
      llhd.wait (%2, %3 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %45 = llhd.prb %pclk_0 : !hw.inout<i1>
      %46 = comb.xor bin %43, %true : i1
      %47 = comb.and bin %46, %45 : i1
      %48 = llhd.prb %presetn_1 : !hw.inout<i1>
      %49 = comb.xor bin %48, %true : i1
      %50 = comb.and bin %44, %49 : i1
      %51 = comb.or bin %47, %50 : i1
      cf.cond_br %51, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %52 = llhd.prb %presetn_1 : !hw.inout<i1>
      %53 = comb.xor %52, %true : i1
      cf.cond_br %53, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %prdata, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %54 = llhd.prb %pwrite_3 : !hw.inout<i1>
      %55 = comb.xor %54, %true : i1
      %56 = llhd.prb %psel_6 : !hw.inout<i1>
      %57 = llhd.prb %penable_2 : !hw.inout<i1>
      %58 = comb.xor %57, %true : i1
      %59 = comb.and %55, %56, %58 : i1
      cf.cond_br %59, ^bb6, ^bb1
    ^bb6:  // pred: ^bb5
      %60 = llhd.prb %paddr_5 : !hw.inout<i7>
      %61 = comb.extract %60 from 2 : (i7) -> i5
      %62 = comb.concat %c0_i27, %61 : i27, i5
      %63 = comb.icmp ceq %62, %c0_i32 : i32
      cf.cond_br %63, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      %64 = llhd.prb %ri_gpio_sw_data : !hw.inout<i32>
      llhd.drv %prdata, %64 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb8:  // pred: ^bb6
      %65 = comb.icmp ceq %62, %c1_i32 : i32
      cf.cond_br %65, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %66 = llhd.prb %ri_gpio_sw_dir : !hw.inout<i32>
      llhd.drv %prdata, %66 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb10:  // pred: ^bb8
      %67 = comb.icmp ceq %62, %c2_i32 : i32
      cf.cond_br %67, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      llhd.drv %prdata, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb12:  // pred: ^bb10
      %68 = comb.icmp ceq %62, %c12_i32 : i32
      cf.cond_br %68, ^bb13, ^bb14
    ^bb13:  // pred: ^bb12
      %69 = llhd.prb %ri_gpio_int_en : !hw.inout<i32>
      llhd.drv %prdata, %69 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb14:  // pred: ^bb12
      %70 = comb.icmp ceq %62, %c13_i32 : i32
      cf.cond_br %70, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      %71 = llhd.prb %ri_gpio_int_mask : !hw.inout<i32>
      llhd.drv %prdata, %71 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb16:  // pred: ^bb14
      %72 = comb.icmp ceq %62, %c14_i32 : i32
      cf.cond_br %72, ^bb17, ^bb18
    ^bb17:  // pred: ^bb16
      %73 = llhd.prb %ri_gpio_int_type : !hw.inout<i32>
      llhd.drv %prdata, %73 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb18:  // pred: ^bb16
      %74 = comb.icmp ceq %62, %c15_i32 : i32
      cf.cond_br %74, ^bb19, ^bb20
    ^bb19:  // pred: ^bb18
      %75 = llhd.prb %ri_gpio_int_pol : !hw.inout<i32>
      llhd.drv %prdata, %75 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb20:  // pred: ^bb18
      %76 = comb.icmp ceq %62, %c16_i32 : i32
      cf.cond_br %76, ^bb21, ^bb22
    ^bb21:  // pred: ^bb20
      %77 = llhd.prb %ri_gpio_int_status : !hw.inout<i32>
      llhd.drv %prdata, %77 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb22:  // pred: ^bb20
      %78 = comb.icmp ceq %62, %c17_i32 : i32
      cf.cond_br %78, ^bb23, ^bb24
    ^bb23:  // pred: ^bb22
      %79 = llhd.prb %ri_gpio_raw_int_status : !hw.inout<i32>
      llhd.drv %prdata, %79 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb24:  // pred: ^bb22
      %80 = comb.icmp ceq %62, %c18_i32 : i32
      cf.cond_br %80, ^bb25, ^bb26
    ^bb25:  // pred: ^bb24
      %81 = llhd.prb %ri_gpio_debounce : !hw.inout<i32>
      llhd.drv %prdata, %81 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb26:  // pred: ^bb24
      %82 = comb.icmp ceq %62, %c24_i32 : i32
      cf.cond_br %82, ^bb27, ^bb28
    ^bb27:  // pred: ^bb26
      %83 = llhd.prb %ri_gpio_int_level_sync : !hw.inout<i32>
      llhd.drv %prdata, %83 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb28:  // pred: ^bb26
      %84 = comb.icmp ceq %62, %c20_i32 : i32
      cf.cond_br %84, ^bb29, ^bb30
    ^bb29:  // pred: ^bb28
      %85 = llhd.prb %ri_gpio_ext_data : !hw.inout<i32>
      llhd.drv %prdata, %85 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb30:  // pred: ^bb28
      llhd.drv %prdata, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    llhd.drv %pclk_0, %pclk after %1 : !hw.inout<i1>
    llhd.drv %presetn_1, %presetn after %1 : !hw.inout<i1>
    llhd.drv %penable_2, %penable after %1 : !hw.inout<i1>
    llhd.drv %pwrite_3, %pwrite after %1 : !hw.inout<i1>
    llhd.drv %pwdata_4, %pwdata after %1 : !hw.inout<i32>
    llhd.drv %paddr_5, %paddr after %1 : !hw.inout<i7>
    llhd.drv %psel_6, %psel after %1 : !hw.inout<i1>
    llhd.drv %gpio_ext_data_7, %gpio_ext_data after %1 : !hw.inout<i32>
    llhd.drv %gpio_int_status_8, %gpio_int_status after %1 : !hw.inout<i32>
    llhd.drv %gpio_raw_int_status_9, %gpio_raw_int_status after %1 : !hw.inout<i32>
    %42 = llhd.prb %prdata : !hw.inout<i32>
    hw.output %37, %39, %18, %20, %22, %24, %34, %26, %28, %42 : i32, i32, i32, i32, i32, i32, i32, i32, i1, i32
  }
  hw.module private @gpio_ctrl(in %pclk : i1, in %pclk_int : i1, in %dbclk : i1, in %presetn : i1, in %dbclk_rstn : i1, in %scan_mode : i1, in %gpio_sw_data : i32, in %gpio_sw_dir : i32, in %gpio_int_en : i32, in %gpio_int_mask : i32, in %gpio_int_type : i32, in %gpio_int_pol : i32, in %gpio_debounce : i32, in %gpio_int_clr : i32, in %gpio_rx_data : i32, in %gpio_int_level_sync : i1, out gpio_tx_data : i32, out gpio_tx_en : i32, out gpio_ext_data : i32, out gpio_int_flag : i1, out gpio_int_flag_n : i1, out gpio_int_status : i32, out gpio_int : i32, out gpio_int_n : i32, out gpio_raw_int_status : i32, out gpio_int_clk_en : i1, out gpio_etb_trig : i32) {
    %c-1_i32 = hw.constant -1 : i32
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %c-1_i5 = hw.constant -1 : i5
    %c0_i27 = hw.constant 0 : i27
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %true = hw.constant true
    %false = hw.constant false
    %c1_i32 = hw.constant 1 : i32
    %c32_i32 = hw.constant 32 : i32
    %c0_i32 = hw.constant 0 : i32
    %pclk_0 = llhd.sig name "pclk" %false : i1
    %2 = llhd.prb %pclk_0 : !hw.inout<i1>
    %pclk_int_1 = llhd.sig name "pclk_int" %false : i1
    %3 = llhd.prb %pclk_int_1 : !hw.inout<i1>
    %presetn_2 = llhd.sig name "presetn" %false : i1
    %4 = llhd.prb %presetn_2 : !hw.inout<i1>
    %gpio_sw_data_3 = llhd.sig name "gpio_sw_data" %c0_i32 : i32
    %gpio_sw_dir_4 = llhd.sig name "gpio_sw_dir" %c0_i32 : i32
    %gpio_int_en_5 = llhd.sig name "gpio_int_en" %c0_i32 : i32
    %gpio_int_type_6 = llhd.sig name "gpio_int_type" %c0_i32 : i32
    %gpio_int_pol_7 = llhd.sig name "gpio_int_pol" %c0_i32 : i32
    %gpio_debounce_8 = llhd.sig name "gpio_debounce" %c0_i32 : i32
    %gpio_int_clr_9 = llhd.sig name "gpio_int_clr" %c0_i32 : i32
    %gpio_rx_data_10 = llhd.sig name "gpio_rx_data" %c0_i32 : i32
    %gpio_int_level_sync_11 = llhd.sig name "gpio_int_level_sync" %false : i1
    %gpio_tx_data = llhd.sig %c0_i32 : i32
    %gpio_tx_en = llhd.sig %c0_i32 : i32
    %gpio_ext_data = llhd.sig %c0_i32 : i32
    %gpio_int_status = llhd.sig %c0_i32 : i32
    %gpio_raw_int_status = llhd.sig %c0_i32 : i32
    %gpio_int_clk_en = llhd.sig %false : i1
    %gpio_rx_data_int = llhd.sig %c0_i32 : i32
    %int_level_sync_in = llhd.sig %c0_i32 : i32
    %int_edge_out = llhd.sig %c0_i32 : i32
    %debounce_d2 = llhd.sig %c0_i32 : i32
    %int_level = llhd.sig %c0_i32 : i32
    %gpio_ext_data_tmp = llhd.sig %c0_i32 : i32
    %zero_value = llhd.sig %false : i1
    %gpio_int_flag_tmp = llhd.sig %false : i1
    llhd.drv %zero_value, %false after %1 : !hw.inout<i1>
    %int_clk_en = llhd.sig %c0_i32 : i32
    %int_level_ff1 = llhd.sig %c0_i32 : i32
    %gpio_int_status_edge = llhd.sig %c0_i32 : i32
    %gpio_int_status_level = llhd.sig %c0_i32 : i32
    %gpio_int_clk_en_tmp = llhd.sig %false : i1
    %int_edge = llhd.sig %c0_i32 : i32
    %int_k = llhd.sig %c0_i32 : i32
    %5 = llhd.prb %gpio_int_en_5 : !hw.inout<i32>
    %6 = llhd.prb %gpio_int_type_6 : !hw.inout<i32>
    %7 = llhd.prb %int_clk_en : !hw.inout<i32>
    %8 = llhd.prb %gpio_int_level_sync_11 : !hw.inout<i1>
    llhd.process {
      cf.br ^bb2(%c0_i32 : i32)
    ^bb1:  // pred: ^bb9
      cf.br ^bb2(%c0_i32 : i32)
    ^bb2(%39: i32):  // 3 preds: ^bb0, ^bb1, ^bb8
      %40 = comb.icmp ult %39, %c32_i32 : i32
      cf.cond_br %40, ^bb3, ^bb9
    ^bb3:  // pred: ^bb2
      %41 = llhd.prb %gpio_int_en_5 : !hw.inout<i32>
      %42 = comb.shru %41, %39 : i32
      %43 = comb.extract %42 from 0 : (i32) -> i1
      cf.cond_br %43, ^bb4, ^bb7
    ^bb4:  // pred: ^bb3
      %44 = llhd.prb %gpio_int_type_6 : !hw.inout<i32>
      %45 = comb.shru %44, %39 : i32
      %46 = comb.extract %45 from 0 : (i32) -> i1
      cf.cond_br %46, ^bb5, ^bb6
    ^bb5:  // pred: ^bb4
      %47 = comb.extract %39 from 5 : (i32) -> i27
      %48 = comb.icmp eq %47, %c0_i27 : i27
      %49 = comb.extract %39 from 0 : (i32) -> i5
      %50 = comb.mux %48, %49, %c-1_i5 : i5
      %51 = llhd.sig.extract %int_clk_en from %50 : (!hw.inout<i32>) -> !hw.inout<i1>
      llhd.drv %51, %true after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb6:  // pred: ^bb4
      %52 = comb.extract %39 from 5 : (i32) -> i27
      %53 = comb.icmp eq %52, %c0_i27 : i27
      %54 = comb.extract %39 from 0 : (i32) -> i5
      %55 = comb.mux %53, %54, %c-1_i5 : i5
      %56 = llhd.sig.extract %int_clk_en from %55 : (!hw.inout<i32>) -> !hw.inout<i1>
      %57 = llhd.prb %gpio_int_level_sync_11 : !hw.inout<i1>
      llhd.drv %56, %57 after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb7:  // pred: ^bb3
      %58 = comb.extract %39 from 5 : (i32) -> i27
      %59 = comb.icmp eq %58, %c0_i27 : i27
      %60 = comb.extract %39 from 0 : (i32) -> i5
      %61 = comb.mux %59, %60, %c-1_i5 : i5
      %62 = llhd.sig.extract %int_clk_en from %61 : (!hw.inout<i32>) -> !hw.inout<i1>
      llhd.drv %62, %false after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb8:  // 3 preds: ^bb5, ^bb6, ^bb7
      %63 = comb.add %39, %c1_i32 : i32
      cf.br ^bb2(%63 : i32)
    ^bb9:  // pred: ^bb2
      llhd.wait (%5, %6, %7, %8 : i32, i32, i32, i1), ^bb1
    }
    %9 = comb.icmp ne %7, %c0_i32 : i32
    llhd.drv %gpio_int_clk_en_tmp, %9 after %1 : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %39 = llhd.prb %pclk_0 : !hw.inout<i1>
      %40 = llhd.prb %presetn_2 : !hw.inout<i1>
      llhd.wait (%2, %4 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %41 = llhd.prb %pclk_0 : !hw.inout<i1>
      %42 = comb.xor bin %39, %true : i1
      %43 = comb.and bin %42, %41 : i1
      %44 = llhd.prb %presetn_2 : !hw.inout<i1>
      %45 = comb.xor bin %44, %true : i1
      %46 = comb.and bin %40, %45 : i1
      %47 = comb.or bin %43, %46 : i1
      cf.cond_br %47, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %48 = llhd.prb %presetn_2 : !hw.inout<i1>
      %49 = comb.xor %48, %true : i1
      cf.cond_br %49, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_int_clk_en, %false after %0 : !hw.inout<i1>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %50 = llhd.prb %gpio_int_clk_en_tmp : !hw.inout<i1>
      llhd.drv %gpio_int_clk_en, %50 after %0 : !hw.inout<i1>
      cf.br ^bb1
    }
    %10 = llhd.prb %gpio_rx_data_int : !hw.inout<i32>
    %11 = llhd.prb %gpio_int_pol_7 : !hw.inout<i32>
    %12 = llhd.prb %gpio_rx_data_10 : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb7
      llhd.drv %gpio_rx_data_int, %c0_i32 after %1 : !hw.inout<i32>
      cf.br ^bb2(%c0_i32 : i32)
    ^bb2(%39: i32):  // 2 preds: ^bb1, ^bb6
      %40 = comb.icmp ult %39, %c32_i32 : i32
      cf.cond_br %40, ^bb3, ^bb7
    ^bb3:  // pred: ^bb2
      %41 = llhd.prb %gpio_int_pol_7 : !hw.inout<i32>
      %42 = comb.shru %41, %39 : i32
      %43 = comb.extract %42 from 0 : (i32) -> i1
      %44 = comb.xor %43, %true : i1
      cf.cond_br %44, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %45 = comb.extract %39 from 5 : (i32) -> i27
      %46 = comb.icmp eq %45, %c0_i27 : i27
      %47 = comb.extract %39 from 0 : (i32) -> i5
      %48 = comb.mux %46, %47, %c-1_i5 : i5
      %49 = llhd.sig.extract %gpio_rx_data_int from %48 : (!hw.inout<i32>) -> !hw.inout<i1>
      %50 = llhd.prb %gpio_rx_data_10 : !hw.inout<i32>
      %51 = comb.shru %50, %39 : i32
      %52 = comb.extract %51 from 0 : (i32) -> i1
      %53 = comb.xor %52, %true : i1
      llhd.drv %49, %53 after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb3
      %54 = comb.extract %39 from 5 : (i32) -> i27
      %55 = comb.icmp eq %54, %c0_i27 : i27
      %56 = comb.extract %39 from 0 : (i32) -> i5
      %57 = comb.mux %55, %56, %c-1_i5 : i5
      %58 = llhd.sig.extract %gpio_rx_data_int from %57 : (!hw.inout<i32>) -> !hw.inout<i1>
      %59 = llhd.prb %gpio_rx_data_10 : !hw.inout<i32>
      %60 = comb.shru %59, %39 : i32
      %61 = comb.extract %60 from 0 : (i32) -> i1
      llhd.drv %58, %61 after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 2 preds: ^bb4, ^bb5
      %62 = comb.add %39, %c1_i32 : i32
      cf.br ^bb2(%62 : i32)
    ^bb7:  // pred: ^bb2
      llhd.wait (%10, %11, %12 : i32, i32, i32), ^bb1
    }
    %13 = llhd.prb %int_level_sync_in : !hw.inout<i32>
    %14 = llhd.prb %gpio_debounce_8 : !hw.inout<i32>
    %15 = llhd.prb %debounce_d2 : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      llhd.drv %int_level_sync_in, %c0_i32 after %1 : !hw.inout<i32>
      %39 = llhd.prb %gpio_rx_data_int : !hw.inout<i32>
      llhd.drv %int_level_sync_in, %39 after %1 : !hw.inout<i32>
      llhd.wait (%13, %10, %14, %15 : i32, i32, i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %39 = llhd.prb %pclk_int_1 : !hw.inout<i1>
      %40 = llhd.prb %presetn_2 : !hw.inout<i1>
      llhd.wait (%3, %4 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %41 = llhd.prb %pclk_int_1 : !hw.inout<i1>
      %42 = comb.xor bin %39, %true : i1
      %43 = comb.and bin %42, %41 : i1
      %44 = llhd.prb %presetn_2 : !hw.inout<i1>
      %45 = comb.xor bin %44, %true : i1
      %46 = comb.and bin %40, %45 : i1
      %47 = comb.or bin %43, %46 : i1
      cf.cond_br %47, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %48 = llhd.prb %presetn_2 : !hw.inout<i1>
      %49 = comb.xor %48, %true : i1
      cf.cond_br %49, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %int_level_ff1, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %50 = llhd.prb %int_level : !hw.inout<i32>
      llhd.drv %int_level_ff1, %50 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    %16 = llhd.prb %int_level : !hw.inout<i32>
    %17 = llhd.prb %int_level_ff1 : !hw.inout<i32>
    %18 = comb.xor %16, %17 : i32
    llhd.drv %int_edge, %18 after %1 : !hw.inout<i32>
    %19 = llhd.prb %int_edge_out : !hw.inout<i32>
    %20 = llhd.prb %int_edge : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb4
      llhd.drv %int_edge_out, %c0_i32 after %1 : !hw.inout<i32>
      cf.br ^bb2(%c0_i32 : i32)
    ^bb2(%39: i32):  // 2 preds: ^bb1, ^bb3
      %40 = comb.icmp ult %39, %c32_i32 : i32
      cf.cond_br %40, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %41 = comb.extract %39 from 5 : (i32) -> i27
      %42 = comb.icmp eq %41, %c0_i27 : i27
      %43 = comb.extract %39 from 0 : (i32) -> i5
      %44 = comb.mux %42, %43, %c-1_i5 : i5
      %45 = llhd.sig.extract %int_edge_out from %44 : (!hw.inout<i32>) -> !hw.inout<i1>
      %46 = llhd.prb %int_edge : !hw.inout<i32>
      %47 = comb.shru %46, %39 : i32
      %48 = comb.extract %47 from 0 : (i32) -> i1
      %49 = llhd.prb %int_level : !hw.inout<i32>
      %50 = comb.shru %49, %39 : i32
      %51 = comb.extract %50 from 0 : (i32) -> i1
      %52 = comb.and %48, %51 : i1
      llhd.drv %45, %52 after %1 : !hw.inout<i1>
      %53 = comb.add %39, %c1_i32 : i32
      cf.br ^bb2(%53 : i32)
    ^bb4:  // pred: ^bb2
      llhd.wait (%19, %20, %16 : i32, i32, i32), ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb6
      %39 = llhd.prb %pclk_0 : !hw.inout<i1>
      %40 = llhd.prb %presetn_2 : !hw.inout<i1>
      llhd.wait (%2, %4 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %41 = llhd.prb %pclk_0 : !hw.inout<i1>
      %42 = comb.xor bin %39, %true : i1
      %43 = comb.and bin %42, %41 : i1
      %44 = llhd.prb %presetn_2 : !hw.inout<i1>
      %45 = comb.xor bin %44, %true : i1
      %46 = comb.and bin %40, %45 : i1
      %47 = comb.or bin %43, %46 : i1
      cf.cond_br %47, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %48 = llhd.prb %presetn_2 : !hw.inout<i1>
      %49 = comb.xor %48, %true : i1
      cf.cond_br %49, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %gpio_int_status_edge, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      llhd.drv %int_k, %c0_i32 after %1 : !hw.inout<i32>
      cf.br ^bb6
    ^bb6:  // 2 preds: ^bb5, ^bb13
      %50 = llhd.prb %int_k : !hw.inout<i32>
      %51 = comb.icmp ult %50, %c32_i32 : i32
      cf.cond_br %51, ^bb7, ^bb1
    ^bb7:  // pred: ^bb6
      %52 = llhd.prb %gpio_int_en_5 : !hw.inout<i32>
      %53 = llhd.prb %int_k : !hw.inout<i32>
      %54 = comb.shru %52, %53 : i32
      %55 = comb.extract %54 from 0 : (i32) -> i1
      %56 = comb.xor %55, %true : i1
      cf.cond_br %56, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %57 = llhd.prb %int_k : !hw.inout<i32>
      %58 = comb.extract %57 from 5 : (i32) -> i27
      %59 = comb.icmp eq %58, %c0_i27 : i27
      %60 = comb.extract %57 from 0 : (i32) -> i5
      %61 = comb.mux %59, %60, %c-1_i5 : i5
      %62 = llhd.sig.extract %gpio_int_status_edge from %61 : (!hw.inout<i32>) -> !hw.inout<i1>
      llhd.drv %62, %false after %0 : !hw.inout<i1>
      cf.br ^bb13
    ^bb9:  // pred: ^bb7
      %63 = llhd.prb %int_edge_out : !hw.inout<i32>
      %64 = llhd.prb %int_k : !hw.inout<i32>
      %65 = comb.shru %63, %64 : i32
      %66 = comb.extract %65 from 0 : (i32) -> i1
      %67 = llhd.prb %gpio_int_en_5 : !hw.inout<i32>
      %68 = comb.shru %67, %64 : i32
      %69 = comb.extract %68 from 0 : (i32) -> i1
      %70 = llhd.prb %gpio_sw_dir_4 : !hw.inout<i32>
      %71 = comb.shru %70, %64 : i32
      %72 = comb.extract %71 from 0 : (i32) -> i1
      %73 = comb.xor %72, %true : i1
      %74 = comb.and %66, %69, %73 : i1
      cf.cond_br %74, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %75 = llhd.prb %int_k : !hw.inout<i32>
      %76 = comb.extract %75 from 5 : (i32) -> i27
      %77 = comb.icmp eq %76, %c0_i27 : i27
      %78 = comb.extract %75 from 0 : (i32) -> i5
      %79 = comb.mux %77, %78, %c-1_i5 : i5
      %80 = llhd.sig.extract %gpio_int_status_edge from %79 : (!hw.inout<i32>) -> !hw.inout<i1>
      llhd.drv %80, %true after %0 : !hw.inout<i1>
      cf.br ^bb13
    ^bb11:  // pred: ^bb9
      %81 = llhd.prb %gpio_int_clr_9 : !hw.inout<i32>
      %82 = llhd.prb %int_k : !hw.inout<i32>
      %83 = comb.shru %81, %82 : i32
      %84 = comb.extract %83 from 0 : (i32) -> i1
      cf.cond_br %84, ^bb12, ^bb13
    ^bb12:  // pred: ^bb11
      %85 = llhd.prb %int_k : !hw.inout<i32>
      %86 = comb.extract %85 from 5 : (i32) -> i27
      %87 = comb.icmp eq %86, %c0_i27 : i27
      %88 = comb.extract %85 from 0 : (i32) -> i5
      %89 = comb.mux %87, %88, %c-1_i5 : i5
      %90 = llhd.sig.extract %gpio_int_status_edge from %89 : (!hw.inout<i32>) -> !hw.inout<i1>
      llhd.drv %90, %false after %0 : !hw.inout<i1>
      cf.br ^bb13
    ^bb13:  // 4 preds: ^bb8, ^bb10, ^bb11, ^bb12
      %91 = llhd.prb %int_k : !hw.inout<i32>
      %92 = comb.add %91, %c1_i32 : i32
      llhd.drv %int_k, %92 after %1 : !hw.inout<i32>
      cf.br ^bb6
    }
    %21 = llhd.prb %gpio_int_status_level : !hw.inout<i32>
    %22 = llhd.prb %gpio_sw_dir_4 : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb9
      llhd.drv %gpio_int_status_level, %c0_i32 after %1 : !hw.inout<i32>
      cf.br ^bb2(%c0_i32 : i32)
    ^bb2(%39: i32):  // 2 preds: ^bb1, ^bb8
      %40 = comb.icmp ult %39, %c32_i32 : i32
      cf.cond_br %40, ^bb3, ^bb9
    ^bb3:  // pred: ^bb2
      %41 = llhd.prb %gpio_sw_dir_4 : !hw.inout<i32>
      %42 = comb.shru %41, %39 : i32
      %43 = comb.extract %42 from 0 : (i32) -> i1
      cf.cond_br %43, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %44 = comb.extract %39 from 5 : (i32) -> i27
      %45 = comb.icmp eq %44, %c0_i27 : i27
      %46 = comb.extract %39 from 0 : (i32) -> i5
      %47 = comb.mux %45, %46, %c-1_i5 : i5
      %48 = llhd.sig.extract %gpio_int_status_level from %47 : (!hw.inout<i32>) -> !hw.inout<i1>
      llhd.drv %48, %false after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb5:  // pred: ^bb3
      %49 = llhd.prb %gpio_int_level_sync_11 : !hw.inout<i1>
      cf.cond_br %49, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %50 = comb.extract %39 from 5 : (i32) -> i27
      %51 = comb.icmp eq %50, %c0_i27 : i27
      %52 = comb.extract %39 from 0 : (i32) -> i5
      %53 = comb.mux %51, %52, %c-1_i5 : i5
      %54 = llhd.sig.extract %gpio_int_status_level from %53 : (!hw.inout<i32>) -> !hw.inout<i1>
      %55 = llhd.prb %int_level : !hw.inout<i32>
      %56 = comb.shru %55, %39 : i32
      %57 = comb.extract %56 from 0 : (i32) -> i1
      llhd.drv %54, %57 after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb7:  // pred: ^bb5
      %58 = comb.extract %39 from 5 : (i32) -> i27
      %59 = comb.icmp eq %58, %c0_i27 : i27
      %60 = comb.extract %39 from 0 : (i32) -> i5
      %61 = comb.mux %59, %60, %c-1_i5 : i5
      %62 = llhd.sig.extract %gpio_int_status_level from %61 : (!hw.inout<i32>) -> !hw.inout<i1>
      %63 = llhd.prb %int_level_sync_in : !hw.inout<i32>
      %64 = comb.shru %63, %39 : i32
      %65 = comb.extract %64 from 0 : (i32) -> i1
      llhd.drv %62, %65 after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb8:  // 3 preds: ^bb4, ^bb6, ^bb7
      %66 = comb.add %39, %c1_i32 : i32
      cf.br ^bb2(%66 : i32)
    ^bb9:  // pred: ^bb2
      llhd.wait (%21, %22, %8, %16, %13 : i32, i32, i1, i32, i32), ^bb1
    }
    %23 = llhd.prb %gpio_raw_int_status : !hw.inout<i32>
    %24 = llhd.prb %gpio_int_status_edge : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb9
      llhd.drv %gpio_raw_int_status, %c0_i32 after %1 : !hw.inout<i32>
      cf.br ^bb2(%c0_i32 : i32)
    ^bb2(%39: i32):  // 2 preds: ^bb1, ^bb8
      %40 = comb.icmp ult %39, %c32_i32 : i32
      cf.cond_br %40, ^bb3, ^bb9
    ^bb3:  // pred: ^bb2
      %41 = llhd.prb %gpio_int_en_5 : !hw.inout<i32>
      %42 = comb.shru %41, %39 : i32
      %43 = comb.extract %42 from 0 : (i32) -> i1
      %44 = comb.xor %43, %true : i1
      cf.cond_br %44, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %45 = comb.extract %39 from 5 : (i32) -> i27
      %46 = comb.icmp eq %45, %c0_i27 : i27
      %47 = comb.extract %39 from 0 : (i32) -> i5
      %48 = comb.mux %46, %47, %c-1_i5 : i5
      %49 = llhd.sig.extract %gpio_raw_int_status from %48 : (!hw.inout<i32>) -> !hw.inout<i1>
      llhd.drv %49, %false after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb5:  // pred: ^bb3
      %50 = llhd.prb %gpio_int_type_6 : !hw.inout<i32>
      %51 = comb.shru %50, %39 : i32
      %52 = comb.extract %51 from 0 : (i32) -> i1
      cf.cond_br %52, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %53 = comb.extract %39 from 5 : (i32) -> i27
      %54 = comb.icmp eq %53, %c0_i27 : i27
      %55 = comb.extract %39 from 0 : (i32) -> i5
      %56 = comb.mux %54, %55, %c-1_i5 : i5
      %57 = llhd.sig.extract %gpio_raw_int_status from %56 : (!hw.inout<i32>) -> !hw.inout<i1>
      %58 = llhd.prb %gpio_int_status_edge : !hw.inout<i32>
      %59 = comb.shru %58, %39 : i32
      %60 = comb.extract %59 from 0 : (i32) -> i1
      llhd.drv %57, %60 after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb7:  // pred: ^bb5
      %61 = comb.extract %39 from 5 : (i32) -> i27
      %62 = comb.icmp eq %61, %c0_i27 : i27
      %63 = comb.extract %39 from 0 : (i32) -> i5
      %64 = comb.mux %62, %63, %c-1_i5 : i5
      %65 = llhd.sig.extract %gpio_raw_int_status from %64 : (!hw.inout<i32>) -> !hw.inout<i1>
      %66 = llhd.prb %gpio_int_status_level : !hw.inout<i32>
      %67 = comb.shru %66, %39 : i32
      %68 = comb.extract %67 from 0 : (i32) -> i1
      llhd.drv %65, %68 after %1 : !hw.inout<i1>
      cf.br ^bb8
    ^bb8:  // 3 preds: ^bb4, ^bb6, ^bb7
      %69 = comb.add %39, %c1_i32 : i32
      cf.br ^bb2(%69 : i32)
    ^bb9:  // pred: ^bb2
      llhd.wait (%23, %5, %6, %24, %21 : i32, i32, i32, i32, i32), ^bb1
    }
    %25 = comb.xor %gpio_int_mask, %c-1_i32 : i32
    %26 = comb.and %23, %25 : i32
    llhd.drv %gpio_int_status, %26 after %1 : !hw.inout<i32>
    %27 = llhd.prb %gpio_int_status : !hw.inout<i32>
    %28 = comb.icmp ne %27, %c0_i32 : i32
    llhd.drv %gpio_int_flag_tmp, %28 after %1 : !hw.inout<i1>
    %29 = llhd.prb %gpio_int_flag_tmp : !hw.inout<i1>
    %SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 = llhd.sig %c0_i32 : i32
    %SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 = llhd.sig %c0_i32 : i32
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %39 = llhd.prb %pclk_int_1 : !hw.inout<i1>
      %40 = llhd.prb %presetn_2 : !hw.inout<i1>
      llhd.wait (%3, %4 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %41 = llhd.prb %pclk_int_1 : !hw.inout<i1>
      %42 = comb.xor bin %39, %true : i1
      %43 = comb.and bin %42, %41 : i1
      %44 = llhd.prb %presetn_2 : !hw.inout<i1>
      %45 = comb.xor bin %44, %true : i1
      %46 = comb.and bin %40, %45 : i1
      %47 = comb.or bin %43, %46 : i1
      cf.cond_br %47, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %48 = llhd.prb %presetn_2 : !hw.inout<i1>
      %49 = comb.xor %48, %true : i1
      cf.cond_br %49, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %50 = llhd.prb %int_level_sync_in : !hw.inout<i32>
      llhd.drv %SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1, %50 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb4, ^bb5
      %39 = llhd.prb %pclk_int_1 : !hw.inout<i1>
      %40 = llhd.prb %presetn_2 : !hw.inout<i1>
      llhd.wait (%3, %4 : i1, i1), ^bb2
    ^bb2:  // pred: ^bb1
      %41 = llhd.prb %pclk_int_1 : !hw.inout<i1>
      %42 = comb.xor bin %39, %true : i1
      %43 = comb.and bin %42, %41 : i1
      %44 = llhd.prb %presetn_2 : !hw.inout<i1>
      %45 = comb.xor bin %44, %true : i1
      %46 = comb.and bin %40, %45 : i1
      %47 = comb.or bin %43, %46 : i1
      cf.cond_br %47, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %48 = llhd.prb %presetn_2 : !hw.inout<i1>
      %49 = comb.xor %48, %true : i1
      cf.cond_br %49, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      llhd.drv %SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2, %c0_i32 after %0 : !hw.inout<i32>
      cf.br ^bb1
    ^bb5:  // pred: ^bb3
      %50 = llhd.prb %SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff1 : !hw.inout<i32>
      llhd.drv %SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2, %50 after %0 : !hw.inout<i32>
      cf.br ^bb1
    }
    %30 = llhd.prb %SUPPORT_INT_LEVEL_SYNC_PROC.int_level_sync_in_ff2 : !hw.inout<i32>
    %31 = comb.mux %8, %30, %13 : i32
    llhd.drv %int_level, %31 after %1 : !hw.inout<i32>
    %32 = llhd.prb %zero_value : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %39 = llhd.prb %zero_value : !hw.inout<i1>
      %40 = comb.replicate %39 : (i1) -> i32
      llhd.drv %debounce_d2, %40 after %1 : !hw.inout<i32>
      llhd.wait (%32, %15 : i1, i32), ^bb1
    }
    %33 = llhd.prb %gpio_tx_data : !hw.inout<i32>
    %34 = llhd.prb %gpio_sw_data_3 : !hw.inout<i32>
    llhd.process {
      cf.br ^bb2(%c0_i32 : i32)
    ^bb1:  // pred: ^bb4
      cf.br ^bb2(%c0_i32 : i32)
    ^bb2(%39: i32):  // 3 preds: ^bb0, ^bb1, ^bb3
      %40 = comb.icmp ult %39, %c32_i32 : i32
      cf.cond_br %40, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %41 = comb.extract %39 from 5 : (i32) -> i27
      %42 = comb.icmp eq %41, %c0_i27 : i27
      %43 = comb.extract %39 from 0 : (i32) -> i5
      %44 = comb.mux %42, %43, %c-1_i5 : i5
      %45 = llhd.sig.extract %gpio_tx_data from %44 : (!hw.inout<i32>) -> !hw.inout<i1>
      %46 = llhd.prb %gpio_sw_data_3 : !hw.inout<i32>
      %47 = comb.shru %46, %39 : i32
      %48 = comb.extract %47 from 0 : (i32) -> i1
      llhd.drv %45, %48 after %1 : !hw.inout<i1>
      %49 = comb.add %39, %c1_i32 : i32
      cf.br ^bb2(%49 : i32)
    ^bb4:  // pred: ^bb2
      llhd.wait (%33, %34 : i32, i32), ^bb1
    }
    %35 = llhd.prb %gpio_tx_en : !hw.inout<i32>
    llhd.process {
      cf.br ^bb2(%c0_i32 : i32)
    ^bb1:  // pred: ^bb4
      cf.br ^bb2(%c0_i32 : i32)
    ^bb2(%39: i32):  // 3 preds: ^bb0, ^bb1, ^bb3
      %40 = comb.icmp ult %39, %c32_i32 : i32
      cf.cond_br %40, ^bb3, ^bb4
    ^bb3:  // pred: ^bb2
      %41 = comb.extract %39 from 5 : (i32) -> i27
      %42 = comb.icmp eq %41, %c0_i27 : i27
      %43 = comb.extract %39 from 0 : (i32) -> i5
      %44 = comb.mux %42, %43, %c-1_i5 : i5
      %45 = llhd.sig.extract %gpio_tx_en from %44 : (!hw.inout<i32>) -> !hw.inout<i1>
      %46 = llhd.prb %gpio_sw_dir_4 : !hw.inout<i32>
      %47 = comb.shru %46, %39 : i32
      %48 = comb.extract %47 from 0 : (i32) -> i1
      llhd.drv %45, %48 after %1 : !hw.inout<i1>
      %49 = comb.add %39, %c1_i32 : i32
      cf.br ^bb2(%49 : i32)
    ^bb4:  // pred: ^bb2
      llhd.wait (%35, %22 : i32, i32), ^bb1
    }
    llhd.drv %gpio_ext_data_tmp, %12 after %1 : !hw.inout<i32>
    %36 = llhd.prb %gpio_ext_data : !hw.inout<i32>
    %37 = llhd.prb %gpio_ext_data_tmp : !hw.inout<i32>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb7
      %39 = llhd.prb %zero_value : !hw.inout<i1>
      %40 = comb.replicate %39 : (i1) -> i32
      llhd.drv %gpio_ext_data, %40 after %1 : !hw.inout<i32>
      cf.br ^bb2(%c0_i32 : i32)
    ^bb2(%41: i32):  // 2 preds: ^bb1, ^bb6
      %42 = comb.icmp ult %41, %c32_i32 : i32
      cf.cond_br %42, ^bb3, ^bb7
    ^bb3:  // pred: ^bb2
      %43 = llhd.prb %gpio_tx_en : !hw.inout<i32>
      %44 = comb.shru %43, %41 : i32
      %45 = comb.extract %44 from 0 : (i32) -> i1
      cf.cond_br %45, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %46 = comb.extract %41 from 5 : (i32) -> i27
      %47 = comb.icmp eq %46, %c0_i27 : i27
      %48 = comb.extract %41 from 0 : (i32) -> i5
      %49 = comb.mux %47, %48, %c-1_i5 : i5
      %50 = llhd.sig.extract %gpio_ext_data from %49 : (!hw.inout<i32>) -> !hw.inout<i1>
      %51 = llhd.prb %gpio_sw_data_3 : !hw.inout<i32>
      %52 = comb.shru %51, %41 : i32
      %53 = comb.extract %52 from 0 : (i32) -> i1
      llhd.drv %50, %53 after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb5:  // pred: ^bb3
      %54 = comb.extract %41 from 5 : (i32) -> i27
      %55 = comb.icmp eq %54, %c0_i27 : i27
      %56 = comb.extract %41 from 0 : (i32) -> i5
      %57 = comb.mux %55, %56, %c-1_i5 : i5
      %58 = llhd.sig.extract %gpio_ext_data from %57 : (!hw.inout<i32>) -> !hw.inout<i1>
      %59 = llhd.prb %gpio_ext_data_tmp : !hw.inout<i32>
      %60 = comb.shru %59, %41 : i32
      %61 = comb.extract %60 from 0 : (i32) -> i1
      llhd.drv %58, %61 after %1 : !hw.inout<i1>
      cf.br ^bb6
    ^bb6:  // 2 preds: ^bb4, ^bb5
      %62 = comb.add %41, %c1_i32 : i32
      cf.br ^bb2(%62 : i32)
    ^bb7:  // pred: ^bb2
      llhd.wait (%32, %36, %35, %34, %37 : i1, i32, i32, i32, i32), ^bb1
    }
    llhd.drv %pclk_0, %pclk after %1 : !hw.inout<i1>
    llhd.drv %pclk_int_1, %pclk_int after %1 : !hw.inout<i1>
    llhd.drv %presetn_2, %presetn after %1 : !hw.inout<i1>
    llhd.drv %gpio_sw_data_3, %gpio_sw_data after %1 : !hw.inout<i32>
    llhd.drv %gpio_sw_dir_4, %gpio_sw_dir after %1 : !hw.inout<i32>
    llhd.drv %gpio_int_en_5, %gpio_int_en after %1 : !hw.inout<i32>
    llhd.drv %gpio_int_type_6, %gpio_int_type after %1 : !hw.inout<i32>
    llhd.drv %gpio_int_pol_7, %gpio_int_pol after %1 : !hw.inout<i32>
    llhd.drv %gpio_debounce_8, %gpio_debounce after %1 : !hw.inout<i32>
    llhd.drv %gpio_int_clr_9, %gpio_int_clr after %1 : !hw.inout<i32>
    llhd.drv %gpio_rx_data_10, %gpio_rx_data after %1 : !hw.inout<i32>
    llhd.drv %gpio_int_level_sync_11, %gpio_int_level_sync after %1 : !hw.inout<i1>
    %38 = llhd.prb %gpio_int_clk_en : !hw.inout<i1>
    hw.output %33, %35, %36, %29, %29, %27, %27, %27, %23, %38, %16 : i32, i32, i32, i1, i1, i32, i32, i32, i32, i1, i32
  }
  hw.module private @gpio_top(out gpio_etb_trig : i32, in %pclk : i1, in %pclk_int : i1, in %dbclk : i1, in %dbclk_rstn : i1, in %scan_mode : i1, in %presetn : i1, in %penable : i1, in %pwrite : i1, in %pwdata : i32, in %paddr : i7, in %psel : i1, in %gpio_in_data : i32, out gpio_out_data : i32, out gpio_out_en : i32, out gpio_int_flag : i1, out gpio_int_clk_en : i1, out prdata : i32) {
    %U_GPIO_APBIF.gpio_sw_data, %U_GPIO_APBIF.gpio_sw_dir, %U_GPIO_APBIF.gpio_int_en, %U_GPIO_APBIF.gpio_int_mask, %U_GPIO_APBIF.gpio_int_type, %U_GPIO_APBIF.gpio_int_pol, %U_GPIO_APBIF.gpio_debounce, %U_GPIO_APBIF.gpio_int_clr, %U_GPIO_APBIF.gpio_int_level_sync, %U_GPIO_APBIF.prdata = hw.instance "U_GPIO_APBIF" @gpio_apbif(pclk: %pclk: i1, presetn: %presetn: i1, penable: %penable: i1, pwrite: %pwrite: i1, pwdata: %pwdata: i32, paddr: %paddr: i7, psel: %psel: i1, gpio_ext_data: %U_GPIO_CTRL.gpio_ext_data: i32, gpio_int_status: %U_GPIO_CTRL.gpio_int_status: i32, gpio_raw_int_status: %U_GPIO_CTRL.gpio_raw_int_status: i32) -> (gpio_sw_data: i32, gpio_sw_dir: i32, gpio_int_en: i32, gpio_int_mask: i32, gpio_int_type: i32, gpio_int_pol: i32, gpio_debounce: i32, gpio_int_clr: i32, gpio_int_level_sync: i1, prdata: i32) {sv.namehint = "gpio_debounce"}
    %U_GPIO_CTRL.gpio_tx_data, %U_GPIO_CTRL.gpio_tx_en, %U_GPIO_CTRL.gpio_ext_data, %U_GPIO_CTRL.gpio_int_flag, %U_GPIO_CTRL.gpio_int_flag_n, %U_GPIO_CTRL.gpio_int_status, %U_GPIO_CTRL.gpio_int, %U_GPIO_CTRL.gpio_int_n, %U_GPIO_CTRL.gpio_raw_int_status, %U_GPIO_CTRL.gpio_int_clk_en, %U_GPIO_CTRL.gpio_etb_trig = hw.instance "U_GPIO_CTRL" @gpio_ctrl(pclk: %pclk: i1, pclk_int: %pclk_int: i1, dbclk: %dbclk: i1, presetn: %presetn: i1, dbclk_rstn: %dbclk_rstn: i1, scan_mode: %scan_mode: i1, gpio_sw_data: %U_GPIO_APBIF.gpio_sw_data: i32, gpio_sw_dir: %U_GPIO_APBIF.gpio_sw_dir: i32, gpio_int_en: %U_GPIO_APBIF.gpio_int_en: i32, gpio_int_mask: %U_GPIO_APBIF.gpio_int_mask: i32, gpio_int_type: %U_GPIO_APBIF.gpio_int_type: i32, gpio_int_pol: %U_GPIO_APBIF.gpio_int_pol: i32, gpio_debounce: %U_GPIO_APBIF.gpio_debounce: i32, gpio_int_clr: %U_GPIO_APBIF.gpio_int_clr: i32, gpio_rx_data: %gpio_in_data: i32, gpio_int_level_sync: %U_GPIO_APBIF.gpio_int_level_sync: i1) -> (gpio_tx_data: i32, gpio_tx_en: i32, gpio_ext_data: i32, gpio_int_flag: i1, gpio_int_flag_n: i1, gpio_int_status: i32, gpio_int: i32, gpio_int_n: i32, gpio_raw_int_status: i32, gpio_int_clk_en: i1, gpio_etb_trig: i32) {sv.namehint = "gpio_int"}
    hw.output %U_GPIO_CTRL.gpio_etb_trig, %U_GPIO_CTRL.gpio_tx_data, %U_GPIO_CTRL.gpio_tx_en, %U_GPIO_CTRL.gpio_int_flag, %U_GPIO_CTRL.gpio_int_clk_en, %U_GPIO_APBIF.prdata : i32, i32, i32, i1, i1, i32
  }
}
