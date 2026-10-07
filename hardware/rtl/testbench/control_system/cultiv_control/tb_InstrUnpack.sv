// tb_InstrUnpack.sv — directed testbench for InstrUnpack.
//
// Written in BOTH styles:
//   (1) an explicit table below with the expected values also passed inline to
//       every check() call, so the Vivado waveform can be read off directly;
//   (2) self-checking with mismatch counting and PASS/FAIL for Verilator / xsim.
//
// Mirrors the SystemC unit test
//   emulator/tests/cultiv_control/leaf/physical_mmio/test_instr_unpack.cpp
//
// WT_W=8, ADDR_W=16 -> INSTR_W = 40. The module is purely combinational, so
// there is no clock: each step applies a word and checks the fields after a
// settling delay.
//
// ============================================================================
//  step | instr (hex, 40-bit)  |   wt  start_addr end_addr | note
// ------+----------------------+--------------------------+---------------------------
//   0   | 00_0000_0000         |    0       0        0     | all zero
//   1   | 00_0005_0007         |    0       5        7     | wt=0 typical
//   2   | 02_0014_0016         |    2      20       22     | wt>0 typical
//   3   | 00_000a_000a         |    0      10       10     | start == end
//   4   | ff_ffff_ffff         |  255   65535    65535     | all ones (field boundaries)
//   5   | 01_0000_ffff         |    1       0    65535     | min/max mix
//   6   | aa_aaaa_5555         |  170   43690    21845     | alternating bit patterns
// ============================================================================
//
// Steps 4 and 6 are the ones that matter: they would catch a field-boundary
// off-by-one or a swapped start/end, which the small values in steps 0-3 would
// not necessarily expose.
`timescale 1ns / 1ps

module tb_InstrUnpack;

    localparam int WT_W    = 8;
    localparam int ADDR_W  = 16;
    localparam int INSTR_W = 2*ADDR_W + WT_W;   // 40

    logic [INSTR_W-1:0] instr;
    logic [WT_W-1:0]    wt;
    logic [ADDR_W-1:0]  start_addr;
    logic [ADDR_W-1:0]  end_addr;

    int errors = 0;
    int step   = 0;

    InstrUnpack #(
        .WT_W   (WT_W),
        .ADDR_W (ADDR_W)
    ) dut (
        .instr      (instr),
        .wt         (wt),
        .start_addr (start_addr),
        .end_addr   (end_addr)
    );

    // pack {wt, start, end} per the documented LSB-first layout, apply, check
    task automatic check(input logic [WT_W-1:0]   exp_wt,
                         input logic [ADDR_W-1:0] exp_start,
                         input logic [ADDR_W-1:0] exp_end,
                         input string             note);
        instr = {exp_wt, exp_start, exp_end};
        #5;
        if (wt !== exp_wt || start_addr !== exp_start || end_addr !== exp_end) begin
            errors++;
            $display("  step %0d MISMATCH  instr=%010h  got wt=%0d start=%0d end=%0d | exp wt=%0d start=%0d end=%0d   %s",
                     step, instr, wt, start_addr, end_addr, exp_wt, exp_start, exp_end, note);
        end else begin
            $display("  step %0d ok        instr=%010h  wt=%0d start=%0d end=%0d   %s",
                     step, instr, wt, start_addr, end_addr, note);
        end
        step++;
    endtask

    initial begin
        $display("tb_InstrUnpack: WT_W=%0d ADDR_W=%0d INSTR_W=%0d", WT_W, ADDR_W, INSTR_W);

        //     wt      start        end        note
        check(8'd0,   16'd0,      16'd0,     "all zero");
        check(8'd0,   16'd5,      16'd7,     "wt=0 typical");
        check(8'd2,   16'd20,     16'd22,    "wt>0 typical");
        check(8'd0,   16'd10,     16'd10,    "start == end");
        check(8'd255, 16'd65535,  16'd65535, "all ones (field boundaries)");
        check(8'd1,   16'd0,      16'd65535, "min/max mix");
        check(8'd170, 16'd43690,  16'd21845, "alternating bit patterns");

        if (errors == 0) $display("tb_InstrUnpack: PASS");
        else             $display("tb_InstrUnpack: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
