// tb_InstrDecode.sv — directed testbench for InstrDecode.
//
// Written in BOTH styles:
//   (1) an explicit cycle-by-cycle table below, and the expected values passed
//       inline to every step() call, so the waveform can be read off directly
//       in Vivado and compared against the comments;
//   (2) self-checking: every step compares the DUT outputs against those same
//       expected values, counts mismatches, and prints PASS/FAIL at the end
//       (so Verilator / xsim can verify it without looking at the waveform).
//
// Mirrors the SystemC unit test
//   emulator/tests/cultiv_control/leaf/physical_mmio/test_instr_decode.cpp
//
// TIMING. Outputs are Mealy (combinational), so each step drives the inputs on
// the NEGEDGE and checks the outputs 1 ns later — i.e. within the same cycle,
// before the posedge that advances the FSM state.
//
// ============================================================================
//  cyc | rst en  w_t start end | r_en r_addr one_instr_end | note
// -----+-----------------------+--------------------------+---------------------------
//   0  |  0   1   0     5    7 |  1      5        0        | A starts, w_t=0 fast path
//   1  |  0   0   -     -    - |  1      6        0        | A streaming
//   2  |  0   0   -     -    - |  1      7        1        | A last read -> one_instr_end
//   3  |  0   1   0    10   10 |  1     10        1        | B: start==end; NO bubble after A
//   4  |  0   1   2    20   22 |  0      0        0        | C: w_t=2, idle cycle 1
//   5  |  0   0   -     -    - |  0      0        0        | C: idle cycle 2
//   6  |  0   0   -     -    - |  1     20        0        | C: stream starts
//   7  |  0   0   -     -    - |  1     21        0        | C: streaming
//   8  |  0   0   -     -    - |  1     22        1        | C last read -> one_instr_end
//   9  |  0   0   -     -    - |  0      0        0        | idle
//  10  |  0   1   0    30   34 |  1     30        0        | D starts (will be aborted)
//  11  |  0   0   -     -    - |  1     31        0        | D streaming
//  12  |  1   0   -     -    - |  0      0        0        | RESET mid-stream kills D
//  13  |  0   1   0    40   41 |  1     40        0        | E starts clean after reset
//  14  |  0   0   -     -    - |  1     41        1        | E last read -> one_instr_end
//  15  |  0   0   -     -    - |  0      0        0        | idle
// ============================================================================
//
// Key things to look for in the waveform:
//   * cycles 2->3: r_addr goes 7 then 10 on CONSECUTIVE cycles (no bubble) —
//     this is the no-stall instruction handoff.
//   * cycle 3: one read only, with one_instr_end in the same cycle (start==end).
//   * cycles 4,5: r_en low for exactly w_t=2 cycles, then the stream begins.
//   * cycle 12: rst forces the outputs idle mid-stream; D never reaches 34.
`timescale 1ns / 1ps

module tb_InstrDecode;

    localparam int WT_W   = 8;
    localparam int ADDR_W = 16;

    logic clk;
    initial begin                       // 100 MHz — single driving process
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    logic                rst;
    logic                en;
    logic [WT_W-1:0]     in_wt;
    logic [ADDR_W-1:0]   in_start;
    logic [ADDR_W-1:0]   in_end;
    logic                r_en;
    logic [ADDR_W-1:0]   r_addr;
    logic                one_instr_end;

    int errors = 0;
    int cyc    = 0;

    InstrDecode #(
        .WT_W   (WT_W),
        .ADDR_W (ADDR_W)
    ) dut (
        .clk           (clk),
        .rst           (rst),
        .en            (en),
        .in_wt         (in_wt),
        .in_start      (in_start),
        .in_end        (in_end),
        .r_en          (r_en),
        .r_addr        (r_addr),
        .one_instr_end (one_instr_end)
    );

    // Drive one cycle of stimulus on the negedge, then check the (combinational)
    // outputs before the posedge advances the FSM.
    task automatic step(input logic              rst_v,
                        input logic              en_v,
                        input logic [WT_W-1:0]   wt_v,
                        input logic [ADDR_W-1:0] st_v,
                        input logic [ADDR_W-1:0] ed_v,
                        input logic              exp_ren,
                        input logic [ADDR_W-1:0] exp_addr,
                        input logic              exp_oie,
                        input string             note);
        @(negedge clk);
        rst      = rst_v;
        en       = en_v;
        in_wt    = wt_v;
        in_start = st_v;
        in_end   = ed_v;
        #1;
        if (r_en !== exp_ren || r_addr !== exp_addr || one_instr_end !== exp_oie) begin
            errors++;
            $display("  cyc %0d MISMATCH  got r_en=%0b r_addr=%0d oie=%0b | exp r_en=%0b r_addr=%0d oie=%0b   %s",
                     cyc, r_en, r_addr, one_instr_end, exp_ren, exp_addr, exp_oie, note);
        end else begin
            $display("  cyc %0d ok        r_en=%0b r_addr=%0d oie=%0b   %s",
                     cyc, r_en, r_addr, one_instr_end, note);
        end
        cyc++;
    endtask

    initial begin
        // power-on reset
        rst = 1'b1; en = 1'b0; in_wt = '0; in_start = '0; in_end = '0;
        @(negedge clk);
        @(negedge clk);
        rst = 1'b0;

        $display("tb_InstrDecode: WT_W=%0d ADDR_W=%0d", WT_W, ADDR_W);

        //         rst en  w_t st  ed  | exp: r_en addr oie | note
        step(1'b0, 1'b1, 8'd0,  16'd5,  16'd7,  1'b1, 16'd5,  1'b0, "A start (w_t=0 fast path)");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b1, 16'd6,  1'b0, "A streaming");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b1, 16'd7,  1'b1, "A last read -> one_instr_end");
        step(1'b0, 1'b1, 8'd0,  16'd10, 16'd10, 1'b1, 16'd10, 1'b1, "B single word, NO bubble after A");
        step(1'b0, 1'b1, 8'd2,  16'd20, 16'd22, 1'b0, 16'd0,  1'b0, "C start, w_t=2 idle 1");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b0, 16'd0,  1'b0, "C idle 2");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b1, 16'd20, 1'b0, "C stream starts");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b1, 16'd21, 1'b0, "C streaming");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b1, 16'd22, 1'b1, "C last read -> one_instr_end");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b0, 16'd0,  1'b0, "idle");

        step(1'b0, 1'b1, 8'd0,  16'd30, 16'd34, 1'b1, 16'd30, 1'b0, "D start (will be aborted)");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b1, 16'd31, 1'b0, "D streaming");
        step(1'b1, 1'b0, 8'd0,  16'd0,  16'd0,  1'b0, 16'd0,  1'b0, "RESET mid-stream kills D");
        step(1'b0, 1'b1, 8'd0,  16'd40, 16'd41, 1'b1, 16'd40, 1'b0, "E start clean after reset");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b1, 16'd41, 1'b1, "E last read -> one_instr_end");
        step(1'b0, 1'b0, 8'd0,  16'd0,  16'd0,  1'b0, 16'd0,  1'b0, "idle");

        if (errors == 0) $display("tb_InstrDecode: PASS");
        else             $display("tb_InstrDecode: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
