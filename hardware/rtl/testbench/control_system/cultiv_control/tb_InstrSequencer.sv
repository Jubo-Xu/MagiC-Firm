// tb_InstrSequencer.sv — directed testbench for InstrSequencer.
//
// Written in BOTH styles:
//   (1) an explicit cycle-by-cycle table below, with the same expected values
//       passed inline to every step() call, so the Vivado waveform can be read
//       off directly and compared against the comments;
//   (2) self-checking: every step compares the DUT outputs against those values,
//       counts mismatches, and prints PASS/FAIL (for Verilator / xsim batch).
//
// Mirrors the SystemC unit test
//   emulator/tests/cultiv_control/leaf/physical_mmio/test_instr_sequencer.cpp
//
// DEPTH = 4, so instruction 3 is the "wait" instruction the pointer saturates on.
//
// TIMING. instr_en/instr_addr are REGISTERED (they reflect the previous cycle's
// inputs); rst_out/cw_gen_finish are COMBINATIONAL (they reflect this cycle's
// inputs). Each step drives on the NEGEDGE and checks 1 ns later, so both kinds
// are sampled within the same cycle, before the posedge advances the FSM.
//
// ============================================================================
//  cyc | ev_valid ev_type one_instr_end | instr_en addr rst_out cw_gen_finish | note
// -----+-------------------------------+------------------------------------+--------------------------
//   0  |   1      START      0          |    0      0     0        0         | START accepted
//   1  |   0       -         0          |    1      0     0        0         | fetch instr 0
//   2  |   0       -         0          |    0      0     0        0         | instr 0 running
//   3  |   0       -         1          |    0      0     0        0         | instr 0 ends
//   4  |   0       -         0          |    1      1     0        0         | fetch instr 1 (no gap)
//   5  |   0       -         1          |    0      0     0        0         | instr 1 ends
//   6  |   0       -         0          |    1      2     0        0         | fetch instr 2
//   7  |   0       -         1          |    0      0     0        0         | instr 2 ends
//   8  |   0       -         0          |    1      3     0        0         | fetch instr 3 (wait instr)
//   9  |   0       -         1          |    0      0     0        0         | instr 3 ends
//  10  |   0       -         0          |    1      3     0        0         | SATURATED: refetch instr 3
//  11  |   0       -         1          |    0      0     0        0         | instr 3 ends again
//  12  |   0       -         0          |    1      3     0        0         | SATURATED again
//  13  |   1      ABORT      0          |    0      0     1        0         | ABORT -> rst_out THIS cycle
//  14  |   0       -         0          |    1      0     0        0         | refetch instr 0
//  15  |   0       -         0          |    0      0     0        0         | instr 0 running
//  16  |   1      FINISH     0          |    0      0     0        0         | FINISH mid-instr -> DRAIN
//  17  |   0       -         0          |    0      0     0        0         | draining
//  18  |   0       -         1          |    0      0     0        0         | in-flight instr ends
//  19  |   0       -         0          |    0      0     1        1         | cw_gen_finish + rst_out
//  20  |   0       -         0          |    0      0     0        0         | pulse gone (1 cycle only)
//  21  |   1      START      0          |    0      0     0        0         | restart
//  22  |   1      FINISH     1          |    1      0     0        0         | FINISH exactly on instr end
//  23  |   0       -         0          |    0      0     1        1         | straight to IDLE, NO DRAIN
//  24  |   0       -         0          |    0      0     0        0         | pulse gone
// ============================================================================
//
// Key things to look for in the waveform:
//   * cycles 3->4, 5->6, 7->8: instr_en re-pulses the cycle right after
//     one_instr_end — the no-stall instruction handoff.
//   * cycles 8..12: instr_addr stays at 3 no matter how many instructions end —
//     the wait-instruction saturation.
//   * cycle 13: rst_out is high in the SAME cycle as ABORT, one cycle before the
//     refetch at 14, so InstrDecode is cleared before it restarts.
//   * cycles 19 and 23: cw_gen_finish is exactly ONE cycle wide.
//   * cycles 22->23: FINISH arriving together with one_instr_end skips DRAIN.
`timescale 1ns / 1ps

module tb_InstrSequencer;

    localparam int DEPTH     = 4;
    localparam int PTR_W     = 2;
    localparam int PAYLOAD_W = 8;

    localparam logic [1:0] EV_START  = 2'd0;
    localparam logic [1:0] EV_ABORT  = 2'd1;
    localparam logic [1:0] EV_FINISH = 2'd2;

    logic clk;
    initial begin                       // 100 MHz — single driving process
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    logic                 rst;
    logic                 ev_valid;
    logic [1:0]           ev_type;
    logic [PAYLOAD_W-1:0] ev_payload;
    logic                 one_instr_end;
    logic                 instr_en;
    logic [PTR_W-1:0]     instr_addr;
    logic                 rst_out;
    logic                 cw_gen_finish;

    int errors = 0;
    int cyc    = 0;

    InstrSequencer #(
        .DEPTH     (DEPTH),
        .PAYLOAD_W (PAYLOAD_W)
    ) dut (
        .clk           (clk),
        .rst           (rst),
        .ev_valid      (ev_valid),
        .ev_type       (ev_type),
        .ev_payload    (ev_payload),
        .one_instr_end (one_instr_end),
        .instr_en      (instr_en),
        .instr_addr    (instr_addr),
        .rst_out       (rst_out),
        .cw_gen_finish (cw_gen_finish)
    );

    // Drive one cycle of stimulus on the negedge, then check the outputs before
    // the posedge advances the FSM.
    task automatic step(input logic       vld,
                        input logic [1:0] etype,
                        input logic       oie,
                        input logic       exp_en,
                        input logic [PTR_W-1:0] exp_addr,
                        input logic       exp_rst_out,
                        input logic       exp_cwf,
                        input string      note);
        @(negedge clk);
        ev_valid      = vld;
        ev_type       = etype;
        one_instr_end = oie;
        #1;
        if (instr_en !== exp_en || instr_addr !== exp_addr ||
            rst_out !== exp_rst_out || cw_gen_finish !== exp_cwf) begin
            errors++;
            $display("  cyc %0d MISMATCH  got en=%0b addr=%0d rst_out=%0b cwf=%0b | exp en=%0b addr=%0d rst_out=%0b cwf=%0b   %s",
                     cyc, instr_en, instr_addr, rst_out, cw_gen_finish,
                     exp_en, exp_addr, exp_rst_out, exp_cwf, note);
        end else begin
            $display("  cyc %0d ok        en=%0b addr=%0d rst_out=%0b cwf=%0b   %s",
                     cyc, instr_en, instr_addr, rst_out, cw_gen_finish, note);
        end
        cyc++;
    endtask

    initial begin
        rst = 1'b1; ev_valid = 1'b0; ev_type = 2'd0; ev_payload = '0; one_instr_end = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst = 1'b0;

        $display("tb_InstrSequencer: DEPTH=%0d", DEPTH);

        //   vld   ev_type    oie  | exp: en   addr  rst_out cwf | note
        step(1'b1, EV_START,  1'b0,  1'b0, 2'd0, 1'b0, 1'b0, "START accepted");
        step(1'b0, 2'd0,      1'b0,  1'b1, 2'd0, 1'b0, 1'b0, "fetch instr 0");
        step(1'b0, 2'd0,      1'b0,  1'b0, 2'd0, 1'b0, 1'b0, "instr 0 running");
        step(1'b0, 2'd0,      1'b1,  1'b0, 2'd0, 1'b0, 1'b0, "instr 0 ends");
        step(1'b0, 2'd0,      1'b0,  1'b1, 2'd1, 1'b0, 1'b0, "fetch instr 1 (no gap)");
        step(1'b0, 2'd0,      1'b1,  1'b0, 2'd0, 1'b0, 1'b0, "instr 1 ends");
        step(1'b0, 2'd0,      1'b0,  1'b1, 2'd2, 1'b0, 1'b0, "fetch instr 2");
        step(1'b0, 2'd0,      1'b1,  1'b0, 2'd0, 1'b0, 1'b0, "instr 2 ends");
        step(1'b0, 2'd0,      1'b0,  1'b1, 2'd3, 1'b0, 1'b0, "fetch instr 3 (wait instr)");
        step(1'b0, 2'd0,      1'b1,  1'b0, 2'd0, 1'b0, 1'b0, "instr 3 ends");
        step(1'b0, 2'd0,      1'b0,  1'b1, 2'd3, 1'b0, 1'b0, "SATURATED: refetch instr 3");
        step(1'b0, 2'd0,      1'b1,  1'b0, 2'd0, 1'b0, 1'b0, "instr 3 ends again");
        step(1'b0, 2'd0,      1'b0,  1'b1, 2'd3, 1'b0, 1'b0, "SATURATED again");
        step(1'b1, EV_ABORT,  1'b0,  1'b0, 2'd0, 1'b1, 1'b0, "ABORT -> rst_out THIS cycle");
        step(1'b0, 2'd0,      1'b0,  1'b1, 2'd0, 1'b0, 1'b0, "refetch instr 0");
        step(1'b0, 2'd0,      1'b0,  1'b0, 2'd0, 1'b0, 1'b0, "instr 0 running");
        step(1'b1, EV_FINISH, 1'b0,  1'b0, 2'd0, 1'b0, 1'b0, "FINISH mid-instr -> DRAIN");
        step(1'b0, 2'd0,      1'b0,  1'b0, 2'd0, 1'b0, 1'b0, "draining");
        step(1'b0, 2'd0,      1'b1,  1'b0, 2'd0, 1'b0, 1'b0, "in-flight instr ends");
        step(1'b0, 2'd0,      1'b0,  1'b0, 2'd0, 1'b1, 1'b1, "cw_gen_finish + rst_out");
        step(1'b0, 2'd0,      1'b0,  1'b0, 2'd0, 1'b0, 1'b0, "pulse gone (1 cycle only)");
        step(1'b1, EV_START,  1'b0,  1'b0, 2'd0, 1'b0, 1'b0, "restart");
        step(1'b1, EV_FINISH, 1'b1,  1'b1, 2'd0, 1'b0, 1'b0, "FINISH exactly on instr end");
        step(1'b0, 2'd0,      1'b0,  1'b0, 2'd0, 1'b1, 1'b1, "straight to IDLE, NO DRAIN");
        step(1'b0, 2'd0,      1'b0,  1'b0, 2'd0, 1'b0, 1'b0, "pulse gone");

        if (errors == 0) $display("tb_InstrSequencer: PASS");
        else             $display("tb_InstrSequencer: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
