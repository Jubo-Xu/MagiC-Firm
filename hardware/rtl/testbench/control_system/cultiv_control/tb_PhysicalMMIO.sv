// tb_PhysicalMMIO.sv — end-to-end testbench for the assembled PhysicalMMIO.
//
// Written in BOTH styles:
//   (1) the explicit cycle-by-cycle table below, with the same expected values
//       passed inline to every step() call, so the Vivado waveform can be read
//       off directly and compared against the comments;
//   (2) self-checking: every step compares the DUT outputs against those values,
//       counts mismatches, and prints PASS/FAIL (for Verilator / xsim batch).
//
// Mirrors the SystemC end-to-end test
//   emulator/tests/cultiv_control/leaf/physical_mmio/test_physical_mmio.cpp
// and replays the IDENTICAL program from the same .mem files.
//
// Config: INSTR_DEPTH=4, CW_DEPTH=16, DATA_W=8 -> ADDR_W=4, INSTR_W=16.
// mmio_cw.mem holds CW[i] = 10*i + 1, so every word identifies its own address.
//
//   instr | wt | start..end | emits
//   ------+----+------------+-----------------------------------------
//     0   | 0  |   0 .. 2   | CW[0]=1, CW[1]=11, CW[2]=21
//     1   | 2  |   4 .. 5   | 2 idle cycles, then CW[4]=41, CW[5]=51
//     2   | 0  |   8 .. 8   | CW[8]=81                 (single word)
//     3   | 1  |  12 .. 13  | 1 idle, CW[12]=121, CW[13]=131   <-- WAIT instr,
//                             repeats forever until FINISH
//
// TIMING. All three outputs are registered. Each step drives the event bus 1 ns
// AFTER the posedge (off the sampling edge — no race, and this is how a real
// combinational source behaves) and checks 1 ns later. So the values checked in
// cycle k are the outputs of the registers updated at posedge k, i.e. the
// response to cycle k-1's inputs.
//
// ============================================================================
//  cyc | ev        | out_valid out_data cw_gen_finish | note
// -----+-----------+---------------------------------+--------------------------------
//   0  | START     |    0        0          0        | START accepted
//   1  |  -        |    0        0          0        | instr0 fetched, CW read issued
//   2  |  -        |    1        1          0        | CW[0]
//   3  |  -        |    1       11          0        | CW[1]
//   4  |  -        |    1       21          0        | CW[2]  (instr0 done)
//   5  |  -        |    0        0          0        | instr1 wt=2 gap
//   6  |  -        |    0        0          0        | instr1 wt=2 gap
//   7  |  -        |    1       41          0        | CW[4]
//   8  |  -        |    1       51          0        | CW[5]  (instr1 done)
//   9  |  -        |    1       81          0        | CW[8]  NO BUBBLE after 51
//  10  |  -        |    0        0          0        | instr3 wt=1 gap
//  11  |  -        |    1      121          0        | CW[12]
//  12  |  -        |    1      131          0        | CW[13] (wait instr done)
//  13  |  -        |    0        0          0        | wait instr repeats: wt=1 gap
//  14  |  -        |    1      121          0        | CW[12]
//  15  |  -        |    1      131          0        | CW[13]
//  16  |  -        |    0        0          0        | ... and again
//  17  |  -        |    1      121          0        |
//  18  |  -        |    1      131          0        |
//  19  | ABORT     |    0        0          0        | ABORT: decode reset this cycle
//  20  |  -        |    0        0          0        | instr0 refetched
//  21  |  -        |    1        1          0        | CW[0]  <-- program restarted
//  22  |  -        |    1       11          0        | CW[1]
//  23  |  -        |    1       21          0        | CW[2]
//  24  | FINISH    |    0        0          0        | FINISH mid-instr (instr1 wt gap)
//  25  |  -        |    0        0          0        | draining: instr1 still in flight
//  26  |  -        |    1       41          0        | CW[4]
//  27  |  -        |    1       51          1        | CW[5] = LAST word + cw_gen_finish
//  28  |  -        |    0        0          0        | pulse gone, stream stopped
//  29  |  -        |    0        0          0        | idle
// ============================================================================
//
// Key things to look for in the waveform:
//   * cycles 8->9: 51 then 81 on CONSECUTIVE cycles — the zero-stall instruction
//     handoff across a wt=0 boundary.
//   * cycles 5,6 and 10,13,16: out_valid low for exactly wt cycles.
//   * cycles 11..18: the wait instruction repeating with period 3 forever.
//   * cycles 19->21: abort restarts the program at CW[0].
//   * cycle 27: out_valid is STILL HIGH in the same cycle as cw_gen_finish —
//     the final command word survives the finish reset (SyncROM has no reset).
//     If it were dropped, the control core would lose the last physical op.
`timescale 1ns / 1ps

module tb_PhysicalMMIO;

    localparam int INSTR_DEPTH = 4;
    localparam int CW_DEPTH    = 16;
    localparam int DATA_W      = 8;
    localparam int WT_W        = 8;
    localparam int PAYLOAD_W   = 8;

    localparam logic [1:0] EV_START  = 2'd0;
    localparam logic [1:0] EV_ABORT  = 2'd1;
    localparam logic [1:0] EV_FINISH = 2'd2;
    localparam logic [1:0] EV_NONE   = 2'd0;   // ignored while ev_valid is low

    logic clk;
    initial begin                       // 100 MHz — single driving process
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    logic                 rst;
    logic                 ev_valid;
    logic [1:0]           ev_type;
    logic [PAYLOAD_W-1:0] ev_payload;
    logic [DATA_W-1:0]    out_data;
    logic                 out_valid;
    logic                 cw_gen_finish;

    int errors = 0;
    int cyc    = 0;

    PhysicalMMIO #(
        .INSTR_DEPTH (INSTR_DEPTH),
        .CW_DEPTH    (CW_DEPTH),
        .DATA_W      (DATA_W),
        .WT_W        (WT_W),
        .PAYLOAD_W   (PAYLOAD_W),
        .INSTR_FILE  ("testbench/control_system/cultiv_control/mmio_instr.mem"),
        .CW_FILE     ("testbench/control_system/cultiv_control/mmio_cw.mem"),
        .INIT_HEX    (1'b0)
    ) dut (
        .clk           (clk),
        .rst           (rst),
        .ev_valid      (ev_valid),
        .ev_type       (ev_type),
        .ev_payload    (ev_payload),
        .out_data      (out_data),
        .out_valid     (out_valid),
        .cw_gen_finish (cw_gen_finish)
    );

    // Drive the event bus just after the posedge, then check the registered
    // outputs before the next edge.
    task automatic step(input logic             vld,
                        input logic [1:0]       etype,
                        input logic             exp_valid,
                        input logic [DATA_W-1:0] exp_data,
                        input logic             exp_cwf,
                        input string            note);
        @(posedge clk);
        #1;                          // off the sampling edge (no race with the DUT)
        ev_valid = vld;
        ev_type  = etype;
        #1;                          // let the registered outputs settle
        if (out_valid !== exp_valid || out_data !== exp_data ||
            cw_gen_finish !== exp_cwf) begin
            errors++;
            $display("  cyc %0d MISMATCH  got valid=%0b data=%0d cwf=%0b | exp valid=%0b data=%0d cwf=%0b   %s",
                     cyc, out_valid, out_data, cw_gen_finish,
                     exp_valid, exp_data, exp_cwf, note);
        end else begin
            $display("  cyc %0d ok        valid=%0b data=%3d cwf=%0b   %s",
                     cyc, out_valid, out_data, cw_gen_finish, note);
        end
        cyc++;
    endtask

    initial begin
        rst = 1'b1; ev_valid = 1'b0; ev_type = EV_NONE; ev_payload = '0;
        @(posedge clk);
        @(posedge clk);
        #1;
        rst = 1'b0;

        $display("tb_PhysicalMMIO: INSTR_DEPTH=%0d CW_DEPTH=%0d DATA_W=%0d",
                 INSTR_DEPTH, CW_DEPTH, DATA_W);

        //    vld   ev_type    | exp: valid  data  cwf | note
        step(1'b1, EV_START,  1'b0, 8'd0,   1'b0, "START accepted");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "instr0 fetched, CW read issued");
        step(1'b0, EV_NONE,   1'b1, 8'd1,   1'b0, "CW[0]");
        step(1'b0, EV_NONE,   1'b1, 8'd11,  1'b0, "CW[1]");
        step(1'b0, EV_NONE,   1'b1, 8'd21,  1'b0, "CW[2] (instr0 done)");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "instr1 wt=2 gap");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "instr1 wt=2 gap");
        step(1'b0, EV_NONE,   1'b1, 8'd41,  1'b0, "CW[4]");
        step(1'b0, EV_NONE,   1'b1, 8'd51,  1'b0, "CW[5] (instr1 done)");
        step(1'b0, EV_NONE,   1'b1, 8'd81,  1'b0, "CW[8] NO BUBBLE after 51");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "instr3 wt=1 gap");
        step(1'b0, EV_NONE,   1'b1, 8'd121, 1'b0, "CW[12]");
        step(1'b0, EV_NONE,   1'b1, 8'd131, 1'b0, "CW[13] (wait instr done)");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "wait instr repeats: wt=1 gap");
        step(1'b0, EV_NONE,   1'b1, 8'd121, 1'b0, "CW[12]");
        step(1'b0, EV_NONE,   1'b1, 8'd131, 1'b0, "CW[13]");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "... and again");
        step(1'b0, EV_NONE,   1'b1, 8'd121, 1'b0, "CW[12]");
        step(1'b0, EV_NONE,   1'b1, 8'd131, 1'b0, "CW[13]");
        step(1'b1, EV_ABORT,  1'b0, 8'd0,   1'b0, "ABORT: decode reset this cycle");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "instr0 refetched");
        step(1'b0, EV_NONE,   1'b1, 8'd1,   1'b0, "CW[0] <-- program restarted");
        step(1'b0, EV_NONE,   1'b1, 8'd11,  1'b0, "CW[1]");
        step(1'b0, EV_NONE,   1'b1, 8'd21,  1'b0, "CW[2]");
        step(1'b1, EV_FINISH, 1'b0, 8'd0,   1'b0, "FINISH mid-instr (instr1 wt gap)");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "draining: instr1 still in flight");
        step(1'b0, EV_NONE,   1'b1, 8'd41,  1'b0, "CW[4]");
        step(1'b0, EV_NONE,   1'b1, 8'd51,  1'b1, "CW[5] = LAST word + cw_gen_finish");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "pulse gone, stream stopped");
        step(1'b0, EV_NONE,   1'b0, 8'd0,   1'b0, "idle");

        if (errors == 0) $display("tb_PhysicalMMIO: PASS");
        else             $display("tb_PhysicalMMIO: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
