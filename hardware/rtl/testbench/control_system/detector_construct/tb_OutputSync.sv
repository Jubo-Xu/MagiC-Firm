// tb_OutputSync.sv — testbench for OutputSync (D=3, D_FIFO=4).
//
// Written in BOTH styles:
//   (1) a per-cycle table below, so the Vivado waveform can be read off directly;
//   (2) self-checking: mismatch count + PASS/FAIL for Verilator / xsim.
//
// Identical logic to MeasurementSync (see tb_MeasurementSync), for detector lines.
// Mirrors emulator/tests/.../test_output_sync.cpp, extended for the P2b parallel
// finish FIFO (single out_finish = OR over used lines) and the P2c SAT_PC clamp.
//
// Convention: bit i == line i. Values MSB-first, so "001" == line0=1. Outputs are
// registered, so out this cycle == the accepted round from the PREVIOUS step's
// inputs. Each step() drives THIS cycle's inputs and checks outputs + pc.
//
// FINISH cases: line1's finish set at pc0 while BUFFERED (not in pc0 mask) must not
// contribute at pc0 and must emerge from the FIFO at pc1; line2 bypass finish at
// pc3; OR-over-used at pc4.
//
// ====================================================================================
//  step | in(det valid finish) | out(=prev): valid used det finish  pc | note
// ------+----------------------+---------------------------------------+---------------
//   0   | 011  011  010        | 0  000 000  0   pc=0 (reset)          | l0 bypass, l1+fin buffered
//   1   | 000  001  000        | 1  001 001  0   pc=1                  | out=pc0 (fin on unused l1 -> 0)
//   2   | 000  000  000        | 1  011 010  1   pc=2                  | out=pc1: l1 from FIFO w/ FINISH
//   3   | 000  000  000        | 1  000 000  0   pc=3                  | out=pc2: fast-forward
//   4   | 000  100  100        | 0  000 000  0   pc=3 (stall holds)    | out=pc3: STALL
//   5   | 111  111  010        | 1  100 000  1   pc=4                  | out=pc3': l2 bypass FINISH
//   6   | 000  000  000        | 1  111 111  1   pc=5                  | out=pc4: all bypass, OR(l1 fin)
//   7   | 000  000  000        | 0  ---  ---  0   pc=5                 | out=pc5: beyond -> stall
// ====================================================================================
`timescale 1ns / 1ps

module tb_OutputSync;

    localparam int D = 3, D_FIFO = 4, PC_W = 16;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    logic            rst;
    logic [D-1:0]    in_det, in_valid, in_finish, sync_mask;
    logic [D-1:0]    out_det, out_used;
    logic            out_valid, out_finish;
    logic [PC_W-1:0] sync_regfile_pc;

    int errors = 0, step_i = 0;

    OutputSync #(.D(D), .D_FIFO(D_FIFO), .PC_W(PC_W)) dut (
        .clk, .rst, .in_det, .in_valid, .in_finish, .sync_mask,
        .out_det, .out_used, .out_valid, .out_finish, .sync_regfile_pc);

    // external output-sync regfile: combinational read of sync_prog[pc]
    localparam int unsigned NPROG = 5;
    localparam int          IDXW  = (NPROG < 2) ? 1 : $clog2(NPROG);
    logic [D-1:0] sync_prog [0:NPROG-1];
    initial begin
        sync_prog[0] = 3'b001;  // pc0: need line0
        sync_prog[1] = 3'b011;  // pc1: need line0, line1
        sync_prog[2] = 3'b000;  // pc2: fast-forward
        sync_prog[3] = 3'b100;  // pc3: need line2
        sync_prog[4] = 3'b111;  // pc4: need all
    end
    always_comb sync_mask = (sync_regfile_pc < PC_W'(NPROG)) ? sync_prog[sync_regfile_pc[IDXW-1:0]] : 3'b111;

    task automatic step(input logic [D-1:0] id,
                        input logic [D-1:0] iv,
                        input logic [D-1:0] ifin,
                        input logic         xv,       // expected out_valid  (= prev round)
                        input logic [D-1:0] xu,       // expected out_used
                        input logic [D-1:0] xd,       // expected out_det
                        input logic         xf,       // expected out_finish
                        input int           xpc,      // expected sync_regfile_pc
                        input string        note);
        @(posedge clk);
        #1;
        if (out_valid !== xv || (xv && (out_used !== xu || out_det !== xd)) ||
            out_finish !== xf || sync_regfile_pc !== PC_W'(xpc)) begin
            errors++;
            $display("  step %0d MISMATCH: val=%b used=%b det=%b fin=%b pc=%0d  exp val=%b used=%b det=%b fin=%b pc=%0d  %s",
                     step_i, out_valid, out_used, out_det, out_finish, sync_regfile_pc, xv, xu, xd, xf, xpc, note);
        end else begin
            $display("  step %0d ok: val=%b used=%b det=%b fin=%b pc=%0d  %s",
                     step_i, out_valid, out_used, out_det, out_finish, sync_regfile_pc, note);
        end
        in_det = id; in_valid = iv; in_finish = ifin;
        step_i++;
    endtask

    // ---- second instance: SAT_PC=2, fast-forward every cycle, pc must clamp at 2 ----
    logic [D-1:0]    s_out_det, s_out_used;
    logic            s_out_valid, s_out_finish;
    logic [PC_W-1:0] s_pc;
    OutputSync #(.D(D), .D_FIFO(D_FIFO), .PC_W(PC_W), .SAT_PC(2)) sdut (
        .clk, .rst, .in_det('0), .in_valid('0), .in_finish('0), .sync_mask('0),
        .out_det(s_out_det), .out_used(s_out_used), .out_valid(s_out_valid),
        .out_finish(s_out_finish), .sync_regfile_pc(s_pc));

    initial begin
        rst = 1'b1; in_det = '0; in_valid = '0; in_finish = '0;
        @(posedge clk); @(posedge clk); #1; rst = 1'b0;
        $display("tb_OutputSync: D=%0d D_FIFO=%0d", D, D_FIFO);

        //    det      valid    finish   xv    xused   xdet    xf    xpc  note
        step(3'b011, 3'b011, 3'b010, 1'b0, 3'b000, 3'b000, 1'b0, 0, "drive pc0 (l0 bypass, l1+fin buffered)");
        step(3'b000, 3'b001, 3'b000, 1'b1, 3'b001, 3'b001, 1'b0, 1, "out=pc0: fin on unused l1 -> 0");
        step(3'b000, 3'b000, 3'b000, 1'b1, 3'b011, 3'b010, 1'b1, 2, "out=pc1: l1 from FIFO carries FINISH");
        step(3'b000, 3'b000, 3'b000, 1'b1, 3'b000, 3'b000, 1'b0, 3, "out=pc2: fast-forward");
        step(3'b000, 3'b100, 3'b100, 1'b0, 3'b000, 3'b000, 1'b0, 3, "out=pc3: STALL (pc holds)");
        step(3'b111, 3'b111, 3'b010, 1'b1, 3'b100, 3'b000, 1'b1, 4, "out=pc3': l2 bypass FINISH");
        step(3'b000, 3'b000, 3'b000, 1'b1, 3'b111, 3'b111, 1'b1, 5, "out=pc4: all bypass, OR(l1 finish)");
        step(3'b000, 3'b000, 3'b000, 1'b0, 3'b000, 3'b000, 1'b0, 5, "out=pc5: beyond program -> stall");

        repeat (4) @(posedge clk);   // trailing cycles: full waveform tail

        if (s_pc !== PC_W'(2)) begin
            errors++;
            $display("  SAT MISMATCH: sdut sync_regfile_pc=%0d, expected clamp at 2", s_pc);
        end else
            $display("  SAT ok: sdut sync_regfile_pc clamped at 2");

        if (errors == 0) $display("tb_OutputSync: PASS");
        else             $display("tb_OutputSync: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
