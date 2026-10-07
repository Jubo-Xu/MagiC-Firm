// tb_Kernel.sv — testbench for Kernel (M=4, N=2, H=2).
//
// Written in BOTH styles:
//   (1) a per-cycle table below, so the Vivado waveform can be read off directly;
//   (2) self-checking: mismatch count + PASS/FAIL for Verilator / xsim.
//
// Mirrors emulator/tests/.../test_kernel.cpp, extended for the P2b finish
// (out_finish = any_emit ? in_finish : 0) and the P2c SAT_PC pc clamp.
//
// Selector picks lines {1,3} onto slots {0,1} (selector_indexes=4'b1101). Cores:
//   core0: A = line1@t0 ^ line1@t1 (emit t1);  C = line1@t2 ^ line1@t3 (emit t3)
//   core1: B = line3@t0 ^ line3@t2 (emit t2)
//   line1 = 1,0,1,1 ; line3 = 1,_,0,_  => A=1, B=1, C=0. A stall after t0 holds pc.
//
// Outputs are registered, so out this cycle == the result of the PREVIOUS step's
// input. Each step() drives THIS cycle's inputs and checks the outputs + pc.
// FINISH: in_finish=1 is driven on t0 (no emit -> out_finish must stay 0) and on
// t3 (emit C -> out_finish=1), proving finish only rides an EMITTED detector.
//
// ==================================================================================
//  step | in(valid used meas fin) | out(=prev): valid det finish  pc | note
// ------+-------------------------+----------------------------------+---------------
//   0   | 1  1010 1010 1          | 0  0  0   pc=0 (reset)            | t0 (fin set, no emit)
//   1   | 0  0000 0000 0          | 0  0  0   pc=1                    | stall (pc holds)
//   2   | 1  0010 0000 0          | 0  0  0   pc=1                    | t1 drive (prev was stall)
//   3   | 1  1010 0010 0          | 1  1  0   pc=2                    | out = t1: emit A=1, fin0
//   4   | 1  0010 0010 1          | 1  1  0   pc=3                    | out = t2: emit B=1, fin0
//   5   | 0  0000 0000 0          | 1  0  1   pc=4                    | out = t3: emit C=0, FIN=1
//   6   | 0  0000 0000 0          | 0  0  0   pc=4                    | drained
// ==================================================================================
`timescale 1ns / 1ps

module tb_Kernel;

    localparam int N = 2, H = 2, M = 4, PC_W = 16;
    localparam int CW = N + 1;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    logic            rst;
    logic            in_valid, in_finish;
    logic [M-1:0]    in_used, in_meas;
    logic [N*2-1:0]  selector_indexes;
    logic [H*CW-1:0] core_mask;
    logic            out_valid, out_det, out_finish;
    logic [PC_W-1:0] core_regfile_pc;

    int errors = 0, step_i = 0;

    Kernel #(.N(N), .H(H), .M(M), .PC_W(PC_W)) dut (
        .clk, .rst, .in_valid, .in_used, .in_meas, .in_finish,
        .selector_indexes, .core_mask, .out_valid, .out_det, .out_finish, .core_regfile_pc);

    // external core regfile: combinational read of prog[pc]
    localparam int unsigned NPROG = 4;
    localparam int          IDXW  = (NPROG < 2) ? 1 : $clog2(NPROG);
    logic [H*CW-1:0] prog [0:NPROG-1];
    initial begin
        prog[0] = 6'b010_001;  // t0: core1 sel=10 emit=0 | core0 sel=01 emit=0
        prog[1] = 6'b000_101;  // t1: core1 idle         | core0 sel=01 emit=1
        prog[2] = 6'b110_001;  // t2: core1 sel=10 emit=1 | core0 sel=01 emit=0
        prog[3] = 6'b000_101;  // t3: core1 idle         | core0 sel=01 emit=1
    end
    always_comb core_mask = (core_regfile_pc < PC_W'(NPROG)) ? prog[core_regfile_pc[IDXW-1:0]] : '0;

    // Drive THIS cycle's inputs just after the posedge, then check outputs (+ pc) —
    // the registered result of the PREVIOUS step's inputs.
    task automatic step(input logic         iv,
                        input logic [M-1:0] iu,
                        input logic [M-1:0] im,
                        input logic         ifin,
                        input logic         xv,       // expected out_valid  (= prev result)
                        input logic         xd,       // expected out_det
                        input logic         xf,       // expected out_finish
                        input int           xpc,      // expected core_regfile_pc
                        input string        note);
        @(posedge clk);
        #1;
        if (out_valid !== xv || out_det !== xd || out_finish !== xf || core_regfile_pc !== PC_W'(xpc)) begin
            errors++;
            $display("  step %0d MISMATCH: val=%b det=%b fin=%b pc=%0d  exp val=%b det=%b fin=%b pc=%0d  %s",
                     step_i, out_valid, out_det, out_finish, core_regfile_pc, xv, xd, xf, xpc, note);
        end else begin
            $display("  step %0d ok: val=%b det=%b fin=%b pc=%0d  %s",
                     step_i, out_valid, out_det, out_finish, core_regfile_pc, note);
        end
        in_valid = iv; in_used = iu; in_meas = im; in_finish = ifin;
        step_i++;
    endtask

    // ---- second instance: SAT_PC=2, free-running valid, pc must clamp at 2 ----
    logic            s_out_valid, s_out_det, s_out_finish;
    logic [PC_W-1:0] s_pc;
    Kernel #(.N(N), .H(H), .M(M), .PC_W(PC_W), .SAT_PC(2)) sdut (
        .clk, .rst, .in_valid(1'b1), .in_used('0), .in_meas('0), .in_finish(1'b0),
        .selector_indexes('0), .core_mask('0),
        .out_valid(s_out_valid), .out_det(s_out_det), .out_finish(s_out_finish), .core_regfile_pc(s_pc));

    initial begin
        selector_indexes = 4'b1101;   // slot0->line1(01), slot1->line3(11)
        rst = 1'b1; in_valid = 1'b0; in_used = '0; in_meas = '0; in_finish = 1'b0;
        @(posedge clk); @(posedge clk); #1; rst = 1'b0;
        $display("tb_Kernel: N=%0d H=%0d M=%0d", N, H, M);

        //    v  used     meas     fin  xv xd xf xpc  note
        step(1'b1, 4'b1010, 4'b1010, 1'b1,  1'b0,1'b0,1'b0, 0, "t0 (fin set, no emit) / out reset");
        step(1'b0, 4'b0000, 4'b0000, 1'b0,  1'b0,1'b0,1'b0, 1, "stall (pc holds) / out = t0 (no emit,fin0)");
        step(1'b1, 4'b0010, 4'b0000, 1'b0,  1'b0,1'b0,1'b0, 1, "t1 drive / out = stall (0)");
        step(1'b1, 4'b1010, 4'b0010, 1'b0,  1'b1,1'b1,1'b0, 2, "out = t1: emit A=1, fin0");
        step(1'b1, 4'b0010, 4'b0010, 1'b1,  1'b1,1'b1,1'b0, 3, "out = t2: emit B=1, fin0");
        step(1'b0, 4'b0000, 4'b0000, 1'b0,  1'b1,1'b0,1'b1, 4, "out = t3: emit C=0, FIN=1 (rides emit)");
        step(1'b0, 4'b0000, 4'b0000, 1'b0,  1'b0,1'b0,1'b0, 4, "drained");

        repeat (4) @(posedge clk);   // trailing cycles: full waveform tail

        // SAT_PC=2 instance: after these cycles pc would be >=6 without the clamp
        if (s_pc !== PC_W'(2)) begin
            errors++;
            $display("  SAT MISMATCH: sdut core_regfile_pc=%0d, expected clamp at 2", s_pc);
        end else
            $display("  SAT ok: sdut core_regfile_pc clamped at 2");

        if (errors == 0) $display("tb_Kernel: PASS");
        else             $display("tb_Kernel: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
