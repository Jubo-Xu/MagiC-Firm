// tb_RootOutputSync.sv — testbench for RootOutputSync.
//
// Written in BOTH styles:
//   (1) a per-cycle table below, so the Vivado waveform can be read off directly;
//   (2) self-checking: mismatch count + PASS/FAIL for Verilator / xsim.
//
// Mirrors emulator/tests/.../test_root_output_sync.cpp (P2c). D=2, INDEX_W=4,
// HW_WIDTH=8, STRIDE=2, SENTINEL=15, SAT_PC=4. All-bypass rounds keep the OutputSync
// front-end predictable so the checks focus on the ADDED logic: markers, global
// indices (zero-extend INDEX_W->HW_WIDTH, unused -> all-ones), the copy-last offset
// (w*STRIDE on the saturating row), and out_finish / last_wait.
//
// Outputs are registered (1-cycle latency, no extra vs OutputSync), so a round's
// outputs appear the cycle AFTER its inputs. Each step() drives THIS cycle's inputs
// and checks the outputs = the PREVIOUS step's round.
//
// gi bus layout: line i at bits [i*HW_WIDTH +: HW_WIDTH]; gi = {line1, line0}.
// marker bus (checked): {last_wait, first_wait, last_normal, first_normal}.
//
// ============================================================================================
//  step | in(det val fin) | out(=prev): v used det fin  gi     mk     pc | note
// ------+-----------------+---------------------------------------------+---------------------
//   0   | 11 11 00        | 0 00  00  0  FFFF 0000 pc=0 (reset)          | drive pc0
//   1   | 11 11 00        | 1 11  11  0  0100 0001 pc=1                  | pc0: first_normal, idx 0,1
//   2   | 00 00 00        | 1 11  11  0  0302 0010 pc=2                  | pc1: last_normal, idx 2,3
//   3   | 11 11 00        | 1 00  00  0  FFFF 0000 pc=3                  | pc2: fast-fwd, unused->all-ones
//   4   | 11 11 00        | 1 11  11  0  0504 0100 pc=4                  | pc3: first_wait (is_wait rises)
//   5   | 11 11 00        | 1 11  11  0  0706 0000 pc=4                  | pc4 sat w=0: idx 6,7 (offset 0)
//   6   | 11 11 11        | 1 11  11  0  0908 0000 pc=4                  | pc4 sat w=1: idx +2 -> 8,9
//   7   | 00 00 00        | 1 11  11  1  0B0A 1000 pc=4                  | pc4 sat w=2 + FINISH: last_wait, +4 -> 10,11
//   8   | 00 00 00        | 0 00  00  0  FFFF 0000 pc=4                  | stall
// ============================================================================================
`timescale 1ns / 1ps

module tb_RootOutputSync;

    localparam int D = 2, D_FIFO = 4, PC_W = 16, INDEX_W = 4, HW_WIDTH = 8;
    localparam int STRIDE = 2, SENTINEL = 15, SAT_PC = 4;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    logic                  rst;
    logic [D-1:0]          in_det, in_valid, in_finish, sync_mask;
    logic [D*INDEX_W-1:0]  global_indexes;
    logic [2:0]            round_marker;
    logic [D-1:0]          out_det, out_used;
    logic                  out_valid, out_finish;
    logic [PC_W-1:0]       sync_regfile_pc;
    logic [D*HW_WIDTH-1:0] out_global_indexes;
    logic                  first_normal, last_normal, first_wait, last_wait;

    int errors = 0, step_i = 0;

    RootOutputSync #(.D(D), .D_FIFO(D_FIFO), .PC_W(PC_W), .INDEX_W(INDEX_W),
                     .HW_WIDTH(HW_WIDTH), .STRIDE(STRIDE), .SENTINEL(SENTINEL), .SAT_PC(SAT_PC)) dut (
        .clk                (clk),
        .rst                (rst),
        .in_det             (in_det),
        .in_valid           (in_valid),
        .in_finish          (in_finish),
        .sync_mask          (sync_mask),
        .global_indexes     (global_indexes),
        .round_marker       (round_marker),
        .out_det            (out_det),
        .out_used           (out_used),
        .out_valid          (out_valid),
        .out_finish         (out_finish),
        .sync_regfile_pc    (sync_regfile_pc),
        .out_global_indexes (out_global_indexes),
        .first_normal       (first_normal),
        .last_normal        (last_normal),
        .first_wait         (first_wait),
        .last_wait          (last_wait)
    );

    // external regfiles (mask / global_index / round_marker) read @ sync_regfile_pc
    localparam int unsigned NPROG = 5;
    localparam int          IDXW  = (NPROG < 2) ? 1 : $clog2(NPROG);
    logic [D-1:0]         mask_prog [0:NPROG-1];
    logic [D*INDEX_W-1:0] gi_prog   [0:NPROG-1];
    logic [2:0]           rm_prog   [0:NPROG-1];
    initial begin
        mask_prog[0]=2'b11; mask_prog[1]=2'b11; mask_prog[2]=2'b00; mask_prog[3]=2'b11; mask_prog[4]=2'b11;
        gi_prog[0]=8'h10;   gi_prog[1]=8'h32;   gi_prog[2]=8'hFF;   gi_prog[3]=8'h54;   gi_prog[4]=8'h76;  // {line1,line0}
        rm_prog[0]=3'b001;  rm_prog[1]=3'b010;  rm_prog[2]=3'b000;  rm_prog[3]=3'b100;  rm_prog[4]=3'b100; // {wait,last,first}
    end
    always_comb begin
        if (sync_regfile_pc < PC_W'(NPROG)) begin
            sync_mask      = mask_prog[sync_regfile_pc[IDXW-1:0]];
            global_indexes = gi_prog  [sync_regfile_pc[IDXW-1:0]];
            round_marker   = rm_prog  [sync_regfile_pc[IDXW-1:0]];
        end else begin
            sync_mask = 2'b11; global_indexes = '0; round_marker = '0;
        end
    end

    logic [3:0] mk_act;
    assign mk_act = {last_wait, first_wait, last_normal, first_normal};

    task automatic step(input logic [D-1:0]          id,
                        input logic [D-1:0]          iv,
                        input logic [D-1:0]          ifin,
                        input logic                  xv,     // expected out_valid  (= prev round)
                        input logic [D-1:0]          xused,  // expected out_used
                        input logic [D-1:0]          xdet,   // expected out_det
                        input logic                  xfin,   // expected out_finish
                        input logic [D*HW_WIDTH-1:0] xgi,    // expected out_global_indexes
                        input logic [3:0]            xmk,    // {last_wait,first_wait,last_normal,first_normal}
                        input int                    xpc,    // expected sync_regfile_pc
                        input string                 note);
        @(posedge clk);
        #1;
        if (out_valid !== xv || (xv && (out_used !== xused || out_det !== xdet)) ||
            out_finish !== xfin || out_global_indexes !== xgi || mk_act !== xmk ||
            sync_regfile_pc !== PC_W'(xpc)) begin
            errors++;
            $display("  step %0d MISMATCH: v=%b used=%b det=%b fin=%b gi=%h mk=%b pc=%0d | exp v=%b used=%b det=%b fin=%b gi=%h mk=%b pc=%0d  %s",
                     step_i, out_valid, out_used, out_det, out_finish, out_global_indexes, mk_act, sync_regfile_pc,
                     xv, xused, xdet, xfin, xgi, xmk, xpc, note);
        end else begin
            $display("  step %0d ok: v=%b used=%b det=%b fin=%b gi=%h mk=%b pc=%0d  %s",
                     step_i, out_valid, out_used, out_det, out_finish, out_global_indexes, mk_act, sync_regfile_pc, note);
        end
        in_det = id; in_valid = iv; in_finish = ifin;
        step_i++;
    endtask

    initial begin
        rst = 1'b1; in_det = '0; in_valid = '0; in_finish = '0;
        @(posedge clk); @(posedge clk); #1; rst = 1'b0;
        $display("tb_RootOutputSync: D=%0d INDEX_W=%0d HW_WIDTH=%0d STRIDE=%0d SAT_PC=%0d", D, INDEX_W, HW_WIDTH, STRIDE, SAT_PC);

        //    det   val   fin   xv    xused  xdet   xfin  xgi        xmk      xpc  note
        step(2'b11,2'b11,2'b00, 1'b0, 2'b00, 2'b00, 1'b0, 16'hFFFF, 4'b0000, 0, "drive pc0 (out reset)");
        step(2'b11,2'b11,2'b00, 1'b1, 2'b11, 2'b11, 1'b0, 16'h0100, 4'b0001, 1, "pc0: first_normal, idx 0,1");
        step(2'b00,2'b00,2'b00, 1'b1, 2'b11, 2'b11, 1'b0, 16'h0302, 4'b0010, 2, "pc1: last_normal, idx 2,3");
        step(2'b11,2'b11,2'b00, 1'b1, 2'b00, 2'b00, 1'b0, 16'hFFFF, 4'b0000, 3, "pc2: fast-fwd, unused->all-ones");
        step(2'b11,2'b11,2'b00, 1'b1, 2'b11, 2'b11, 1'b0, 16'h0504, 4'b0100, 4, "pc3: first_wait (is_wait rises)");
        step(2'b11,2'b11,2'b00, 1'b1, 2'b11, 2'b11, 1'b0, 16'h0706, 4'b0000, 4, "pc4 sat w=0: idx 6,7 (offset 0)");
        step(2'b11,2'b11,2'b11, 1'b1, 2'b11, 2'b11, 1'b0, 16'h0908, 4'b0000, 4, "pc4 sat w=1: +2 -> idx 8,9");
        step(2'b00,2'b00,2'b00, 1'b1, 2'b11, 2'b11, 1'b1, 16'h0B0A, 4'b1000, 4, "pc4 sat w=2 + FINISH: last_wait, +4 -> 10,11");
        step(2'b00,2'b00,2'b00, 1'b0, 2'b00, 2'b00, 1'b0, 16'hFFFF, 4'b0000, 4, "stall");

        repeat (4) @(posedge clk);   // trailing cycles: full waveform tail

        if (errors == 0) $display("tb_RootOutputSync: PASS");
        else             $display("tb_RootOutputSync: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
