// tb_RawSelector.sv — testbench for RawSelector (M=4, R=2).
//
// Written in BOTH styles:
//   (1) a per-cycle table below, so the Vivado waveform can be read off directly;
//   (2) self-checking: mismatch count + PASS/FAIL for Verilator / xsim.
//
// Mirrors emulator/tests/.../test_raw_selector.cpp, extended for the P2b single
// out_finish = OR over forwarded (valid) lines of their in_finish.
//
// Selector: out0 <- line2, out1 <- line0  (selector_indexes = {idx1=0, idx0=2} = 4'b0010).
// Outputs are registered, so out this cycle == the selection of the PREVIOUS step's
// inputs. Each step() drives THIS cycle's inputs and checks the outputs.
//
// Bit order: in_*[i] = line i; out_*[i] = output i (out0=line2, out1=line0).
// ==============================================================================
//  step | in_meas in_valid in_finish | out(=prev): meas valid finish | note
// ------+----------------------------+-------------------------------+-----------
//   0   | 1101    0101     0000       | 00   00    0  (reset)         | line0,2 valid; no finish
//   1   | 1010    1010     0000       | 11   11    0                  | l2=1,l0=1 fwd
//   2   | 0001    0101     0100       | 00   00    0                  | l2,l0 not valid -> 0
//   3   | 0001    0001     0100       | 10   11    1                  | l2 finish=1 & valid -> out_finish
//   4   | 0000    0000     0000       | 10   10    0                  | l2 finish set but l2 NOT valid -> 0
//   5   | 0000    0000     0000       | 00   00    0                  | drained
// ==============================================================================
`timescale 1ns / 1ps

module tb_RawSelector;

    localparam int R = 2, M = 4, INDEX_W = 2;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    logic                 rst;
    logic [R*INDEX_W-1:0] selector_indexes;
    logic [M-1:0]         in_meas, in_valid, in_finish;
    logic [R-1:0]         out_meas, out_valid;
    logic                 out_finish;

    int errors = 0, step_i = 0;

    RawSelector #(.R(R), .M(M)) dut (
        .clk, .rst, .selector_indexes, .in_meas, .in_valid, .in_finish,
        .out_meas, .out_valid, .out_finish);

    // Drive THIS cycle's inputs just after the posedge, then check the outputs —
    // the registered selection of the PREVIOUS step's inputs (xm/xv/xf).
    task automatic step(input logic [M-1:0] im,
                        input logic [M-1:0] iv,
                        input logic [M-1:0] ifin,
                        input logic [R-1:0] xm,      // expected out_meas  (= prev selection)
                        input logic [R-1:0] xv,      // expected out_valid
                        input logic         xf,      // expected out_finish
                        input string note);
        @(posedge clk);
        #1;
        if (out_meas !== xm || out_valid !== xv || out_finish !== xf) begin
            errors++;
            $display("  step %0d MISMATCH: out meas=%b val=%b fin=%b  exp meas=%b val=%b fin=%b  %s",
                     step_i, out_meas, out_valid, out_finish, xm, xv, xf, note);
        end else begin
            $display("  step %0d ok: out meas=%b val=%b fin=%b  %s",
                     step_i, out_meas, out_valid, out_finish, note);
        end
        in_meas = im; in_valid = iv; in_finish = ifin;   // drive next inputs (registered @ next posedge)
        step_i++;
    endtask

    initial begin
        selector_indexes = 4'b0010;   // idx0=2 (bits[1:0]=10), idx1=0 (bits[3:2]=00)
        rst = 1'b1; in_meas = '0; in_valid = '0; in_finish = '0;
        @(posedge clk); @(posedge clk); #1; rst = 1'b0;
        $display("tb_RawSelector: M=%0d R=%0d", M, R);

        //    in_meas  in_valid in_finish  exp_meas exp_val exp_fin  note
        step(4'b1101, 4'b0101, 4'b0000,  2'b00,   2'b00,  1'b0, "first drive (out reset)");
        step(4'b1010, 4'b1010, 4'b0000,  2'b11,   2'b11,  1'b0, "out = step0: l2=1,l0=1 fwd");
        step(4'b0001, 4'b0101, 4'b0100,  2'b00,   2'b00,  1'b0, "out = step1: l2,l0 not valid");
        step(4'b0001, 4'b0001, 4'b0100,  2'b10,   2'b11,  1'b1, "out = step2: l2 finish&valid -> out_finish");
        step(4'b0000, 4'b0000, 4'b0000,  2'b10,   2'b10,  1'b0, "out = step3: l2 finish set but l2 NOT valid -> 0");
        step(4'b0000, 4'b0000, 4'b0000,  2'b00,   2'b00,  1'b0, "drained");

        repeat (4) @(posedge clk);   // trailing cycles: full waveform tail

        if (errors == 0) $display("tb_RawSelector: PASS");
        else             $display("tb_RawSelector: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
