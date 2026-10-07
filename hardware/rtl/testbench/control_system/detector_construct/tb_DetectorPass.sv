// tb_DetectorPass.sv — testbench for DetectorPass (D=4).
//
// Written in BOTH styles:
//   (1) a per-cycle table below, so the Vivado waveform can be read off directly;
//   (2) self-checking: mismatch count + PASS/FAIL for Verilator / xsim.
//
// Mirrors the SystemC unit test emulator/tests/.../test_detector_pass.cpp, extended
// for the P2b per-line finish bus (in_finish -> out_finish, delayed one cycle).
//
// DetectorPass is a pure 1-cycle register of three independent buses, so the output
// this cycle == the input driven the PREVIOUS cycle. Each step() drives THIS cycle's
// inputs and checks the outputs, which reflect the PREVIOUS step's inputs.
//
// ==========================================================================
//  step | in_det in_valid in_finish | out(=prev in): det valid finish | note
// ------+---------------------------+---------------------------------+--------
//   0   | 1101   0101     0100      | 0000  0000  0000  (reset)       | first drive
//   1   | 1010   1010     0010      | 1101  0101  0100                | delay step0
//   2   | 0011   1111     1111      | 1010  1010  0010                | delay step1
//   3   | 0000   0000     0000      | 0011  1111  1111                | delay step2
//   4   | 0000   0000     0000      | 0000  0000  0000                | drained
// ==========================================================================
// Bit order: bit i == line i. "1101" = lines 0,2,3 set.
`timescale 1ns / 1ps

module tb_DetectorPass;

    localparam int D = 4;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    logic         rst;
    logic [D-1:0] in_valid, in_det, in_finish;
    logic [D-1:0] out_valid, out_det, out_finish;

    int errors = 0, step_i = 0;

    DetectorPass #(.D(D)) dut (
        .clk, .rst, .in_valid, .in_det, .in_finish,
        .out_valid, .out_det, .out_finish);

    // Drive THIS cycle's inputs just after the posedge, then check the outputs —
    // which are the registered result of the PREVIOUS step's inputs (xd/xv/xf).
    task automatic step(input logic [D-1:0] id,      // this cycle's inputs
                        input logic [D-1:0] iv,
                        input logic [D-1:0] ifin,
                        input logic [D-1:0] xd,      // expected outputs this cycle (= prev inputs)
                        input logic [D-1:0] xv,
                        input logic [D-1:0] xf,
                        input string note);
        @(posedge clk);
        #1;
        if (out_det !== xd || out_valid !== xv || out_finish !== xf) begin
            errors++;
            $display("  step %0d MISMATCH: out det=%b val=%b fin=%b  exp det=%b val=%b fin=%b  %s",
                     step_i, out_det, out_valid, out_finish, xd, xv, xf, note);
        end else begin
            $display("  step %0d ok: out det=%b val=%b fin=%b  %s",
                     step_i, out_det, out_valid, out_finish, note);
        end
        in_det = id; in_valid = iv; in_finish = ifin;   // drive next inputs (registered @ next posedge)
        step_i++;
    endtask

    initial begin
        rst = 1'b1; in_det = '0; in_valid = '0; in_finish = '0;
        @(posedge clk); @(posedge clk); #1; rst = 1'b0;
        $display("tb_DetectorPass: D=%0d", D);

        //    in_det   in_valid in_finish  exp_det  exp_val  exp_fin  note
        step(4'b1101, 4'b0101, 4'b0100,  4'b0000, 4'b0000, 4'b0000, "first drive (out still reset)");
        step(4'b1010, 4'b1010, 4'b0010,  4'b1101, 4'b0101, 4'b0100, "out = step0 inputs");
        step(4'b0011, 4'b1111, 4'b1111,  4'b1010, 4'b1010, 4'b0010, "out = step1 inputs");
        step(4'b0000, 4'b0000, 4'b0000,  4'b0011, 4'b1111, 4'b1111, "out = step2 inputs");
        step(4'b0000, 4'b0000, 4'b0000,  4'b0000, 4'b0000, 4'b0000, "drained");

        repeat (4) @(posedge clk);   // trailing cycles: full waveform tail

        if (errors == 0) $display("tb_DetectorPass: PASS");
        else             $display("tb_DetectorPass: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
