// tb_DrainAggregator.sv — testbench for DrainAggregator (N=3).
//
// Written in BOTH styles:
//   (1) a per-cycle table below, with expected all_drain_done passed inline to
//       every step() call, so the Vivado waveform can be read off directly;
//   (2) self-checking: mismatch count + PASS/FAIL for Verilator / xsim.
//
// Mirrors the SystemC unit test
//   emulator/tests/control_system/cultiv_control/board_control/test_drain_aggregator.cpp
//
// all_drain_done = all_latched & ~all_latched_q is derived from REGISTERED signals
// only, so it is edge-aligned and stays HIGH for the full cycle (a positive-edge
// flop catches it reliably). It fires ONE cycle AFTER the last drain_done is
// latched. Each step drives inputs 1 ns after the posedge (registered-source
// timing) and checks 1 ns later.
//
// ============================================================================
//  step | clear drain_done_in | all_drain_done | note
// ------+---------------------+----------------+---------------------------------
//   0   |   0     000         |      0         | idle
//   1   |   0     001         |      0         | src0 pulses (latched at this posedge)
//   2   |   0     000         |      0         | wait
//   3   |   0     100         |      0         | src2 pulses
//   4   |   0     010         |      0         | src1 pulses (LAST) — join fills at this posedge
//   5   |   0     000         |      1  <<<     | FIRE, one cycle after the last arrival
//   6   |   0     000         |      0         | no re-pulse (latch stays full)
//   7   |   1     000         |      0         | clear (new trial)
//   8   |   0     111         |      0         | all three pulse at once
//   9   |   0     000         |      1  <<<     | FIRE, one cycle later
//  10   |   0     000         |      0         | no re-pulse
//  11   |   0     111         |      0         | pulse again, NO clear -> latch already full, no fire
//  12   |   0     000         |      0         | still no fire
//  13   |   1     000         |      0         | clear
//  14   |   0     101         |      0         | partial (2 of 3)
//  15   |   0     010         |      0         | completing source latched at this posedge
//  16   |   0     000         |      1  <<<     | FIRE
//  17   |   0     000         |      0         | done
// ============================================================================
//
// Bit order: drain_done_in[0]=src0, [1]=src1, [2]=src2. "100"=src2, "010"=src1.
`timescale 1ns / 1ps

module tb_DrainAggregator;

    localparam int N = 3;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    logic         rst, clear, all_drain_done;
    logic [N-1:0] drain_done_in;

    int errors = 0, step_i = 0;

    DrainAggregator #(.N(N)) dut (
        .clk, .rst, .clear, .drain_done_in, .all_drain_done);

    task automatic step(input logic c, input logic [N-1:0] d, input logic exp, input string note);
        @(posedge clk);
        #1;                       // drive just after the edge (registered-source timing)
        clear = c; drain_done_in = d;
        #1;                       // settle the combinational output
        if (all_drain_done !== exp) begin
            errors++;
            $display("  step %0d MISMATCH: clear=%0b in=%03b got=%0b exp=%0b  %s",
                     step_i, c, d, all_drain_done, exp, note);
        end else begin
            $display("  step %0d ok: clear=%0b in=%03b all_drain_done=%0b  %s",
                     step_i, c, d, all_drain_done, note);
        end
        step_i++;
    endtask

    initial begin
        rst = 1'b1; clear = 1'b0; drain_done_in = '0;
        @(posedge clk); @(posedge clk); #1; rst = 1'b0;
        $display("tb_DrainAggregator: N=%0d", N);

        //   clear  in     exp  note
        step(1'b0, 3'b000, 1'b0, "idle");
        step(1'b0, 3'b001, 1'b0, "src0 pulses");
        step(1'b0, 3'b000, 1'b0, "wait");
        step(1'b0, 3'b100, 1'b0, "src2 pulses");
        step(1'b0, 3'b010, 1'b0, "src1 pulses (LAST) - join fills");
        step(1'b0, 3'b000, 1'b1, "FIRE (one cycle after last arrival)");
        step(1'b0, 3'b000, 1'b0, "no re-pulse");
        step(1'b1, 3'b000, 1'b0, "clear (new trial)");
        step(1'b0, 3'b111, 1'b0, "all three pulse at once");
        step(1'b0, 3'b000, 1'b1, "FIRE (one cycle later)");
        step(1'b0, 3'b000, 1'b0, "no re-pulse");
        step(1'b0, 3'b111, 1'b0, "pulse again, no clear -> no fire");
        step(1'b0, 3'b000, 1'b0, "still no fire");
        step(1'b1, 3'b000, 1'b0, "clear");
        step(1'b0, 3'b101, 1'b0, "partial (2 of 3)");
        step(1'b0, 3'b010, 1'b0, "completing source latched");
        step(1'b0, 3'b000, 1'b1, "FIRE");
        step(1'b0, 3'b000, 1'b0, "done");

        if (errors == 0) $display("tb_DrainAggregator: PASS");
        else             $display("tb_DrainAggregator: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
