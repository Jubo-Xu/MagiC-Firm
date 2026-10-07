// DrainAggregator.sv — join N drain-done pulses into one "all received" pulse.
// 1:1 port of emulator/include/control_system/cultiv_control/board_control/drain_aggregator.hpp.
//
// Used to aggregate a leaf's P PhysicalMMIO cw_gen_finish pulses into one
// "command generation done" pulse. (The old router role — joining children's
// out_drain_done — is RETIRED: drain completion now rides the DATA path via the
// DetectorConstructBlock's det out_finish, so BoardControl no longer emits
// out_drain_done.) The inputs are one-cycle pulses arriving at DIFFERENT cycles;
// the output is a single one-cycle pulse on the cycle the LAST one arrives.
//
// CLEAN, EDGE-ALIGNED PULSE. The output is derived ONLY from the registered
// latch, never from the combinational input:
//     all_latched    = &latched
//     all_drain_done = all_latched & ~all_latched_q
// so all_drain_done rises ~clock-to-Q after the posedge and stays HIGH across the
// whole cycle including the next posedge — any positive-edge flop catches it by
// construction. A same-cycle &(latched | drain_done_in) would instead track the
// input's arrival time within the cycle, so a flop could miss it. It fires ONE
// cycle after the last drain_done arrives; +1 cycle is negligible on the drain path.
//
// `clear` (driven by START at board level) wipes the latches for the next trial.
// No RUN-time false fire: the sources only pulse on the finish path.
`timescale 1ns / 1ps

module DrainAggregator #(
    parameter int N = 2                 // number of drain-done sources
) (
    input  logic         clk,
    input  logic         rst,
    input  logic         clear,         // wipe latches for a new trial
    input  logic [N-1:0] drain_done_in, // one-cycle pulses, different cycles
    output logic         all_drain_done // 1-cycle pulse the cycle the LAST input arrives
);

    logic [N-1:0] latched;        // which sources have pulsed so far
    logic         all_latched;    // registered latch fully set
    logic         all_latched_q;  // registered all_latched (for the rising edge)

    assign all_latched    = &latched;                       // registered signal only
    assign all_drain_done = all_latched & ~all_latched_q;   // edge-aligned, full-cycle

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            latched       <= '0;
            all_latched_q <= 1'b0;
        end else if (clear) begin
            latched       <= '0;
            all_latched_q <= 1'b0;
        end else begin
            latched       <= latched | drain_done_in;
            all_latched_q <= all_latched;
        end
    end

endmodule
