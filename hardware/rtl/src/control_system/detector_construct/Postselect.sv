// Postselect.sv — stage-board postselect (reject) block (RTL).
//
// 1:1 RTL mapping of the SystemC model (emulator/src/detector_construct/
// postselect.cpp). Taps the synced detector word (OutputSync.out_det) and its
// mask: post_select = 1 iff any masked postselect detector fired this det-time
// ("reject now"). No in_used needed (unused lines are 0 in in_det).
//
// NO internal program counter: the postselect regfile is addressed by the SHARED
// OutputSync read pointer (osync_pc) at assembly — the postselect and output-sync
// regfiles are built entry-for-entry by det-time, so they stay aligned and the
// copy-last saturation of osync_pc is inherited for free (the appended zero row =>
// no reject during the wait rounds). The mask is registered one cycle (the
// global_index gi_reg pattern) to line up with OutputSync's REGISTERED detector
// word; the OR-reduce on top is purely combinational.
//
// Uses the FLAT postselect regfile (one d-bit mask per det-time).
//
// Parameters:
//   D  number of detector lines
module Postselect #(
    parameter int D = 4
) (
    input  logic         clk,
    input  logic         rst,              // asynchronous, active high
    input  logic         in_valid,         // OutputSync.out_valid (synced word valid this cycle)
    input  logic [D-1:0] in_det,           // synced detectors (OutputSync.out_det)
    input  logic [D-1:0] postselect_mask,  // flat mask, ps_rom @ shared osync_pc
    output logic         post_select       // combinational: 1 when a postselect detector fired
);

    // one-cycle mask-align register: the mask read at osync_pc on the edge that
    // produced the current registered out_det is available alongside that word one
    // cycle later (same alignment global_index uses via gi_reg).
    logic [D-1:0] mask_reg;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) mask_reg <= '0;
        else     mask_reg <= postselect_mask;
    end

    // combinational OR-reduce of (aligned mask & synced detectors), gated by valid
    assign post_select = in_valid & (|(mask_reg & in_det));

endmodule
