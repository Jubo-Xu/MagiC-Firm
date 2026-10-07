// postselect.hpp — stage-board postselect (reject) block.
//
// Taps the synced detector word (from OutputSync). Each accepted det-time it ANDs
// the incoming detectors with the postselect mask and OR-reduces: post_select=1
// iff any postselect detector fired this det-time ("reject now"). The shot's
// overall reject is the OR of these pulses (accumulated downstream), giving the
// earliest possible rejection.
//
// No internal program counter: the postselect regfile is addressed by the SHARED
// OutputSync read pointer (osync_pc) — the postselect and output-sync regfiles are
// built entry-for-entry by det-time, so they stay aligned, and the copy-last
// saturation of osync_pc is inherited for free (the appended zero row => no reject
// during the wait rounds). The mask is registered one cycle (the global_index
// gi_reg pattern) to line up with OutputSync's REGISTERED detector word; the
// OR-reduce on top is purely combinational.
//
// Uses the FLAT postselect regfile (one d-bit mask per det-time; the compiler's
// --postselect-layout flat), so unused/idle lines are simply 0 in in_det and need
// no separate in_used input.
//
// Parameters:
//   d  number of detector lines
#pragma once

#include <systemc>

#include "signals.hpp"

namespace emu {

SC_MODULE(Postselect) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- inputs ---
    sc_core::sc_in<bool> in_valid;         // OutputSync.out_valid (synced word valid this cycle)
    sc_core::sc_in<Bits> in_det;           // [d-1:0] synced detectors (OutputSync.out_det)
    sc_core::sc_in<Bits> postselect_mask;  // [d-1:0] flat mask, ps_rom @ shared osync_pc

    // --- outputs ---
    sc_core::sc_out<bool> post_select;     // combinational: 1 when a postselect detector fired

    Postselect(sc_core::sc_module_name nm, int d);

  private:
    void reg_mask();  // clocked: register the mask one cycle to align with the registered det word
    void drive();     // comb: OR-reduce(in_det & aligned mask), gated by in_valid

    const int d_;
    sc_core::sc_signal<Bits> mask_reg_;   // postselect_mask delayed one cycle
};

}  // namespace emu
