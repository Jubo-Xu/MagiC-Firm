// drain_aggregator.hpp — join N drain-done pulses into one "all received" pulse.
//
// Used to aggregate a leaf's P PhysicalMMIO cw_gen_finish pulses into one
// "command generation done" pulse. (The old router role — joining children's
// out_drain_done — is RETIRED: drain completion now rides the DATA path via the
// DetectorConstructBlock's det out_finish, so BoardControl no longer emits
// out_drain_done and there is nothing to join on the control path.)
// The inputs are one-cycle pulses that arrive at DIFFERENT cycles; the output is
// a single one-cycle pulse on the cycle the LAST one arrives.
//
// CLEAN, EDGE-ALIGNED PULSE. The output is derived ONLY from the registered
// latch, never from the combinational input:
//     all_latched    = &latched
//     all_drain_done = all_latched & ~all_latched_q
// so all_drain_done rises ~clock-to-Q after the posedge and is guaranteed HIGH
// across the ENTIRE cycle, including the next posedge — any positive-edge flop
// catches it by construction. (A same-cycle `&(latched | drain_done_in)` would
// track the input's arrival time within the cycle, so a flop could miss it — the
// hazard this design avoids.) It fires ONE cycle after the last drain_done
// arrives (the latch captures that pulse at the posedge, then all_latched goes
// true next cycle); +1 cycle is negligible on the finish/drain path.
//
// `clear` (driven by START at the board level) wipes the latches for the next
// trial. No RUN-time false fire is possible: the sources only pulse on the finish
// path, so the latch cannot fill mid-run.
//
// Parameters:
//   n   number of drain-done sources (bus width)
#pragma once

#include <systemc>

#include "signals.hpp"

namespace emu {

SC_MODULE(DrainAggregator) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- inputs ---
    sc_core::sc_in<bool> clear;          // wipe latches for a new trial (e.g. on START)
    sc_core::sc_in<Bits> drain_done_in;  // [n] one-cycle pulses, arriving at different cycles

    // --- output ---
    sc_core::sc_out<bool> all_drain_done;  // 1-cycle pulse the cycle the LAST input arrives

    DrainAggregator(sc_core::sc_module_name nm, int n);

  private:
    void comb();          // all_drain_done = all_latched & ~all_latched_q (registered signals only)
    void seq();           // clocked: latched, all_latched_q
    bool all_latched();   // &latched  (registered latch fully set)

    const int n_;

    sc_core::sc_signal<Bits> latched_;        // which sources have pulsed so far
    sc_core::sc_signal<bool> all_latched_q_;  // registered all_latched (for the rising-edge pulse)
};

}  // namespace emu
