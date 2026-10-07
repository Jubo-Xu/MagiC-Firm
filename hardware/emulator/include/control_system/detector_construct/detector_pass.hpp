// detector_pass.hpp — detector forwarding block (detector-pass).
//
// Forwards the children's input detectors up the tree, registered one cycle so
// the pass path is a defined pipeline stage (aligned with the construct path).
// Pure 1-cycle register of the whole d-line bus + its per-line valids — no
// selection, no regfile (the identity form of RawSelector).
//
// Parameters:
//   d  number of detector lines
#pragma once

#include <systemc>

#include "signals.hpp"

namespace emu {

SC_MODULE(DetectorPass) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- inputs ---
    sc_core::sc_in<Bits> in_valid;   // [d-1:0] per-line valid
    sc_core::sc_in<Bits> in_det;     // [d-1:0] input detectors
    sc_core::sc_in<Bits> in_finish;  // [d-1:0] per-line finish (child link bit broadcast at board assembly)

    // --- outputs (registered) ---
    sc_core::sc_out<Bits> out_valid;   // [d-1:0] in_valid delayed one cycle
    sc_core::sc_out<Bits> out_det;     // [d-1:0] in_det delayed one cycle
    sc_core::sc_out<Bits> out_finish;  // [d-1:0] in_finish delayed one cycle

    DetectorPass(sc_core::sc_module_name nm, int d);

  private:
    void tick();

    const int d_;
};

}  // namespace emu
