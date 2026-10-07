// raw_selector.hpp — raw-measurement forwarding selector (raw-measure-filter).
//
// A parallel tap of the board's raw measurement input: it forwards r of the m
// raw measurement lines (chosen by a fixed selector) up the tree, each with its
// valid. Pure select + one-cycle register — independent of the detector path
// (no synchronization with detectors is needed; each carries its own valid).
//
// Parameters:
//   r  number of raw output measurements
//   m  number of raw input measurement lines
// Derived: index_width = ceil(log2 m) (bits per selector index).
//
// Regfile format (matches the compiler serializer):
//   selector_indexes[r*index_width-1:0] = {index r-1, ..., index 0}
#pragma once

#include <systemc>

#include "signals.hpp"

namespace emu {

SC_MODULE(RawSelector) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;   // asynchronous, active high

    // --- inputs ---
    sc_core::sc_in<Bits> selector_indexes;  // [r*index_width-1:0] static (preconfigured)
    sc_core::sc_in<Bits> in_meas;           // [m-1:0] raw measurements
    sc_core::sc_in<Bits> in_valid;          // [m-1:0] per-line valid
    sc_core::sc_in<Bits> in_finish;         // [m-1:0] per-line finish

    // --- outputs (registered) ---
    sc_core::sc_out<Bits> out_meas;    // [r-1:0] out_meas[i] = in_meas[index i]
    sc_core::sc_out<Bits> out_valid;   // [r-1:0] out_valid[i] = in_valid[index i]
    sc_core::sc_out<bool> out_finish;  // single: OR over forwarded (valid) lines of their finish

    RawSelector(sc_core::sc_module_name nm, int r, int m);

  private:
    void tick();

    const int r_;
    const int m_;
    const int index_width_;
};

}  // namespace emu
