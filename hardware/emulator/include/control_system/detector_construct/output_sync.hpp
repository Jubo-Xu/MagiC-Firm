// output_sync.hpp — detector output synchronization block.
//
// Groups incoming detectors (from local kernels + forwarded children) into output
// rounds. Functionally identical to MeasurementSync (per-line FIFO, sync-mask
// barrier, bypass for just-in-time arrivals, all-zero fast-forward, registered
// outputs, regfile pointer) — just detector lines instead of measurement lines.
//
// Shared front-end for both output paths; the root attaches a global-index part
// and a stage attaches an OR-reduce on top of the synced detector word.
//
// Parameters:
//   d       number of detector lines
//   d_fifo  depth of each per-line sync FIFO
#pragma once

#include <systemc>

#include <cstdint>
#include <deque>
#include <vector>

#include "signals.hpp"

namespace emu {

SC_MODULE(OutputSync) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;   // asynchronous, active high

    // --- inputs ---
    sc_core::sc_in<Bits> in_det;     // [d-1:0] incoming detectors
    sc_core::sc_in<Bits> in_valid;   // [d-1:0] per-line valid (1 for one cycle)
    sc_core::sc_in<Bits> in_finish;  // [d-1:0] per-line finish tag (rides with each detector)
    sc_core::sc_in<Bits> sync_mask;  // [d-1:0] mask from the output-sync regfile @ pc

    // --- outputs ---
    sc_core::sc_out<Bits>     out_det;          // [d-1:0] synced detectors (0 where unused)
    sc_core::sc_out<Bits>     out_used;         // [d-1:0] which outputs are used this round (= mask)
    sc_core::sc_out<bool>     out_valid;        // 1 for one cycle when out_det is a valid synced word
    sc_core::sc_out<bool>     out_finish;       // 1 when the emitted det-time is the last round
    sc_core::sc_out<uint32_t> sync_regfile_pc;  // read address into the output-sync regfile

    // sat_pc: if >= 0, the pc saturates (holds) at this index instead of advancing
    // past it — used for the copy-last saturating wait row. -1 disables (normal).
    OutputSync(sc_core::sc_module_name nm, int d, int d_fifo, int sat_pc = -1);

  private:
    void tick();  // clocked process (async reset)

    const int d_;
    const int d_fifo_;
    const int sat_pc_;
    uint32_t  pc_ = 0;
    std::vector<std::deque<uint8_t>> fifo_;      // per-line buffered detector bits
    std::vector<std::deque<uint8_t>> fifo_fin_;  // per-line buffered finish tags (parallel to fifo_)
};

}  // namespace emu
