// measurement_sync.hpp — input measurement synchronization block.
//
// One FIFO per raw measurement line. Incoming (valid) measurements are buffered;
// a shared regfile pointer `sync_regfile_pc` walks the sync-mask program. At each
// step the mask says which lines this measurement-time needs; when all needed
// lines have data ready, the block pops them, emits one synced word, and advances
// the pointer. An all-zero mask is a fast-forward (emit nothing, just advance).
//
// Parameters:
//   m       number of raw measurement input lines
//   d_fifo  depth of each per-line sync FIFO
#pragma once

#include <systemc>

#include <cstdint>
#include <deque>
#include <vector>

#include "signals.hpp"

namespace emu {

SC_MODULE(MeasurementSync) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;   // asynchronous, active high

    // --- inputs ---
    sc_core::sc_in<Bits> in_meas;    // [m-1:0] raw measurement bits
    sc_core::sc_in<Bits> in_valid;   // [m-1:0] per-line valid (1 for one cycle)
    sc_core::sc_in<Bits> in_finish;  // [m-1:0] per-line "last round" tag, rides with each measurement
    sc_core::sc_in<Bits> sync_mask;  // [m-1:0] mask read from the sync regfile @ pc

    // --- outputs ---
    sc_core::sc_out<Bits>     out_meas;         // [m-1:0] synced measurements (0 where unused)
    sc_core::sc_out<Bits>     out_used;         // [m-1:0] which outputs are used this round (= mask)
    sc_core::sc_out<bool>     out_valid;        // 1 for one cycle when out_meas is a valid synced word
    sc_core::sc_out<bool>     out_finish;       // 1 when the emitted meas-time is the last round
    sc_core::sc_out<uint32_t> sync_regfile_pc;  // read address into the sync regfile

    // sat_pc: if >= 0, the pc saturates (holds) at this index instead of advancing
    // past it — used for the copy-last saturating wait row. -1 disables (normal).
    MeasurementSync(sc_core::sc_module_name nm, int m, int d_fifo, int sat_pc = -1);

  private:
    void tick();  // clocked process (async reset)

    const int m_;
    const int d_fifo_;
    const int sat_pc_;
    uint32_t  pc_ = 0;
    std::vector<std::deque<uint8_t>> fifo_;      // per-line buffered measurement bits
    std::vector<std::deque<uint8_t>> fifo_fin_;  // per-line buffered finish tags (parallel to fifo_)
};

}  // namespace emu
