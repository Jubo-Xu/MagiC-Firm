// kernel.hpp — one detector-construction channel (kernel).
//
// Takes the board's synced measurement word (from MeasurementSync) and builds
// this channel's detectors. A fixed selector picks n of the m measurement lines;
// h cores each XOR-accumulate a masked subset of those n and emit-and-clear when
// their detector window completes. At most one core emits per meas-time (compiler
// guarantee), so the emit bits are a one-hot select for the single detector out.
//
// Datapath (selector -> mask-AND -> XOR-tree -> XOR-with-reg) is combinational;
// core registers, the core-regfile pointer, and the outputs are registered, so
// the block has one-cycle latency. The pointer advances once per in_valid.
//
// Parameters:
//   n  selector width (measurements selected per kernel)
//   h  number of cores
//   m  number of board measurement input lines
// Derived: index_width = ceil(log2 m) (bits per selector index).
//
// Regfile field formats (match the compiler serializer):
//   selector_indexes[n*index_width-1:0] = {index n-1, ..., index 0}
//   core_mask[h*(n+1)-1:0]              = {core h-1, ..., core 0}, core c at bit
//                                         c*(n+1); within a core bits[n-1:0]=select,
//                                         bit[n]=emit.
#pragma once

#include <systemc>

#include <cstdint>
#include <vector>

#include "signals.hpp"

namespace emu {

SC_MODULE(Kernel) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;   // asynchronous, active high

    // --- inputs (from MeasurementSync) ---
    sc_core::sc_in<bool> in_valid;   // out_valid of MeasurementSync
    sc_core::sc_in<Bits> in_used;    // [m-1:0] out_used of MeasurementSync
    sc_core::sc_in<Bits> in_meas;    // [m-1:0] out_meas of MeasurementSync
    sc_core::sc_in<bool> in_finish;  // out_finish of MeasurementSync (finish of the consumed meas-time)

    // --- regfile inputs ---
    sc_core::sc_in<Bits> selector_indexes;  // [n*index_width-1:0] static (preconfigured)
    sc_core::sc_in<Bits> core_mask;         // [h*(n+1)-1:0] read @ core_regfile_pc

    // --- outputs ---
    sc_core::sc_out<bool>     out_valid;         // 1 for one cycle when a detector is emitted
    sc_core::sc_out<bool>     out_det;           // the constructed detector bit
    sc_core::sc_out<bool>     out_finish;        // finish of the meas-time this detector completed on
    sc_core::sc_out<uint32_t> core_regfile_pc;   // read address into the core regfile

    // sat_pc: if >= 0, the core-regfile pc saturates (holds) at this index — the
    // copy-last saturating wait row. -1 disables (normal). Advances with the shared
    // MeasurementSync valids, so it takes the same saturation index.
    Kernel(sc_core::sc_module_name nm, int n, int h, int m, int sat_pc = -1);

  private:
    void tick();

    const int n_;
    const int h_;
    const int m_;
    const int index_width_;
    const int sat_pc_;
    uint32_t  pc_ = 0;
    std::vector<uint8_t> reg_;  // per-core accumulators (size h)
};

}  // namespace emu
