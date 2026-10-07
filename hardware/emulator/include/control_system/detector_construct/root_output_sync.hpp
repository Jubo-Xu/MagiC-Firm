// root_output_sync.hpp — root-node output sync with global detector indices,
// round markers, and the copy-last wait offset.
//
// Wraps OutputSync (the shared detector-sync front-end) and adds, on top of the
// registered synced detector word:
//   * global indices — global_index[nt] read at the same sync_regfile_pc, then
//     registered one cycle (gi_reg) to line up with OutputSync's registered out_det;
//   * round markers  — round_marker[nt] {is_first_normal,is_last_normal,is_wait}
//     read/registered the same way (rm_reg), decoded into the four boundary pulses
//     first_normal / last_normal / first_wait / last_wait for the control core;
//   * copy-last wait offset — the output pc saturates at sat_pc (the appended
//     copy-last row). On the saturating row the global index is the regfile base
//     plus w*stride, where w counts saturating rounds so each wait repeat gets a
//     fresh detector-number block. Non-saturating rounds get offset 0.
//
// The output global-index datapath is hw_width bits/line (>= regfile index_width):
// the regfile value is zero-extended and the offset added; unused slots (regfile
// sentinel or not in out_used) are one-extended to all-ones.
//
// Parameters:
//   d            number of detector lines
//   d_fifo       depth of each per-line sync FIFO (passed to OutputSync)
//   index_width  bits per global index in the REGFILE (= ceil(log2(ndet+stride+1))(+1))
//   hw_width     bits per global index on the OUTPUT datapath (>= index_width)
//   stride       detectors per wait round (used lines at the last det-time)
//   sentinel     all-ones regfile index value marking an unused slot
//   sat_pc       copy-last saturating row index (-1 = no wait row)
#pragma once

#include <systemc>

#include <cstdint>

#include "control_system/detector_construct/output_sync.hpp"
#include "signals.hpp"

namespace emu {

SC_MODULE(RootOutputSync) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- inputs ---
    sc_core::sc_in<Bits> in_det;          // [d-1:0]
    sc_core::sc_in<Bits> in_valid;        // [d-1:0]
    sc_core::sc_in<Bits> in_finish;       // [d-1:0] per-line finish (rides with each detector)
    sc_core::sc_in<Bits> sync_mask;       // [d-1:0] output-sync mask @ pc
    sc_core::sc_in<Bits> global_indexes;  // [d*index_width-1:0] global-index word @ pc
    sc_core::sc_in<Bits> round_marker;    // [2:0] {is_first_normal,is_last_normal,is_wait} @ pc

    // --- outputs ---
    sc_core::sc_out<Bits>     out_det;             // [d-1:0]
    sc_core::sc_out<Bits>     out_used;            // [d-1:0]
    sc_core::sc_out<bool>     out_valid;
    sc_core::sc_out<bool>     out_finish;          // 1 when the emitted det-time is the last round
    sc_core::sc_out<uint32_t> sync_regfile_pc;     // addresses sync + global-index + round-marker regfiles
    sc_core::sc_out<Bits>     out_global_indexes;  // [d*hw_width-1:0] aligned with out_det

    // --- boundary markers for the control core (all one-cycle pulses, aligned w/ out_det) ---
    sc_core::sc_out<bool> first_normal;  // is_first_normal & valid
    sc_core::sc_out<bool> last_normal;   // is_last_normal  & valid
    sc_core::sc_out<bool> first_wait;    // rising edge of is_wait (first wait det-time)
    sc_core::sc_out<bool> last_wait;     // is_wait & finish (last wait round completes)

    RootOutputSync(sc_core::sc_module_name nm, int d, int d_fifo, int index_width,
                   int hw_width, int stride, int sentinel, int sat_pc);

  private:
    void reg_pipe();   // clocked: gi_reg / rm_reg / pc_reg / is_wait_q / w, aligned to out_det
    void drive_out();  // comb: out_valid + markers + global-index (with offset)

    const int d_;
    const int index_width_;
    const int hw_width_;
    const int stride_;
    const int sentinel_;
    const int sat_pc_;

    OutputSync os_;   // shared detector-sync front-end

    sc_core::sc_signal<bool>     ov_int_;     // OutputSync.out_valid
    sc_core::sc_signal<bool>     of_int_;     // OutputSync.out_finish
    sc_core::sc_signal<Bits>     used_int_;   // OutputSync.out_used (tapped, also forwarded)
    sc_core::sc_signal<Bits>     gi_reg_;     // global_indexes delayed one cycle
    sc_core::sc_signal<Bits>     rm_reg_;     // round_marker delayed one cycle
    sc_core::sc_signal<uint32_t> pc_reg_;     // producing pc, delayed one cycle (= current det-time index)
    sc_core::sc_signal<bool>     is_wait_q_;  // is_wait of the previous VALID output round
    sc_core::sc_signal<uint32_t> w_;          // saturating-round counter (wait repeats)
};

}  // namespace emu
