// detector_construct_block.hpp — the universal per-board detector-construct block.
//
// ONE parameterized module for every board kind (leaf / router / root / stage /
// monolithic). Which sub-blocks exist is derived from the config; a case is just
// different params + regfile data:
//
//   raw path    (raw_out>0)  : RawSelector                  -> fwd_raw
//   construct   (k>0)        : MeasurementSync -> k Kernels  -> construct dets
//   pass        (d_in>0)     : DetectorPass                  -> pass dets
//   out bus                  : d_out = pass(d_in) ++ construct(k)  -> fwd_det (if !root)
//   output sync (root||stage): OutputSync / RootOutputSync (root: + global index)
//   postselect  (stage)      : Postselect
//
// Storage split: ADDRESS-DRIVEN regfiles (sync, per-kernel core, output_sync,
// global_index, postselect) are RegFileROMs read at a running pc. STATIC selector
// indices (kernel + raw) are preconfiguration registers -> held Bits constants
// (drive_constants). Absent buses are 0-width (Bits(0)).
#pragma once

#include <systemc>

#include <memory>
#include <vector>

#include "control_system/detector_construct/detector_pass.hpp"
#include "control_system/detector_construct/kernel.hpp"
#include "control_system/detector_construct/measurement_sync.hpp"
#include "control_system/detector_construct/output_sync.hpp"
#include "control_system/detector_construct/postselect.hpp"
#include "control_system/detector_construct/raw_selector.hpp"
#include "control_system/detector_construct/root_output_sync.hpp"
#include "lib/regfile_rom.hpp"
#include "signals.hpp"

namespace emu {

struct KernelConfig {
    Bits              selector_indexes;  // n*index_width bits (packed, static)
    std::vector<Bits> core_words;        // T words, each h*(n+1) bits
};

struct DcbConfig {
    // --- sizes / roles ---
    int  m = 0;              // raw measurement input lines
    int  d_in = 0;           // input detectors from children (0 for leaf/monolithic)
    int  k = 0;              // kernels
    int  n = 0;              // selector width (n_cap)
    int  h = 0;              // cores per kernel
    int  raw_out = 0;        // raw measurements forwarded up (0 for root)
    int  idx_w = 0;          // global-index REGFILE width (root)
    int  hw_width = 0;       // global-index OUTPUT datapath width (root, >= idx_w)
    int  stride = 0;         // detectors per wait round (root)
    int  sentinel = 0;       // all-ones regfile index = unused slot (root)
    bool copy_last = false;  // copy-last wait row present -> pc saturates at each regfile's last row
    int  sync_fifo = 4;
    int  out_fifo = 4;
    bool is_root = false;
    bool has_postselect = false;
    bool has_osync = false;   // board carries an output_sync regfile (with --output-sync all: every board)

    // --- regfile data (empty when the corresponding block is absent) ---
    std::vector<Bits>         sync_words;            // T x m
    std::vector<KernelConfig> kernels;               // k
    Bits                      raw_selector_indexes;  // raw_out*index_width (static)
    std::vector<Bits>         output_sync_words;     // ndt x d_out
    std::vector<Bits>         global_index_words;    // ndt x d_out*idx_w (root)
    std::vector<Bits>         round_marker_words;    // ndt x 3 (root)
    std::vector<Bits>         postselect_words;      // ndt x d_out (flat)

    int  d_out() const { return d_in + k; }
    // OutputSync is the output stage on every board that carries an output_sync regfile
    // (root + stage always; all boards when compiled with --output-sync all).
    bool has_output_sync() const { return is_root || has_postselect || has_osync; }
};

SC_MODULE(DetectorConstructBlock) {
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- inputs ---
    sc_core::sc_in<Bits> in_meas;         // [m] raw measurements
    sc_core::sc_in<Bits> in_valid;        // [m] per-line valid
    sc_core::sc_in<Bits> in_meas_finish;  // [m] per-line finish (leaf: stimulus; else child raw-bit broadcast)
    sc_core::sc_in<Bits> in_det;          // [d_in] children detectors
    sc_core::sc_in<Bits> in_det_valid;    // [d_in] per-line valid
    sc_core::sc_in<Bits> in_det_finish;   // [d_in] per-line finish (child det-link bit broadcast at assembly)

    // --- forward-up outputs (to parent; meaningful when !is_root / raw_out>0) ---
    sc_core::sc_out<Bits> fwd_det;         // [d_out]
    sc_core::sc_out<Bits> fwd_det_valid;   // [d_out]
    sc_core::sc_out<bool> fwd_det_finish;  // single: this board's detector-output finish (to parent det link)
    sc_core::sc_out<Bits> fwd_raw;         // [raw_out]
    sc_core::sc_out<Bits> fwd_raw_valid;   // [raw_out]
    sc_core::sc_out<bool> fwd_raw_finish;  // single: this board's raw-output finish (to parent raw link)

    // --- root final output (meaningful when is_root) ---
    sc_core::sc_out<Bits>     out_det;             // [d_out] synced detectors
    sc_core::sc_out<Bits>     out_used;            // [d_out]
    sc_core::sc_out<bool>     out_valid;
    sc_core::sc_out<bool>     out_finish;          // 1 when the emitted det-time is the last round
    sc_core::sc_out<Bits>     out_global_indexes;  // [d_out*hw_width]

    // --- root boundary markers for the control core (meaningful when is_root) ---
    sc_core::sc_out<bool> first_normal, last_normal, first_wait, last_wait;

    // --- stage postselect (meaningful when has_postselect) ---
    sc_core::sc_out<bool> post_select;

    DetectorConstructBlock(sc_core::sc_module_name nm, const DcbConfig& cfg);

  private:
    void drive_constants();  // hold static selector indices (kernel + raw)
    void gather();           // d_out bus = pass(d_in) ++ construct(k)
    void drive_output();     // root: internal synced det/valid -> block output ports
    void drive_fwd();        // non-root: OutputSync-synced det/used/valid -> fwd_det ports

    DcbConfig cfg_;

    // --- construct path ---
    std::unique_ptr<RegFileROM>       sync_rom_;
    std::unique_ptr<MeasurementSync>  ms_;
    std::vector<std::unique_ptr<RegFileROM>> core_roms_;
    std::vector<std::unique_ptr<Kernel>>     kernels_;
    // --- pass path ---
    std::unique_ptr<DetectorPass>     pass_;
    // --- raw path ---
    std::unique_ptr<RawSelector>      raw_;
    // --- output path ---
    std::unique_ptr<RegFileROM>       osync_rom_, gidx_rom_, rm_rom_, ps_rom_;
    std::unique_ptr<OutputSync>       osync_;
    std::unique_ptr<RootOutputSync>   rosync_;
    std::unique_ptr<Postselect>       ps_;

    // --- signals ---
    sc_core::sc_signal<uint32_t> sync_pc_;
    sc_core::sc_signal<Bits>     sync_mask_, ms_meas_, ms_used_, raw_sel_const_;
    sc_core::sc_signal<bool>     ms_valid_, ms_finish_;
    sc_core::sc_vector<sc_core::sc_signal<uint32_t>> core_pc_;
    sc_core::sc_vector<sc_core::sc_signal<Bits>>     core_mask_, sel_const_;
    sc_core::sc_vector<sc_core::sc_signal<bool>>     kern_det_, kern_valid_, kern_finish_;
    sc_core::sc_signal<Bits>     pass_det_, pass_valid_, pass_finish_;
    sc_core::sc_signal<Bits>     dbus_det_, dbus_valid_, dbus_finish_;   // the d_out bus
    sc_core::sc_signal<uint32_t> osync_pc_;
    sc_core::sc_signal<Bits>     osync_mask_, gidx_word_, rm_word_, osync_det_, osync_used_;
    sc_core::sc_signal<bool>     osync_valid_, osync_finish_;
    sc_core::sc_signal<Bits>     ps_mask_;
};

}  // namespace emu
