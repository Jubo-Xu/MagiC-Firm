// detector_construct_block.cpp — wiring of the universal detector-construct block.
//
// Pure structural composition: instantiate the sub-blocks the config calls for,
// wire them, load each RegFileROM. No new datapath logic beyond the `gather`
// (pack the d_out bus = pass ++ construct) and holding the static selectors.
#include "control_system/detector_construct/detector_construct_block.hpp"

#include <string>

namespace emu {

DetectorConstructBlock::DetectorConstructBlock(sc_core::sc_module_name nm, const DcbConfig& cfg)
    : sc_module(nm), cfg_(cfg) {
    const int m = cfg.m, k = cfg.k, n = cfg.n, h = cfg.h, d_in = cfg.d_in;
    const int d_out = cfg.d_out();
    const int T   = static_cast<int>(cfg.sync_words.size());          // meas-times (incl. copy-last row)
    const int ndt = static_cast<int>(cfg.output_sync_words.size());   // det-times  (incl. copy-last row)
    const int core_w = h * (n + 1);
    // copy-last: each block's pc saturates at its own regfile's last row.
    const int ms_sat  = cfg.copy_last ? (T   - 1) : -1;
    const int out_sat = cfg.copy_last ? (ndt - 1) : -1;

    core_pc_.init(k);   core_mask_.init(k);  sel_const_.init(k);
    kern_det_.init(k);  kern_valid_.init(k); kern_finish_.init(k);

    SC_HAS_PROCESS(DetectorConstructBlock);

    // ---------------- raw path ----------------
    if (cfg.raw_out > 0) {
        raw_ = std::make_unique<RawSelector>("raw", cfg.raw_out, m);
        raw_->clk(clk); raw_->rst(rst);
        raw_->selector_indexes(raw_sel_const_);
        raw_->in_meas(in_meas); raw_->in_valid(in_valid); raw_->in_finish(in_meas_finish);
        raw_->out_meas(fwd_raw); raw_->out_valid(fwd_raw_valid); raw_->out_finish(fwd_raw_finish);
    }

    // ---------------- construct path ----------------
    if (k > 0) {
        sync_rom_ = std::make_unique<RegFileROM>("sync_rom", T, m);
        sync_rom_->load(cfg.sync_words);
        sync_rom_->addr(sync_pc_); sync_rom_->data(sync_mask_);

        ms_ = std::make_unique<MeasurementSync>("ms", m, cfg.sync_fifo, ms_sat);
        ms_->clk(clk); ms_->rst(rst);
        ms_->in_meas(in_meas); ms_->in_valid(in_valid); ms_->in_finish(in_meas_finish);
        ms_->sync_mask(sync_mask_);
        ms_->out_meas(ms_meas_); ms_->out_used(ms_used_); ms_->out_valid(ms_valid_);
        ms_->out_finish(ms_finish_);
        ms_->sync_regfile_pc(sync_pc_);

        for (int i = 0; i < k; ++i) {
            auto rom = std::make_unique<RegFileROM>(("core_rom" + std::to_string(i)).c_str(), T, core_w);
            rom->load(cfg.kernels[i].core_words);
            rom->addr(core_pc_[i]); rom->data(core_mask_[i]);
            core_roms_.push_back(std::move(rom));

            auto kn = std::make_unique<Kernel>(("kernel" + std::to_string(i)).c_str(), n, h, m, ms_sat);
            kn->clk(clk); kn->rst(rst);
            kn->in_valid(ms_valid_); kn->in_used(ms_used_); kn->in_meas(ms_meas_);
            kn->in_finish(ms_finish_);
            kn->selector_indexes(sel_const_[i]); kn->core_mask(core_mask_[i]);
            kn->out_valid(kern_valid_[i]); kn->out_det(kern_det_[i]);
            kn->out_finish(kern_finish_[i]);
            kn->core_regfile_pc(core_pc_[i]);
            kernels_.push_back(std::move(kn));
        }
    }

    // ---------------- pass path ----------------
    if (d_in > 0) {
        pass_ = std::make_unique<DetectorPass>("pass", d_in);
        pass_->clk(clk); pass_->rst(rst);
        pass_->in_valid(in_det_valid); pass_->in_det(in_det); pass_->in_finish(in_det_finish);
        pass_->out_valid(pass_valid_); pass_->out_det(pass_det_); pass_->out_finish(pass_finish_);
    }

    // ---------------- gather (d_out bus) + static selectors ----------------
    SC_METHOD(gather);
    if (d_in > 0) sensitive << pass_det_ << pass_valid_ << pass_finish_;
    for (int i = 0; i < k; ++i) sensitive << kern_det_[i] << kern_valid_[i] << kern_finish_[i];

    SC_METHOD(drive_constants);   // runs once at init

    // ---------------- output path ----------------
    if (cfg.has_output_sync()) {
        osync_rom_ = std::make_unique<RegFileROM>("osync_rom", ndt, d_out);
        osync_rom_->load(cfg.output_sync_words);
        osync_rom_->addr(osync_pc_); osync_rom_->data(osync_mask_);

        if (cfg.is_root) {
            gidx_rom_ = std::make_unique<RegFileROM>("gidx_rom", ndt, d_out * cfg.idx_w);
            gidx_rom_->load(cfg.global_index_words);
            gidx_rom_->addr(osync_pc_); gidx_rom_->data(gidx_word_);

            rm_rom_ = std::make_unique<RegFileROM>("rm_rom", ndt, 3);
            rm_rom_->load(cfg.round_marker_words);
            rm_rom_->addr(osync_pc_); rm_rom_->data(rm_word_);

            rosync_ = std::make_unique<RootOutputSync>("rosync", d_out, cfg.out_fifo, cfg.idx_w,
                                                       cfg.hw_width, cfg.stride, cfg.sentinel, out_sat);
            rosync_->clk(clk); rosync_->rst(rst);
            rosync_->in_det(dbus_det_); rosync_->in_valid(dbus_valid_);
            rosync_->in_finish(dbus_finish_);
            rosync_->sync_mask(osync_mask_); rosync_->global_indexes(gidx_word_);
            rosync_->round_marker(rm_word_);
            rosync_->out_det(osync_det_);           // internal (also feeds postselect)
            rosync_->out_used(out_used);            // -> block port
            rosync_->out_valid(osync_valid_);       // internal (also feeds postselect)
            rosync_->out_finish(out_finish);        // -> block port (root final finish)
            rosync_->sync_regfile_pc(osync_pc_);
            rosync_->out_global_indexes(out_global_indexes);   // -> block port
            rosync_->first_normal(first_normal); rosync_->last_normal(last_normal);
            rosync_->first_wait(first_wait);     rosync_->last_wait(last_wait);

            SC_METHOD(drive_output);                // copy internal det/valid to block ports
            sensitive << osync_det_ << osync_valid_;
        } else {
            osync_ = std::make_unique<OutputSync>("osync", d_out, cfg.out_fifo, out_sat);
            osync_->clk(clk); osync_->rst(rst);
            osync_->in_det(dbus_det_); osync_->in_valid(dbus_valid_);
            osync_->in_finish(dbus_finish_);
            osync_->sync_mask(osync_mask_);
            osync_->out_det(osync_det_); osync_->out_used(osync_used_);
            osync_->out_valid(osync_valid_); osync_->out_finish(osync_finish_);
            osync_->sync_regfile_pc(osync_pc_);

            // non-root: the OutputSync-synced detectors ARE this board's forwarded output.
            // Collapse (out_valid, out_used[]) back to per-line fwd_det_valid[] so the link
            // interface to the parent is identical to before (the sync stage is invisible).
            SC_METHOD(drive_fwd);
            sensitive << osync_det_ << osync_used_ << osync_valid_ << osync_finish_;
        }
    }

    // ---------------- postselect ----------------
    if (cfg.has_postselect) {
        ps_rom_ = std::make_unique<RegFileROM>("ps_rom", ndt, d_out);
        ps_rom_->load(cfg.postselect_words);
        ps_rom_->addr(osync_pc_); ps_rom_->data(ps_mask_);   // shared output-sync pc (aligned regfiles)

        ps_ = std::make_unique<Postselect>("ps", d_out);
        ps_->clk(clk); ps_->rst(rst);
        ps_->in_valid(osync_valid_); ps_->in_det(osync_det_);
        ps_->postselect_mask(ps_mask_);
        ps_->post_select(post_select);
    }
}

// static selector registers (kernel + raw) — held at their configured value
void DetectorConstructBlock::drive_constants() {
    for (int i = 0; i < cfg_.k; ++i)
        sel_const_[i].write(cfg_.kernels[i].selector_indexes);
    if (cfg_.raw_out > 0)
        raw_sel_const_.write(cfg_.raw_selector_indexes);
}

// d_out bus = pass(d_in) ++ construct(k); also drive fwd_det (unless root)
void DetectorConstructBlock::gather() {
    const int d_in = cfg_.d_in, k = cfg_.k, d_out = cfg_.d_out();
    Bits det(d_out), val(d_out), fin(d_out);
    if (d_in > 0) {
        const Bits pd = pass_det_.read(), pv = pass_valid_.read(), pf = pass_finish_.read();
        for (int i = 0; i < d_in; ++i) {
            det[i] = (i < static_cast<int>(pd.size())) ? pd[i] : 0;
            val[i] = (i < static_cast<int>(pv.size())) ? pv[i] : 0;
            fin[i] = (i < static_cast<int>(pf.size())) ? pf[i] : 0;
        }
    }
    for (int i = 0; i < k; ++i) {
        det[d_in + i] = kern_det_[i].read() ? 1 : 0;
        val[d_in + i] = kern_valid_[i].read() ? 1 : 0;
        fin[d_in + i] = kern_finish_[i].read() ? 1 : 0;
    }
    dbus_det_.write(det); dbus_valid_.write(val); dbus_finish_.write(fin);
    // fwd_det for non-root now comes from drive_fwd() (OutputSync output); root doesn't forward.
    if (cfg_.is_root) { fwd_det.write(Bits(0)); fwd_det_valid.write(Bits(0)); fwd_det_finish.write(false); }
}

// root: forward the synced det/valid (internal signals) to the block output ports
void DetectorConstructBlock::drive_output() {
    out_det.write(osync_det_.read());
    out_valid.write(osync_valid_.read());
}

// non-root: OutputSync-synced detectors -> fwd_det, collapsing the single out_valid and
// per-line out_used back into the familiar per-line fwd_det_valid[] link interface.
void DetectorConstructBlock::drive_fwd() {
    const int d_out = cfg_.d_out();
    const Bits det  = osync_det_.read();
    const Bits used = osync_used_.read();
    const bool ov   = osync_valid_.read();
    Bits val(d_out);
    for (int i = 0; i < d_out; ++i)
        val[i] = (ov && i < (int)used.size() && used[i]) ? 1 : 0;
    fwd_det.write(det);
    fwd_det_valid.write(val);
    fwd_det_finish.write(osync_finish_.read());   // single: re-synced detector-output finish
}

}  // namespace emu
