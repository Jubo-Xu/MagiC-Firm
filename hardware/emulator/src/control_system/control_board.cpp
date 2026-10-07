// control_board.cpp — wiring of the universal ControlBoard wrapper.
//
// Pure structural composition: instantiate DCB + BoardControl + (leaf) an array
// of PhysicalMMIOs, then bind them through a handful of small glue processes.
// No new datapath logic beyond the aggregation glue described in the header.
#include "control_system/control_board.hpp"

#include <string>

namespace emu {

ControlBoard::ControlBoard(sc_core::sc_module_name nm, const BoardConfig& cfg)
    : sc_module(nm), cfg_(cfg),
      // The DCB samples these gated nets combinationally by bit-index at every clock
      // edge — including the sc_clock's posedge at t=0, which precedes the driver
      // thread asserting rst. Give them their fixed widths here (they are otherwise
      // default width-0 until drive_gating_broadcast first runs) so that first edge
      // never indexes a zero-width vector. Widths mirror the DCB ports they feed.
      gin_det_valid_("gin_det_valid", Bits(cfg.dcb.d_in)),
      gin_meas_valid_("gin_meas_valid", Bits(cfg.dcb.m)),
      gin_det_finish_("gin_det_finish", Bits(cfg.dcb.d_in)),
      gin_meas_finish_("gin_meas_finish", Bits(cfg.dcb.m)) {
    const int P = cfg_.p();

    // sc_vector output ports: one entry per leaf control core (size 0 otherwise).
    mmio_out_data.init(P);
    mmio_out_valid.init(P);
    cw_gen_finish.init(P);

    SC_HAS_PROCESS(ControlBoard);

    // ---------------- DetectorConstructBlock ----------------
    dcb_ = std::make_unique<DetectorConstructBlock>("dcb", cfg_.dcb);
    dcb_->clk(clk);
    dcb_->rst(board_reset_);                 // board-local reset (folds in rst / abort / finish)
    dcb_->in_meas(in_meas);
    dcb_->in_valid(gin_meas_valid_);         // gated
    dcb_->in_meas_finish(gin_meas_finish_);  // leaf stimulus / child-raw broadcast
    dcb_->in_det(in_det);
    dcb_->in_det_valid(gin_det_valid_);      // gated
    dcb_->in_det_finish(gin_det_finish_);    // child-det broadcast
    dcb_->fwd_det(fwd_det);
    dcb_->fwd_det_valid(dcb_fwd_det_valid_);
    dcb_->fwd_det_finish(dcb_fwd_det_finish_);
    dcb_->fwd_raw(fwd_raw);
    dcb_->fwd_raw_valid(dcb_fwd_raw_valid_);
    dcb_->fwd_raw_finish(fwd_raw_finish);
    dcb_->out_det(out_det);
    dcb_->out_used(out_used);
    dcb_->out_valid(dcb_out_valid_);
    dcb_->out_finish(dcb_out_finish_);
    dcb_->out_global_indexes(out_global_indexes);
    dcb_->first_normal(first_normal);
    dcb_->last_normal(last_normal);
    dcb_->first_wait(first_wait);
    dcb_->last_wait(last_wait);
    dcb_->post_select(dcb_post_select_);

    // ---------------- BoardControl ----------------
    ctrl_ = std::make_unique<BoardControl>("ctrl", cfg_.event_mode, cfg_.data_src,
                                           cfg_.m_ps(), cfg_.payload_w);
    ctrl_->clk(clk);
    ctrl_->rst(rst);                         // global async reset
    ctrl_->in_valid(bc_in_valid_);
    ctrl_->data_attempt(bc_data_attempt_);
    ctrl_->ev_valid(bc_ev_valid_);
    ctrl_->ev_type(bc_ev_type_);
    ctrl_->ev_payload(bc_ev_payload_);
    ctrl_->ev_attempt(bc_ev_attempt_);
    ctrl_->post_select(bc_ps_);
    ctrl_->post_select_attempt(bc_ps_attempt_);
    ctrl_->drain_done(bc_drain_done_);
    ctrl_->data_valid_output(bc_dvo_);
    ctrl_->out_attempt(out_attempt);
    ctrl_->cur_attempt(cur_attempt_);
    ctrl_->discard(discard);
    ctrl_->out_ev_valid(oev_valid_);
    ctrl_->out_ev_type(oev_type_);
    ctrl_->out_ev_payload(oev_payload_);
    ctrl_->out_ev_attempt(oev_attempt_);
    ctrl_->reset(board_reset_);
    ctrl_->set_in_data_to_zero(set_zero_);

    // ---------------- leaf PhysicalMMIO array ----------------
    // Driven by the SAME event that enters BoardControl (bc_ev_*), see header. Each
    // core's instruction / command-word program is loaded from its MmioConfig (filled
    // by the board loader from MMIO_instr_<p>/MMIO_cw_<p>). The measured command words
    // loop back into in_meas via an external StimReadout in a whole-system stim test.
    const bool mmio_from_out_ev = (cfg_.event_mode == BoardControl::ORIGINATE);
    for (int p = 0; p < P; ++p) {
        const MmioConfig& mc = cfg_.mmios[p];
        auto mm = std::make_unique<PhysicalMMIO>(("mmio" + std::to_string(p)).c_str(), mc);
        if (!mc.instr_words.empty()) mm->load_instr(mc.instr_words);
        if (!mc.cw_words.empty())    mm->load_cw(mc.cw_words);
        mm->clk(clk);
        // System reset ONLY — NOT board_reset_. The MMIO handles abort/finish itself via
        // the event bus (sequencer EXEC restarts on EV_ABORT, drains on EV_FINISH) and its
        // own internal rst_out. Folding abort/finish into its top reset would slam the
        // sequencer to IDLE, where it ignores EV_ABORT, so a gap/abort would never re-issue
        // the program and the readout would never advance to the next shot.
        mm->rst(rst);
        // ORIGINATE: generated out_ev (registered, carries internal ABORT).
        // FORWARD  : the received event bus (parent's relayed ABORT).
        if (mmio_from_out_ev) { mm->ev_valid(oev_valid_); mm->ev_type(oev_type_); mm->ev_payload(oev_payload_); }
        else                  { mm->ev_valid(bc_ev_valid_); mm->ev_type(bc_ev_type_); mm->ev_payload(bc_ev_payload_); }
        mm->out_data(mmio_out_data[p]);
        mm->out_valid(mmio_out_valid[p]);
        mm->cw_gen_finish(cw_gen_finish[p]);
        mmios_.push_back(std::move(mm));
    }

    // ---------------- glue processes ----------------
    SC_METHOD(drive_gating_broadcast);
    sensitive << set_zero_ << in_det_valid << in_meas_valid
              << in_det_finish_child << in_raw_finish_child << in_meas_finish;

    SC_METHOD(drive_bc_status);
    sensitive << in_det_valid << in_meas_valid << in_child_attempt;

    SC_METHOD(drive_ev);
    if (cfg_.is_root) sensitive << start << finish;
    else              sensitive << ev_valid << ev_type << ev_payload << ev_attempt;

    SC_METHOD(drive_ps_bus);
    sensitive << dcb_post_select_ << cur_attempt_ << ps_in << ps_in_attempt << gap_post_select;

    SC_METHOD(drive_drain_status);
    sensitive << dcb_out_finish_ << dcb_fwd_det_finish_ << dcb_out_valid_
              << dcb_fwd_det_valid_ << dcb_fwd_raw_valid_;

    SC_METHOD(drive_board_outputs);
    sensitive << dcb_fwd_det_finish_ << dcb_fwd_det_valid_ << dcb_fwd_raw_valid_
              << dcb_out_valid_ << dcb_out_finish_ << dcb_post_select_ << cur_attempt_
              << oev_valid_ << oev_type_ << oev_payload_ << oev_attempt_;
}

// DCB input valids gated by set_in_data_to_zero, plus per-child single-finish
// broadcast across each child's slice of the concatenated bus.
void ControlBoard::drive_gating_broadcast() {
    const bool z = set_zero_.read();

    // Emit FIXED-width vectors (sized by config, not by the input's runtime size):
    // the top-level valid ports are transiently width-0 before their driver first
    // runs, and the DCB indexes these nets by bit at the sc_clock's t=0 posedge.
    // Copy whatever bits the source currently carries; zero when gated or absent.
    const int DI = cfg_.dcb.d_in, M = cfg_.dcb.m;

    Bits dv(DI);
    if (!z) { const Bits s = in_det_valid.read();
              for (int i = 0; i < DI && i < (int)s.size(); ++i) dv[i] = s[i]; }
    gin_det_valid_.write(dv);

    Bits mv(M);
    if (!z) { const Bits s = in_meas_valid.read();
              for (int i = 0; i < M && i < (int)s.size(); ++i) mv[i] = s[i]; }
    gin_meas_valid_.write(mv);

    // detector-finish broadcast (children only; leaf d_in == 0 -> empty)
    Bits idf(cfg_.dcb.d_in);
    const Bits cdf = in_det_finish_child.read();
    for (int c = 0, base = 0; c < cfg_.nchild(); ++c) {
        const bool f = (c < (int)cdf.size()) && cdf[c];
        for (int j = 0; j < cfg_.child_dw[c]; ++j) idf[base + j] = f ? 1 : 0;
        base += cfg_.child_dw[c];
    }
    gin_det_finish_.write(idf);

    // measurement-finish: leaf uses the direct stimulus; a router broadcasts each
    // child's single raw-finish across that child's raw slice.
    if (cfg_.is_leaf) {
        Bits imf(M);
        const Bits s = in_meas_finish.read();
        for (int i = 0; i < M && i < (int)s.size(); ++i) imf[i] = s[i];
        gin_meas_finish_.write(imf);
    } else {
        Bits imf(cfg_.dcb.m);
        const Bits crf = in_raw_finish_child.read();
        for (int c = 0, base = 0; c < cfg_.nchild(); ++c) {
            const bool f = (c < (int)crf.size()) && crf[c];
            for (int j = 0; j < cfg_.child_raw[c]; ++j) imf[base + j] = f ? 1 : 0;
            base += cfg_.child_raw[c];
        }
        gin_meas_finish_.write(imf);
    }
}

// Incoming-data status into BoardControl.
void ControlBoard::drive_bc_status() {
    bc_in_valid_.write(in_det_valid.read().any() || in_meas_valid.read().any());
    bc_data_attempt_.write(cfg_.nchild() > 0 ? in_child_attempt.read().any() : false);
}

// Event bus into BoardControl: root turns host start/finish into an event; a
// non-root board relays the parent's event bus straight through.
void ControlBoard::drive_ev() {
    if (cfg_.is_root) {
        const bool s = start.read(), f = finish.read();
        bc_ev_valid_.write(s || f);
        bc_ev_type_.write(f ? static_cast<uint32_t>(BoardControl::EV_FINISH)
                            : static_cast<uint32_t>(BoardControl::EV_START));
        bc_ev_payload_.write(Bits(cfg_.payload_w));
        bc_ev_attempt_.write(false);  // ORIGINATE ignores this
    } else {
        bc_ev_valid_.write(ev_valid.read());
        bc_ev_type_.write(ev_type.read());
        bc_ev_payload_.write(ev_payload.read());
        bc_ev_attempt_.write(ev_attempt.read());
    }
}

// Root post_select bus assembly: [ own DCB ps (has_postselect) | nps stage boards
// | gap ]. Own + gap slots are stamped with cur_attempt so their XNOR-equality
// always matches (a set bit is a live reject); stage slots keep their own attempt.
void ControlBoard::drive_ps_bus() {
    const int mps = cfg_.m_ps();
    Bits ps(mps), pa(mps);
    const bool cur = cur_attempt_.read();

    if (cfg_.has_postselect) {
        ps[cfg_.own_base()] = dcb_post_select_.read() ? 1 : 0;
        pa[cfg_.own_base()] = cur ? 1 : 0;
    }
    const Bits sin = ps_in.read(), sia = ps_in_attempt.read();
    for (int i = 0; i < cfg_.nps; ++i) {
        ps[cfg_.stage_base() + i] = (i < (int)sin.size() && sin[i]) ? 1 : 0;
        pa[cfg_.stage_base() + i] = (i < (int)sia.size() && sia[i]) ? 1 : 0;
    }
    ps[cfg_.gap_index()] = gap_post_select.read() ? 1 : 0;
    pa[cfg_.gap_index()] = cur ? 1 : 0;

    bc_ps_.write(ps);
    bc_ps_attempt_.write(pa);
}

// drain_done + data_valid_output into BoardControl. drain_done is this board's own
// DCB detector-finish (root: out_finish; non-root: fwd_det_finish).
void ControlBoard::drive_drain_status() {
    bc_drain_done_.write(cfg_.is_root ? dcb_out_finish_.read() : dcb_fwd_det_finish_.read());
    const bool dvo = cfg_.is_root
                         ? dcb_out_valid_.read()
                         : (dcb_fwd_det_valid_.read().any() || dcb_fwd_raw_valid_.read().any());
    bc_dvo_.write(dvo);
}

// Fan the DCB-internal nets we also needed for glue out to the top ports, plus
// this board's post-select fast path (stamped cur_attempt).
void ControlBoard::drive_board_outputs() {
    fwd_det_finish.write(dcb_fwd_det_finish_.read());
    fwd_det_valid.write(dcb_fwd_det_valid_.read());
    fwd_raw_valid.write(dcb_fwd_raw_valid_.read());
    out_valid.write(dcb_out_valid_.read());
    out_finish.write(dcb_out_finish_.read());
    ps_out.write(dcb_post_select_.read());
    ps_out_attempt.write(cur_attempt_.read());
    // BoardControl's generated event bus -> children (and, for ORIGINATE, the MMIOs)
    out_ev_valid.write(oev_valid_.read());
    out_ev_type.write(oev_type_.read());
    out_ev_payload.write(oev_payload_.read());
    out_ev_attempt.write(oev_attempt_.read());
}

}  // namespace emu
