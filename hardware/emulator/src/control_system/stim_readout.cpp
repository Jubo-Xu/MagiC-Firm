// stim_readout.cpp — sim-only measurement readout model (see the header).
#include "control_system/stim_readout.hpp"

namespace emu {

StimReadout::StimReadout(sc_core::sc_module_name nm, const StimReadoutConfig& cfg)
    : sc_module(nm), cfg_(cfg),
      shot_reg_("shot_reg_"), base_reg_("base_reg_"), addr_("addr_"),
      en_("en_"), finish_q_("finish_q_") {
    SC_HAS_PROCESS(StimReadout);

    const int total = (cfg_.shots * cfg_.depth < 1) ? 1 : cfg_.shots * cfg_.depth;
    meas_rom_ = std::make_unique<SyncROM>("meas_rom", total, cfg_.m);
    meas_rom_->load(cfg_.meas);
    meas_rom_->clk(clk); meas_rom_->en(en_); meas_rom_->addr(addr_);
    meas_rom_->data(meas_data_); meas_rom_->data_valid(meas_dv_);

    valid_rom_ = std::make_unique<SyncROM>("valid_rom", total, cfg_.m);
    valid_rom_->load(cfg_.valid);
    valid_rom_->clk(clk); valid_rom_->en(en_); valid_rom_->addr(addr_);
    valid_rom_->data(valid_data_); valid_rom_->data_valid(valid_dv_);

    SC_METHOD(comb_addr);
    sensitive << mmio_out_valid << mmio_out_data << rst << shot_reg_ << base_reg_;

    SC_METHOD(seq);
    sensitive << clk.pos() << rst;
    dont_initialize();

    SC_METHOD(drive_out);
    sensitive << meas_data_ << valid_data_ << finish_q_ << shot_reg_;
}

// this cycle's read: enable + address = base_reg + index (base moved by abort, see seq()).
void StimReadout::comb_addr() {
    const Bits d = mmio_out_data.read();
    const uint32_t idx  = extract(d, 0, d.size());
    const bool active = mmio_out_valid.read() && !rst.read();
    en_.write(active);
    addr_.write(active ? base_reg_.read() + idx : 0);
}

// registered shot progression + the 1-cycle finish pipe (aligned to the BRAM data).
// The POINTER (shot_reg / base_reg) advances on the ABORT event — global and in-sync across
// leaves — NOT on the MMIO index; the read address just follows base_reg. See the header.
void StimReadout::seq() {
    if (rst.read()) {
        shot_reg_.write(0);
        base_reg_.write(0);
        finish_q_.write(false);
        return;
    }
    if (!clk.posedge()) return;

    const bool is_abort = ev_valid.read() && (ev_type.read() == EV_ABORT);
    const bool fin      = cw_gen_finish.read();

    finish_q_.write(mmio_out_valid.read() && fin);   // delay finish to align with the read data

    if (fin) {                                        // trial done -> reset for the next trial
        shot_reg_.write(0);
        base_reg_.write(0);
    } else if (is_abort) {                            // discard current shot -> jump to next shot's base
        const uint32_t next = shot_reg_.read() + 1;
        if (next < static_cast<uint32_t>(cfg_.shots)) {   // guard: hold at the last shot
            shot_reg_.write(next);
            base_reg_.write(next * static_cast<uint32_t>(cfg_.depth));
        }
    }
}

// outputs land one cycle after the read; SyncROM already zeroes data when !data_valid.
void StimReadout::drive_out() {
    const Bits vd = valid_data_.read();
    out_meas.write(meas_data_.read());
    out_meas_valid.write(vd);
    out_meas_finish.write(finish_q_.read() ? vd : Bits(cfg_.m));
    shot_reg.write(shot_reg_.read());
}

}  // namespace emu
