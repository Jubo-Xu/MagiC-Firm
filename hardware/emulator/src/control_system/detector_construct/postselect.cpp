// postselect.cpp — implementation of Postselect.
//
// reg_mask (clocked): mask_reg <= postselect_mask, so the mask read at osync_pc on
//   the edge that produced the current registered det word is available alongside
//   that word one cycle later (same alignment global_index uses via gi_reg).
// drive (comb): post_select = in_valid ? OR_i (mask_reg[i] & in_det[i]) : 0.
// No program counter: saturation of osync_pc (copy-last) is inherited — the
// appended zero row makes mask_reg 0 during the wait rounds, so no reject fires.
#include "control_system/detector_construct/postselect.hpp"

namespace emu {

Postselect::Postselect(sc_core::sc_module_name nm, int d)
    : sc_module(nm), d_(d) {
    SC_HAS_PROCESS(Postselect);
    SC_METHOD(reg_mask); sensitive << clk.pos() << rst; dont_initialize();
    SC_METHOD(drive);    sensitive << in_valid << in_det << mask_reg_;
}

void Postselect::reg_mask() {
    if (rst.read()) { mask_reg_.write(Bits(d_)); return; }
    if (!clk.posedge()) return;
    mask_reg_.write(postselect_mask.read());
}

void Postselect::drive() {
    if (!in_valid.read()) { post_select.write(false); return; }
    const Bits det  = in_det.read();
    const Bits mask = mask_reg_.read();
    bool any = false;                    // any postselect detector fired this det-time
    for (int i = 0; i < d_ && i < static_cast<int>(det.size()) && i < static_cast<int>(mask.size()); ++i)
        if (mask[i] && det[i]) { any = true; break; }
    post_select.write(any);
}

}  // namespace emu
