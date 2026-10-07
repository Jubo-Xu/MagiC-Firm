// detector_pass.cpp — implementation of DetectorPass.
//
// Each cycle: out_det <= in_det, out_valid <= in_valid. Pure 1-cycle register.
#include "control_system/detector_construct/detector_pass.hpp"

namespace emu {

DetectorPass::DetectorPass(sc_core::sc_module_name nm, int d)
    : sc_module(nm), d_(d) {
    SC_HAS_PROCESS(DetectorPass);
    SC_METHOD(tick);
    sensitive << clk.pos() << rst;
    dont_initialize();
}

void DetectorPass::tick() {
    if (rst.read()) {
        out_det.write(Bits(d_));
        out_valid.write(Bits(d_));
        out_finish.write(Bits(d_));
        return;
    }
    if (!clk.posedge()) return;

    const Bits det = in_det.read();
    const Bits val = in_valid.read();
    const Bits fin = in_finish.read();
    sc_assert(det.size() == static_cast<std::size_t>(d_) &&
              val.size() == static_cast<std::size_t>(d_) &&
              fin.size() == static_cast<std::size_t>(d_));
    out_det.write(det);
    out_valid.write(val);
    out_finish.write(fin);
}

}  // namespace emu
