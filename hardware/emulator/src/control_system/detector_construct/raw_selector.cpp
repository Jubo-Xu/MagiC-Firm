// raw_selector.cpp — implementation of RawSelector.
//
// Each cycle: out_meas[i]  <= in_meas[idx_i]
//             out_valid[i] <= in_valid[idx_i]
// where idx_i is selector index i. Registered (one-cycle delay); no gating (the
// valid rides along with each line).
#include "control_system/detector_construct/raw_selector.hpp"

#include "common.hpp"   // index_width
// extract() from signals.hpp (via the header)

namespace emu {

RawSelector::RawSelector(sc_core::sc_module_name nm, int r, int m)
    : sc_module(nm), r_(r), m_(m), index_width_(index_width(m)) {
    SC_HAS_PROCESS(RawSelector);
    SC_METHOD(tick);
    sensitive << clk.pos() << rst;
    dont_initialize();
}

void RawSelector::tick() {
    if (rst.read()) {
        out_meas.write(Bits(r_));
        out_valid.write(Bits(r_));
        out_finish.write(false);
        return;
    }
    if (!clk.posedge()) return;

    const Bits meas = in_meas.read();
    const Bits val  = in_valid.read();
    const Bits fin  = in_finish.read();
    const Bits sidx = selector_indexes.read();
    sc_assert(meas.size() == static_cast<std::size_t>(m_) &&
              val.size()  == static_cast<std::size_t>(m_) &&
              fin.size()  == static_cast<std::size_t>(m_) &&
              sidx.size() == static_cast<std::size_t>(r_ * index_width_));

    Bits om(r_), ov(r_);
    bool of = false;
    for (int i = 0; i < r_; ++i) {
        uint32_t idx = extract(sidx, static_cast<std::size_t>(i) * index_width_, index_width_);
        if (idx < static_cast<uint32_t>(m_)) {
            om[i] = meas[idx];
            ov[i] = val[idx];
            if (val[idx] && fin[idx]) of = true;   // finish of the forwarded round
        }
    }
    out_meas.write(om);
    out_valid.write(ov);
    out_finish.write(of);
}

}  // namespace emu
