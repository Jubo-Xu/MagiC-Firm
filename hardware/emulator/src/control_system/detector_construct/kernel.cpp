// kernel.cpp — implementation of Kernel (one detector-construction channel).
//
// Per accepted meas-time (in_valid=1):
//   b_m[j] = in_used[idx_j] & in_meas[idx_j]     (selector picks line idx_j)
//   for each core c:
//     result_c = XOR_j (sel_c[j] & b_m[j])        (masked XOR-tree)
//     b_c      = result_c ^ reg_c                 (accumulate)
//     if emit_c: out_det = b_c; reg_c <= 0        (emit-and-clear)
//     else:      reg_c <= b_c
//   out_valid = (any emit); out_det = emitting core's b_c (one-hot select)
//   core_regfile_pc advances by 1
// When in_valid=0: pc holds, regs hold, outputs are 0.
#include "control_system/detector_construct/kernel.hpp"

#include <algorithm>

#include "common.hpp"   // index_width
// extract() comes from signals.hpp (via the header)

namespace emu {

Kernel::Kernel(sc_core::sc_module_name nm, int n, int h, int m, int sat_pc)
    : sc_module(nm), n_(n), h_(h), m_(m),
      index_width_(index_width(m)), sat_pc_(sat_pc), reg_(h, 0) {
    SC_HAS_PROCESS(Kernel);
    SC_METHOD(tick);
    sensitive << clk.pos() << rst;
    dont_initialize();
}

void Kernel::tick() {
    if (rst.read()) {
        pc_ = 0;
        std::fill(reg_.begin(), reg_.end(), 0);
        out_valid.write(false);
        out_det.write(false);
        out_finish.write(false);
        core_regfile_pc.write(0);
        return;
    }
    if (!clk.posedge()) return;

    if (!in_valid.read()) {                 // no synced word this cycle: hold everything
        out_valid.write(false);
        out_det.write(false);
        out_finish.write(false);
        core_regfile_pc.write(pc_);
        return;
    }

    const Bits used  = in_used.read();
    const Bits meas  = in_meas.read();
    const Bits sidx  = selector_indexes.read();
    const Bits cmask = core_mask.read();
    sc_assert(used.size()  == static_cast<std::size_t>(m_) &&
              meas.size()  == static_cast<std::size_t>(m_));
    sc_assert(sidx.size()  == static_cast<std::size_t>(n_ * index_width_));
    sc_assert(cmask.size() == static_cast<std::size_t>(h_ * (n_ + 1)));

    // --- selector: gather the n selected (and used) measurement bits ---
    std::vector<uint8_t> b_m(n_, 0);
    for (int j = 0; j < n_; ++j) {
        uint32_t idx = extract(sidx, static_cast<std::size_t>(j) * index_width_, index_width_);
        b_m[j] = (idx < static_cast<uint32_t>(m_) && used[idx] && meas[idx]) ? 1 : 0;
    }

    // --- cores: masked XOR-tree, accumulate, emit-and-clear ---
    const int cw = n_ + 1;
    bool    any_emit = false;
    uint8_t out_bit  = 0;
    for (int c = 0; c < h_; ++c) {
        uint8_t result = 0;
        for (int j = 0; j < n_; ++j)
            result ^= (cmask[c * cw + j] & b_m[j]);
        uint8_t emit = cmask[c * cw + n_];
        uint8_t b_c  = result ^ reg_[c];
        if (emit) { out_bit ^= b_c; any_emit = true; reg_[c] = 0; }
        else      { reg_[c] = b_c; }
    }

    out_valid.write(any_emit);
    out_det.write(any_emit ? static_cast<bool>(out_bit) : false);
    out_finish.write(any_emit ? in_finish.read() : false);   // emitted detector carries this meas-time's finish
    pc_ = pc_ + 1;
    if (sat_pc_ >= 0 && pc_ > static_cast<uint32_t>(sat_pc_))
        pc_ = sat_pc_;   // saturate at the copy-last wait row
    core_regfile_pc.write(pc_);
}

}  // namespace emu
