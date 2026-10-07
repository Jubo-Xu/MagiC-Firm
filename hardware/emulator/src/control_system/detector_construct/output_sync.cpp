// output_sync.cpp — implementation of OutputSync.
//
// Identical logic to MeasurementSync (see measurement_sync.cpp for the detailed
// modelling notes), for detector lines:
//  * b_valid[i] == "FIFO i non-empty"; needed line's value from FIFO head, else
//    bypass straight from the input when it arrives this cycle (FIFO empty).
//  * A bypassed arrival is consumed, not buffered; FIFO-head consume still buffers
//    a co-arriving detector (later round).
//  * Ready = every needed line available (buffered OR arriving); all-zero mask is
//    trivially ready -> fast-forward.
//  * Outputs registered (one-cycle pulse); pc advances per accepted round.
#include "control_system/detector_construct/output_sync.hpp"

#include <string>
#include <vector>

#include "log.hpp"

namespace emu {

OutputSync::OutputSync(sc_core::sc_module_name nm, int d, int d_fifo, int sat_pc)
    : sc_module(nm), d_(d), d_fifo_(d_fifo), sat_pc_(sat_pc), fifo_(d), fifo_fin_(d) {
    SC_HAS_PROCESS(OutputSync);
    SC_METHOD(tick);
    sensitive << clk.pos() << rst;
    dont_initialize();
}

void OutputSync::tick() {
    // --- asynchronous reset ---
    if (rst.read()) {
        pc_ = 0;
        for (auto& f : fifo_) f.clear();
        for (auto& f : fifo_fin_) f.clear();
        out_det.write(Bits(d_));
        out_used.write(Bits(d_));
        out_valid.write(false);
        out_finish.write(false);
        sync_regfile_pc.write(0);
        return;
    }
    if (!clk.posedge()) return;

    // --- sample inputs ---
    const Bits ind  = in_det.read();
    const Bits inv  = in_valid.read();
    const Bits inf  = in_finish.read();
    const Bits mask = sync_mask.read();
    sc_assert(ind.size()  == static_cast<std::size_t>(d_) &&
              inv.size()  == static_cast<std::size_t>(d_) &&
              inf.size()  == static_cast<std::size_t>(d_) &&
              mask.size() == static_cast<std::size_t>(d_));

    // --- resolve each needed line's source; decide readiness ---
    std::vector<char> from_fifo(d_, 0);
    std::vector<char> bypass(d_, 0);
    bool ready = true;
    for (int i = 0; i < d_; ++i) {
        if (!mask[i]) continue;
        if (!fifo_[i].empty())      from_fifo[i] = 1;
        else if (inv[i])            bypass[i]    = 1;
        else                        ready = false;
    }

    Bits     out_d(d_), out_u(d_);
    bool     ov      = false;
    bool     of      = false;      // finish of the emitted det-time (OR over used lines)
    uint32_t next_pc = pc_;

    if (ready) {
        for (int i = 0; i < d_; ++i) {
            if (!mask[i]) continue;
            out_u[i] = 1;
            out_d[i] = from_fifo[i] ? fifo_[i].front() : ind[i];
            if (from_fifo[i] ? fifo_fin_[i].front() : (bool)inf[i]) of = true;  // finish rides with the bit
        }
        for (int i = 0; i < d_; ++i)
            if (from_fifo[i]) { fifo_[i].pop_front(); fifo_fin_[i].pop_front(); }
        ov = true;
        next_pc = pc_ + 1;
        if (sat_pc_ >= 0 && next_pc > static_cast<uint32_t>(sat_pc_))
            next_pc = sat_pc_;   // saturate at the copy-last wait row
    }

    // --- buffer this round's arrivals, except those consumed by bypass ---
    for (int i = 0; i < d_; ++i) {
        if (!inv[i]) continue;
        if (ready && bypass[i]) continue;
        if (static_cast<int>(fifo_[i].size()) >= d_fifo_)
            log_error(name(), "output-sync FIFO overflow on line " + std::to_string(i));
        else
            { fifo_[i].push_back(ind[i]); fifo_fin_[i].push_back(inf[i] ? 1 : 0); }
    }

    // --- drive registered outputs ---
    out_det.write(out_d);
    out_used.write(out_u);
    out_valid.write(ov);
    out_finish.write(of);
    pc_ = next_pc;
    sync_regfile_pc.write(pc_);
}

}  // namespace emu
