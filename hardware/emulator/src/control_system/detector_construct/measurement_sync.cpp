// measurement_sync.cpp — implementation of MeasurementSync.
//
// Modelling notes (see header for the block's intent):
//  * The FIFOs are plain C++ members, so head-reads and pointer state are always
//    "current" within a clock edge — this is the combinational-read assumption
//    (no 1-cycle read latency on the FIFO/regfile).
//  * A needed line's value is taken from its FIFO head if the FIFO is non-empty
//    (the oldest buffered measurement = the one for this meas-time), OTHERWISE,
//    if a measurement is arriving this very cycle, straight from the input
//    (BYPASS). Bypass avoids the extra FIFO round-trip so a just-in-time
//    measurement reaches the (registered) output in one cycle, not two.
//  * A bypassed arrival is consumed, not buffered — only NON-consumed arrivals
//    are pushed. FIFO-head consumption still pushes any co-arriving measurement
//    (it belongs to a later meas-time), preserving per-line order.
//  * Readiness = every needed line is available (buffered OR arriving). An
//    all-zero mask has no needed lines, so it is trivially ready: it emits an
//    all-zero word and advances pc — the "fast-forward" case falls out for free.
//  * Outputs are registered: written at a clock edge, visible next cycle, so
//    out_valid is a one-cycle pulse per accepted step.
#include "control_system/detector_construct/measurement_sync.hpp"

#include <string>
#include <vector>

#include "log.hpp"

namespace emu {

MeasurementSync::MeasurementSync(sc_core::sc_module_name nm, int m, int d_fifo, int sat_pc)
    : sc_module(nm), m_(m), d_fifo_(d_fifo), sat_pc_(sat_pc), fifo_(m), fifo_fin_(m) {
    SC_HAS_PROCESS(MeasurementSync);
    SC_METHOD(tick);
    sensitive << clk.pos() << rst;  // clk edge for steps; rst change for async reset
    dont_initialize();
}

void MeasurementSync::tick() {
    // --- asynchronous reset: clear pointer, state machine, and all FIFOs ---
    if (rst.read()) {
        pc_ = 0;
        for (auto& f : fifo_) f.clear();
        for (auto& f : fifo_fin_) f.clear();
        out_meas.write(Bits(m_));
        out_used.write(Bits(m_));
        out_valid.write(false);
        out_finish.write(false);
        sync_regfile_pc.write(0);
        return;
    }
    // Triggered by rst deasserting (not a clock edge): nothing to do.
    if (!clk.posedge()) return;

    // --- sample inputs ---
    const Bits inm  = in_meas.read();
    const Bits inv  = in_valid.read();
    const Bits inf  = in_finish.read();
    const Bits mask = sync_mask.read();
    sc_assert(inm.size() == static_cast<std::size_t>(m_) &&
              inv.size() == static_cast<std::size_t>(m_) &&
              mask.size() == static_cast<std::size_t>(m_));

    // --- resolve each needed line's source; decide readiness ---
    std::vector<char> from_fifo(m_, 0);  // consume the FIFO head
    std::vector<char> bypass(m_, 0);      // consume straight from the input
    bool ready = true;
    for (int i = 0; i < m_; ++i) {
        if (!mask[i]) continue;                 // line not needed this meas-time
        if (!fifo_[i].empty())      from_fifo[i] = 1;
        else if (inv[i])            bypass[i]    = 1;
        else                        ready = false;  // needed but unavailable -> stall
    }

    Bits     out_m(m_), out_u(m_);
    bool     ov      = false;
    bool     of      = false;      // finish of the emitted meas-time (OR over used lines)
    uint32_t next_pc = pc_;

    if (ready) {
        for (int i = 0; i < m_; ++i) {
            if (!mask[i]) continue;
            out_u[i] = 1;
            out_m[i] = from_fifo[i] ? fifo_[i].front() : inm[i];               // fifo head or bypass
            if (from_fifo[i] ? fifo_fin_[i].front() : (bool)inf[i]) of = true;  // finish rides with the bit
        }
        for (int i = 0; i < m_; ++i)
            if (from_fifo[i]) { fifo_[i].pop_front(); fifo_fin_[i].pop_front(); }
        ov = true;
        next_pc = pc_ + 1;
        if (sat_pc_ >= 0 && next_pc > static_cast<uint32_t>(sat_pc_))
            next_pc = sat_pc_;   // saturate at the copy-last wait row
    }
    // else: stall — outputs idle, pc holds, but arrivals are still buffered below.

    // --- buffer this cycle's arrivals, except those consumed by bypass ---
    for (int i = 0; i < m_; ++i) {
        if (!inv[i]) continue;
        if (ready && bypass[i]) continue;  // consumed directly this cycle
        if (static_cast<int>(fifo_[i].size()) >= d_fifo_)
            log_error(name(), "sync FIFO overflow on line " + std::to_string(i));
        else
            { fifo_[i].push_back(inm[i]); fifo_fin_[i].push_back(inf[i] ? 1 : 0); }
    }

    // --- drive registered outputs ---
    out_meas.write(out_m);
    out_used.write(out_u);
    out_valid.write(ov);
    out_finish.write(of);
    pc_ = next_pc;
    sync_regfile_pc.write(pc_);
}

}  // namespace emu
