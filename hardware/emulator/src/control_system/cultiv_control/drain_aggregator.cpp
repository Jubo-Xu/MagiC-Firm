// drain_aggregator.cpp — implementation of DrainAggregator.
#include "control_system/cultiv_control/drain_aggregator.hpp"

namespace emu {

DrainAggregator::DrainAggregator(sc_core::sc_module_name nm, int n)
    : sc_module(nm), n_(n), latched_("latched"), all_latched_q_("all_latched_q") {
    SC_HAS_PROCESS(DrainAggregator);

    SC_METHOD(comb);
    sensitive << latched_ << all_latched_q_;   // registered signals only (no combinational input)

    SC_METHOD(seq);
    sensitive << clk.pos() << rst;
    dont_initialize();
}

// all N sources have been LATCHED (registered), i.e. the join is complete.
// Bits signals default to width 0 at time 0, so treat any out-of-range bit as
// "not latched" — otherwise the unchecked operator[] segfaults.
bool DrainAggregator::all_latched() {
    const Bits l = latched_.read();
    for (int i = 0; i < n_; ++i)
        if (!((i < (int)l.size()) && l[i])) return false;
    return true;
}

void DrainAggregator::comb() {
    // rising edge of all_latched -> exactly one clean, edge-aligned cycle
    all_drain_done.write(all_latched() && !all_latched_q_.read());
}

void DrainAggregator::seq() {
    if (rst.read()) {
        latched_.write(Bits(n_));
        all_latched_q_.write(false);
        return;
    }
    if (!clk.posedge()) return;

    if (clear.read()) {
        latched_.write(Bits(n_));
        all_latched_q_.write(false);
        return;
    }

    // all_latched_q registers this cycle's all_latched (computed from the
    // pre-update latch), so it is the one-cycle-delayed all_latched.
    all_latched_q_.write(all_latched());

    // accumulate this cycle's pulses into the latch
    const Bits l = latched_.read();
    const Bits d = drain_done_in.read();
    Bits nl(n_);
    for (int i = 0; i < n_; ++i) {
        const bool lb = (i < (int)l.size()) && l[i];
        const bool db = (i < (int)d.size()) && d[i];
        nl[i] = (lb || db) ? 1 : 0;
    }
    latched_.write(nl);
}

}  // namespace emu
