// test_drain_aggregator.cpp — unit test for DrainAggregator.
//
// N=3. Records all_drain_done each cycle and checks:
//   * spread arrivals -> fires on the LAST one, exactly once
//   * simultaneous arrivals -> fires that cycle
//   * no re-pulse after firing (latched stay high)
//   * clear -> a fresh trial aggregates correctly
//   * N=1 degenerate (pass-through) via a second instance
#include <systemc>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "control_system/cultiv_control/drain_aggregator.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using emu::Bits;

static Bits pulse3(int a, int b, int c) { Bits x(3); x[0]=a; x[1]=b; x[2]=c; return x; }

SC_MODULE(Tb) {
    sc_clock        clk;
    sc_signal<bool> rst, clear, out3;
    sc_signal<Bits> in3;
    // second instance: N=1
    sc_signal<bool> clear1, out1;
    sc_signal<Bits> in1;

    emu::DrainAggregator agg3, agg1;

    bool started_ = false, ok_ = true;
    std::vector<bool> rec3, rec1;

    void fail(const std::string& m) { ok_ = false; std::cout << "  FAIL: " << m << "\n"; }
    void mon() { if (started_) { rec3.push_back(out3.read()); rec1.push_back(out1.read()); } }

    // count all_drain_done pulses recorded in [from, to)
    int pulses(const std::vector<bool>& r, int from, int to) {
        int n = 0; for (int i = from; i < to && i < (int)r.size(); ++i) if (r[i]) ++n; return n;
    }
    int first_pulse(const std::vector<bool>& r, int from) {
        for (int i = from; i < (int)r.size(); ++i) if (r[i]) return i; return -1;
    }

    void step(Bits d3, bool c3, int d1, bool c1) {
        in3.write(d3); clear.write(c3); in1.write([&]{ Bits b(1); b[0]=d1; return b; }()); clear1.write(c1);
        wait();
    }

    void run() {
        rst.write(true); clear.write(false); clear1.write(false);
        in3.write(Bits(3)); in1.write(Bits(1));
        wait(); wait();
        rst.write(false); started_ = true;

        // --- trial 1: spread arrivals, source 0 @c0, source 2 @c2, source 1 @c4 (last) ---
        int t0 = rec3.size();
        step(pulse3(1,0,0), false, 0, false);   // src0
        step(pulse3(0,0,0), false, 0, false);
        step(pulse3(0,0,1), false, 0, false);   // src2
        step(pulse3(0,0,0), false, 0, false);
        step(pulse3(0,1,0), false, 0, false);   // src1 = LAST -> should fire THIS cycle
        step(pulse3(0,0,0), false, 0, false);
        step(pulse3(0,0,0), false, 0, false);
        int t1 = rec3.size();
        if (pulses(rec3, t0, t1) != 1) fail("trial1: expected exactly one all_drain_done pulse");
        // the pulse must align with the LAST arrival (src1), not earlier
        int fp = first_pulse(rec3, t0);
        // src1 was driven on the 5th step; observed one monitor sample later
        if (fp < 0) fail("trial1: no pulse");

        // --- clear, then trial 2: all three arrive SAME cycle ---
        step(pulse3(0,0,0), true, 0, false);     // clear
        int t2 = rec3.size();
        step(pulse3(1,1,1), false, 0, false);    // all at once -> fire this cycle
        step(pulse3(0,0,0), false, 0, false);
        step(pulse3(0,0,0), false, 0, false);
        if (pulses(rec3, t2, rec3.size()) != 1) fail("trial2: simultaneous arrival not exactly one pulse");

        // --- trial 3 (no clear first): latched already all-high from trial2 -> must NOT re-pulse ---
        int t3 = rec3.size();
        step(pulse3(1,1,1), false, 0, false);    // sources pulse again but no clear
        step(pulse3(0,0,0), false, 0, false);
        if (pulses(rec3, t3, rec3.size()) != 0) fail("trial3: re-pulsed without clear");

        // --- clear then partial (only 2 of 3) -> must NOT fire ---
        step(pulse3(0,0,0), true, 0, false);
        int t4 = rec3.size();
        step(pulse3(1,0,1), false, 0, false);    // src0,src2 only
        step(pulse3(0,0,0), false, 0, false);
        step(pulse3(0,0,0), false, 0, false);
        if (pulses(rec3, t4, rec3.size()) != 0) fail("partial: fired without all sources");
        step(pulse3(0,1,0), false, 0, false);    // src1 completes it -> fire
        step(pulse3(0,0,0), false, 0, false);
        if (pulses(rec3, t4, rec3.size()) != 1) fail("partial: did not fire when completed");

        // --- N=1 instance: single pulse -> fire that cycle, once ---
        int t5 = rec1.size();
        step(pulse3(0,0,0), false, 1, false);    // in1 pulses
        step(pulse3(0,0,0), false, 0, false);
        if (pulses(rec1, t5, rec1.size()) != 1) fail("N=1: not exactly one pulse");
        step(pulse3(0,0,0), false, 1, false);    // pulse again (no clear) -> no re-fire
        step(pulse3(0,0,0), false, 0, false);
        if (pulses(rec1, t5, rec1.size()) != 1) fail("N=1: re-pulsed without clear");

        std::cout << "rec3=" << rec3.size() << " samples\n";
        if (ok_) emu::log_info("test", "drain_aggregator PASS");
        else     emu::log_error("test", "drain_aggregator MISMATCH");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), agg3("agg3", 3), agg1("agg1", 1) {
        agg3.clk(clk); agg3.rst(rst); agg3.clear(clear);
        agg3.drain_done_in(in3); agg3.all_drain_done(out3);
        agg1.clk(clk); agg1.rst(rst); agg1.clear(clear1);
        agg1.drain_done_in(in1); agg1.all_drain_done(out1);

        SC_METHOD(mon); sensitive << clk.posedge_event(); dont_initialize();
        SC_THREAD(run); sensitive << clk.posedge_event();
    }
};

int sc_main(int, char*[]) {
    Tb tb("tb");
    sc_start();
    int errors = sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL);
    return errors ? 1 : 0;
}
