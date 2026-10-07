// test_raw_selector.cpp — unit test for RawSelector.
//
// m=4, r=2. Selector: out0 <- line2, out1 <- line0 (indices {0, 2},
// index_width=2 => selector_indexes = {idx1=0, idx0=2}).
// Drives three input cycles + idle and checks the registered (1-cycle-delayed)
// selected outputs, confirming per-line valid rides along with the measurement.
#include <systemc>

#include <initializer_list>
#include <utility>
#include <vector>

#include "control_system/detector_construct/raw_selector.hpp"
#include "signals.hpp"
#include "log.hpp"

using namespace sc_core;
using emu::Bits;

static Bits mk(std::initializer_list<int> bits) {
    Bits b(bits.size());
    std::size_t i = 0;
    for (int x : bits) b[i++] = x ? 1 : 0;
    return b;
}

SC_MODULE(Tb) {
    static constexpr int M = 4, R = 2;

    sc_clock        clk;
    sc_signal<bool> rst;
    sc_signal<Bits> selector_indexes, in_meas, in_valid, in_finish, out_meas, out_valid;
    sc_signal<bool> out_finish;

    emu::RawSelector dut;

    bool started_ = false;
    std::vector<std::pair<Bits, Bits>> expected;   // (meas, valid)
    std::vector<std::pair<Bits, Bits>> collected;

    void monitor() {
        if (started_)
            collected.emplace_back(out_meas.read(), out_valid.read());
    }

    void stim() {
        // idx0 = 2 (bits [0,1]), idx1 = 0 (bits [0,0])  => mk({0,1, 0,0})
        selector_indexes.write(mk({0, 1, 0, 0}));

        rst.write(true); in_meas.write(Bits(M)); in_valid.write(Bits(M)); in_finish.write(Bits(M));
        wait();                        // P1 reset
        rst.write(false); started_ = true;

        // c1: line2=(1,valid1), line0=(1,valid1)
        in_meas.write(mk({1,0,1,1})); in_valid.write(mk({1,0,1,0})); wait();  // P2
        // c2: line2=(0,valid0), line0=(0,valid0)
        in_meas.write(mk({0,1,0,1})); in_valid.write(mk({0,1,0,1})); wait();  // P3
        // c3: line2=(0,valid1), line0=(1,valid1)
        in_meas.write(mk({1,0,0,0})); in_valid.write(mk({1,0,1,0})); wait();  // P4
        // idle
        in_meas.write(Bits(M)); in_valid.write(Bits(M)); wait();              // P5
        wait(); wait();

        // Expected registered stream, one cycle behind the input (first entry is
        // the post-reset zero the monitor sees before c1 propagates).
        expected = {
            { mk({0,0}), mk({0,0}) },   // out after reset
            { mk({1,1}), mk({1,1}) },   // sel(c1): line2=1/1, line0=1/1
            { mk({0,0}), mk({0,0}) },   // sel(c2): both invalid
            { mk({0,1}), mk({1,1}) },   // sel(c3): line2 meas0/valid1, line0 1/1
            { mk({0,0}), mk({0,0}) },   // sel(idle)
        };

        bool ok = collected.size() >= expected.size();
        for (std::size_t i = 0; ok && i < expected.size(); ++i)
            ok = (collected[i] == expected[i]);

        std::cout << "collected " << collected.size() << " (checking first " << expected.size() << "):\n";
        for (std::size_t i = 0; i < expected.size() && i < collected.size(); ++i)
            std::cout << "  [" << i << "] meas=" << collected[i].first
                      << " valid=" << collected[i].second
                      << "   expected meas=" << expected[i].first
                      << " valid=" << expected[i].second << "\n";
        if (!ok) emu::log_error("test", "raw_selector MISMATCH");
        else     emu::log_info("test", "raw_selector PASS");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", R, M) {
        dut.clk(clk); dut.rst(rst);
        dut.selector_indexes(selector_indexes);
        dut.in_meas(in_meas); dut.in_valid(in_valid); dut.in_finish(in_finish);
        dut.out_meas(out_meas); dut.out_valid(out_valid); dut.out_finish(out_finish);

        SC_METHOD(monitor); sensitive << clk.posedge_event(); dont_initialize();
        SC_THREAD(stim);    sensitive << clk.posedge_event();
    }
};

int sc_main(int, char*[]) {
    Tb tb("tb");
    sc_start();
    int errors = sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL);
    return errors ? 1 : 0;
}
