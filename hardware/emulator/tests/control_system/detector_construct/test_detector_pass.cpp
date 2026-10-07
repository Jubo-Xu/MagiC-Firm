// test_detector_pass.cpp — unit test for DetectorPass.
//
// d=4. Drives a few cycles and checks the whole (det, valid) bus comes out
// exactly one cycle later, unchanged.
#include <systemc>

#include <initializer_list>
#include <utility>
#include <vector>

#include "control_system/detector_construct/detector_pass.hpp"
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
    static constexpr int D = 4;

    sc_clock        clk;
    sc_signal<bool> rst;
    sc_signal<Bits> in_valid, in_det, in_finish, out_valid, out_det, out_finish;

    emu::DetectorPass dut;

    bool started_ = false;
    std::vector<std::pair<Bits, Bits>> expected, collected;   // (det, valid)

    void monitor() {
        if (started_) collected.emplace_back(out_det.read(), out_valid.read());
    }

    void stim() {
        rst.write(true); in_det.write(Bits(D)); in_valid.write(Bits(D)); in_finish.write(Bits(D)); wait();  // P1 reset
        rst.write(false); started_ = true;

        in_det.write(mk({1,0,1,1})); in_valid.write(mk({1,0,1,0})); wait();  // c1
        in_det.write(mk({0,1,0,1})); in_valid.write(mk({0,1,0,1})); wait();  // c2
        in_det.write(mk({1,1,0,0})); in_valid.write(mk({1,1,1,1})); wait();  // c3
        in_det.write(Bits(D));       in_valid.write(Bits(D));       wait();  // idle
        wait(); wait();

        // one cycle behind the input; first entry is the post-reset zero
        expected = {
            { mk({0,0,0,0}), mk({0,0,0,0}) },   // out after reset
            { mk({1,0,1,1}), mk({1,0,1,0}) },   // c1
            { mk({0,1,0,1}), mk({0,1,0,1}) },   // c2
            { mk({1,1,0,0}), mk({1,1,1,1}) },   // c3
            { mk({0,0,0,0}), mk({0,0,0,0}) },   // idle
        };

        bool ok = collected.size() >= expected.size();
        for (std::size_t i = 0; ok && i < expected.size(); ++i)
            ok = (collected[i] == expected[i]);

        std::cout << "collected " << collected.size() << " (checking first " << expected.size() << "):\n";
        for (std::size_t i = 0; i < expected.size() && i < collected.size(); ++i)
            std::cout << "  [" << i << "] det=" << collected[i].first
                      << " valid=" << collected[i].second << "\n";
        if (!ok) emu::log_error("test", "detector_pass MISMATCH");
        else     emu::log_info("test", "detector_pass PASS");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", D) {
        dut.clk(clk); dut.rst(rst);
        dut.in_valid(in_valid); dut.in_det(in_det); dut.in_finish(in_finish);
        dut.out_valid(out_valid); dut.out_det(out_det); dut.out_finish(out_finish);

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
