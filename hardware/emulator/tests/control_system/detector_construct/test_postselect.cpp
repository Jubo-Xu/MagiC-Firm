// test_postselect.cpp — unit test for Postselect.
//
// d=4. Postselect has NO internal pc now: the flat mask is presented one cycle
// AHEAD of the detector word it applies to (in the block it comes from the shared
// osync_pc, and Postselect registers it one cycle to align with OutputSync's
// registered det word). post_select is then a combinational OR-reduce.
//   mask M0 then det D0 (l1 set, M0 masks l0,l1) -> fire
//   mask M1(empty) then det D1(all ones)         -> no fire
//   mask M2(l3)   then det D2(l3=0)              -> no fire
//   mask M3(l2)   then det D3(l2=1)              -> fire
#include <systemc>

#include <initializer_list>
#include <vector>

#include "control_system/detector_construct/postselect.hpp"
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
    sc_signal<bool> rst, in_valid, post_select;
    sc_signal<Bits> in_det, postselect_mask;

    emu::Postselect dut;

    bool             started_ = false;
    std::vector<int> expected, collected;

    void monitor() {
        if (started_) collected.push_back(post_select.read() ? 1 : 0);
    }

    void stim() {
        rst.write(true); in_valid.write(false); in_det.write(Bits(D)); postselect_mask.write(Bits(D)); wait();
        rst.write(false); started_ = true;
        // mask leads the det word by one cycle (Postselect registers it internally to align)
        postselect_mask.write(mk({1,1,0,0})); in_det.write(Bits(D));       in_valid.write(false); wait();  // prime M0
        postselect_mask.write(mk({0,0,0,0})); in_det.write(mk({0,1,0,1})); in_valid.write(true);  wait();  // D0 & M0 -> fire
        postselect_mask.write(mk({0,0,0,1})); in_det.write(mk({1,1,1,1})); in_valid.write(true);  wait();  // D1 & M1(empty) -> no
        postselect_mask.write(mk({0,0,1,0})); in_det.write(mk({0,0,0,0})); in_valid.write(true);  wait();  // D2 & M2 -> no
        postselect_mask.write(Bits(D));       in_det.write(mk({0,0,1,0})); in_valid.write(true);  wait();  // D3 & M3 -> fire
        in_det.write(Bits(D)); in_valid.write(false);
        for (int i = 0; i < 10; ++i) wait();

        expected = {0, 1, 0, 0, 1, 0};   // [prime, D0, D1, D2, D3, idle]

        bool ok = collected.size() >= expected.size();
        for (std::size_t i = 0; ok && i < expected.size(); ++i)
            ok = (collected[i] == expected[i]);

        std::cout << "collected: ";
        for (int v : collected) std::cout << v;
        std::cout << "  (expected prefix ";
        for (int v : expected) std::cout << v;
        std::cout << ")\n";
        if (!ok) emu::log_error("test", "postselect sequence MISMATCH");
        else     emu::log_info("test", "postselect PASS");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", D) {
        dut.clk(clk); dut.rst(rst);
        dut.in_valid(in_valid); dut.in_det(in_det); dut.postselect_mask(postselect_mask);
        dut.post_select(post_select);

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
