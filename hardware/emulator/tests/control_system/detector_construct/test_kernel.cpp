// test_kernel.cpp — unit test for Kernel (one channel).
//
// m=4, n=2, h=2. Selector picks measurement lines {1, 3} onto selector slots
// {0, 1}. Two cores alternate emission:
//   core0: detector A = line1@t0 ^ line1@t1  (emit t1)
//          detector C = line1@t2 ^ line1@t3  (emit t3)
//   core1: detector B = line3@t0 ^ line3@t2  (emit t2)
// Measurements: line1 = 1,0,1,1 (t0..t3); line3 = 1,_,0,_ .
//   A = 1^0 = 1 ; B = 1^0 = 1 ; C = 1^1 = 0.
// A stall (in_valid=0) is inserted after t0 to check pc/reg hold. Expected
// emitted detector stream (on out_valid): [1, 1, 0].
#include <systemc>

#include <initializer_list>
#include <vector>

#include "control_system/detector_construct/kernel.hpp"
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
    static constexpr int M = 4, N = 2, H = 2;
    static constexpr int CW = N + 1;      // core word width (n+1)

    sc_clock            clk;
    sc_signal<bool>     rst, in_valid, in_finish, out_valid, out_det, out_finish;
    sc_signal<Bits>     in_used, in_meas, selector_indexes, core_mask;
    sc_signal<uint32_t> core_regfile_pc;

    emu::Kernel dut;

    std::vector<Bits>    prog;        // core regfile: mask per meas-time
    std::vector<uint8_t> expected;    // emitted detector stream
    std::vector<uint8_t> collected;

    // external core regfile: combinational read of prog[pc]
    void regfile() {
        uint32_t p = core_regfile_pc.read();
        core_mask.write(p < prog.size() ? prog[p] : Bits(H * CW));  // beyond -> all-zero (no emit)
    }

    void monitor() {
        if (out_valid.read())
            collected.push_back(out_det.read() ? 1 : 0);
    }

    void stim() {
        // selector_indexes: slot0 -> line1 (idx 1), slot1 -> line3 (idx 3).
        // index_width = 2, so {index1=11, index0=01} = bits [1,0, 1,1].
        selector_indexes.write(mk({1, 0, 1, 1}));

        rst.write(true); in_valid.write(false); in_finish.write(false);
        in_used.write(Bits(M)); in_meas.write(Bits(M));
        wait();                                     // reset
        rst.write(false);

        // t0: line1=1, line3=1 ; no emit
        in_valid.write(true); in_used.write(mk({0,1,0,1})); in_meas.write(mk({0,1,0,1})); wait();
        // stall: in_valid=0 -> pc and regs must hold
        in_valid.write(false); in_used.write(Bits(M)); in_meas.write(Bits(M)); wait();
        // t1: line1=0 ; core0 emits A=1
        in_valid.write(true); in_used.write(mk({0,1,0,0})); in_meas.write(mk({0,0,0,0})); wait();
        // t2: line1=1, line3=0 ; core1 emits B=1
        in_valid.write(true); in_used.write(mk({0,1,0,1})); in_meas.write(mk({0,1,0,0})); wait();
        // t3: line1=1 ; core0 emits C=0
        in_valid.write(true); in_used.write(mk({0,1,0,0})); in_meas.write(mk({0,1,0,0})); wait();
        // idle
        in_valid.write(false); in_used.write(Bits(M)); in_meas.write(Bits(M));
        for (int i = 0; i < 10; ++i) wait();

        bool ok = (collected == expected);
        std::cout << "collected " << collected.size() << " / expected " << expected.size() << ":  ";
        for (uint8_t d : collected) std::cout << int(d);
        std::cout << "  (expected ";
        for (uint8_t d : expected) std::cout << int(d);
        std::cout << ")\n";
        if (!ok) emu::log_error("test", "kernel detector stream MISMATCH");
        else     emu::log_info("test", "kernel PASS");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", N, H, M) {
        dut.clk(clk); dut.rst(rst);
        dut.in_valid(in_valid); dut.in_used(in_used); dut.in_meas(in_meas); dut.in_finish(in_finish);
        dut.selector_indexes(selector_indexes); dut.core_mask(core_mask);
        dut.out_valid(out_valid); dut.out_det(out_det); dut.out_finish(out_finish);
        dut.core_regfile_pc(core_regfile_pc);

        // core regfile: within a core, bits[0..n-1]=select, bit[n]=emit; core c at c*(n+1).
        prog = {
            mk({1,0,0, 0,1,0}),   // t0: core0 sel=01 emit=0 ; core1 sel=10 emit=0
            mk({1,0,1, 0,0,0}),   // t1: core0 sel=01 emit=1 ; core1 idle
            mk({1,0,0, 0,1,1}),   // t2: core0 sel=01 emit=0 ; core1 sel=10 emit=1
            mk({1,0,1, 0,0,0}),   // t3: core0 sel=01 emit=1 ; core1 idle
        };
        expected = {1, 1, 0};   // A, B, C

        SC_METHOD(regfile); sensitive << core_regfile_pc;              // runs at init -> prog[0]
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
