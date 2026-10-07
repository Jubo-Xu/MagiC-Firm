// test_output_sync.cpp — unit test for OutputSync.
//
// Same structure as the MeasurementSync test (identical logic), for d=3 detector
// lines. Exercises every path: bypass, buffer+FIFO, fast-forward, stall+mixed.
// Expected emitted (out_used, out_det) sequence: same as the sync test.
#include <systemc>

#include <initializer_list>
#include <utility>
#include <vector>

#include "control_system/detector_construct/output_sync.hpp"
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
    static constexpr int D = 3;
    static constexpr int D_FIFO = 4;

    sc_clock                clk;
    sc_signal<bool>         rst;
    sc_signal<Bits>         in_det, in_valid, in_finish, sync_mask;
    sc_signal<Bits>         out_det, out_used;
    sc_signal<bool>         out_valid, out_finish;
    sc_signal<uint32_t>     pc;

    emu::OutputSync         dut;

    std::vector<Bits>                    program;
    std::vector<std::pair<Bits, Bits>>   expected;   // (used, det)
    std::vector<std::pair<Bits, Bits>>   collected;

    void regfile() {
        uint32_t p = pc.read();
        sync_mask.write(p < program.size() ? program[p] : Bits(D, 1));  // out of range -> stall
    }

    void monitor() {
        if (out_valid.read())
            collected.emplace_back(out_used.read(), out_det.read());
    }

    void stim() {
        rst.write(true);  in_det.write(Bits(D)); in_valid.write(Bits(D)); in_finish.write(Bits(D)); wait();  // reset
        rst.write(false);
        in_det.write(mk({1, 1, 0})); in_valid.write(mk({1, 1, 0})); wait();  // line0(bypass)+line1(buffer)
        in_det.write(Bits(D));       in_valid.write(Bits(D));        wait();  // pc1 consumes buffered line1
        wait();                                                               // pc2 fast-forward
        in_det.write(mk({0, 1, 0})); in_valid.write(mk({1, 1, 0})); wait();  // pc3 stall (line2 missing)
        in_det.write(mk({0, 0, 0})); in_valid.write(mk({0, 0, 1})); wait();  // line2 arrives -> pc3 done
        in_det.write(Bits(D));       in_valid.write(Bits(D));
        for (int i = 0; i < 20; ++i) wait();

        bool ok = (collected.size() == expected.size());
        for (std::size_t i = 0; ok && i < expected.size(); ++i)
            ok = (collected[i].first == expected[i].first) &&
                 (collected[i].second == expected[i].second);

        std::cout << "collected " << collected.size() << " / expected " << expected.size() << "\n";
        for (std::size_t i = 0; i < collected.size(); ++i)
            std::cout << "  [" << i << "] used=" << collected[i].first
                      << " det=" << collected[i].second << "\n";
        if (!ok) emu::log_error("test", "output_sync sequence MISMATCH");
        else     emu::log_info("test", "output_sync PASS");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", D, D_FIFO) {
        dut.clk(clk);   dut.rst(rst);
        dut.in_det(in_det); dut.in_valid(in_valid); dut.in_finish(in_finish); dut.sync_mask(sync_mask);
        dut.out_det(out_det); dut.out_used(out_used);
        dut.out_valid(out_valid); dut.out_finish(out_finish); dut.sync_regfile_pc(pc);

        program = { mk({1,0,0}), mk({0,1,0}), mk({0,0,0}), mk({1,1,1}) };
        expected = {
            { mk({1,0,0}), mk({1,0,0}) },   // pc0
            { mk({0,1,0}), mk({0,1,0}) },   // pc1
            { mk({0,0,0}), mk({0,0,0}) },   // pc2 fast-forward
            { mk({1,1,1}), mk({0,1,0}) },   // pc3
        };

        SC_METHOD(regfile); sensitive << pc;
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
