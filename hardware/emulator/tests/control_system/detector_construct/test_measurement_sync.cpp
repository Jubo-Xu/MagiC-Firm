// test_measurement_sync.cpp — unit test for MeasurementSync.
//
// Drives one scenario (m=3) that exercises every path:
//   pc0: bypass (line0 arrives exactly when needed, FIFO empty)
//   pc1: buffer + FIFO (line1 arrived early during pc0, consumed from FIFO)
//   pc2: fast-forward (all-zero mask)
//   pc3: stall then mixed consume (line0/line1 from FIFO, line2 by bypass)
//
// The emitted (out_used, out_meas) sequence is path-independent (bypass changes
// latency, not values), so we score the ordered sequence against the expected.
#include <systemc>

#include <initializer_list>
#include <utility>
#include <vector>

#include "control_system/detector_construct/measurement_sync.hpp"
#include "signals.hpp"
#include "log.hpp"

using namespace sc_core;
using emu::Bits;

// Build Bits from {line0, line1, line2, ...} (index 0 = line 0 = bit 0).
static Bits mk(std::initializer_list<int> bits) {
    Bits b(bits.size());
    std::size_t i = 0;
    for (int x : bits) b[i++] = x ? 1 : 0;
    return b;
}

SC_MODULE(Tb) {
    static constexpr int M = 3;
    static constexpr int D_FIFO = 4;

    sc_clock                clk;
    sc_signal<bool>         rst;
    sc_signal<Bits>         in_meas, in_valid, in_finish, sync_mask;
    sc_signal<Bits>         out_meas, out_used;
    sc_signal<bool>         out_valid, out_finish;
    sc_signal<uint32_t>     pc;

    emu::MeasurementSync    dut;

    std::vector<Bits>                    program;    // sync mask per pc
    std::vector<std::pair<Bits, Bits>>   expected;   // (used, meas)
    std::vector<std::pair<Bits, Bits>>   collected;

    // The external sync regfile: combinational read of program[pc].
    void regfile() {
        uint32_t p = pc.read();
        sync_mask.write(p < program.size() ? program[p] : Bits(M, 1));  // out of range -> all-ones -> stall
    }

    // Collect one synced word per out_valid pulse (registered => reads previous edge).
    void monitor() {
        if (out_valid.read())
            collected.emplace_back(out_used.read(), out_meas.read());
    }

    void stim() {
        rst.write(true);  in_meas.write(Bits(M)); in_valid.write(Bits(M)); in_finish.write(Bits(M)); wait();  // reset
        rst.write(false);
        in_meas.write(mk({1, 1, 0})); in_valid.write(mk({1, 1, 0})); wait();  // cycle1: line0(need,bypass)+line1(buffer)
        in_meas.write(Bits(M));       in_valid.write(Bits(M));        wait();  // cycle2: pc1 consumes buffered line1
        wait();                                                                // cycle3: pc2 fast-forward
        in_meas.write(mk({0, 1, 0})); in_valid.write(mk({1, 1, 0})); wait();  // cycle4: pc3 stall (line2 missing), buffer line0,line1
        in_meas.write(mk({0, 0, 0})); in_valid.write(mk({0, 0, 1})); wait();  // cycle5: line2 arrives -> pc3 completes
        in_meas.write(Bits(M));       in_valid.write(Bits(M));
        for (int i = 0; i < 20; ++i) wait();

        // --- check ---
        bool ok = (collected.size() == expected.size());
        for (std::size_t i = 0; ok && i < expected.size(); ++i)
            ok = (collected[i].first == expected[i].first) &&
                 (collected[i].second == expected[i].second);

        std::cout << "collected " << collected.size() << " / expected " << expected.size() << "\n";
        for (std::size_t i = 0; i < collected.size(); ++i)
            std::cout << "  [" << i << "] used=" << collected[i].first
                      << " meas=" << collected[i].second << "\n";
        if (!ok) emu::log_error("test", "measurement_sync sequence MISMATCH");
        else     emu::log_info("test", "measurement_sync PASS");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", M, D_FIFO) {
        dut.clk(clk);   dut.rst(rst);
        dut.in_meas(in_meas); dut.in_valid(in_valid); dut.in_finish(in_finish); dut.sync_mask(sync_mask);
        dut.out_meas(out_meas); dut.out_used(out_used);
        dut.out_valid(out_valid); dut.out_finish(out_finish); dut.sync_regfile_pc(pc);

        program = { mk({1,0,0}), mk({0,1,0}), mk({0,0,0}), mk({1,1,1}) };
        expected = {
            { mk({1,0,0}), mk({1,0,0}) },   // pc0
            { mk({0,1,0}), mk({0,1,0}) },   // pc1
            { mk({0,0,0}), mk({0,0,0}) },   // pc2 fast-forward
            { mk({1,1,1}), mk({0,1,0}) },   // pc3
        };

        SC_METHOD(regfile); sensitive << pc;                     // runs at init -> program[0]
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
