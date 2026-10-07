// test_sync_rom.cpp — unit test for SyncROM (synchronous / BRAM-style read).
//
// Checks: 1-cycle read latency, back-to-back reads with no bubble, data_valid
// following en by one cycle, data HOLDING while en is low, out-of-range -> zeros,
// and .mem loading in bin and hex.
#include <systemc>

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "lib/sync_rom.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using emu::Bits;

static Bits val(uint32_t v, int width) {
    Bits b(width);
    for (int i = 0; i < width; ++i) b[i] = (v >> i) & 1u;
    return b;
}

SC_MODULE(Tb) {
    static constexpr int DEPTH = 8, WIDTH = 8;

    sc_clock            clk;
    sc_signal<bool>     en, data_valid;
    sc_signal<uint32_t> addr;
    sc_signal<Bits>     data;

    emu::SyncROM dut;

    bool started_ = false;
    std::vector<std::pair<Bits, bool>> collected;

    void monitor() {
        if (started_) collected.emplace_back(data.read(), data_valid.read());
    }

    // drive (en, addr) for one cycle
    void drive(bool e, uint32_t a) { en.write(e); addr.write(a); wait(); }

    void stim() {
        en.write(false); addr.write(0); wait();
        started_ = true;

        // --- load(vector): mem[i] = 3i+1 ---
        drive(false, 0);      // idle
        drive(true, 2);       // -> 7
        drive(true, 3);       // -> 10   back-to-back, no bubble
        drive(true, 4);       // -> 13
        drive(false, 0);      // -> data forced to 0, valid 0
        drive(true, 100);     // -> out of range: zeros, valid 1
        drive(false, 0);      // -> 0, valid 0

        // --- reload from a bin .mem: 1,2,4,8 ---
        dut.load_mem_file("sync_rom_test.mem", "bin");
        drive(true, 0);       // -> 1
        drive(true, 3);       // -> 8

        // --- reload from a hex .mem: 0x2a = 42 ---
        dut.load_mem_file("sync_rom_hex.mem", "hex");
        drive(true, 0);       // -> 42
        drive(false, 0);
        wait(); wait();

        const std::vector<std::pair<Bits, bool>> expected = {
            {val(0,  WIDTH), false},   // initial
            {val(0,  WIDTH), false},   // idle
            {val(7,  WIDTH), true},    // addr 2
            {val(10, WIDTH), true},    // addr 3
            {val(13, WIDTH), true},    // addr 4
            {val(0,  WIDTH), false},   // en low -> data forced to 0
            {val(0,  WIDTH), true},    // out of range -> zeros
            {val(0,  WIDTH), false},   // en low -> 0
            {val(1,  WIDTH), true},    // bin mem[0]
            {val(8,  WIDTH), true},    // bin mem[3]
            {val(42, WIDTH), true},    // hex mem[0]
            {val(0,  WIDTH), false},   // en low -> 0
        };

        bool ok = collected.size() >= expected.size();
        for (std::size_t i = 0; ok && i < expected.size(); ++i)
            ok = (collected[i].first == expected[i].first) &&
                 (collected[i].second == expected[i].second);

        std::cout << "collected " << collected.size()
                  << " (checking first " << expected.size() << "):\n";
        for (std::size_t i = 0; i < expected.size() && i < collected.size(); ++i)
            std::cout << "  [" << i << "] data=" << collected[i].first
                      << " valid=" << collected[i].second
                      << "   exp data=" << expected[i].first
                      << " valid=" << expected[i].second
                      << (collected[i].first == expected[i].first &&
                          collected[i].second == expected[i].second ? "" : "   <-- MISMATCH")
                      << "\n";

        if (!ok) emu::log_error("test", "sync_rom MISMATCH");
        else     emu::log_info("test", "sync_rom PASS");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", DEPTH, WIDTH) {
        dut.clk(clk); dut.en(en); dut.addr(addr);
        dut.data(data); dut.data_valid(data_valid);

        std::vector<Bits> words;
        for (int i = 0; i < DEPTH; ++i) words.push_back(val(3 * i + 1, WIDTH));
        dut.load(words);

        { std::ofstream f("sync_rom_test.mem"); f << "00000001\n00000010\n00000100\n00001000\n"; }
        { std::ofstream f("sync_rom_hex.mem");  f << "2a\n"; }

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
