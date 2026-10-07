// test_regfile_rom.cpp — unit test for RegFileROM.
//
// Checks async read from load(vector), out-of-range -> zeros, and .mem file
// loading in both bin and hex.
#include <systemc>

#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#include "lib/regfile_rom.hpp"
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

int sc_main(int, char*[]) {
    static constexpr int DEPTH = 4, WIDTH = 6;
    sc_signal<uint32_t> addr;
    sc_signal<Bits>     data;
    emu::RegFileROM rf("rf", DEPTH, WIDTH);
    rf.addr(addr);
    rf.data(data);

    bool ok = true;
    // wiggle addr so the async read re-fires even when the target equals the
    // previous address (needed after reloading contents).
    auto chk = [&](uint32_t a, const Bits& exp, const char* tag) {
        addr.write(0xFFFFFFFFu); sc_start(1, SC_NS);
        addr.write(a);           sc_start(1, SC_NS);
        if (data.read() != exp) {
            ok = false;
            std::cout << "  MISMATCH @" << tag << " addr=" << a
                      << " got=" << data.read() << " exp=" << exp << "\n";
        }
    };

    // --- load(vector) + async read + out-of-range ---
    Bits w0 = mk({1,0,1,0,0,0}), w1 = mk({0,1,0,1,0,1}),
         w2 = mk({1,1,1,1,1,1}), w3 = mk({0,0,0,0,0,0});
    rf.load({w0, w1, w2, w3});
    chk(0, w0, "v0"); chk(1, w1, "v1"); chk(2, w2, "v2"); chk(3, w3, "v3");
    chk(5, Bits(WIDTH), "oob");                       // out of range -> zeros

    // --- load_mem_file (bin): distinct values 1,2,4,8 ---
    { std::ofstream f("regfile_rom_test.mem"); f << "000001\n000010\n000100\n001000\n"; }
    rf.load_mem_file("regfile_rom_test.mem", "bin");
    chk(0, mk({1,0,0,0,0,0}), "bin0"); chk(1, mk({0,1,0,0,0,0}), "bin1");
    chk(2, mk({0,0,1,0,0,0}), "bin2"); chk(3, mk({0,0,0,1,0,0}), "bin3");

    // --- load_mem_file (hex): value 42 = 0x2a ---
    { std::ofstream f("regfile_rom_hex.mem"); f << "2a\n"; }
    rf.load_mem_file("regfile_rom_hex.mem", "hex");
    chk(0, mk({0,1,0,1,0,1}), "hex0");                // 42 = bits 1,3,5

    std::cout << "regfile_rom: depth/width " << rf.depth() << "/" << rf.width() << "\n";
    if (!ok) emu::log_error("test", "regfile_rom MISMATCH");
    else     emu::log_info("test", "regfile_rom PASS");

    int errors = sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL);
    return errors ? 1 : 0;
}
