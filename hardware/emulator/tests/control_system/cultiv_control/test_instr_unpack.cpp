// test_instr_unpack.cpp — unit test for InstrUnpack.
//
// WT_W=8, ADDR_W=16 -> instr_width = 40. Packs known {wt, start, end} triples
// into a word and checks they come back out, including the all-zero and
// all-ones (field-boundary) cases.
#include <systemc>

#include <cstdint>
#include <iostream>
#include <vector>

#include "control_system/cultiv_control/instr_unpack.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using emu::Bits;

static constexpr int WT_W = 8, ADDR_W = 16;
static const int INSTR_W = emu::InstrUnpack::instr_width(WT_W, ADDR_W);

// pack {wt, start, end} per the documented LSB-first layout
static Bits pack(uint32_t wt, uint32_t start, uint32_t end) {
    Bits b(INSTR_W);
    for (int i = 0; i < ADDR_W; ++i) b[i] = (end >> i) & 1u;
    for (int i = 0; i < ADDR_W; ++i) b[ADDR_W + i] = (start >> i) & 1u;
    for (int i = 0; i < WT_W;  ++i)  b[2 * ADDR_W + i] = (wt >> i) & 1u;
    return b;
}

int sc_main(int, char*[]) {
    sc_signal<Bits>     instr;
    sc_signal<uint32_t> wt, start_addr, end_addr;

    emu::InstrUnpack dut("dut", WT_W, ADDR_W);
    dut.instr(instr); dut.wt(wt); dut.start_addr(start_addr); dut.end_addr(end_addr);

    bool ok = true;
    auto chk = [&](uint32_t w, uint32_t s, uint32_t e, const char* tag) {
        instr.write(pack(w, s, e));
        sc_start(1, SC_NS);
        if (wt.read() != w || start_addr.read() != s || end_addr.read() != e) {
            ok = false;
            std::cout << "  MISMATCH " << tag << ": got wt=" << wt.read()
                      << " start=" << start_addr.read() << " end=" << end_addr.read()
                      << "  exp wt=" << w << " start=" << s << " end=" << e << "\n";
        } else {
            std::cout << "  ok " << tag << ": wt=" << w << " start=" << s << " end=" << e << "\n";
        }
    };

    std::cout << "instr_width=" << INSTR_W << " (WT_W=" << WT_W << " ADDR_W=" << ADDR_W << ")\n";
    chk(0, 0, 0, "all zero");
    chk(0, 5, 7, "wt=0 typical");
    chk(2, 20, 22, "wt>0 typical");
    chk(0, 10, 10, "start==end");
    chk(255, 65535, 65535, "all ones (field boundaries)");
    chk(1, 0, 65535, "min/max mix");
    chk(170, 43690, 21845, "alternating bit patterns");

    if (!ok) emu::log_error("test", "instr_unpack MISMATCH");
    else     emu::log_info("test", "instr_unpack PASS");

    int errors = sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL);
    return errors ? 1 : 0;
}
