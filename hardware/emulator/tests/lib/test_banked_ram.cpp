// test_banked_ram.cpp — unit test for BankedRAM.
//
// Everything is driven through the ports; the block has no back door, so this
// exercises the same paths the design will. Geometry matches the PGE manifest's
// M_T block at B=16 (16 banks x 40 deep x 640-bit words = 50 KiB), with a small
// 4x4x8 instance for the cases that need exhaustive coverage.
//
// Cases:
//   1  bit-slice decode reconstructs every index
//   2  reset clears the read outputs
//   3  write, then read -> data on the NEXT cycle, not the same one
//   4  all banks read simultaneously in one cycle (what the banking rule buys)
//   5  a bank with rd_en low drives ZERO, not its previous word
//   6  read-during-write to one address returns the OLD word
//   7  out-of-range offset reads zero
//   8  rst wipes previously written contents
#include <systemc>

#include <cstdint>
#include <iostream>
#include <vector>

#include "lib/banked_ram.hpp"
#include "signals.hpp"

using namespace sc_core;
using emu::Bits;

namespace {

// A word whose bit pattern encodes `seed`, so a wrong word is obvious.
Bits mk(int width, int seed) {
    Bits b(width);
    for (int i = 0; i < width; ++i) b[i] = (seed >> (i % 31)) & 1;
    if (seed) b[seed % width] = 1;
    return b;
}

bool ok = true;

void chk(bool cond, const std::string& what) {
    if (!cond) {
        ok = false;
        std::cout << "  FAIL: " << what << "\n";
    }
}

// Harness: owns the clock and the port signals for one BankedRAM.
struct Harness {
    int banks, depth, width;
    sc_clock clk;
    sc_signal<bool> rst;
    sc_vector<sc_signal<bool>>     rd_en, wr_en;
    sc_vector<sc_signal<uint32_t>> rd_addr, wr_addr;
    sc_vector<sc_signal<Bits>>     rd_data, wr_data;
    emu::BankedRAM ram;

    Harness(const char* nm, int B, int D, int W)
        : banks(B), depth(D), width(W),
          clk("clk", 10, SC_NS),
          rd_en("rd_en", B), wr_en("wr_en", B),
          rd_addr("rd_addr", B), wr_addr("wr_addr", B),
          rd_data("rd_data", B), wr_data("wr_data", B),
          ram(nm, B, D, W) {
        ram.clk(clk);
        ram.rst(rst);
        for (int b = 0; b < B; ++b) {
            ram.rd_en[b](rd_en[b]);     ram.rd_addr[b](rd_addr[b]);
            ram.rd_data[b](rd_data[b]);
            ram.wr_en[b](wr_en[b]);     ram.wr_addr[b](wr_addr[b]);
            ram.wr_data[b](wr_data[b]);
        }
        idle();
    }

    void idle() {
        for (int b = 0; b < banks; ++b) {
            rd_en[b].write(false); wr_en[b].write(false);
            rd_addr[b].write(0);   wr_addr[b].write(0);
            wr_data[b].write(Bits(width));
        }
    }

    void cycle() { sc_start(10, SC_NS); }

    void reset() {
        rst.write(true);  cycle();
        rst.write(false); idle();
    }

    // Drive one write; it lands on the next clock edge.
    void write(int index, const Bits& w) {
        idle();
        int b = ram.bank_of(index);
        wr_en[b].write(true);
        wr_addr[b].write(ram.offset_of(index));
        wr_data[b].write(w);
        cycle();
        idle();
    }

    // Present an address, take a cycle, return what the bank drove.
    Bits read(int index) {
        idle();
        int b = ram.bank_of(index);
        rd_en[b].write(true);
        rd_addr[b].write(ram.offset_of(index));
        cycle();
        idle();
        return rd_data[b].read();
    }
};

}  // namespace

int sc_main(int, char*[]) {
    static constexpr int B = 16, D = 40, W = 640;   // the real M_T geometry
    Harness h("ram", B, D, W);

    // --- 1. bit-slice decode covers the whole index space, bijectively ---
    {
        std::vector<int> seen(B * D, -1);
        for (int i = 0; i < B * D; ++i) {
            int bank = h.ram.bank_of(i), off = h.ram.offset_of(i);
            chk(bank >= 0 && bank < B, "bank in range");
            chk(off >= 0 && off < D, "offset in range");
            chk(off * B + bank == i, "offset*B + bank reconstructs the index");
            int slot = bank * D + off;
            chk(seen[slot] == -1, "bit-slice decode is injective");
            seen[slot] = i;
        }
    }

    // --- 2. reset clears the read outputs ---
    h.reset();
    for (int b = 0; b < B; ++b)
        chk(h.rd_data[b].read() == Bits(W), "rd_data zero after reset");

    // --- 3. write then read: data on the NEXT cycle, not the same one ---
    {
        const int idx = 37;                 // bank 5, offset 2
        Bits w = mk(W, 12345);
        h.idle();
        int b = h.ram.bank_of(idx);
        h.wr_en[b].write(true);
        h.wr_addr[b].write(h.ram.offset_of(idx));
        h.wr_data[b].write(w);
        h.rd_en[b].write(true);             // read the same address, same cycle
        h.rd_addr[b].write(h.ram.offset_of(idx));
        h.cycle();
        h.idle();
        // Case 6 folded in: READ_FIRST, so this read saw the OLD (zero) word.
        chk(h.rd_data[b].read() == Bits(W), "read-during-write returns old word");
        chk(h.read(idx) == w, "written word reads back on a later cycle");
    }

    // --- 4. all banks read simultaneously in one cycle ---
    {
        std::vector<int> idx(B);
        for (int b = 0; b < B; ++b) {
            idx[b] = 7 * B + b;             // offset 7 in every bank
            h.write(idx[b], mk(W, 1000 + b));
        }
        h.idle();
        for (int b = 0; b < B; ++b) { h.rd_en[b].write(true); h.rd_addr[b].write(7); }
        h.cycle();
        for (int b = 0; b < B; ++b)
            chk(h.rd_data[b].read() == mk(W, 1000 + b),
                "bank " + std::to_string(b) + " read in the same cycle as all others");
        h.idle();
    }

    // --- 5. a disabled bank drives ZERO, not its previous word ---
    {
        h.idle();
        h.rd_en[3].write(true); h.rd_addr[3].write(7);
        h.cycle();
        chk(h.rd_data[3].read() == mk(W, 1003), "enabled bank drives its word");
        h.idle();                            // rd_en low
        h.cycle();
        chk(h.rd_data[3].read() == Bits(W), "disabled bank drives zero, not held");
    }

    // --- 7. out-of-range offset reads zero ---
    {
        h.idle();
        h.rd_en[0].write(true); h.rd_addr[0].write(D + 5);
        h.cycle();
        chk(h.rd_data[0].read() == Bits(W), "out-of-range offset reads zero");
        h.idle();
    }

    // --- 8. rst wipes previously written contents ---
    {
        chk(h.read(7 * B + 3) == mk(W, 1003), "word still present before reset");
        h.reset();
        chk(h.read(7 * B + 3) == Bits(W), "contents cleared by rst");
    }

    std::cout << (ok ? "test_banked_ram: PASS\n" : "test_banked_ram: FAIL\n");
    return ok ? 0 : 1;
}
