// banked_ram.hpp — banked read/write RAM (the PGE's M_T column store).
//
// Generic memory primitive. `banks * depth` words of `width` bits, addressed by
// one index whose BIT-SLICES pick bank and offset:
//
//     bank   = index & (banks - 1)
//     offset = index >> log2(banks)
//
// valid only for a power-of-two `banks`. The PGE compiler enforces that, and
// its relabelling pi(c) = offset*B + bank is what turns the split into fixed
// wiring instead of a runtime address lookup.
//
// One read port and one write port PER BANK. That is exactly what the
// compiler's banking rule buys: for every fault, at most P=1 of its checks
// lands in any one bank, so all of a fault's columns are readable in one cycle.
//
// SYNCHRONOUS read (1-cycle latency): data appears the cycle AFTER the address
// is presented. Unlike RegFileROM's asynchronous read this is not a modelling
// preference — at ndet=640 the store is 640*640 bits = 50 KiB, which is BRAM on
// any FPGA, and BRAM registers its output.
//
// A bank whose rd_en is low drives ZERO, not its previous word, so a consumer's
// XOR tree can take all banks unconditionally — rd_en is the mask. In RTL that
// is one AND against the registered enable.
//
// Read-during-write to the same (bank, offset) returns the OLD word (BRAM
// READ_FIRST).
//
// Parameters:
//   banks  number of banks (power of two)
//   depth  words per bank
//   width  bits per word
#pragma once

#include <systemc>

#include <cstdint>
#include <vector>

#include "signals.hpp"

namespace emu {

SC_MODULE(BankedRAM) {
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;   // asynchronous, active high

    sc_core::sc_vector<sc_core::sc_in<bool>>     rd_en;    // [banks]
    sc_core::sc_vector<sc_core::sc_in<uint32_t>> rd_addr;  // [banks] offsets
    sc_core::sc_vector<sc_core::sc_out<Bits>>    rd_data;  // [banks] next cycle

    sc_core::sc_vector<sc_core::sc_in<bool>>     wr_en;    // [banks]
    sc_core::sc_vector<sc_core::sc_in<uint32_t>> wr_addr;  // [banks] offsets
    sc_core::sc_vector<sc_core::sc_in<Bits>>     wr_data;  // [banks]

    BankedRAM(sc_core::sc_module_name nm, int banks, int depth, int width);

    // Address decode, exposed so callers never open-code the bit-slice.
    int bank_of(int index)   const { return index & (banks_ - 1); }
    int offset_of(int index) const { return index >> bank_shift_; }

    int banks() const { return banks_; }
    int depth() const { return depth_; }
    int width() const { return width_; }

    // TESTS AND DEBUG ONLY — not a port, not a process, drives nothing and
    // costs no simulation time. This is the emulator's counterpart of the
    // hierarchical reference an RTL testbench uses (dut.g_bank[b].mem[off]);
    // it exists so a test can inspect the array without the design growing a
    // read path that synthesis would have to build. Never call it from a
    // module.
    const Bits& debug_word(int bank, int offset) const {
        return mem_[bank][offset];
    }

  private:
    void tick();

    int banks_, depth_, width_, bank_shift_;
    std::vector<std::vector<Bits>> mem_;   // [bank][offset]
};

}  // namespace emu
