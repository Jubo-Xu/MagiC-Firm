// regfile_rom.hpp — read-only regfile / ROM (the "BRAM" holding compiler masks).
//
// Generic memory primitive (not detector-construct-specific): address in, data
// out, contents loaded once at config time (like BRAM init) and read-only during
// operation. ASYNCHRONOUS read (data = mem[addr] combinationally) — this matches
// every datapath module, whose registered `pc` drives `addr` and consumes the
// word the same cycle. On FPGA that's distributed RAM / LUTRAM; a true
// (synchronous) BRAM would add a cycle and need the address prefetched.
//
// One instance per regfile: wire block.<pc> -> addr and data -> block.<mask>.
//
// Parameters:
//   depth  number of words
//   width  bits per word
#pragma once

#include <systemc>

#include <cstdint>
#include <string>
#include <vector>

#include "signals.hpp"

namespace emu {

SC_MODULE(RegFileROM) {
    sc_core::sc_in<uint32_t> addr;
    sc_core::sc_out<Bits>    data;

    RegFileROM(sc_core::sc_module_name nm, int depth, int width);

    // config-time init
    void load(const std::vector<Bits>& words);   // from parsed words
    void load_mem_file(const std::string& path,  // from a compiler .mem
                       const std::string& fmt);   // fmt = "hex" | "bin"

    std::size_t depth() const { return mem_.size(); }
    int         width() const { return width_; }

  private:
    void read();  // combinational: data = mem[addr]

    int               width_;
    std::vector<Bits> mem_;
};

}  // namespace emu
