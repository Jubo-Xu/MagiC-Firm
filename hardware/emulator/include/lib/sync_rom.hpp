// sync_rom.hpp — synchronous-read ROM (a true BRAM).
//
// Generic memory primitive, the SYNCHRONOUS sibling of RegFileROM:
//
//   RegFileROM  asynchronous read  data = mem[addr] combinationally
//               -> FPGA distributed RAM / LUTRAM; used for the small
//                  compiler regfiles whose registered pc drives addr and
//                  which are consumed the same cycle.
//   SyncROM     synchronous read   data = mem[addr] registered, 1 cycle later
//               -> FPGA block RAM; used for large memories (e.g. the command
//                  word memory), where a 1-cycle read latency is the norm.
//
// Protocol: assert `en` with `addr`; one cycle later `data` holds mem[addr] and
// `data_valid` is high. Back-to-back reads (en held high with a walking addr)
// produce a contiguous data stream with no bubble.
//
// NO RESET, deliberately — this models a BRAM output register, which typically
// has none. `data_valid` simply follows `en` by one cycle, so a consumer that
// stops driving `en` sees data_valid fall one cycle later on its own. Callers
// rely on this: resetting the output register would drop the word that was
// already fetched before the reset (see PhysicalMMIO, where that word is the
// final command of a round).
//
// `data` is ZEROED whenever `data_valid` is low, rather than holding the last
// word as a raw BRAM would. That keeps (data, data_valid) self-consistent so
// consumers never have to mask the bus themselves — one extra AND gate in RTL.
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

SC_MODULE(SyncROM) {
    sc_core::sc_in<bool>     clk;
    sc_core::sc_in<bool>     en;    // read enable
    sc_core::sc_in<uint32_t> addr;  // read address (sampled when en)

    sc_core::sc_out<Bits> data;        // mem[addr] one cycle after en (0 when !data_valid)
    sc_core::sc_out<bool> data_valid;  // en, delayed one cycle

    SyncROM(sc_core::sc_module_name nm, int depth, int width);

    // config-time init (same interface as RegFileROM)
    void load(const std::vector<Bits>& words);
    void load_mem_file(const std::string& path, const std::string& fmt);  // fmt = "hex" | "bin"

    std::size_t depth() const { return mem_.size(); }
    int         width() const { return width_; }

  private:
    void tick();  // clocked read (no reset — see header comment)

    int               width_;
    std::vector<Bits> mem_;
};

}  // namespace emu
