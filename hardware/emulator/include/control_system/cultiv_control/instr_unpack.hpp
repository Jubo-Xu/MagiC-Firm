// instr_unpack.hpp — instruction word field extraction for PhysicalMMIO.
//
// Pure combinational. Splits one packed instruction word (as read from the
// instruction regfile) into its fields. This is the SINGLE place that knows the
// instruction encoding: when instructions gain fields (branch select, round
// boundary flag, ...), only this module and the compiler's emitter change —
// PhysicalMMIO's wiring and InstrDecode stay untouched.
//
// LAYOUT (LSB-first, matching how the compiler packs kernel selectors):
//
//   bits [ADDR_W-1 : 0]                 end_addr    last CW address
//   bits [2*ADDR_W-1 : ADDR_W]          start_addr  first CW address
//   bits [2*ADDR_W+WT_W-1 : 2*ADDR_W]   wt          idle cycles before first read
//
//   instr_width = 2*ADDR_W + WT_W
//
// Parameters:
//   wt_w    width of the wt field
//   addr_w  width of start_addr / end_addr (they share it)
#pragma once

#include <systemc>

#include <cstdint>

#include "signals.hpp"

namespace emu {

SC_MODULE(InstrUnpack) {
    // --- input: one packed instruction word ---
    sc_core::sc_in<Bits> instr;   // [2*addr_w + wt_w - 1 : 0]

    // --- outputs: the decoded fields ---
    sc_core::sc_out<uint32_t> wt;
    sc_core::sc_out<uint32_t> start_addr;
    sc_core::sc_out<uint32_t> end_addr;

    InstrUnpack(sc_core::sc_module_name nm, int wt_w, int addr_w);

    // instruction word width for this parameterisation (sizes the regfile)
    static int instr_width(int wt_w, int addr_w) { return 2 * addr_w + wt_w; }

  private:
    void unpack();  // combinational

    const int wt_w_;
    const int addr_w_;
};

}  // namespace emu
