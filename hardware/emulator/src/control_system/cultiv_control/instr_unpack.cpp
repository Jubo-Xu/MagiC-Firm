// instr_unpack.cpp — implementation of InstrUnpack (pure combinational).
#include "control_system/cultiv_control/instr_unpack.hpp"

namespace emu {

InstrUnpack::InstrUnpack(sc_core::sc_module_name nm, int wt_w, int addr_w)
    : sc_module(nm), wt_w_(wt_w), addr_w_(addr_w) {
    SC_HAS_PROCESS(InstrUnpack);
    SC_METHOD(unpack);
    sensitive << instr;
}

void InstrUnpack::unpack() {
    const Bits w = instr.read();
    end_addr.write(extract(w, 0, addr_w_));
    start_addr.write(extract(w, addr_w_, addr_w_));
    wt.write(extract(w, 2 * addr_w_, wt_w_));
}

}  // namespace emu
