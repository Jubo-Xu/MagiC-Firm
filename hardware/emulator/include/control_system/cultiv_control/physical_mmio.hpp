// physical_mmio.hpp — command-word generator for one physical control channel.
//
// One PhysicalMMIO per control core on a leaf board (P per board; a core may
// drive one qubit or a group — that is a user configuration choice). It turns
// the board's control events into a stream of command words for the physical
// control core, and reports when its command generation has finished.
//
// Pure assembly of four verified blocks:
//
//   ev_valid/ev_type/ev_payload
//         |
//         v
//   +-----------------+  instr_en ------------------------------+
//   | InstrSequencer  |  instr_addr --> InstrRegFile            |
//   |  IDLE/EXEC/DRAIN|                 (RegFileROM, async)     |
//   +-----------------+                      | instr[INSTR_W]   |
//         ^       | rst_out                  v                  |
//         |       |                    +------------+           |
//         |       |                    | InstrUnpack|           |
//         |       |                    +------------+           |
//         |       |                     wt/start/end            |
//         |       |                          |                  |
//         |       +--------------->  +----------------+ <-------+
//         |                          |  InstrDecode   |  (en)
//         +---- one_instr_end -------|                |
//                                    +----------------+
//                                      r_en | r_addr
//                                           v
//                                     +-----------+
//                                     | CwMemory  |  (SyncROM, 1-cycle read)
//                                     +-----------+
//                                           |
//                                           v
//                               out_data / out_valid    cw_gen_finish
//
// OUTPUT CONTRACT. out_valid alone carries "there is a command this cycle";
// when it is low, out_data is 0 and the control core does nothing. No explicit
// NOP command word is needed, so idle cycles cost no CW memory.
//
// The instruction memory is small and read combinationally at a registered
// address (RegFileROM = LUTRAM); the command-word memory is large and read with
// one cycle of latency (SyncROM = BRAM).
//
// NO COMBINATIONAL LOOP: one_instr_end feeds only InstrSequencer's *clocked*
// process, and instr_en/instr_addr come only from its registers, so the ring is
// broken at the flops.
#pragma once

#include <systemc>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "control_system/cultiv_control/instr_decode.hpp"
#include "control_system/cultiv_control/instr_sequencer.hpp"
#include "control_system/cultiv_control/instr_unpack.hpp"
#include "lib/regfile_rom.hpp"
#include "lib/sync_rom.hpp"
#include "signals.hpp"

namespace emu {

struct MmioConfig {
    int instr_depth = 4;    // instruction regfile depth (last entry = wait instruction)
    int cw_depth    = 16;   // command-word memory depth
    int data_w      = 8;    // command word width
    int wt_w        = 8;    // width of the instruction's wt field
    int payload_w   = 8;    // reserved ev_payload width

    // program data (optional; empty = leave the memories zero). Loaded by the board
    // loader from MMIO_instr_*/MMIO_cw_* so ControlBoard can load each core's MMIO,
    // the same way DcbConfig bundles its regfile data.
    std::vector<Bits> instr_words;  // instruction regfile words
    std::vector<Bits> cw_words;     // command-word memory words

    // derived — keeping these in one place stops the compiler's packing and the
    // hardware's decoding from drifting apart
    int addr_w()  const { return index_width(cw_depth); }
    int instr_w() const { return InstrUnpack::instr_width(wt_w, addr_w()); }
};

SC_MODULE(PhysicalMMIO) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- control event bus (from the board's control block) ---
    sc_core::sc_in<bool>     ev_valid;
    sc_core::sc_in<uint32_t> ev_type;     // InstrSequencer::EvType
    sc_core::sc_in<Bits>     ev_payload;  // RESERVED (JUMP); unused today

    // --- to the physical control core ---
    sc_core::sc_out<Bits> out_data;   // command word (0 when !out_valid)
    sc_core::sc_out<bool> out_valid;

    // --- to the board's control block ---
    sc_core::sc_out<bool> cw_gen_finish;  // 1-cycle pulse, aligned with the LAST out_valid

    PhysicalMMIO(sc_core::sc_module_name nm, const MmioConfig& cfg);

    // --- config-time memory init ---
    void load_instr(const std::vector<Bits>& words);
    void load_instr_mem(const std::string& path, const std::string& fmt);
    void load_cw(const std::vector<Bits>& words);
    void load_cw_mem(const std::string& path, const std::string& fmt);

    // pack one instruction per the InstrUnpack layout (also used by tests)
    static Bits pack_instr(uint32_t wt, uint32_t start, uint32_t end, int wt_w, int addr_w);
    Bits pack_instr(uint32_t wt, uint32_t start, uint32_t end) const {
        return pack_instr(wt, start, end, cfg_.wt_w, cfg_.addr_w());
    }

    const MmioConfig& config() const { return cfg_; }

  private:
    MmioConfig cfg_;

    std::unique_ptr<InstrSequencer> seq_;
    std::unique_ptr<RegFileROM>     instr_rom_;
    std::unique_ptr<InstrUnpack>    unpack_;
    std::unique_ptr<InstrDecode>    decode_;
    std::unique_ptr<SyncROM>        cw_rom_;

    sc_core::sc_signal<bool>     instr_en_, rst_out_, one_instr_end_, r_en_;
    sc_core::sc_signal<uint32_t> instr_addr_, wt_, start_addr_, end_addr_, r_addr_;
    sc_core::sc_signal<Bits>     instr_word_;
};

}  // namespace emu
