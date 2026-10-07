// stim_readout.hpp — measurement readout model for whole-system stim testing.
//
// Closes the leaf loop for a stim whole-system test: it consumes a leaf board's
// PhysicalMMIO command stream and replays recorded stim measurements back into that
// board's ControlBoard (in_meas / in_valid / in_meas_finish). It is a REAL,
// synthesizable module (compiled onto the FPGA) that STANDS IN for the qubits +
// readout during a hardware-in-the-loop test — test/sim infrastructure, but hardware.
// Driven by the GLOBAL rst (and cw_gen_finish), never by the board's internal reset.
//
// The MMIO drives one command per measurement round; its command word (mmio_out_data)
// is the ROUND INDEX (from the compiler's sim CW memory). The recorded measurements
// live in a per-board BRAM (SyncROM) laid out shot-major (shots * DEPTH rows, DEPTH =
// N or N+1 with copy-last) — large multi-shot data, so a 1-cycle read latency is the
// norm; the module is synchronous around it. The latency is harmless: once the loop is
// closed the whole datapath is pipelined, so it just shifts the pipeline.
//
// SHOT ADVANCE (on ABORT). An abort — the leaf's ev_valid & ev_type==EV_ABORT, the same
// event that restarts the CW program in BoardControl/PhysicalMMIO — DISCARDS the current
// shot: jump base_reg to the NEXT recorded shot's first row and bump shot_reg. This is the
// only thing the abort does; it is DECOUPLED from the read. The abort may arrive several
// cycles BEFORE the MMIO's first command (a leaf used in a later cultivation stage idles a
// large `wt` before its first read), so it only moves the pointer and waits there; the
// actual read still happens on each mmio_out_valid at base_reg + index. Advancing on the
// (global, in-sync) abort — instead of on the MMIO's wt-delayed index==0 — keeps every
// leaf's readout jumping to the next shot on the SAME cycle, in lockstep.
//
//   reset (rst | cw_gen_finish) : shot_reg <= 0, base_reg <= 0
//     (cw_gen_finish = trial done -> reset so the next trial replays from shot 0)
//   on abort (guarded, only while shot_reg+1 < SHOTS):
//     shot_reg <= shot_reg+1, base_reg <= (shot_reg+1)*DEPTH
//   on mmio_out_valid : addr = base_reg + index   (the read; data lands one cycle later)
//   one cycle later: out_meas / out_meas_valid = mem[addr] (SyncROM zeroes them when the
//     read wasn't enabled); out_meas_finish = (cw_gen_finish at the read cycle)
//     ? out_meas_valid : 0.
#pragma once

#include <systemc>

#include <cstdint>
#include <memory>
#include <vector>

#include "lib/sync_rom.hpp"
#include "signals.hpp"

namespace emu {

struct StimReadoutConfig {
    int m = 0;      // measurement width (board ports)
    int shots = 0;  // number of recorded shots in the memory
    int depth = 0;  // rows per shot (N, or N+1 with copy-last)
    std::vector<Bits> meas;   // shots*depth words, width m  (sim_readout_meas)
    std::vector<Bits> valid;  // shots*depth words, width m  (sim_readout_valid)
};

SC_MODULE(StimReadout) {
    // --- clock / GLOBAL reset (not board_reset) ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- from the leaf board's PhysicalMMIO (via ControlBoard mmio_* outputs) ---
    sc_core::sc_in<bool> mmio_out_valid;
    sc_core::sc_in<Bits> mmio_out_data;   // command word = round index
    sc_core::sc_in<bool> cw_gen_finish;   // trial done (aligned with the last command)

    // --- the leaf's event bus (the SAME ev that feeds BoardControl + PhysicalMMIO): an
    //     EV_ABORT advances the readout to the next shot, in sync across all leaves ---
    sc_core::sc_in<bool>     ev_valid;
    sc_core::sc_in<uint32_t> ev_type;

    // --- to the leaf board's ControlBoard measurement inputs (one cycle after the read) ---
    sc_core::sc_out<Bits> out_meas;         // [m]
    sc_core::sc_out<Bits> out_meas_valid;   // [m]
    sc_core::sc_out<Bits> out_meas_finish;  // [m] = valid on the finish round, else 0

    // --- observability for the testbench ---
    sc_core::sc_out<uint32_t> shot_reg;     // shots started so far (== SHOTS while the last is active)

    StimReadout(sc_core::sc_module_name nm, const StimReadoutConfig& cfg);

  private:
    static constexpr uint32_t EV_ABORT = 1;   // matches BoardControl::EV_ABORT

    void comb_addr();   // combinational: read enable + address into the BRAMs
    void seq();         // registered: shot_reg / base_reg / finish pipeline
    void drive_out();   // outputs from the (1-cycle-late) BRAM data + finish pipe

    StimReadoutConfig cfg_;
    std::unique_ptr<SyncROM> meas_rom_, valid_rom_;

    sc_core::sc_signal<uint32_t> shot_reg_, base_reg_, addr_;
    sc_core::sc_signal<bool>     en_, finish_q_;
    sc_core::sc_signal<Bits>     meas_data_, valid_data_;
    sc_core::sc_signal<bool>     meas_dv_, valid_dv_;
};

}  // namespace emu
