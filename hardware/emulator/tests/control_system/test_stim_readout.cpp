// test_stim_readout.cpp — unit test for the sim-only StimReadout model.
//
// Hand-built BRAM: SHOTS=2, DEPTH=3, m=3, meas[addr]=addr, valid[addr]=all-ones.
// Drive the MMIO command stream (round index + cw_gen_finish) and the leaf event bus
// (ev_valid/ev_type) and check the 1-cycle-delayed measurement outputs + shot_reg. Covers:
// read address = base + index, ABORT advancing to the next shot's base (decoupled from the
// read), the last-shot advance guard, cw_gen_finish (finish bit + reset), and idle.
#include <systemc>

#include <iostream>

#include "control_system/stim_readout.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using namespace emu;

static constexpr uint32_t EV_ABORT = 1;   // matches BoardControl::EV_ABORT

static Bits mk(uint32_t v, int w) {
    Bits b(w);
    for (int i = 0; i < w; ++i) b[i] = (v >> i) & 1;
    return b;
}
static uint32_t val(const Bits& b) { return extract(b, 0, b.size()); }

SC_MODULE(Tb) {
    sc_clock        clk;
    sc_signal<bool> rst, mmio_v, cw_fin, ev_v;
    sc_signal<uint32_t> ev_t;
    sc_signal<Bits> mmio_data, o_meas, o_valid, o_finish;
    sc_signal<uint32_t> shot;
    std::unique_ptr<StimReadout> dut;
    int errors = 0;

    void chk(bool ok, const std::string& tag) {
        if (!ok) { ++errors; std::cout << "  MISMATCH: " << tag << "\n"; }
    }

    // Outputs (registered + BRAM read) are observed one step after the read/abort is
    // driven, so we DEFER each step's expectation and check it on the next call.
    struct Exp { int meas, valid, finish; uint32_t shot; std::string tag; };
    Exp  pend; bool have = false;

    void check_pending() {
        if (!have) return;
        chk((int)val(o_meas.read())   == pend.meas,   pend.tag + " meas");
        chk((int)val(o_valid.read())  == pend.valid,  pend.tag + " valid");
        chk((int)val(o_finish.read()) == pend.finish, pend.tag + " finish");
        chk(shot.read() == pend.shot,                 pend.tag + " shot_reg");
    }

    // drive one cycle (read idx + finish, and/or an abort); expected = what it produces:
    // meas[base+idx], valid, finish bit, and shot_reg AFTER this cycle's update.
    void step(bool v, uint32_t idx, bool abort, bool fin,
              int exp_meas, int exp_valid, int exp_finish, uint32_t exp_shot,
              const std::string& tag) {
        mmio_v.write(v); mmio_data.write(mk(idx, 2)); cw_fin.write(fin);
        ev_v.write(abort); ev_t.write(abort ? EV_ABORT : 0u);
        wait();
        check_pending();
        pend = {exp_meas, exp_valid, exp_finish, exp_shot, tag}; have = true;
    }

    void run() {
        rst.write(true); mmio_v.write(false); cw_fin.write(false); ev_v.write(false);
        mmio_data.write(mk(0, 2)); ev_t.write(0);
        wait(); wait();
        rst.write(false);

        // shot 0 (base 0): read rounds 0,1,2 -> addr base+idx = 0,1,2. No abort -> shot_reg holds 0.
        step(true, 0, false, false, 0, 7, 0, 0, "s0 r0");
        step(true, 1, false, false, 1, 7, 0, 0, "s0 r1");
        step(true, 2, false, false, 2, 7, 0, 0, "s0 r2");
        // ABORT (no read): jump to shot 1's base (3), shot_reg 0->1.
        step(false, 0, true, false, 0, 0, 0, 1, "abort->s1");
        // shot 1 (base 3): read rounds 0,1,2 -> addr 3,4,5.
        step(true, 0, false, false, 3, 7, 0, 1, "s1 r0");
        step(true, 1, false, false, 4, 7, 0, 1, "s1 r1");
        step(true, 2, false, false, 5, 7, 0, 1, "s1 r2");
        // ABORT at the last shot (shot_reg+1 == SHOTS) -> GUARD: no advance, base/shot hold.
        step(false, 0, true, false, 0, 0, 0, 1, "abort-guard");
        // a live round + cw_gen_finish: reads base(3)+2=5, finish=valid, shot_reg resets to 0.
        step(true, 2, false, true, 5, 7, 7, 0, "finish");
        // idle: mmio_out_valid=0 -> outputs 0.
        step(false, 0, false, false, 0, 0, 0, 0, "idle");
        wait(); check_pending();   // flush the last deferred expectation

        if (errors == 0) log_info("test", "stim_readout PASS");
        else             log_error("test", "stim_readout MISMATCH");
        sc_stop();
    }

    Tb(sc_module_name nm) : sc_module(nm), clk("clk", 10, SC_NS) {
        StimReadoutConfig cfg;
        cfg.m = 3; cfg.shots = 2; cfg.depth = 3;
        for (uint32_t a = 0; a < 6; ++a) { cfg.meas.push_back(mk(a, 3)); cfg.valid.push_back(mk(7, 3)); }
        dut = std::make_unique<StimReadout>("dut", cfg);
        dut->clk(clk); dut->rst(rst);
        dut->mmio_out_valid(mmio_v); dut->mmio_out_data(mmio_data); dut->cw_gen_finish(cw_fin);
        dut->ev_valid(ev_v); dut->ev_type(ev_t);
        dut->out_meas(o_meas); dut->out_meas_valid(o_valid); dut->out_meas_finish(o_finish);
        dut->shot_reg(shot);

        SC_HAS_PROCESS(Tb);
        SC_THREAD(run); sensitive << clk.posedge_event();
    }
};

int sc_main(int, char*[]) {
    Tb tb("tb");
    sc_start();
    return sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL) ? 1 : 0;
}
