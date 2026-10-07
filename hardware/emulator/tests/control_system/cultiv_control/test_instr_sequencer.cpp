// test_instr_sequencer.cpp — unit test for InstrSequencer.
//
// DEPTH=4. Records (instr_en, instr_addr, rst_out, cw_gen_finish) every cycle
// and matches an exact expected pattern. Because rst_out/cw_gen_finish are
// combinational and the monitor samples at posedge, the whole stream can be
// shifted by one cycle relative to the stimulus; the checker locates the first
// instr_en=1 and matches from there, so it is immune to that global offset.
//
// Covered:
//   * START -> instruction 0 fetched the next cycle
//   * advance on one_instr_end: 0 -> 1 -> 2 -> 3 with en re-pulsing each time
//   * SATURATION at DEPTH-1: further one_instr_end keeps refetching instr 3
//     (the compiled "wait" instruction)
//   * ABORT mid-instruction: rst_out high that cycle, refetch of instr 0 next
//   * FINISH mid-instruction: -> DRAIN, then one_instr_end -> IDLE + cw_gen_finish
//   * FINISH exactly on one_instr_end: straight to IDLE, no DRAIN
//   * cw_gen_finish is a ONE-cycle pulse
#include <systemc>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "control_system/cultiv_control/instr_sequencer.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using emu::Bits;
using S = emu::InstrSequencer;

struct Sample { bool en; uint32_t addr; bool rst_o; bool cwf; };

SC_MODULE(Tb) {
    static constexpr int DEPTH = 4, PAYLOAD_W = 8;

    sc_clock            clk;
    sc_signal<bool>     rst, ev_valid, one_instr_end, instr_en, rst_out, cw_gen_finish;
    sc_signal<uint32_t> ev_type, instr_addr;
    sc_signal<Bits>     ev_payload;

    emu::InstrSequencer dut;

    bool started_ = false;
    std::vector<Sample> rec;
    bool ok_ = true;

    void monitor() {
        if (started_)
            rec.push_back({instr_en.read(), instr_addr.read(),
                           rst_out.read(), cw_gen_finish.read()});
    }

    void fail(const std::string& m) { ok_ = false; std::cout << "  FAIL: " << m << "\n"; }

    // one cycle of stimulus
    void put(bool valid, uint32_t type, bool oie) {
        ev_valid.write(valid); ev_type.write(type); one_instr_end.write(oie); wait();
    }
    void idle(bool oie = false) { put(false, 0, oie); }

    void run() {
        rst.write(true); ev_valid.write(false); ev_type.write(0);
        ev_payload.write(Bits(PAYLOAD_W)); one_instr_end.write(false);
        wait(); wait();
        rst.write(false); started_ = true;

        put(true, S::EV_START, false);  // c0  START
        idle();                         // c1  (en=1 addr=0)
        idle();                         // c2
        idle(true);                     // c3  instr 0 done
        idle();                         // c4  (en=1 addr=1)
        idle(true);                     // c5  instr 1 done
        idle();                         // c6  (en=1 addr=2)
        idle(true);                     // c7  instr 2 done
        idle();                         // c8  (en=1 addr=3)
        idle(true);                     // c9  instr 3 done -> SATURATE
        idle();                         // c10 (en=1 addr=3 again)
        idle(true);                     // c11 saturate again
        idle();                         // c12 (en=1 addr=3)
        put(true, S::EV_ABORT, false);  // c13 ABORT mid-instruction
        idle();                         // c14 (en=1 addr=0)
        idle();                         // c15
        put(true, S::EV_FINISH, false); // c16 FINISH mid-instruction -> DRAIN
        idle();                         // c17 draining
        idle(true);                     // c18 in-flight instruction ends
        idle();                         // c19 (cw_gen_finish + rst_out)
        idle();                         // c20 pulse must be gone
        put(true, S::EV_START, false);  // c21 restart
        put(true, S::EV_FINISH, true);  // c22 FINISH exactly on one_instr_end
        idle();                         // c23 (cw_gen_finish, NO drain)
        idle();                         // c24 pulse gone
        idle(); idle();

        // expected outputs per cycle, starting at the first instr_en=1 (c1)
        const std::vector<Sample> pat = {
            {true,  0, false, false},   // c1  fetch instr 0
            {false, 0, false, false},   // c2
            {false, 0, false, false},   // c3
            {true,  1, false, false},   // c4  fetch instr 1
            {false, 0, false, false},   // c5
            {true,  2, false, false},   // c6  fetch instr 2
            {false, 0, false, false},   // c7
            {true,  3, false, false},   // c8  fetch instr 3
            {false, 0, false, false},   // c9
            {true,  3, false, false},   // c10 SATURATED at DEPTH-1
            {false, 0, false, false},   // c11
            {true,  3, false, false},   // c12 saturated again
            {false, 0, true,  false},   // c13 ABORT -> rst_out this cycle
            {true,  0, false, false},   // c14 refetch instr 0
            {false, 0, false, false},   // c15
            {false, 0, false, false},   // c16 FINISH seen
            {false, 0, false, false},   // c17 DRAIN
            {false, 0, false, false},   // c18 DRAIN, instruction ends
            {false, 0, true,  true},    // c19 cw_gen_finish + rst_out
            {false, 0, false, false},   // c20 pulse gone (1 cycle only)
            {false, 0, false, false},   // c21 START seen
            {true,  0, false, false},   // c22 fetch instr 0
            {false, 0, true,  true},    // c23 FINISH+oie -> straight to IDLE
            {false, 0, false, false},   // c24 pulse gone
        };

        int p0 = -1;
        for (int i = 0; i < (int)rec.size(); ++i) if (rec[i].en) { p0 = i; break; }
        if (p0 < 0) fail("no instruction fetch recorded");
        else if (p0 + (int)pat.size() > (int)rec.size()) fail("stream too short");
        else {
            for (int i = 0; i < (int)pat.size(); ++i) {
                const Sample& g = rec[p0 + i];
                const Sample& e = pat[i];
                if (g.en != e.en || g.addr != e.addr || g.rst_o != e.rst_o || g.cwf != e.cwf)
                    fail("cycle " + std::to_string(i) +
                         " got(en=" + std::to_string(g.en) + ",addr=" + std::to_string(g.addr) +
                         ",rst=" + std::to_string(g.rst_o) + ",cwf=" + std::to_string(g.cwf) +
                         ") exp(en=" + std::to_string(e.en) + ",addr=" + std::to_string(e.addr) +
                         ",rst=" + std::to_string(e.rst_o) + ",cwf=" + std::to_string(e.cwf) + ")");
            }
        }

        std::cout << "first-fetch idx=" << p0 << ", total samples=" << rec.size() << "\n";
        if (ok_) emu::log_info("test", "instr_sequencer PASS");
        else     emu::log_error("test", "instr_sequencer MISMATCH");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", DEPTH, PAYLOAD_W) {
        dut.clk(clk); dut.rst(rst);
        dut.ev_valid(ev_valid); dut.ev_type(ev_type); dut.ev_payload(ev_payload);
        dut.one_instr_end(one_instr_end);
        dut.instr_en(instr_en); dut.instr_addr(instr_addr);
        dut.rst_out(rst_out); dut.cw_gen_finish(cw_gen_finish);

        SC_METHOD(monitor); sensitive << clk.posedge_event(); dont_initialize();
        SC_THREAD(run);     sensitive << clk.posedge_event();
    }
};

int sc_main(int, char*[]) {
    Tb tb("tb");
    sc_start();
    int errors = sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL);
    return errors ? 1 : 0;
}
