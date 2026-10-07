// test_physical_mmio.cpp — end-to-end test of the assembled PhysicalMMIO.
//
// This is the first test that exercises the whole chain
//   InstrSequencer -> InstrRegFile -> InstrUnpack -> InstrDecode -> CwMemory
// so it checks the properties no unit test could: the command-word stream a
// control core would actually see, and the reset timing across module borders.
//
// Config: instr_depth=4, cw_depth=16, data_w=8  ->  addr_w=4, instr_w=16.
// CW[i] = 10*i + 1, so every command word identifies its own address.
//
//   instr | wt | start..end | emits
//   ------+----+------------+---------------------------------------
//     0   | 0  |   0 .. 2   | CW[0]=1, CW[1]=11, CW[2]=21
//     1   | 2  |   4 .. 5   | 2 idle cycles, then CW[4]=41, CW[5]=51
//     2   | 0  |   8 .. 8   | CW[8]=81                (single word)
//     3   | 1  |  12 .. 13  | 1 idle, CW[12]=121, CW[13]=131  <-- WAIT instr,
//                             repeats forever until FINISH
//
// Checks:
//   1. exact command-word stream, including the wt gaps and the wait repeat
//   2. no bubble across a wt=0 instruction handoff (51 then 81 back-to-back)
//   3. ABORT restarts the program from CW[0]
//   4. FINISH still delivers the LAST command word (SyncROM has no reset)
//   5. cw_gen_finish pulses in exactly the same cycle as that final out_valid
#include <systemc>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "control_system/cultiv_control/physical_mmio.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using emu::Bits;
using S = emu::InstrSequencer;

struct Sample { bool valid; uint32_t data; bool cwf; };

SC_MODULE(Tb) {
    sc_clock            clk;
    sc_signal<bool>     rst, ev_valid, out_valid, cw_gen_finish;
    sc_signal<uint32_t> ev_type;
    sc_signal<Bits>     ev_payload, out_data;

    emu::MmioConfig  cfg;
    emu::PhysicalMMIO dut;

    bool started_ = false;
    std::vector<Sample> rec;
    bool ok_ = true;

    void fail(const std::string& m) { ok_ = false; std::cout << "  FAIL: " << m << "\n"; }

    void monitor() {
        if (started_)
            rec.push_back({out_valid.read(),
                           emu::extract(out_data.read(), 0, cfg.data_w),
                           cw_gen_finish.read()});
    }

    void put(bool v, uint32_t t) { ev_valid.write(v); ev_type.write(t); wait(); }
    void idle(int n = 1) { for (int i = 0; i < n; ++i) put(false, 0); }

    void run() {
        rst.write(true); ev_valid.write(false); ev_type.write(0);
        ev_payload.write(Bits(cfg.payload_w));
        wait(); wait();
        rst.write(false); started_ = true;

        // ---- phase 1: START, free-run through the program into the wait loop
        put(true, S::EV_START);
        idle(20);
        int abort_at = rec.size();
        // ---- phase 2: ABORT -> program restarts at instruction 0
        put(true, S::EV_ABORT);
        idle(3);
        // ---- phase 3: FINISH mid-instruction (during instr1's wt gap) -> DRAIN,
        //      the in-flight instruction still completes and emits its last word
        put(true, S::EV_FINISH);
        idle(12);

        // --- 1/2: exact stream from the first valid ---
        const std::vector<Sample> pat = {
            {true,   1, false},   // instr0: CW[0]
            {true,  11, false},   //         CW[1]
            {true,  21, false},   //         CW[2]
            {false,  0, false},   // instr1: wt=2 gap
            {false,  0, false},
            {true,  41, false},   //         CW[4]
            {true,  51, false},   //         CW[5]
            {true,  81, false},   // instr2: CW[8]  <-- NO bubble after 51
            {false,  0, false},   // instr3: wt=1 gap
            {true, 121, false},   //         CW[12]
            {true, 131, false},   //         CW[13]
            {false,  0, false},   // wait instr repeats...
            {true, 121, false},
            {true, 131, false},
            {false,  0, false},
            {true, 121, false},
            {true, 131, false},
        };
        int p0 = -1;
        for (int i = 0; i < (int)rec.size(); ++i) if (rec[i].valid) { p0 = i; break; }
        if (p0 < 0) fail("no command words emitted");
        else if (p0 + (int)pat.size() > (int)rec.size()) fail("stream too short");
        else
            for (int i = 0; i < (int)pat.size(); ++i) {
                const Sample& g = rec[p0 + i];
                if (g.valid != pat[i].valid || (pat[i].valid && g.data != pat[i].data))
                    fail("stream cycle " + std::to_string(i) + " got(v=" +
                         std::to_string(g.valid) + ",d=" + std::to_string(g.data) +
                         ") exp(v=" + std::to_string(pat[i].valid) + ",d=" +
                         std::to_string(pat[i].data) + ")");
            }

        // --- 3: after ABORT the program restarts at CW[0] ---
        // NOTE: at most ONE in-flight command word may still emerge after an
        // abort — InstrDecode is reset the same cycle, but SyncROM (which has no
        // reset, by design) still delivers the word fetched the cycle before.
        // That is intended and harmless: the qubits are reset right afterwards.
        // So allow one leading word, then require a clean restart at CW[0].
        std::vector<uint32_t> after_abort;
        for (int i = abort_at; i < (int)rec.size(); ++i)
            if (rec[i].valid) after_abort.push_back(rec[i].data);
        std::size_t k = (!after_abort.empty() && after_abort[0] != 1) ? 1 : 0;  // skip in-flight
        if (k == 1) std::cout << "  abort: 1 in-flight word (" << after_abort[0]
                              << ") emitted after the abort, as designed\n";
        if (after_abort.size() < k + 3 ||
            after_abort[k] != 1 || after_abort[k + 1] != 11 || after_abort[k + 2] != 21)
            fail("ABORT did not restart the program at CW[0],CW[1],CW[2]");

        // --- 4/5: cw_gen_finish aligns with the FINAL out_valid ---
        int fin = -1, n_fin = 0, last_valid = -1;
        for (int i = 0; i < (int)rec.size(); ++i) {
            if (rec[i].cwf) { if (fin < 0) fin = i; ++n_fin; }
            if (rec[i].valid) last_valid = i;
        }
        if (fin < 0)                fail("cw_gen_finish never pulsed");
        else if (n_fin != 1)        fail("cw_gen_finish pulsed " + std::to_string(n_fin) +
                                         " times (expected exactly 1)");
        else if (!rec[fin].valid)   fail("cw_gen_finish cycle has out_valid=0: the LAST "
                                         "command word was dropped");
        else if (fin != last_valid) fail("out_valid continued after cw_gen_finish");
        else std::cout << "  finish: last command word = " << rec[fin].data
                       << " delivered in the same cycle as cw_gen_finish\n";

        std::cout << "samples=" << rec.size() << " first-valid=" << p0 << "\n";
        if (ok_) emu::log_info("test", "physical_mmio PASS");
        else     emu::log_error("test", "physical_mmio MISMATCH");
        sc_stop();
    }

    static emu::MmioConfig mk_cfg() {
        emu::MmioConfig c;
        c.instr_depth = 4; c.cw_depth = 16; c.data_w = 8; c.wt_w = 8; c.payload_w = 8;
        return c;
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), cfg(mk_cfg()), dut("dut", cfg) {
        dut.clk(clk); dut.rst(rst);
        dut.ev_valid(ev_valid); dut.ev_type(ev_type); dut.ev_payload(ev_payload);
        dut.out_data(out_data); dut.out_valid(out_valid); dut.cw_gen_finish(cw_gen_finish);

        // command words: CW[i] = 10*i + 1
        std::vector<Bits> cw;
        for (int i = 0; i < cfg.cw_depth; ++i) {
            Bits w(cfg.data_w);
            const uint32_t v = 10u * i + 1u;
            for (int b = 0; b < cfg.data_w; ++b) w[b] = (v >> b) & 1u;
            cw.push_back(w);
        }
        dut.load_cw(cw);

        dut.load_instr({
            dut.pack_instr(0,  0,  2),   // instr0
            dut.pack_instr(2,  4,  5),   // instr1
            dut.pack_instr(0,  8,  8),   // instr2
            dut.pack_instr(1, 12, 13),   // instr3 = wait instruction
        });

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
