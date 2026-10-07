// test_instr_decode.cpp — unit test for InstrDecode.
//
// Records (r_en, r_addr, one_instr_end) every cycle, then checks the CW read
// stream against an expected per-cycle pattern. Because the outputs are Mealy
// (combinational) and the monitor samples at posedge, the whole stream can be
// shifted by one cycle relative to the stimulus; the checker locates the first
// read and matches the pattern from there, so it is immune to that global
// offset while still verifying contiguity, wait gaps, and one_instr_end timing.
//
// Scenarios:
//   A  wt=0  start=5  end=7   -> reads 5,6,7        (multi-word, fast path)
//   B  wt=0  start=10 end=10  -> read  10           (start==end, single word)
//      A->B has NO bubble (7 then 10 on consecutive cycles)
//   C  wt=2  start=20 end=22  -> idle,idle,20,21,22 (wait then stream)
//   reset mid-stream: outputs drop to idle and a later instruction starts clean.
#include <systemc>

#include <cstdint>
#include <iostream>
#include <vector>

#include "control_system/cultiv_control/instr_decode.hpp"
#include "log.hpp"

using namespace sc_core;

struct Sample { bool en; uint32_t addr; bool oie; };

SC_MODULE(Tb) {
    sc_clock            clk;
    sc_signal<bool>     rst, en, r_en, one_instr_end;
    sc_signal<uint32_t> in_wt, in_start, in_end, r_addr;

    emu::InstrDecode dut;

    bool started_ = false;
    std::vector<Sample> rec;

    void monitor() {
        if (started_) rec.push_back({r_en.read(), r_addr.read(), one_instr_end.read()});
    }

    // drive one cycle of instruction input
    void put(bool e, uint32_t wt, uint32_t st, uint32_t ed) {
        en.write(e); in_wt.write(wt); in_start.write(st); in_end.write(ed); wait();
    }
    void idle() { put(false, 0, 0, 0); }

    bool ok_ = true;
    void fail(const std::string& m) { ok_ = false; std::cout << "  FAIL: " << m << "\n"; }

    // find first r_en=1 in [from, end); -1 if none
    int first_read(int from) {
        for (int i = from; i < (int)rec.size(); ++i) if (rec[i].en) return i;
        return -1;
    }

    void run() {
        rst.write(true); en.write(false); in_wt.write(0); in_start.write(0); in_end.write(0);
        wait(); wait();
        rst.write(false); started_ = true;

        // --- Phase 1: A, B, C back-to-back (state-machine handoff emulated) ---
        // A: en@cycle0, len 3 (reads 5,6,7)
        put(true, 0, 5, 7);       // c0: en A
        idle();                   // c1: A streaming
        idle();                   // c2: A last read (oie)
        // B: en one cycle after A's oie -> c3
        put(true, 0, 10, 10);     // c3: en B (single word, oie same cycle)
        // C: en one cycle after B's oie -> c4
        put(true, 2, 20, 22);     // c4: en C (wt=2)
        idle();                   // c5: C wait
        idle();                   // c6: C read 20
        idle();                   // c7: C read 21
        idle();                   // c8: C read 22 (oie)
        idle(); idle(); idle();   // drain

        // expected per-cycle pattern from the first read
        const std::vector<Sample> pat = {
            {true,  5, false},
            {true,  6, false},
            {true,  7, true},     // A end
            {true, 10, true},     // B (contiguous, single word)
            {false, 0, false},    // C wait
            {false, 0, false},    // C wait
            {true, 20, false},
            {true, 21, false},
            {true, 22, true},     // C end
        };
        int p0 = first_read(0);
        if (p0 < 0) fail("no reads recorded");
        else if (p0 + (int)pat.size() > (int)rec.size()) fail("stream too short");
        else {
            for (int i = 0; i < (int)pat.size(); ++i) {
                const Sample& g = rec[p0 + i];
                const Sample& e = pat[i];
                bool match = (g.en == e.en) && (g.oie == e.oie) && (!e.en || g.addr == e.addr);
                if (!match)
                    fail("phase1 cycle " + std::to_string(i) +
                         " got(en=" + std::to_string(g.en) + ",addr=" + std::to_string(g.addr) +
                         ",oie=" + std::to_string(g.oie) + ")");
            }
        }

        // --- Phase 2: reset mid-stream ---
        int base = rec.size();
        put(true, 0, 30, 34);     // start D (reads 30..34)
        idle();                   // D read 31
        rst.write(true); idle();  // assert reset mid-stream
        rst.write(false);
        // after reset, launch E cleanly
        put(true, 0, 40, 41);     // E: reads 40,41
        idle(); idle(); idle();

        // D must NOT complete (no read of 34); E must read 40,41 in order.
        std::vector<uint32_t> reads_after_reset;
        bool saw_34 = false;
        for (int i = base; i < (int)rec.size(); ++i) {
            if (rec[i].en) {
                reads_after_reset.push_back(rec[i].addr);
                if (rec[i].addr == 34) saw_34 = true;
            }
        }
        if (saw_34) fail("reset failed: instruction D completed (read 34)");
        // the tail of the read list should end with E's 40,41
        int n = reads_after_reset.size();
        if (n < 2 || reads_after_reset[n-2] != 40 || reads_after_reset[n-1] != 41)
            fail("post-reset instruction E did not read 40,41 cleanly");

        std::cout << "phase1 first-read idx=" << p0 << ", total samples=" << rec.size() << "\n";
        if (ok_) emu::log_info("test", "instr_decode PASS");
        else     emu::log_error("test", "instr_decode MISMATCH");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", /*wt_w=*/8, /*addr_w=*/16) {
        dut.clk(clk); dut.rst(rst);
        dut.en(en); dut.in_wt(in_wt); dut.in_start(in_start); dut.in_end(in_end);
        dut.r_en(r_en); dut.r_addr(r_addr); dut.one_instr_end(one_instr_end);

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
