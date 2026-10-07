// test_board_control.cpp — unit test for BoardControl.
//
// Instantiates all FOUR board combinations and drives each through targeted
// scenarios, recording every cycle and checking invariants (robust to the
// one-cycle sampling offset of the combinational outputs):
//
//   ORIGINATE+EXTERNAL  distributed root
//   ORIGINATE+INTERNAL  monolithic (root+leaf)
//   FORWARD  +EXTERNAL   distributed mid (router)
//   FORWARD  +INTERNAL   distributed leaf
//
// Key things checked:
//   * abort at attempt==0 IS detected (the XNOR-not-AND bug)
//   * a stale post_select (attempt mismatch) is IGNORED
//   * ORIGINATE flips attempt and emits ABORT with the NEW attempt
//   * FORWARD adopts ev_attempt and relays the event one cycle later
//   * EXTERNAL gates on attempt mismatch; INTERNAL gates only on abort
//   * FINISH -> DRAIN -> IDLE, where drain_done (= DCB det-finish) yields only reset
#include <systemc>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "control_system/cultiv_control/board_control.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using emu::Bits;
using B = emu::BoardControl;

// one board instance + its nets + a per-cycle record
struct Dut {
    sc_signal<bool>     rst, in_valid, data_attempt, ev_valid, ev_attempt, drain_done,
                        data_valid_output, out_attempt, cur_attempt, discard,
                        out_ev_valid, out_ev_attempt, reset, set_zero;
    sc_signal<uint32_t> ev_type, out_ev_type;
    sc_signal<Bits>     ev_payload, post_select, post_select_attempt, out_ev_payload;
    std::unique_ptr<B>  dut;

    struct Rec { bool reset, setz, oev_v, oev_a, o_att, o_disc; uint32_t oev_t; };
    std::vector<Rec> rec;

    void bind(sc_in<bool>& clk_in, sc_clock& clk, int em, int ds, int m, int pw, const char* nm) {
        dut = std::make_unique<B>(nm, em, ds, m, pw);
        dut->clk(clk); dut->rst(rst);
        dut->in_valid(in_valid); dut->data_attempt(data_attempt);
        dut->ev_valid(ev_valid); dut->ev_type(ev_type); dut->ev_payload(ev_payload);
        dut->ev_attempt(ev_attempt);
        dut->post_select(post_select); dut->post_select_attempt(post_select_attempt);
        dut->drain_done(drain_done); dut->data_valid_output(data_valid_output);
        dut->out_attempt(out_attempt); dut->cur_attempt(cur_attempt); dut->discard(discard);
        dut->out_ev_valid(out_ev_valid); dut->out_ev_type(out_ev_type);
        dut->out_ev_payload(out_ev_payload); dut->out_ev_attempt(out_ev_attempt);
        dut->reset(reset); dut->set_in_data_to_zero(set_zero);
        (void)clk_in;
    }
    void snap() {
        rec.push_back({reset.read(), set_zero.read(), out_ev_valid.read(),
                       out_ev_attempt.read(), out_attempt.read(), discard.read(),
                       out_ev_type.read()});
    }
    // did an out_ev of this type appear anywhere in [from, to)? return its attempt (or -1)
    int ev_attempt_of(uint32_t t, int from, int to) {
        for (int i = from; i < to && i < (int)rec.size(); ++i)
            if (rec[i].oev_v && rec[i].oev_t == t) return rec[i].oev_a ? 1 : 0;
        return -1;
    }
    bool any_reset(int from, int to) {
        for (int i = from; i < to && i < (int)rec.size(); ++i) if (rec[i].reset) return true;
        return false;
    }
    bool any_discard(int from, int to) {
        for (int i = from; i < to && i < (int)rec.size(); ++i) if (rec[i].o_disc) return true;
        return false;
    }
    bool any_setz(int from, int to) {
        for (int i = from; i < to && i < (int)rec.size(); ++i) if (rec[i].setz) return true;
        return false;
    }
};

SC_MODULE(Sys) {
    sc_clock clk;
    Dut root, mono, mid, leaf;
    bool ok_ = true;
    void fail(const std::string& m) { ok_ = false; std::cout << "  FAIL: " << m << "\n"; }

    void mon() { root.snap(); mono.snap(); mid.snap(); leaf.snap(); }

    // helpers to set common defaults on a Dut for one cycle
    static void ps_none(Dut& d) { d.post_select.write(Bits(1)); d.post_select_attempt.write(Bits(1)); }

    void run() {
        // ---- global reset ----
        for (Dut* d : {&root, &mono, &mid, &leaf}) {
            d->rst.write(true); d->in_valid.write(false); d->data_attempt.write(false);
            d->ev_valid.write(false); d->ev_type.write(0); d->ev_payload.write(Bits(2));
            d->ev_attempt.write(false); d->drain_done.write(false);
            d->data_valid_output.write(false);
            d->post_select.write(Bits(1)); d->post_select_attempt.write(Bits(1));
        }
        wait(); wait();
        for (Dut* d : {&root, &mono, &mid, &leaf}) d->rst.write(false);
        wait();

        // ================= ORIGINATE boards: root & mono =================
        // START -> EXEC, emit START(att=0)
        int t0 = root.rec.size();
        for (Dut* d : {&root, &mono}) { d->ev_valid.write(true); d->ev_type.write(B::EV_START); }
        wait();
        for (Dut* d : {&root, &mono}) { d->ev_valid.write(false); }
        wait(); wait();
        if (root.ev_attempt_of(B::EV_START, t0, root.rec.size()) != 0) fail("root START att!=0");
        if (mono.ev_attempt_of(B::EV_START, t0, mono.rec.size()) != 0) fail("mono START att!=0");

        // --- THE BUG: abort at attempt==0 must be detected ---
        int t1 = root.rec.size();
        for (Dut* d : {&root, &mono}) {
            Bits ps(1), pa(1); ps[0] = 1; pa[0] = 0;      // post_select set, attempt 0 == internal 0
            d->post_select.write(ps); d->post_select_attempt.write(pa);
        }
        wait();
        for (Dut* d : {&root, &mono}) ps_none(*d);
        wait(); wait();
        if (!root.any_reset(t1, root.rec.size())) fail("root abort@0 NOT detected (XNOR bug)");
        if (root.ev_attempt_of(B::EV_ABORT, t1, root.rec.size()) != 1) fail("root ABORT att!=1 (flip)");
        if (!mono.any_reset(t1, mono.rec.size())) fail("mono abort@0 NOT detected (XNOR bug)");
        if (mono.ev_attempt_of(B::EV_ABORT, t1, mono.rec.size()) != 1) fail("mono ABORT att!=1 (flip)");
        if (!root.any_discard(t1, root.rec.size())) fail("root discard not pulsed on abort@0");
        if (!mono.any_discard(t1, mono.rec.size())) fail("mono discard not pulsed on abort@0");

        // --- stale post_select: attempt now 1, post_select stamped 0 -> IGNORED ---
        int t2 = root.rec.size();
        for (Dut* d : {&root, &mono}) {
            Bits ps(1), pa(1); ps[0] = 1; pa[0] = 0;      // stamped 0 != internal 1
            d->post_select.write(ps); d->post_select_attempt.write(pa);
        }
        wait();
        for (Dut* d : {&root, &mono}) ps_none(*d);
        wait(); wait();
        if (root.any_reset(t2, root.rec.size())) fail("root stale post_select NOT ignored");
        if (mono.any_reset(t2, mono.rec.size())) fail("mono stale post_select NOT ignored");

        // --- abort at attempt==1 (matching) -> detect, flip to 0 ---
        int t3 = root.rec.size();
        for (Dut* d : {&root, &mono}) {
            Bits ps(1), pa(1); ps[0] = 1; pa[0] = 1;      // stamped 1 == internal 1
            d->post_select.write(ps); d->post_select_attempt.write(pa);
        }
        wait();
        for (Dut* d : {&root, &mono}) ps_none(*d);
        wait(); wait();
        if (!root.any_reset(t3, root.rec.size())) fail("root abort@1 not detected");
        if (root.ev_attempt_of(B::EV_ABORT, t3, root.rec.size()) != 0) fail("root ABORT att!=0 (flip back)");

        // --- FINISH -> DRAIN -> IDLE; drain_done (= DCB det-finish) produces only reset ---
        int t4 = root.rec.size();
        for (Dut* d : {&root, &mono}) { d->ev_valid.write(true); d->ev_type.write(B::EV_FINISH); }
        wait();
        for (Dut* d : {&root, &mono}) { d->ev_valid.write(false); }
        wait(); wait();                             // draining
        int t5 = root.rec.size();
        for (Dut* d : {&root, &mono}) d->drain_done.write(true);
        wait();
        for (Dut* d : {&root, &mono}) d->drain_done.write(false);
        wait(); wait();
        if (root.ev_attempt_of(B::EV_FINISH, t4, t5) < 0) fail("root FINISH not emitted down");
        if (!root.any_reset(t5, root.rec.size())) fail("root finish reset not pulsed");
        if (!mono.any_reset(t5, mono.rec.size())) fail("mono finish reset not pulsed");
        if (root.any_discard(t5, root.rec.size())) fail("root discard wrongly pulsed on finish");
        if (mono.any_discard(t5, mono.rec.size())) fail("mono discard wrongly pulsed on finish");

        // ================= FORWARD boards: mid & leaf =================
        // START forwarded + attempt adopted
        int u0 = mid.rec.size();
        for (Dut* d : {&mid, &leaf}) {
            d->ev_valid.write(true); d->ev_type.write(B::EV_START); d->ev_attempt.write(false);
        }
        wait();
        for (Dut* d : {&mid, &leaf}) d->ev_valid.write(false);
        wait(); wait();
        if (mid.ev_attempt_of(B::EV_START, u0, mid.rec.size()) != 0) fail("mid START not forwarded");
        if (leaf.ev_attempt_of(B::EV_START, u0, leaf.rec.size()) != 0) fail("leaf START not forwarded");

        // ABORT forwarded, attempt adopted to 1, reset pulsed
        int u1 = mid.rec.size();
        for (Dut* d : {&mid, &leaf}) {
            d->ev_valid.write(true); d->ev_type.write(B::EV_ABORT); d->ev_attempt.write(true);
        }
        wait();
        for (Dut* d : {&mid, &leaf}) d->ev_valid.write(false);
        wait(); wait();
        if (mid.ev_attempt_of(B::EV_ABORT, u1, mid.rec.size()) != 1) fail("mid ABORT not forwarded w/ att=1");
        if (!mid.any_reset(u1, mid.rec.size())) fail("mid abort no reset");
        if (!leaf.any_reset(u1, leaf.rec.size())) fail("leaf abort no reset");

        // ---- data gating difference: EXTERNAL(mid) vs INTERNAL(leaf) ----
        // Both boards now at internal attempt 1. Present valid data tagged attempt 0.
        int u2 = mid.rec.size();
        for (Dut* d : {&mid, &leaf}) { d->in_valid.write(true); d->data_attempt.write(false); }
        wait(); wait();
        for (Dut* d : {&mid, &leaf}) d->in_valid.write(false);
        wait();
        // mid (EXTERNAL): data_attempt 0 != internal 1 -> gate
        if (!mid.any_setz(u2, mid.rec.size())) fail("mid did not gate mismatched-attempt data");
        // leaf (INTERNAL): data_att_eff == internal, no abort -> NO gate
        if (leaf.any_setz(u2, leaf.rec.size())) fail("leaf wrongly gated (INTERNAL should ignore data_attempt)");

        // leaf DOES gate when abort_now, even without attempt mismatch
        int u3 = leaf.rec.size();
        leaf.in_valid.write(true);
        leaf.ev_valid.write(true); leaf.ev_type.write(B::EV_ABORT); leaf.ev_attempt.write(false);
        wait();
        leaf.in_valid.write(false); leaf.ev_valid.write(false);
        wait();
        if (!leaf.any_setz(u3, leaf.rec.size())) fail("leaf did not gate on abort");

        std::cout << "samples=" << root.rec.size() << "\n";
        if (ok_) emu::log_info("test", "board_control PASS");
        else     emu::log_error("test", "board_control MISMATCH");
        sc_stop();
    }

    SC_CTOR(Sys) : clk("clk", 10, SC_NS) {
        sc_in<bool> dummy;
        root.bind(dummy, clk, B::ORIGINATE, B::EXTERNAL, 1, 2, "root");
        mono.bind(dummy, clk, B::ORIGINATE, B::INTERNAL, 1, 2, "mono");
        mid .bind(dummy, clk, B::FORWARD,   B::EXTERNAL, 0, 2, "mid");
        leaf.bind(dummy, clk, B::FORWARD,   B::INTERNAL, 0, 2, "leaf");

        SC_METHOD(mon); sensitive << clk.posedge_event(); dont_initialize();
        SC_THREAD(run); sensitive << clk.posedge_event();
    }
};

int sc_main(int, char*[]) {
    Sys sys("sys");
    sc_start();
    int errors = sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL);
    return errors ? 1 : 0;
}
