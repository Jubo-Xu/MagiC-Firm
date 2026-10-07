// test_control_board_stim.cpp — whole-system stim test: a TREE of ControlBoards +
// per-leaf StimReadout loops, driven from a compiled example (like test_dcb_stim, but the
// full boards — control FSM + MMIO + measurement-readout loop, not the bare datapath).
//
// GENERIC over topology (monolithic = 1-board tree, distributed = root/routers/leaves wired
// per connections.json). test_dcb_stim already validates the DCB DATA plane across the tree;
// this adds the CONTROL plane: events ripple DOWN (root.out_ev -> child.ev), data/attempt/
// finish/post-select ripple UP, and each LEAF closes its own mmio_out -> StimReadout -> in_meas
// loop. START kicks the root; from there the whole tree runs itself.
//
// Multi-shot, multi-trial state machine at the ROOT (identical to the monolithic driver):
//   * INTERNAL post-select reject -> board aborts and auto-advances every leaf's StimReadout;
//     check the oracle (expected_postselect at the root) agreed it's a reject.
//   * CLEAN completion (last_normal + wait_rounds real rounds) -> compare out_det vs
//     expected_dets[s], then gap_post_select to force-advance to the next shot.
// The LAST shot ends the trial conditionally (matching real operation): clean -> host finish
// (drain via EXEC->DRAIN->out_finish, which resets the readouts so the next START restarts with
// no reset); discard -> nothing to drain, so rst to restart.
//
//   ./test_control_board_stim <example_dir> [root_board_id]
#include <systemc>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "control_system/board_config.hpp"
#include "control_system/control_board.hpp"
#include "control_system/control_board_loader.hpp"
#include "control_system/detector_construct/board_loader.hpp"   // load_stim_vectors
#include "control_system/stim_readout.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using namespace emu;
using nlohmann::json;

struct PortMap { int child, child_port; };

// All ControlBoard port nets for one board, plus its leaf loop nets.
struct Nets {
    sc_signal<bool>     ev_valid, ev_attempt, start, finish, gap_ps;
    sc_signal<uint32_t> ev_type;
    sc_signal<Bits>     ev_payload;
    sc_signal<Bits>     in_det, in_det_valid, in_meas, in_meas_valid, in_meas_finish;
    sc_signal<Bits>     in_det_fin_ch, in_raw_fin_ch, in_child_att, ps_in, ps_in_att;
    sc_signal<Bits>     fwd_det, fwd_det_valid, fwd_raw, fwd_raw_valid;
    sc_signal<bool>     fwd_det_finish, fwd_raw_finish, out_attempt, ps_out, ps_out_att;
    sc_signal<bool>     out_ev_valid, out_ev_attempt;
    sc_signal<uint32_t> out_ev_type;
    sc_signal<Bits>     out_ev_payload;
    sc_signal<Bits>     out_det, out_used, out_gidx;
    sc_signal<bool>     out_valid, out_finish, discard, first_normal, last_normal, first_wait, last_wait;
    // leaf loop (P=1): mmio_out -> StimReadout -> in_meas
    sc_signal<Bits>     mmio_data, sr_meas, sr_valid, sr_finish;
    sc_signal<bool>     mmio_valid, cw_finish;
    sc_signal<uint32_t> sr_shot;
};

SC_MODULE(Sys) {
    sc_clock        clk;
    sc_signal<bool> rst;

    // per-board structures (keyed by board_id)
    std::map<int, std::unique_ptr<Nets>>          net;
    std::map<int, std::unique_ptr<ControlBoard>>  board;
    std::map<int, std::unique_ptr<StimReadout>>   sr;      // leaves only
    std::map<int, BoardConfig>                    cfg;
    std::map<int, std::vector<PortMap>>           meas_port, det_port;   // parent input line -> (child, child_port)
    std::map<int, std::vector<int>>               children;             // ordered child list (matches loader)
    std::map<int, int>                            parent_of;            // child -> parent
    std::vector<int> board_ids, leaf_ids, stage_ids;   // stage_ids feed root.ps_in (order)
    int root_id = -1;

    // ROOT host controls (the only board the test drives)
    sc_signal<bool> h_start, h_finish, h_gap_ps;
    Nets* rn = nullptr;   // root's nets (monitored)

    // ---- checking state ----
    int ndet = 0, wait_rounds = 0;
    std::vector<Bits> expected_dets, expected_ps;
    std::vector<int>  recon;
    bool ln_seen = false, disc_seen = false, complete = false, fin_seen = false;
    int  post_ln_rounds = 0;

    // combinational tree data plane: each parent input port <- its child's forward bus,
    // plus per-child attempt/finish. Fixed-width vectors (children may be transiently width-0).
    void wire() {
        for (int p : board_ids) {
            if (children[p].empty()) continue;                     // leaves have no children
            const int M = cfg[p].dcb.m, DI = cfg[p].dcb.d_in;
            Bits im(M), iv(M), imf(M), id(DI), idv(DI), idf(DI);
            const auto& mp = meas_port[p];
            for (int i = 0; i < (int)mp.size() && i < M; ++i) {
                const Nets& c = *net[mp[i].child];
                const Bits f = c.fwd_raw.read(), fv = c.fwd_raw_valid.read();
                const int cp = mp[i].child_port;
                im[i]  = (cp < (int)f.size())  ? f[cp]  : 0;
                iv[i]  = (cp < (int)fv.size()) ? fv[cp] : 0;
                imf[i] = c.fwd_raw_finish.read() ? 1 : 0;          // broadcast child's single raw-finish
            }
            const auto& dp = det_port[p];
            for (int j = 0; j < (int)dp.size() && j < DI; ++j) {
                const Nets& c = *net[dp[j].child];
                const Bits f = c.fwd_det.read(), fv = c.fwd_det_valid.read();
                const int cp = dp[j].child_port;
                id[j]  = (cp < (int)f.size())  ? f[cp]  : 0;
                idv[j] = (cp < (int)fv.size()) ? fv[cp] : 0;
                idf[j] = c.fwd_det_finish.read() ? 1 : 0;          // broadcast child's single det-finish
            }
            net[p]->in_meas.write(im);   net[p]->in_meas_valid.write(iv);
            net[p]->in_det.write(id);    net[p]->in_det_valid.write(idv);

            // per-child single-bit finishes + attempt (indexed by child order)
            const int NC = (int)children[p].size();
            Bits dfc(NC), rfc(NC), att(NC);
            for (int c = 0; c < NC; ++c) {
                const Nets& cn = *net[children[p][c]];
                dfc[c] = cn.fwd_det_finish.read() ? 1 : 0;
                rfc[c] = cn.fwd_raw_finish.read() ? 1 : 0;
                att[c] = cn.out_attempt.read() ? 1 : 0;
            }
            net[p]->in_det_fin_ch.write(dfc);
            net[p]->in_raw_fin_ch.write(rfc);
            net[p]->in_child_att.write(att);
            // in_meas_finish is the broadcast raw-finish for a router; the DCB re-broadcasts it,
            // but the ControlBoard's gating uses in_raw_finish_child for routers, so imf is unused
            // there. Drive it to keep the net sized.
            net[p]->in_meas_finish.write(imf);
        }
        // post-select fast path: each stage board's ps_out -> root.ps_in[slot]
        if (root_id >= 0 && !stage_ids.empty()) {
            const int NS = (int)stage_ids.size();
            Bits ps(NS), pa(NS);
            for (int i = 0; i < NS; ++i) {
                ps[i] = net[stage_ids[i]]->ps_out.read() ? 1 : 0;
                pa[i] = net[stage_ids[i]]->ps_out_att.read() ? 1 : 0;
            }
            net[root_id]->ps_in.write(ps);
            net[root_id]->ps_in_att.write(pa);
        }
    }

    // each leaf: close its mmio_out -> StimReadout -> in_meas loop. FIXED-width (the StimReadout
    // output is transiently width-0 at t=0, and the leaf DCB indexes in_meas by bit at that edge).
    void loopback() {
        for (int l : leaf_ids) {
            const int M = cfg[l].dcb.m;
            Bits m(M), v(M), f(M);
            const Bits sm = net[l]->sr_meas.read(), sv = net[l]->sr_valid.read(), sf = net[l]->sr_finish.read();
            for (int i = 0; i < M && i < (int)sm.size(); ++i) m[i] = sm[i];
            for (int i = 0; i < M && i < (int)sv.size(); ++i) v[i] = sv[i];
            for (int i = 0; i < M && i < (int)sf.size(); ++i) f[i] = sf[i];
            net[l]->in_meas.write(m);
            net[l]->in_meas_valid.write(v);
            net[l]->in_meas_finish.write(f);
        }
    }

    // collect the ROOT output each cycle -> recon[global_index] = detector; track completion.
    void monitor() {
        if (rn->discard.read()) disc_seen = true;
        const bool ln_this = rn->last_normal.read();
        if (ln_this) ln_seen = true;
        if (rn->out_finish.read()) fin_seen = true;

        if (!rn->out_valid.read()) return;
        const Bits used = rn->out_used.read(), det = rn->out_det.read(), gi = rn->out_gidx.read();
        const int dout = cfg[root_id].dcb.d_out(), iw = cfg[root_id].dcb.hw_width;
        int real_used = 0;
        for (int l = 0; l < dout; ++l) {
            if (!used[l]) continue;
            uint32_t idx = extract(gi, (std::size_t)l * iw, iw);
            if (idx < recon.size()) { ++real_used; recon[idx] = det[l] ? 1 : 0; }
        }
        if (ln_seen && !ln_this && real_used > 0) ++post_ln_rounds;
        if (ln_seen && post_ln_rounds >= wait_rounds) complete = true;
    }

    bool exp_ps_of(int s) {
        return s < (int)expected_ps.size() && expected_ps[s].size() && expected_ps[s][0];
    }

    void run() {
        const int N = expected_dets.size();
        const int TRIALS = 2;
        std::cout << "[t] control_board_stim: boards=" << board_ids.size() << " leaves=" << leaf_ids.size()
                  << " root=" << root_id << " stage=" << stage_ids.size() << " shots=" << N
                  << " trials=" << TRIALS << " ndet=" << ndet << " wait_rounds=" << wait_rounds << "\n" << std::flush;

        int checked = 0, passed = 0, det_mismatch = 0, det_missing = 0, ps_mismatch = 0, internal = 0;
        int finishes = 0, finish_attempts = 0;

        rst.write(true); wait(); wait(); rst.write(false);   // power-on reset (once)

        for (int trial = 0; trial < TRIALS; ++trial) {
            h_start.write(true); wait(); h_start.write(false);

            for (int s = 0; s < N; ++s) {
                const bool is_last = (s == N - 1);
                ln_seen = disc_seen = complete = false;
                post_ln_rounds = 0;
                std::fill(recon.begin(), recon.end(), -1);

                int w = 0;
                while (!disc_seen && !complete && w < 12000) { wait(); ++w; }
                if (trial == 0 && s < 20) {   // DBG: are all leaves' StimReadouts on the same shot?
                    std::cout << "  [sync] s" << s << " (" << (disc_seen && !complete ? "disc" : "clean") << ") sr_shot:";
                    for (int l : leaf_ids) std::cout << " " << net[l]->sr_shot.read();
                    std::cout << "\n" << std::flush;
                }
                const bool exp_ps      = exp_ps_of(s);
                const bool was_discard = disc_seen && !complete;

                if (was_discard) {                            // internal post-select reject
                    ++internal;
                    if (!exp_ps) { ++ps_mismatch;
                        std::cout << "  trial " << trial << " shot " << s << ": discarded but expected_ps=0\n"; }
                    for (int k = 0; k < 80; ++k) wait();       // let reset/advance settle
                } else {                                       // clean completion -> compare
                    int mism = 0, miss = 0;
                    for (int d = 0; d < ndet; ++d) {
                        if (recon[d] < 0) ++miss;
                        else if (recon[d] != (expected_dets[s][d] ? 1 : 0)) ++mism;
                    }
                    ++checked; det_mismatch += mism; det_missing += miss;
                    if (mism == 0 && miss == 0) ++passed;
                    else std::cout << "  trial " << trial << " shot " << s
                                   << ": det_mismatch=" << mism << " det_missing=" << miss << "\n";
                    if (exp_ps) { ++ps_mismatch;
                        std::cout << "  trial " << trial << " shot " << s << ": ran clean but expected_ps=1\n"; }
                    if (!is_last) {                            // gap post-select -> discard + advance
                        h_gap_ps.write(true); wait(); h_gap_ps.write(false);
                        for (int k = 0; k < 80; ++k) wait();
                    }
                }

                if (is_last) {
                    // End the trial matching real operation: finish only an ACCEPTED (clean) attempt.
                    // The async host-finish in the loop also needs a SATURATING detector-emit round to
                    // ride (the copy-last wait rows); wr=0/normal saturates on the last real round, which
                    // is not re-emitted, so a late finish yields no out_finish. So finish is exercised on
                    // clean copy-last shots; a reject, or normal (no saturating emit), restarts via reset.
                    if (!was_discard && wait_rounds > 0) {
                        ++finish_attempts;
                        fin_seen = false;
                        h_finish.write(true); wait(); h_finish.write(false);
                        int fw = 0;
                        while (!fin_seen && fw < 12000) { wait(); ++fw; }
                        if (fin_seen) ++finishes;
                        else std::cout << "  trial " << trial << ": out_finish never fired after finish\n";
                        for (int k = 0; k < 10; ++k) wait();
                    } else {
                        rst.write(true); wait(); wait(); rst.write(false);   // reject / normal -> reset to restart
                    }
                }
            }
        }

        std::cout << "control_board_stim: trials=" << TRIALS << " shots/trial=" << N
                  << " checked=" << checked << " passed=" << passed
                  << " internal_ps=" << internal << " finishes=" << finishes << "/" << finish_attempts
                  << " det_mismatch=" << det_mismatch << " det_missing=" << det_missing
                  << " ps_mismatch=" << ps_mismatch << "\n";
        if (passed == checked && checked > 0 && ps_mismatch == 0 && finishes == finish_attempts)
            log_info("test", "control_board_stim PASS");
        else
            log_error("test", "control_board_stim MISMATCH");
        sc_stop();
    }

    // bind one ControlBoard to its Nets. Root host-controls are the shared h_* signals; every
    // other board's start/finish/gap and its parent-event bus are handled in the ctor.
    void bind_board(int id) {
        ControlBoard& b = *board[id];
        Nets& n = *net[id];
        b.clk(clk); b.rst(rst);
        b.ev_valid(n.ev_valid); b.ev_type(n.ev_type); b.ev_payload(n.ev_payload); b.ev_attempt(n.ev_attempt);
        b.start(id == root_id ? h_start : n.start);
        b.finish(id == root_id ? h_finish : n.finish);
        b.gap_post_select(id == root_id ? h_gap_ps : n.gap_ps);
        b.in_det(n.in_det); b.in_det_valid(n.in_det_valid);
        b.in_meas(n.in_meas); b.in_meas_valid(n.in_meas_valid); b.in_meas_finish(n.in_meas_finish);
        b.in_det_finish_child(n.in_det_fin_ch); b.in_raw_finish_child(n.in_raw_fin_ch);
        b.in_child_attempt(n.in_child_att); b.ps_in(n.ps_in); b.ps_in_attempt(n.ps_in_att);
        b.fwd_det(n.fwd_det); b.fwd_det_valid(n.fwd_det_valid); b.fwd_det_finish(n.fwd_det_finish);
        b.fwd_raw(n.fwd_raw); b.fwd_raw_valid(n.fwd_raw_valid); b.fwd_raw_finish(n.fwd_raw_finish);
        b.out_attempt(n.out_attempt); b.ps_out(n.ps_out); b.ps_out_attempt(n.ps_out_att);
        b.out_ev_valid(n.out_ev_valid); b.out_ev_type(n.out_ev_type);
        b.out_ev_payload(n.out_ev_payload); b.out_ev_attempt(n.out_ev_attempt);
        b.out_det(n.out_det); b.out_used(n.out_used); b.out_valid(n.out_valid);
        b.out_finish(n.out_finish); b.out_global_indexes(n.out_gidx);
        b.first_normal(n.first_normal); b.last_normal(n.last_normal);
        b.first_wait(n.first_wait); b.last_wait(n.last_wait); b.discard(n.discard);
        if (b.mmio_out_data.size() > 0) {
            b.mmio_out_data[0](n.mmio_data); b.mmio_out_valid[0](n.mmio_valid); b.cw_gen_finish[0](n.cw_finish);
        }
    }

    Sys(sc_module_name nm, const std::string& dir, int rid)
        : sc_module(nm), clk("clk", 10, SC_NS), rst("rst", true) {   // power-on reset: held at t=0
        const json man = detail::read_json(dir + "/manifest.json");
        ndet        = man["detectors"].get<int>();
        wait_rounds = man.value("wait_rounds", 0);
        recon.assign(ndet, -1);

        // ---- roles + topology per board ----
        for (const auto& be : man["boards"]) {
            const int id = be["board_id"].get<int>();
            board_ids.push_back(id);
            cfg[id] = load_board_config(dir, id);
            cfg[id].dcb.sync_fifo = 64; cfg[id].dcb.out_fifo = 64;   // generous for tree propagation
            if (cfg[id].is_leaf) leaf_ids.push_back(id);
            if (cfg[id].is_root) root_id = id;
            // connections.json: children (in loader order) + input-line port maps
            std::ifstream cj(dir + "/board" + std::to_string(id) + "/json/connections.json");
            if (cj) {
                json c; cj >> c;
                std::set<int> seen; std::vector<int> order;
                auto note = [&](int ch) { if (seen.insert(ch).second) order.push_back(ch); };
                for (const auto& dp : c["detector_ports"])
                    { meas_port[id]; det_port[id].push_back({dp["child"].get<int>(), dp["child_port"].get<int>()}); note(dp["child"].get<int>()); }
                for (const auto& mp : c["measurement_ports"])
                    { meas_port[id].push_back({mp["child"].get<int>(), mp["child_port"].get<int>()}); note(mp["child"].get<int>()); }
                children[id] = order;
                for (int ch : order) parent_of[ch] = id;
            }
        }
        // stage boards feeding the root ps_in (manifest stage_boards, excluding the root), in order
        if (man.contains("stage_boards")) {
            std::map<int,int> by_stage;   // stage_index -> board_id
            for (auto& kv : man["stage_boards"].items()) by_stage[std::stoi(kv.key())] = kv.value().get<int>();
            for (auto& kv : by_stage) if (kv.second != root_id) stage_ids.push_back(kv.second);
        }
        if (rid >= 0) root_id = rid;
        rn = net[root_id].get();  // set after nets built (below) — placeholder, fixed post-loop

        // ---- instantiate boards + nets + leaf StimReadouts ----
        for (int id : board_ids) {
            net[id]   = std::make_unique<Nets>();
            board[id] = std::make_unique<ControlBoard>(("b" + std::to_string(id)).c_str(), cfg[id]);
        }
        rn = net[root_id].get();
        for (int id : board_ids) bind_board(id);

        // event bus DOWN: a child's ev_* reads its parent's out_ev_* net (re-bind those ports).
        // Done by rebinding: ControlBoard ev_* ports were bound to the child's own ev_* nets in
        // bind_board; instead drive the child's ev_* nets from the parent via a small method.

        // leaf StimReadouts + loop
        for (int l : leaf_ids) {
            StimReadoutConfig scfg = load_stim_readout(dir, l);
            sr[l] = std::make_unique<StimReadout>(("sr" + std::to_string(l)).c_str(), scfg);
            Nets& n = *net[l];
            sr[l]->clk(clk); sr[l]->rst(rst);
            sr[l]->mmio_out_valid(n.mmio_valid); sr[l]->mmio_out_data(n.mmio_data); sr[l]->cw_gen_finish(n.cw_finish);
            // abort -> advance readout, in sync. Same event source as the MMIO: ORIGINATE
            // (monolithic root+leaf) generates it on out_ev; FORWARD (distributed leaf) relays ev.
            const bool from_out = (cfg[l].event_mode == BoardControl::ORIGINATE);
            sr[l]->ev_valid(from_out ? n.out_ev_valid : n.ev_valid);
            sr[l]->ev_type (from_out ? n.out_ev_type  : n.ev_type);
            sr[l]->out_meas(n.sr_meas); sr[l]->out_meas_valid(n.sr_valid); sr[l]->out_meas_finish(n.sr_finish);
            sr[l]->shot_reg(n.sr_shot);
        }

        // expected detectors from the ROOT. The post-select oracle is the OR of EVERY stage
        // board's expected_postselect: the board aborts a shot if ANY stage board rejects it
        // (each stage board's post_select feeds the root's abort), so a single-board oracle
        // would miss rejects that originate on another stage board (e.g. leaf board 0).
        expected_dets = load_stim_vectors(dir, root_id).expected_dets;
        if (man.contains("stage_boards")) {
            for (auto& kv : man["stage_boards"].items()) {
                const std::vector<Bits>& ps = load_stim_vectors(dir, kv.value().get<int>()).expected_postselect;
                for (int s = 0; s < (int)ps.size(); ++s) {
                    if ((int)expected_ps.size() <= s) expected_ps.push_back(Bits(1));
                    if (ps[s].size() && ps[s][0]) expected_ps[s][0] = 1;
                }
            }
        }

        SC_HAS_PROCESS(Sys);
        SC_METHOD(wire);
        for (int id : board_ids)
            sensitive << net[id]->fwd_det << net[id]->fwd_det_valid << net[id]->fwd_det_finish
                      << net[id]->fwd_raw << net[id]->fwd_raw_valid << net[id]->fwd_raw_finish
                      << net[id]->out_attempt << net[id]->ps_out << net[id]->ps_out_att;
        SC_METHOD(loopback);
        for (int l : leaf_ids) sensitive << net[l]->sr_meas << net[l]->sr_valid << net[l]->sr_finish;
        SC_METHOD(drive_events);
        for (int id : board_ids)
            sensitive << net[id]->out_ev_valid << net[id]->out_ev_type
                      << net[id]->out_ev_payload << net[id]->out_ev_attempt;
        SC_METHOD(monitor); sensitive << clk.posedge_event(); dont_initialize();
        SC_THREAD(run); sensitive << clk.posedge_event();
    }

    // event bus DOWN: each non-root board's ev_* = its parent's out_ev_*.
    void drive_events() {
        for (int id : board_ids) {
            if (id == root_id) continue;
            auto it = parent_of.find(id);
            if (it == parent_of.end()) continue;
            const Nets& p = *net[it->second];
            Nets& n = *net[id];
            n.ev_valid.write(p.out_ev_valid.read());
            n.ev_type.write(p.out_ev_type.read());
            n.ev_payload.write(p.out_ev_payload.read());
            n.ev_attempt.write(p.out_ev_attempt.read());
        }
    }
};

int sc_main(int argc, char* argv[]) {
    // Compiled example, relative to the repository root (run from there or pass a directory).
    const std::string dir = (argc > 1) ? argv[1]
        : "hardware/compiler/control_system/results/"
          "example_compiler_config__d1=3,d2=9,b=Y,p=0.001__osync=all,wr=3,wrow=copy-last,ps=flat";
    const int rid = (argc > 2) ? std::stoi(argv[2]) : -1;
    Sys sys("sys", dir, rid);
    sc_start();
    return sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL) ? 1 : 0;
}
