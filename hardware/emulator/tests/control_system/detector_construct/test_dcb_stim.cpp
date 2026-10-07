// test_dcb_stim.cpp — replay stim vectors through the board tree and check the
// root output against the ground truth. General: one DetectorConstructBlock per
// board, wired per connections.json; handles monolithic (1-board tree) and
// distributed alike.  Correctness only — no per-hop transport delay is modelled
// (that's a later full-system sim); the blocks' own latencies + FIFOs settle.
//
//   ./test_dcb_stim <example_dir> [--shots N] [--input-gap G] [--shot-gap G]
#include <systemc>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "control_system/detector_construct/board_loader.hpp"
#include "log.hpp"
#include "signals.hpp"

using namespace sc_core;
using namespace emu;
using nlohmann::json;

struct Args { std::string dir; int shots = -1, input_gap = 0, shot_gap = 0; };

// all of a block's port nets
struct BoardSig {
    sc_signal<Bits> in_meas, in_valid, in_meas_finish, in_det, in_det_valid, in_det_finish;
    sc_signal<Bits> fwd_det, fwd_det_valid, fwd_raw, fwd_raw_valid;
    sc_signal<bool> fwd_det_finish, fwd_raw_finish;
    sc_signal<Bits> out_det, out_used, out_global_indexes;
    sc_signal<bool> out_valid, out_finish, post_select;
    sc_signal<bool> first_normal, last_normal, first_wait, last_wait;
};
struct PortMap { int child, child_port; };
struct Conn    { std::vector<PortMap> meas, det; };   // parent input port -> (child, child output port)

SC_MODULE(Sys) {
    sc_clock        clk;
    sc_signal<bool> rst;

    Args args;
    int  T = 0, ndet = 0, root_id = -1, wait_rounds = 0;
    bool copy_last = false;
    std::vector<int>                                   board_ids, leaf_ids;
    std::map<int, DcbConfig>                           cfg;
    std::map<int, BoardVectors>                        vec;
    std::map<int, std::unique_ptr<BoardSig>>           sig;
    std::map<int, std::unique_ptr<DetectorConstructBlock>> blk;
    std::map<int, Conn>                                conn;
    std::vector<int> recon;
    std::vector<int>    stage_ids;          // boards with a Postselect block
    std::map<int, bool> ps_fired;           // per stage board: did post_select fire this shot?
    std::set<int>       finish_idx;         // per shot: detector indices seen on a root out_finish round
    bool                finish_fired = false;  // per shot: out_finish pulsed at least once
    std::set<int>       wait_idx;           // per shot: global indices >= ndet (offset-accumulated wait dets)
    int mk_first_normal = 0, mk_last_normal = 0, mk_first_wait = 0, mk_last_wait = 0;  // per-shot marker counts
    int mk_lastwait_nofinish = 0;           // per-shot: last_wait pulses NOT coincident with out_finish
    int total_mismatch = 0, total_missing = 0, total_ps_mismatch = 0, shots_run = 0;
    int shots_finish_ok = 0;                // shots where out_finish fired at least once
    int shots_marker_ok = 0;                // shots where the four markers fired as expected
    int total_marker_bad = 0;               // shots with a wrong marker count / missing wait index

    // combinational tree wiring: each parent input port <- its child's forward bus
    void wire() {
        for (auto& kv : conn) {
            const int p = kv.first; const Conn& c = kv.second;
            Bits im(cfg[p].m), iv(cfg[p].m), imf(cfg[p].m);
            Bits id(cfg[p].d_in), idv(cfg[p].d_in), idf(cfg[p].d_in);
            for (int i = 0; i < (int)c.meas.size() && i < cfg[p].m; ++i) {
                const BoardSig& s = *sig[c.meas[i].child];
                const Bits f = s.fwd_raw.read(), fv = s.fwd_raw_valid.read();
                const int cp = c.meas[i].child_port;
                im[i]  = (cp < (int)f.size())  ? f[cp]  : 0;
                iv[i]  = (cp < (int)fv.size()) ? fv[cp] : 0;
                imf[i] = s.fwd_raw_finish.read() ? 1 : 0;   // broadcast child's single raw finish
            }
            for (int j = 0; j < (int)c.det.size() && j < cfg[p].d_in; ++j) {
                const BoardSig& s = *sig[c.det[j].child];
                const Bits f = s.fwd_det.read(), fv = s.fwd_det_valid.read();
                const int cp = c.det[j].child_port;
                id[j]  = (cp < (int)f.size())  ? f[cp]  : 0;
                idv[j] = (cp < (int)fv.size()) ? fv[cp] : 0;
                idf[j] = s.fwd_det_finish.read() ? 1 : 0;   // broadcast child's single det finish
            }
            sig[p]->in_meas.write(im);  sig[p]->in_valid.write(iv);  sig[p]->in_meas_finish.write(imf);
            sig[p]->in_det.write(id);   sig[p]->in_det_valid.write(idv); sig[p]->in_det_finish.write(idf);
        }
    }

    // collect the root output each cycle -> recon[global_index] = detector
    void monitor() {
        // postselect: OR each stage board's post_select over the shot
        for (int b : stage_ids)
            if (sig[b]->post_select.read()) ps_fired[b] = true;

        const BoardSig& s = *sig[root_id];
        if (!s.out_valid.read()) return;
        const Bits used = s.out_used.read(), det = s.out_det.read(), gi = s.out_global_indexes.read();
        const bool fin = s.out_finish.read();
        if (fin) finish_fired = true;   // out_finish pulsed (rides last real round in wr=0, wait row in copy-last)
        const int dout = cfg[root_id].d_out(), iw = cfg[root_id].hw_width;   // output datapath width
        for (int l = 0; l < dout; ++l) {
            if (!used[l]) continue;
            uint32_t idx = extract(gi, (std::size_t)l * iw, iw);
            if (idx < recon.size()) recon[idx] = det[l] ? 1 : 0;
            else if (idx >= (uint32_t)ndet && idx < (uint32_t)ndet + 4096)
                wait_idx.insert((int)idx);                                // offset-accumulated wait detector
            if (fin && idx < recon.size()) finish_idx.insert((int)idx);   // detectors carried on a finish round
        }
        // boundary markers (root; each gated by out_valid inside the block)
        if (s.first_normal.read()) ++mk_first_normal;
        if (s.last_normal.read())  ++mk_last_normal;
        if (s.first_wait.read())   ++mk_first_wait;
        if (s.last_wait.read())  { ++mk_last_wait; if (!fin) ++mk_lastwait_nofinish; }
    }

    void run() {
        const int flush  = 12 * T + 200;   // generous for multi-hop propagation
        const int nshots = (args.shots > 0) ? std::min(args.shots, vec[root_id].shots) : vec[root_id].shots;

        for (int s = 0; s < nshots; ++s) {
            rst.write(true);
            for (int leaf : leaf_ids) {
                sig[leaf]->in_meas.write(Bits(cfg[leaf].m)); sig[leaf]->in_valid.write(Bits(cfg[leaf].m));
                sig[leaf]->in_meas_finish.write(Bits(cfg[leaf].m));
            }
            wait(); wait();
            rst.write(false);
            std::fill(recon.begin(), recon.end(), -1);
            for (int b : stage_ids) ps_fired[b] = false;
            finish_idx.clear(); wait_idx.clear(); finish_fired = false;
            mk_first_normal = mk_last_normal = mk_first_wait = mk_last_wait = mk_lastwait_nofinish = 0;

            // Real rounds. wr=0: finish rides the last real round (= last_normal). copy-last:
            // real rounds carry NO finish; it rides the appended saturating wait rounds below.
            for (int t = 0; t < T; ++t) {
                const bool fin = (!copy_last) && (t == T - 1);
                for (int leaf : leaf_ids) {
                    sig[leaf]->in_meas.write(vec[leaf].in_meas[s * T + t]);
                    sig[leaf]->in_valid.write(vec[leaf].in_valid[s * T + t]);
                    sig[leaf]->in_meas_finish.write(fin ? vec[leaf].in_valid[s * T + t] : Bits(cfg[leaf].m));
                }
                wait();
                for (int leaf : leaf_ids) {
                    sig[leaf]->in_meas.write(Bits(cfg[leaf].m)); sig[leaf]->in_valid.write(Bits(cfg[leaf].m));
                    sig[leaf]->in_meas_finish.write(Bits(cfg[leaf].m));
                }
                for (int g = 0; g < args.input_gap; ++g) wait();
            }
            // copy-last: drive wait_rounds extra saturating rounds (re-feed the last real round so the
            // saturating regfile row emits). Finish rides the LAST one -> last_wait fires on the wait row.
            for (int w = 0; copy_last && w < wait_rounds; ++w) {
                const bool fin = (w == wait_rounds - 1);
                for (int leaf : leaf_ids) {
                    sig[leaf]->in_meas.write(vec[leaf].in_meas[s * T + (T - 1)]);
                    sig[leaf]->in_valid.write(vec[leaf].in_valid[s * T + (T - 1)]);
                    sig[leaf]->in_meas_finish.write(fin ? vec[leaf].in_valid[s * T + (T - 1)] : Bits(cfg[leaf].m));
                }
                wait();
                for (int leaf : leaf_ids) {
                    sig[leaf]->in_meas.write(Bits(cfg[leaf].m)); sig[leaf]->in_valid.write(Bits(cfg[leaf].m));
                    sig[leaf]->in_meas_finish.write(Bits(cfg[leaf].m));
                }
                for (int g = 0; g < args.input_gap; ++g) wait();
            }
            for (int f = 0; f < flush; ++f) wait();

            int mism = 0, miss = 0;
            for (int d = 0; d < ndet; ++d) {
                if (recon[d] < 0) ++miss;
                else if (recon[d] != (vec[root_id].expected_dets[s][d] ? 1 : 0)) ++mism;
            }
            total_mismatch += mism; total_missing += miss; ++shots_run;
            if (mism || miss)
                std::cout << "  shot " << s << ": mismatch=" << mism << " missing=" << miss << "\n";

            // finish: out_finish must pulse once per shot (last real round in wr=0, wait row in copy-last)
            if (finish_fired) ++shots_finish_ok;
            else std::cout << "  shot " << s << ": out_finish never fired\n";

            // markers: first_normal/last_normal fire once each; first_wait/last_wait only for copy-last,
            // last_wait must coincide with out_finish; wait rounds emit offset indices ndet + w*stride.
            const int exp_fw = copy_last ? 1 : 0;
            bool marker_ok = (mk_first_normal == 1 && mk_last_normal == 1 &&
                              mk_first_wait == exp_fw && mk_last_wait == exp_fw &&
                              mk_lastwait_nofinish == 0);
            if (copy_last)
                for (int w = 0; w < wait_rounds; ++w)
                    if (!wait_idx.count(ndet + w * cfg[root_id].stride)) marker_ok = false;
            if (marker_ok) ++shots_marker_ok;
            else {
                ++total_marker_bad;
                std::cout << "  shot " << s << ": marker fn=" << mk_first_normal << " ln=" << mk_last_normal
                          << " fw=" << mk_first_wait << " lw=" << mk_last_wait
                          << " lw_nofin=" << mk_lastwait_nofinish << " wait_idx=" << wait_idx.size() << "\n";
            }

            // per stage board: emulator's OR of post_select vs the OR of its postselect detectors
            for (int b : stage_ids) {
                bool emu = ps_fired[b];
                bool exp = !vec[b].expected_postselect.empty() && vec[b].expected_postselect[s][0];
                if (emu != exp) {
                    ++total_ps_mismatch;
                    std::cout << "  shot " << s << " board" << b << ": postselect emu=" << emu
                              << " exp=" << exp << "\n";
                }
            }

            for (int g = 0; g < args.shot_gap; ++g) wait();
        }

        std::cout << "boards=" << board_ids.size() << " leaves=" << leaf_ids.size()
                  << " root=" << root_id << " stage_boards=" << stage_ids.size()
                  << "  shots=" << shots_run << "  det_mismatch=" << total_mismatch
                  << " det_missing=" << total_missing << " postselect_mismatch=" << total_ps_mismatch
                  << " finish_ok=" << shots_finish_ok << "/" << shots_run
                  << " marker_ok=" << shots_marker_ok << "/" << shots_run
                  << (copy_last ? " [copy-last]" : "") << "\n";
        if (total_mismatch == 0 && total_missing == 0 && total_ps_mismatch == 0 &&
            shots_finish_ok == shots_run && shots_marker_ok == shots_run)
             log_info("test", "dcb_stim PASS");
        else log_error("test", "dcb_stim MISMATCH");
        sc_stop();
    }

    Sys(sc_module_name nm, const Args& a) : sc_module(nm), clk("clk", 10, SC_NS), args(a) {
        std::ifstream mf(a.dir + "/manifest.json"); json man; mf >> man;
        T = man["meas_times"].get<int>();
        ndet = man["detectors"].get<int>();
        wait_rounds = man.value("wait_rounds", 0);
        copy_last = (man.value("wait_row", std::string("normal")) == "copy-last");
        recon.assign(ndet, -1);

        // load configs + vectors
        for (const auto& b : man["boards"]) {
            const int id = b["board_id"].get<int>();
            board_ids.push_back(id);
            std::set<std::string> rf; for (const auto& r : b["regfiles"]) rf.insert(r.get<std::string>());
            cfg[id] = load_dcb_config(a.dir, id);
            cfg[id].sync_fifo = 64; cfg[id].out_fifo = 64;      // generous for tree propagation
            vec[id] = load_stim_vectors(a.dir, id);
            if (rf.count("qubit_scope"))          leaf_ids.push_back(id);
            if (rf.count("postselect"))           stage_ids.push_back(id);
            if (b["type"].get<std::string>() == "root") root_id = id;
        }

        // instantiate + bind blocks
        for (int id : board_ids) {
            sig[id] = std::make_unique<BoardSig>();
            BoardSig& s = *sig[id];
            blk[id] = std::make_unique<DetectorConstructBlock>(("blk" + std::to_string(id)).c_str(), cfg[id]);
            DetectorConstructBlock& B = *blk[id];
            B.clk(clk); B.rst(rst);
            B.in_meas(s.in_meas); B.in_valid(s.in_valid); B.in_meas_finish(s.in_meas_finish);
            B.in_det(s.in_det); B.in_det_valid(s.in_det_valid); B.in_det_finish(s.in_det_finish);
            B.fwd_det(s.fwd_det); B.fwd_det_valid(s.fwd_det_valid); B.fwd_det_finish(s.fwd_det_finish);
            B.fwd_raw(s.fwd_raw); B.fwd_raw_valid(s.fwd_raw_valid); B.fwd_raw_finish(s.fwd_raw_finish);
            B.out_det(s.out_det); B.out_used(s.out_used); B.out_valid(s.out_valid); B.out_finish(s.out_finish);
            B.out_global_indexes(s.out_global_indexes); B.post_select(s.post_select);
            B.first_normal(s.first_normal); B.last_normal(s.last_normal);
            B.first_wait(s.first_wait);     B.last_wait(s.last_wait);
        }

        // load tree wiring
        for (int id : board_ids) {
            std::ifstream cj(a.dir + "/board" + std::to_string(id) + "/json/connections.json");
            if (!cj) continue;
            json c; cj >> c;
            Conn cc;
            for (const auto& mp : c["measurement_ports"]) cc.meas.push_back({mp["child"].get<int>(), mp["child_port"].get<int>()});
            for (const auto& dp : c["detector_ports"])    cc.det.push_back({dp["child"].get<int>(), dp["child_port"].get<int>()});
            conn[id] = std::move(cc);
        }

        SC_HAS_PROCESS(Sys);
        SC_METHOD(wire);
        for (int id : board_ids)
            sensitive << sig[id]->fwd_det << sig[id]->fwd_det_valid << sig[id]->fwd_det_finish
                      << sig[id]->fwd_raw << sig[id]->fwd_raw_valid << sig[id]->fwd_raw_finish;
        SC_METHOD(monitor); sensitive << clk.posedge_event(); dont_initialize();
        SC_THREAD(run);     sensitive << clk.posedge_event();
    }
};

int sc_main(int argc, char* argv[]) {
    Args a;
    // Compiled example, relative to the repository root (run from there or pass a directory).
    a.dir = (argc > 1) ? argv[1]
        : "hardware/compiler/control_system/results/monolithic__d1=3,d2=9,b=Y,p=0.001";
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--shots" && i + 1 < argc)          a.shots = std::stoi(argv[++i]);
        else if (s == "--input-gap" && i + 1 < argc) a.input_gap = std::stoi(argv[++i]);
        else if (s == "--shot-gap" && i + 1 < argc)  a.shot_gap = std::stoi(argv[++i]);
    }
    Sys sys("sys", a);
    sc_start();
    int errors = sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL);
    return errors ? 1 : 0;
}
