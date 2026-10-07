// control_board_loader.hpp — build a ControlBoard's BoardConfig + its leaf
// StimReadoutConfig from a compiled example dir (control-system compiler output).
//
// System-level: assembles a whole board from the per-board regfiles (via the
// detector-construct board_loader), the MMIO program (MMIO_instr_*/MMIO_cw_*), the
// leaf readout memories (sim_readout_*), and the roles/topology in manifest.json /
// connections.json. Used by the whole-system stim test to instantiate ControlBoards
// + StimReadouts directly from the compiler output.
#pragma once

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "control_system/board_config.hpp"
#include "control_system/cultiv_control/board_control.hpp"   // BoardControl::{ORIGINATE,FORWARD,...}
#include "control_system/detector_construct/board_loader.hpp"  // load_dcb_config, load_mem, detail::*
#include "control_system/stim_readout.hpp"

namespace emu {

// leaf measurement replay memories (sim_readout_meas / sim_readout_valid). Returns
// an empty config (shots==0) for a non-leaf board (no readout mems).
inline StimReadoutConfig load_stim_readout(const std::string& example_dir, int board_id) {
    const std::string bdir = example_dir + "/board" + std::to_string(board_id);
    StimReadoutConfig sr;
    std::ifstream f(bdir + "/json/sim_readout_meas.json");
    if (!f) return sr;
    nlohmann::json m;
    f >> m;
    sr.m     = m["word_width"].get<int>();
    sr.depth = m["depth"].get<int>();
    sr.shots = m["shots"].get<int>();
    sr.meas  = load_mem(bdir + "/mem/sim_readout_meas.mem",  sr.m, "bin");
    sr.valid = load_mem(bdir + "/mem/sim_readout_valid.mem", sr.m, "bin");
    return sr;
}

// full BoardConfig for one board (roles, dcb, child topology, leaf MMIO program).
inline BoardConfig load_board_config(const std::string& example_dir, int board_id) {
    using nlohmann::json;
    const std::string bdir = example_dir + "/board" + std::to_string(board_id);
    const json man = detail::read_json(example_dir + "/manifest.json");
    const json be  = detail::board_entry(man, board_id);

    std::set<std::string> rf;
    for (const auto& r : be["regfiles"]) rf.insert(r.get<std::string>());

    BoardConfig cfg;
    cfg.is_root        = (be["type"].get<std::string>() == "root");
    cfg.is_leaf        = rf.count("qubit_scope") > 0;
    cfg.has_postselect = rf.count("postselect") > 0;
    cfg.event_mode     = cfg.is_root ? BoardControl::ORIGINATE : BoardControl::FORWARD;
    cfg.data_src       = cfg.is_leaf ? BoardControl::INTERNAL  : BoardControl::EXTERNAL;
    cfg.payload_w      = 2;   // reserved ev_payload width (matches ControlBoard default)

    cfg.dcb = load_dcb_config(example_dir, board_id);

    // child topology (routers/root): from connections.json. Children in port order;
    // child_dw = detector-output lines fed up, child_raw = raw-forward lines.
    std::ifstream cj(bdir + "/json/connections.json");
    if (cj) {
        json c;
        cj >> c;
        std::vector<int> order;
        std::map<int, int> dw, raw;
        std::set<int> seen;
        auto note = [&](int ch) { if (seen.insert(ch).second) order.push_back(ch); };
        for (const auto& dp : c["detector_ports"])    { int ch = dp["child"].get<int>(); note(ch); dw[ch]++; }
        for (const auto& mp : c["measurement_ports"]) { int ch = mp["child"].get<int>(); note(ch); raw[ch]++; }
        for (int ch : order) {
            cfg.child_dw.push_back(dw.count(ch)  ? dw[ch]  : 0);
            cfg.child_raw.push_back(raw.count(ch) ? raw[ch] : 0);
        }
    }

    // nps = # of OTHER stage boards routing post_select straight to this root (the
    // root's own postselect, if any, is the own slot in the ps bus, not a fast path).
    if (cfg.is_root && man.contains("stage_boards")) {
        std::set<int> stage_ids;
        for (const auto& kv : man["stage_boards"].items()) stage_ids.insert(kv.value().get<int>());
        stage_ids.erase(board_id);
        cfg.nps = static_cast<int>(stage_ids.size());
    }

    // leaf: one PhysicalMMIO (P=1) with its instr/cw program loaded
    if (cfg.is_leaf) {
        const json ij  = detail::read_json(bdir + "/json/MMIO_instr_0.json");
        const json cwj = detail::read_json(bdir + "/json/MMIO_cw_0.json");
        MmioConfig mc;
        mc.instr_depth = ij["depth"].get<int>();
        mc.cw_depth    = cwj["depth"].get<int>();
        mc.data_w      = cwj["word_width"].get<int>();
        mc.wt_w        = ij["wt_w"].get<int>();
        mc.payload_w   = cfg.payload_w;
        mc.instr_words = load_mem(bdir + "/mem/MMIO_instr_0.mem", ij["word_width"].get<int>(),  "bin");
        mc.cw_words    = load_mem(bdir + "/mem/MMIO_cw_0.mem",    cwj["word_width"].get<int>(), "bin");
        cfg.mmios.push_back(std::move(mc));
    }

    return cfg;
}

}  // namespace emu
