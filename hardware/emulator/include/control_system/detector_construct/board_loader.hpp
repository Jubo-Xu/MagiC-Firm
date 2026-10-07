// board_loader.hpp — build a DcbConfig + stimulus from a compiled results/<example>.
//
// Turns the compiler's on-disk artifacts (manifest.json + board<id>/mem/*.mem +
// the gen_stim_vectors output) into in-memory structures, so the test harness
// has no parsing logic. Header-only (pulls in nlohmann/json), included by tests.
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "common.hpp"   // index_width
#include "control_system/detector_construct/detector_construct_block.hpp"
#include "lib/memfile.hpp"
#include "signals.hpp"

namespace emu {

// stimulus vectors for one board (rows = shots*T for meas, shots for dets)
struct BoardVectors {
    int shots = 0, T = 0, m = 0, ndet = 0;
    std::vector<Bits> in_meas;             // shots*T words (m bits)
    std::vector<Bits> in_valid;            // shots*T words (m bits)
    std::vector<Bits> expected_dets;       // shots words (ndet bits) — root only
    std::vector<Bits> expected_postselect; // shots words (1 bit) — stage boards only
};

namespace detail {
inline nlohmann::json read_json(const std::string& path) {
    std::ifstream f(path);
    nlohmann::json j;
    f >> j;
    return j;
}
inline nlohmann::json board_entry(const nlohmann::json& man, int board_id) {
    for (const auto& b : man["boards"])
        if (b["board_id"].get<int>() == board_id) return b;
    throw std::runtime_error("board_id not found in manifest");
}
}  // namespace detail

inline DcbConfig load_dcb_config(const std::string& example_dir, int board_id) {
    using nlohmann::json;
    json man = detail::read_json(example_dir + "/manifest.json");
    json be  = detail::board_entry(man, board_id);
    const std::string fmt = man["mem_format"].get<std::string>();
    const int T   = man["meas_times"].get<int>();
    const int ndt = man["det_times"].get<int>();
    const std::string mdir = example_dir + "/board" + std::to_string(board_id) + "/mem/";

    DcbConfig c;
    c.m       = be["m_phys"].get<int>();
    c.k       = be["kernels_avail"].get<int>();
    c.n       = be.value("n_cap", 0);
    c.h       = be.value("h_cap", 0);
    c.idx_w   = be.value("global_index_width", 0);
    c.is_root = (be["type"].get<std::string>() == "root");
    c.copy_last = (man.value("wait_row", std::string("normal")) == "copy-last");
    c.has_postselect = be.contains("postselect_stages");
    // OutputSync is the output stage on every board that carries an output_sync regfile
    // (all boards with --output-sync all; root+stage otherwise).
    if (be.contains("regfiles"))
        for (const auto& r : be["regfiles"])
            if (r.get<std::string>() == "output_sync") c.has_osync = true;
    c.d_in    = be.value("detector_input_ports", 0);
    c.raw_out = (be.contains("raw_out_cap") && !be["raw_out_cap"].is_null())
                    ? be["raw_out_cap"].get<int>() : 0;

    const int d_out = c.d_out();
    const int iw    = index_width(c.m);

    if (c.k > 0) {
        c.sync_words = load_mem(mdir + "sync.mem", c.m, fmt);
        for (int i = 0; i < c.k; ++i) {
            KernelConfig kc;
            auto sel = load_mem(mdir + "k" + std::to_string(i) + "_selector.mem", iw, fmt);
            kc.selector_indexes = pack_words(sel, iw);
            kc.core_words = load_mem(mdir + "k" + std::to_string(i) + "_core.mem", c.h * (c.n + 1), fmt);
            c.kernels.push_back(std::move(kc));
        }
    }
    if (c.raw_out > 0) {
        auto raw = load_mem(mdir + "raw_selector.mem", iw, fmt);
        c.raw_selector_indexes = pack_words(raw, iw);
    }
    if (c.has_output_sync())
        c.output_sync_words = load_mem(mdir + "output_sync.mem", d_out, fmt);
    if (c.is_root) {
        c.hw_width = man.value("global_index_hw_width", c.idx_w);
        c.stride   = be.value("global_index_stride", 0);
        c.sentinel = be.value("global_index_sentinel", c.idx_w > 0 ? (1 << c.idx_w) - 1 : 0);
        c.global_index_words = load_mem(mdir + "global_index.mem", d_out * c.idx_w, fmt);
        c.round_marker_words = load_mem(mdir + "round_marker.mem", 3, fmt);
    }
    if (c.has_postselect)
        c.postselect_words = load_mem(mdir + "postselect.mem", d_out, fmt);

    (void)ndt;  // depth of the output regfiles == ndt (checked implicitly by row counts)
    return c;
}

inline BoardVectors load_stim_vectors(const std::string& example_dir, int board_id) {
    using nlohmann::json;
    const std::string bdir = example_dir + "/board" + std::to_string(board_id);
    const std::string mdir = bdir + "/mem/";
    BoardVectors v;

    std::ifstream imj(bdir + "/json/in_meas.json");   // leaves (and monolithic) only
    if (imj) {
        json im; imj >> im;
        v.shots = im["shots"].get<int>();
        v.T     = im["T"].get<int>();
        v.m     = im["word_width"].get<int>();
        v.in_meas  = load_mem(mdir + "in_meas.mem",  v.m, "bin");
        v.in_valid = load_mem(mdir + "in_valid.mem", v.m, "bin");
    }
    std::ifstream edj(bdir + "/json/expected_dets.json");   // root (and monolithic) only
    if (edj) {
        json ed; edj >> ed;
        v.ndet = ed["word_width"].get<int>();
        v.expected_dets = load_mem(mdir + "expected_dets.mem", v.ndet, "bin");
        if (v.shots == 0) v.shots = ed["shots"].get<int>();
    }
    std::ifstream psj(bdir + "/json/expected_postselect.json");   // stage boards only
    if (psj) {
        json ps; psj >> ps;
        v.expected_postselect = load_mem(mdir + "expected_postselect.mem", 1, "bin");
        if (v.shots == 0) v.shots = ps["shots"].get<int>();
    }
    return v;
}

}  // namespace emu
