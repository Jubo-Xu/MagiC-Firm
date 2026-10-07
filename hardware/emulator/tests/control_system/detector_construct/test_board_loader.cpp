// test_board_loader.cpp — sanity-check the loader against a compiled example.
//
// Loads a board's DcbConfig + stimulus and prints/asserts the shapes. Not a
// SystemC simulation (uses sc_main only as the entry point).
//   ./test_board_loader [example_dir] [board_id]
#include <systemc>

#include <iostream>
#include <string>

#include "control_system/detector_construct/board_loader.hpp"

using namespace emu;

// Compiled example, relative to the repository root (run from there or pass a directory).
static const char* kDefaultDir =
    "hardware/compiler/control_system/results/monolithic__d1=3,d2=9,b=Y,p=0.001";

static int dim0(const std::vector<Bits>& v) { return static_cast<int>(v.size()); }
static int dim1(const std::vector<Bits>& v) { return v.empty() ? 0 : static_cast<int>(v[0].size()); }

int sc_main(int argc, char* argv[]) {
    std::string dir = (argc > 1) ? argv[1] : kDefaultDir;
    int board_id    = (argc > 2) ? std::stoi(argv[2]) : 0;

    DcbConfig c = load_dcb_config(dir, board_id);
    BoardVectors v = load_stim_vectors(dir, board_id);

    std::cout << "--- DcbConfig (board " << board_id << ") ---\n";
    std::cout << "  m=" << c.m << " k=" << c.k << " n=" << c.n << " h=" << c.h
              << " idx_w=" << c.idx_w << " is_root=" << c.is_root
              << " has_postselect=" << c.has_postselect << " d_in=" << c.d_in
              << " raw_out=" << c.raw_out << " d_out=" << c.d_out() << "\n";
    std::cout << "  sync_words   " << dim0(c.sync_words) << " x " << dim1(c.sync_words) << "\n";
    std::cout << "  kernels      " << c.kernels.size()
              << "  (k0 selector=" << c.kernels[0].selector_indexes.size() << "b"
              << " core=" << dim0(c.kernels[0].core_words) << "x" << dim1(c.kernels[0].core_words) << ")\n";
    std::cout << "  output_sync  " << dim0(c.output_sync_words)  << " x " << dim1(c.output_sync_words)  << "\n";
    std::cout << "  global_index " << dim0(c.global_index_words) << " x " << dim1(c.global_index_words) << "\n";
    std::cout << "  postselect   " << dim0(c.postselect_words)   << " x " << dim1(c.postselect_words)   << "\n";
    std::cout << "--- stimulus ---\n";
    std::cout << "  shots=" << v.shots << " T=" << v.T << " m=" << v.m << " ndet=" << v.ndet << "\n";
    std::cout << "  in_meas       " << dim0(v.in_meas)  << " x " << dim1(v.in_meas)  << "\n";
    std::cout << "  in_valid      " << dim0(v.in_valid) << " x " << dim1(v.in_valid) << "\n";
    std::cout << "  expected_dets " << dim0(v.expected_dets) << " x " << dim1(v.expected_dets) << "\n";

    // --- assertions (monolithic d3d9) ---
    bool ok = true;
    auto chk = [&](bool cond, const char* what) { if (!cond) { ok = false; std::cout << "  FAIL: " << what << "\n"; } };
    chk(c.m == 85 && c.k == 92 && c.n == 4 && c.h == 2 && c.idx_w == 10, "params");
    chk(c.is_root && c.has_postselect && c.d_in == 0 && c.raw_out == 0 && c.d_out() == 92, "roles");
    chk(dim0(c.sync_words) == v.T && dim1(c.sync_words) == c.m, "sync shape");
    chk((int)c.kernels.size() == c.k, "kernel count");
    chk(dim1(c.kernels[0].core_words) == c.h * (c.n + 1) && dim0(c.kernels[0].core_words) == v.T, "core shape");
    chk((int)c.kernels[0].selector_indexes.size() == c.n * index_width(c.m), "selector width");
    chk(dim1(c.output_sync_words) == c.d_out(), "output_sync width");
    chk(dim1(c.global_index_words) == c.d_out() * c.idx_w, "global_index width");
    chk(dim1(c.postselect_words) == c.d_out(), "postselect width");
    chk(dim0(v.in_meas) == v.shots * v.T && dim1(v.in_meas) == c.m, "in_meas shape");
    chk(dim0(v.expected_dets) == v.shots && dim1(v.expected_dets) == 631, "expected_dets shape");

    std::cout << (ok ? "board_loader PASS\n" : "board_loader FAIL\n");
    return ok ? 0 : 1;
}
