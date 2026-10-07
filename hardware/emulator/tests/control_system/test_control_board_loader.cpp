// test_control_board_loader.cpp — sanity-check the compiled-dir -> config loaders.
//   ./test_control_board_loader <example_dir> [board_id]
#include <systemc>

#include <iostream>
#include <string>

#include "control_system/control_board_loader.hpp"

using namespace emu;

int sc_main(int argc, char* argv[]) {
    // Compiled example, relative to the repository root (run from there or pass a directory).
    const std::string dir = (argc > 1) ? argv[1]
        : "hardware/compiler/control_system/results/"
          "monolithic__d1=3,d2=9,b=Y,p=0.001__osync=all,wr=0,wrow=normal,ps=flat";
    const int bid = (argc > 2) ? std::stoi(argv[2]) : 0;

    const BoardConfig bc = load_board_config(dir, bid);
    const StimReadoutConfig sr = load_stim_readout(dir, bid);

    std::cout << "board" << bid
              << ": is_root=" << bc.is_root << " is_leaf=" << bc.is_leaf
              << " has_ps=" << bc.has_postselect
              << " event_mode=" << bc.event_mode << " data_src=" << bc.data_src
              << " nchild=" << bc.nchild() << " nps=" << bc.nps << " m_ps=" << bc.m_ps() << "\n";
    std::cout << "  dcb: m=" << bc.dcb.m << " d_in=" << bc.dcb.d_in << " k=" << bc.dcb.k
              << " raw_out=" << bc.dcb.raw_out << " is_root=" << bc.dcb.is_root << "\n";

    bool ok = true;
    if (bc.is_leaf) {
        const MmioConfig& mc = bc.mmios.at(0);
        std::cout << "  mmio: instr_depth=" << mc.instr_depth << " cw_depth=" << mc.cw_depth
                  << " data_w=" << mc.data_w << " wt_w=" << mc.wt_w
                  << " instr_words=" << mc.instr_words.size() << " cw_words=" << mc.cw_words.size() << "\n";
        std::cout << "  readout: m=" << sr.m << " shots=" << sr.shots << " depth=" << sr.depth
                  << " meas=" << sr.meas.size() << " valid=" << sr.valid.size() << "\n";
        ok &= (int)mc.instr_words.size() == mc.instr_depth;
        ok &= (int)mc.cw_words.size()    == mc.cw_depth;
        ok &= (int)sr.meas.size()  == sr.shots * sr.depth;
        ok &= (int)sr.valid.size() == sr.shots * sr.depth;
        ok &= mc.instr_words.empty() ? false : (int)mc.instr_words[0].size() == mc.instr_w();
        ok &= sr.m == bc.dcb.m;   // readout width == board measurement width
    }
    std::cout << (ok ? "control_board_loader OK" : "control_board_loader MISMATCH") << "\n";
    return ok ? 0 : 1;
}
