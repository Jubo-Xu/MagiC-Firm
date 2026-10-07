// test_root_output_sync.cpp — unit test for RootOutputSync.
//
// Reuses the OutputSync scenario (d=3) and adds a global-index word per round
// (regfile index_width=4, hw_width=4, no wait offset) plus a round_marker word.
// Checks that (out_used, out_det, out_global_indexes) come out aligned per round,
// that unused slots read all-ones (sentinel), that out_global_indexes is all-zero
// when out_valid=0, and that first_normal/last_normal pulse on the marked rounds.
#include <systemc>

#include <initializer_list>
#include <tuple>
#include <vector>

#include "control_system/detector_construct/root_output_sync.hpp"
#include "signals.hpp"
#include "log.hpp"

using namespace sc_core;
using emu::Bits;

static Bits mk(std::initializer_list<int> bits) {
    Bits b(bits.size());
    std::size_t i = 0;
    for (int x : bits) b[i++] = x ? 1 : 0;
    return b;
}
// pack per-line indices, each `iw` bits, line i at bits [i*iw +: iw]
static Bits gipack(std::initializer_list<int> vals, int iw) {
    Bits b(vals.size() * iw);
    int i = 0;
    for (int v : vals) { for (int k = 0; k < iw; ++k) b[i * iw + k] = (v >> k) & 1; ++i; }
    return b;
}
// round_marker word: {is_first_normal(bit0), is_last_normal(bit1), is_wait(bit2)}
static Bits rm3(int first, int last, int wait) {
    Bits b(3); b[0] = first ? 1 : 0; b[1] = last ? 1 : 0; b[2] = wait ? 1 : 0; return b;
}

SC_MODULE(Tb) {
    static constexpr int D = 3, D_FIFO = 4, IW = 4, HW = 4, STRIDE = 0, SENT = (1 << IW) - 1;

    sc_clock                clk;
    sc_signal<bool>         rst;
    sc_signal<Bits>         in_det, in_valid, in_finish, sync_mask, global_indexes, round_marker;
    sc_signal<Bits>         out_det, out_used, out_global_indexes;
    sc_signal<bool>         out_valid, out_finish, first_normal, last_normal, first_wait, last_wait;
    sc_signal<uint32_t>     pc;

    emu::RootOutputSync     dut;

    std::vector<Bits>       mask_prog, gi_prog, rm_prog;
    std::vector<std::tuple<Bits, Bits, Bits>> expected, collected;   // (used, det, gi)
    std::vector<std::pair<bool, bool>> mk_expected, mk_collected;    // (first_normal, last_normal)

    // three parallel regfiles read at the same pc
    void regfile() {
        uint32_t p = pc.read();
        sync_mask.write(p < mask_prog.size() ? mask_prog[p] : Bits(D, 1));       // out of range -> stall
        global_indexes.write(p < gi_prog.size() ? gi_prog[p] : Bits(D * IW));    // out of range -> zeros
        round_marker.write(p < rm_prog.size() ? rm_prog[p] : Bits(3));
    }

    void monitor() {
        if (out_valid.read()) {
            collected.emplace_back(out_used.read(), out_det.read(), out_global_indexes.read());
            mk_collected.emplace_back(first_normal.read(), last_normal.read());
        }
        // when out_valid=0 every line is unused -> all-ones (sentinel encoding), not zero
    }

    void stim() {
        rst.write(true);  in_det.write(Bits(D)); in_valid.write(Bits(D)); in_finish.write(Bits(D)); wait();
        rst.write(false);
        in_det.write(mk({1,1,0})); in_valid.write(mk({1,1,0})); wait();  // pc0 bypass line0 + buffer line1
        in_det.write(Bits(D));     in_valid.write(Bits(D));     wait();  // pc1 consume line1 from FIFO
        wait();                                                          // pc2 fast-forward
        in_det.write(mk({0,1,0})); in_valid.write(mk({1,1,0})); wait();  // pc3 stall (line2 missing)
        in_det.write(mk({0,0,0})); in_valid.write(mk({0,0,1})); wait();  // line2 arrives -> pc3 done
        in_det.write(Bits(D));     in_valid.write(Bits(D));
        for (int i = 0; i < 20; ++i) wait();

        bool ok = (collected.size() == expected.size()) && (mk_collected.size() == mk_expected.size());
        for (std::size_t i = 0; ok && i < expected.size(); ++i)
            ok = (collected[i] == expected[i]) && (mk_collected[i] == mk_expected[i]);

        std::cout << "collected " << collected.size() << " / expected " << expected.size() << "\n";
        for (std::size_t i = 0; i < collected.size(); ++i) {
            auto& [u, d, g] = collected[i];
            std::cout << "  used=" << u << " det=" << d << " gi=" << g
                      << " first_normal=" << mk_collected[i].first
                      << " last_normal=" << mk_collected[i].second << "\n";
        }
        if (!ok) emu::log_error("test", "root_output_sync sequence MISMATCH");
        else     emu::log_info("test", "root_output_sync PASS");
        sc_stop();
    }

    SC_CTOR(Tb) : clk("clk", 10, SC_NS), dut("dut", D, D_FIFO, IW, HW, STRIDE, SENT, /*sat_pc*/-1) {
        dut.clk(clk); dut.rst(rst);
        dut.in_det(in_det); dut.in_valid(in_valid); dut.in_finish(in_finish); dut.sync_mask(sync_mask);
        dut.global_indexes(global_indexes); dut.round_marker(round_marker);
        dut.out_det(out_det); dut.out_used(out_used); dut.out_valid(out_valid); dut.out_finish(out_finish);
        dut.sync_regfile_pc(pc); dut.out_global_indexes(out_global_indexes);
        dut.first_normal(first_normal); dut.last_normal(last_normal);
        dut.first_wait(first_wait);     dut.last_wait(last_wait);

        mask_prog = { mk({1,0,0}), mk({0,1,0}), mk({0,0,0}), mk({1,1,1}) };
        gi_prog = {
            gipack({5, 15, 15}, IW),   // pc0: line0 index 5
            gipack({1, 2, 15}, IW),    // pc1: line0=1(unused), line1=2
            gipack({15, 15, 15}, IW),  // pc2: fast-forward (all sentinel)
            gipack({7, 8, 9}, IW),     // pc3: all three
        };
        rm_prog = { rm3(1,0,0), rm3(0,0,0), rm3(0,0,0), rm3(0,1,0) };   // first_normal@pc0, last_normal@pc3
        // unused slots come out all-ones (sentinel), regardless of the regfile value there
        expected = {
            { mk({1,0,0}), mk({1,0,0}), gipack({5,15,15}, IW) },    // pc0
            { mk({0,1,0}), mk({0,1,0}), gipack({15,2,15}, IW) },    // pc1 (line0 unused -> 15, not 1)
            { mk({0,0,0}), mk({0,0,0}), gipack({15,15,15}, IW) },   // pc2 fast-forward
            { mk({1,1,1}), mk({0,1,0}), gipack({7,8,9}, IW) },      // pc3
        };
        mk_expected = { {true,false}, {false,false}, {false,false}, {false,true} };

        SC_METHOD(regfile); sensitive << pc;
        SC_METHOD(monitor); sensitive << clk.posedge_event(); dont_initialize();
        SC_THREAD(stim);    sensitive << clk.posedge_event();
    }
};

int sc_main(int, char*[]) {
    Tb tb("tb");
    sc_start();
    int errors = sc_report_handler::get_count(SC_ERROR) + sc_report_handler::get_count(SC_FATAL);
    return errors ? 1 : 0;
}
