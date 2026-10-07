// root_output_sync.cpp — implementation of RootOutputSync.
//
// OutputSync drives out_det directly and taps out_valid/out_finish/out_used to
// internal signals. reg_pipe registers global_indexes, round_marker, and the
// producing pc one cycle so all three line up with OutputSync's registered out_det
// (the same alignment global_index has always used). drive_out decodes the markers
// and builds the hw_width global-index bus, adding w*stride on the saturating row.
#include "control_system/detector_construct/root_output_sync.hpp"

namespace emu {

RootOutputSync::RootOutputSync(sc_core::sc_module_name nm, int d, int d_fifo, int index_width,
                               int hw_width, int stride, int sentinel, int sat_pc)
    : sc_module(nm), d_(d), index_width_(index_width), hw_width_(hw_width),
      stride_(stride), sentinel_(sentinel), sat_pc_(sat_pc),
      os_("output_sync", d, d_fifo, sat_pc) {
    // OutputSync: det passes straight through to our output; valid/finish/used are
    // tapped so we can decode markers and mask the indices; pc drives all regfiles.
    os_.clk(clk);               os_.rst(rst);
    os_.in_det(in_det);         os_.in_valid(in_valid);   os_.in_finish(in_finish);
    os_.sync_mask(sync_mask);
    os_.out_det(out_det);       os_.out_used(used_int_);
    os_.out_valid(ov_int_);     os_.out_finish(of_int_);
    os_.sync_regfile_pc(sync_regfile_pc);

    SC_HAS_PROCESS(RootOutputSync);
    SC_METHOD(reg_pipe);  sensitive << clk.pos() << rst; dont_initialize();
    SC_METHOD(drive_out); sensitive << ov_int_ << of_int_ << used_int_
                                    << gi_reg_ << rm_reg_ << pc_reg_ << is_wait_q_ << w_;
}

void RootOutputSync::reg_pipe() {
    if (rst.read()) {
        gi_reg_.write(Bits(d_ * index_width_));
        rm_reg_.write(Bits(3));
        pc_reg_.write(0);
        is_wait_q_.write(false);
        w_.write(0);
        return;
    }
    if (!clk.posedge()) return;

    // Reads see the pre-update values (= those that produced the current out_det).
    const bool     ov       = ov_int_.read();
    const Bits     rm       = rm_reg_.read();
    const bool     is_wait  = (rm.size() > 2) && rm[2];
    const uint32_t pc       = pc_reg_.read();
    const bool     sat_here = (sat_pc_ >= 0) && (pc == static_cast<uint32_t>(sat_pc_));

    if (ov) {
        is_wait_q_.write(is_wait);                 // remember this round's is_wait for the next edge test
        if (sat_here) w_.write(w_.read() + 1);     // count saturating (wait-repeat) rounds
    }
    // pipeline registers (align global_index / round_marker / pc with the registered out_det)
    gi_reg_.write(global_indexes.read());
    rm_reg_.write(round_marker.read());
    pc_reg_.write(sync_regfile_pc.read());
}

void RootOutputSync::drive_out() {
    const bool ov = ov_int_.read();
    out_valid.write(ov);
    out_finish.write(of_int_.read());
    out_used.write(used_int_.read());

    const Bits rm      = rm_reg_.read();
    const bool is_first = (rm.size() > 0) && rm[0];
    const bool is_last  = (rm.size() > 1) && rm[1];
    const bool is_wait  = (rm.size() > 2) && rm[2];

    first_normal.write(ov && is_first);
    last_normal.write(ov && is_last);
    first_wait.write(ov && is_wait && !is_wait_q_.read());   // rising edge of is_wait
    last_wait.write(ov && is_wait && of_int_.read());        // is_wait & finish

    // --- global-index bus (hw_width bits/line), with the copy-last offset ---
    const uint32_t pc       = pc_reg_.read();
    const bool     sat_here = (sat_pc_ >= 0) && (pc == static_cast<uint32_t>(sat_pc_));
    const uint32_t offset   = sat_here ? (w_.read() * static_cast<uint32_t>(stride_)) : 0;
    const Bits     gi       = gi_reg_.read();
    const Bits     used     = used_int_.read();

    Bits out(d_ * hw_width_);
    const uint32_t all_ones = (hw_width_ >= 32) ? 0xFFFFFFFFu : ((1u << hw_width_) - 1u);
    for (int l = 0; l < d_; ++l) {
        uint32_t idx = extract(gi, static_cast<std::size_t>(l) * index_width_, index_width_);
        const bool is_used = ov && (l < static_cast<int>(used.size())) && used[l];
        uint32_t hw = (is_used && idx != static_cast<uint32_t>(sentinel_)) ? (idx + offset) : all_ones;
        for (int b = 0; b < hw_width_; ++b)
            out[static_cast<std::size_t>(l) * hw_width_ + b] = (hw >> b) & 1u;
    }
    out_global_indexes.write(out);
}

}  // namespace emu
