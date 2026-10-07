// physical_mmio.cpp — implementation of PhysicalMMIO (structural only).
#include "control_system/cultiv_control/physical_mmio.hpp"

namespace emu {

PhysicalMMIO::PhysicalMMIO(sc_core::sc_module_name nm, const MmioConfig& cfg)
    : sc_module(nm), cfg_(cfg),
      instr_en_("instr_en"), rst_out_("rst_out"), one_instr_end_("one_instr_end"),
      r_en_("r_en"), instr_addr_("instr_addr"), wt_("wt"), start_addr_("start_addr"),
      end_addr_("end_addr"), r_addr_("r_addr"), instr_word_("instr_word") {
    const int aw = cfg_.addr_w();
    const int iw = cfg_.instr_w();

    seq_       = std::make_unique<InstrSequencer>("seq", cfg_.instr_depth, cfg_.payload_w);
    instr_rom_ = std::make_unique<RegFileROM>("instr_rom", cfg_.instr_depth, iw);
    unpack_    = std::make_unique<InstrUnpack>("unpack", cfg_.wt_w, aw);
    decode_    = std::make_unique<InstrDecode>("decode", cfg_.wt_w, aw);
    cw_rom_    = std::make_unique<SyncROM>("cw_rom", cfg_.cw_depth, cfg_.data_w);

    // --- sequencer: events in, instruction fetch + local reset out ---
    seq_->clk(clk);            seq_->rst(rst);
    seq_->ev_valid(ev_valid);  seq_->ev_type(ev_type);  seq_->ev_payload(ev_payload);
    seq_->one_instr_end(one_instr_end_);
    seq_->instr_en(instr_en_); seq_->instr_addr(instr_addr_);
    seq_->rst_out(rst_out_);
    seq_->cw_gen_finish(cw_gen_finish);          // straight out of the block

    // --- instruction regfile (async read at the registered address) ---
    instr_rom_->addr(instr_addr_);
    instr_rom_->data(instr_word_);

    // --- field extraction ---
    unpack_->instr(instr_word_);
    unpack_->wt(wt_); unpack_->start_addr(start_addr_); unpack_->end_addr(end_addr_);

    // --- decoder: instruction -> CW read stream. rst_out_ is its SINGLE reset
    //     source (it already ORs the system reset, abort and finish). ---
    decode_->clk(clk);  decode_->rst(rst_out_);
    decode_->en(instr_en_);
    decode_->in_wt(wt_); decode_->in_start(start_addr_); decode_->in_end(end_addr_);
    decode_->r_en(r_en_); decode_->r_addr(r_addr_);
    decode_->one_instr_end(one_instr_end_);

    // --- command-word memory (BRAM, 1-cycle). No reset by design: that is what
    //     lets the final command word still be delivered on a finish. ---
    cw_rom_->clk(clk);
    cw_rom_->en(r_en_); cw_rom_->addr(r_addr_);
    cw_rom_->data(out_data); cw_rom_->data_valid(out_valid);
}

void PhysicalMMIO::load_instr(const std::vector<Bits>& words) { instr_rom_->load(words); }

void PhysicalMMIO::load_instr_mem(const std::string& path, const std::string& fmt) {
    instr_rom_->load_mem_file(path, fmt);
}

void PhysicalMMIO::load_cw(const std::vector<Bits>& words) { cw_rom_->load(words); }

void PhysicalMMIO::load_cw_mem(const std::string& path, const std::string& fmt) {
    cw_rom_->load_mem_file(path, fmt);
}

Bits PhysicalMMIO::pack_instr(uint32_t wt, uint32_t start, uint32_t end, int wt_w, int addr_w) {
    Bits b(InstrUnpack::instr_width(wt_w, addr_w));
    for (int i = 0; i < addr_w; ++i) b[i] = (end >> i) & 1u;
    for (int i = 0; i < addr_w; ++i) b[addr_w + i] = (start >> i) & 1u;
    for (int i = 0; i < wt_w;   ++i) b[2 * addr_w + i] = (wt >> i) & 1u;
    return b;
}

}  // namespace emu
