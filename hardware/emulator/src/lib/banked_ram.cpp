#include "lib/banked_ram.hpp"

#include <stdexcept>

namespace emu {

BankedRAM::BankedRAM(sc_core::sc_module_name nm, int banks, int depth, int width)
    : sc_core::sc_module(nm),
      rd_en("rd_en", banks), rd_addr("rd_addr", banks), rd_data("rd_data", banks),
      wr_en("wr_en", banks), wr_addr("wr_addr", banks), wr_data("wr_data", banks),
      banks_(banks), depth_(depth), width_(width), bank_shift_(0) {
    if (banks < 1 || (banks & (banks - 1)) != 0)
        throw std::invalid_argument("BankedRAM: banks must be a power of two");
    if (depth < 1 || width < 1)
        throw std::invalid_argument("BankedRAM: depth and width must be >= 1");
    while ((1 << bank_shift_) < banks_) ++bank_shift_;
    mem_.assign(banks_, std::vector<Bits>(depth_, Bits(width_)));

    SC_HAS_PROCESS(BankedRAM);
    // Clock ONLY: with rst in the sensitivity list this would also run on reset
    // DEASSERTION, and that firing is indistinguishable from a clock edge
    // (clk.posedge() reports true for the whole simulation time, not the delta),
    // so it would perform a spurious access. Reset is therefore synchronous;
    // identical behaviour whenever rst is held a full cycle.
    SC_METHOD(tick);
    sensitive << clk.pos();
    dont_initialize();
}

void BankedRAM::tick() {
    if (rst.read()) {
        for (auto& bank : mem_)
            for (auto& w : bank) w = Bits(width_);
        for (int b = 0; b < banks_; ++b) rd_data[b].write(Bits(width_));
        return;
    }

    // READ_FIRST: every read is sampled before any write lands, so a read and a
    // write to the same address in one cycle returns the old word.  A disabled
    // bank drives zero rather than holding.
    for (int b = 0; b < banks_; ++b) {
        Bits out(width_);
        if (rd_en[b].read()) {
            uint32_t a = rd_addr[b].read();
            if (a < static_cast<uint32_t>(depth_)) out = mem_[b][a];
        }
        rd_data[b].write(out);
    }

    for (int b = 0; b < banks_; ++b) {
        if (!wr_en[b].read()) continue;
        uint32_t a = wr_addr[b].read();
        if (a < static_cast<uint32_t>(depth_)) mem_[b][a] = wr_data[b].read();
    }
}

}  // namespace emu
