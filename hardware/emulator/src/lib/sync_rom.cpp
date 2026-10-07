// sync_rom.cpp — implementation of SyncROM.
//
// Reuses the shared .mem parser in lib/memfile.hpp (words MSB-first per line,
// LSB = bit 0, arbitrary width) rather than duplicating it.
#include "lib/sync_rom.hpp"

#include <fstream>

#include "lib/memfile.hpp"
#include "log.hpp"

namespace emu {

SyncROM::SyncROM(sc_core::sc_module_name nm, int depth, int width)
    : width_(width), mem_(depth, Bits(width)) {
    SC_HAS_PROCESS(SyncROM);
    SC_METHOD(tick);
    sensitive << clk.pos();   // no reset: the output register models a BRAM's
}

void SyncROM::tick() {
    const bool e = en.read();
    if (e) {
        const uint32_t a = addr.read();
        data.write(a < mem_.size() ? mem_[a] : Bits(width_));  // out of range -> zeros
    } else {
        data.write(Bits(width_));   // keep (data, data_valid) consistent
    }
    data_valid.write(e);
}

void SyncROM::load(const std::vector<Bits>& words) {
    for (const auto& w : words)
        sc_assert(w.size() == static_cast<std::size_t>(width_));
    mem_ = words;
}

void SyncROM::load_mem_file(const std::string& path, const std::string& fmt) {
    std::ifstream probe(path);
    if (!probe) { log_error(name(), "cannot open mem file: " + path); return; }
    probe.close();
    mem_ = load_mem(path, width_, fmt);
}

}  // namespace emu
