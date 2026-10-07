// regfile_rom.cpp — implementation of RegFileROM.
//
// The .mem parser handles arbitrary word widths (regfiles can be very wide, e.g.
// a packed global_index word), so it works bit-by-bit / nibble-by-nibble rather
// than via fixed-width integer conversion. Words are MSB-first per line with
// LSB = bit 0 (matching the serializer's `_word`).
#include "lib/regfile_rom.hpp"

#include <fstream>

#include "log.hpp"

namespace emu {

namespace {
int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Parse one .mem line (MSB-first) into a width-bit word, bit i = value bit i.
Bits parse_word(const std::string& line, int width, const std::string& fmt) {
    Bits b(width);
    const int L = static_cast<int>(line.size());
    if (fmt == "bin") {
        for (int i = 0; i < width; ++i) {          // bit i <- char at (L-1-i)
            int ci = L - 1 - i;
            if (ci >= 0) b[i] = (line[ci] == '1');
        }
    } else {  // hex: 4 bits per nibble, nibble k (from LSB) = char at (L-1-k)
        for (int i = 0; i < width; ++i) {
            int ci = L - 1 - (i / 4);
            if (ci >= 0) {
                int v = hex_val(line[ci]);
                if (v >= 0) b[i] = (v >> (i % 4)) & 1;
            }
        }
    }
    return b;
}
}  // namespace

RegFileROM::RegFileROM(sc_core::sc_module_name nm, int depth, int width)
    : width_(width), mem_(depth, Bits(width)) {
    SC_HAS_PROCESS(RegFileROM);
    SC_METHOD(read);
    sensitive << addr;   // asynchronous read
}

void RegFileROM::read() {
    uint32_t a = addr.read();
    data.write(a < mem_.size() ? mem_[a] : Bits(width_));
}

void RegFileROM::load(const std::vector<Bits>& words) {
    for (const auto& w : words)
        sc_assert(w.size() == static_cast<std::size_t>(width_));
    mem_ = words;
}

void RegFileROM::load_mem_file(const std::string& path, const std::string& fmt) {
    std::ifstream f(path);
    if (!f) { log_error(name(), "cannot open mem file: " + path); return; }
    std::vector<Bits> words;
    std::string line;
    while (std::getline(f, line)) {
        // trim trailing whitespace / CR
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (!line.empty()) words.push_back(parse_word(line, width_, fmt));
    }
    mem_ = std::move(words);
}

}  // namespace emu
