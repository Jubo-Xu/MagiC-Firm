// memfile.hpp — parse compiler .mem files into Bits words.
//
// Words are MSB-first per line, LSB = bit 0 (matching the serializer's _word).
// Arbitrary width (bit/nibble-by-nibble, no fixed-int conversion) so very wide
// words (e.g. a packed global_index row) load correctly.
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "signals.hpp"

namespace emu {

inline int mem_hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// one .mem line (MSB-first) -> width-bit word, bit i = value bit i
inline Bits parse_mem_word(const std::string& line, int width, const std::string& fmt) {
    Bits b(width);
    const int L = static_cast<int>(line.size());
    if (fmt == "bin") {
        for (int i = 0; i < width; ++i) {
            int ci = L - 1 - i;
            if (ci >= 0) b[i] = (line[ci] == '1');
        }
    } else {  // hex: 4 bits/nibble, nibble k (from LSB) = char at (L-1-k)
        for (int i = 0; i < width; ++i) {
            int ci = L - 1 - (i / 4);
            if (ci >= 0) { int v = mem_hex_val(line[ci]); if (v >= 0) b[i] = (v >> (i % 4)) & 1; }
        }
    }
    return b;
}

// load a whole .mem file -> vector of width-bit words
inline std::vector<Bits> load_mem(const std::string& path, int width, const std::string& fmt) {
    std::ifstream f(path);
    std::vector<Bits> words;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (!line.empty()) words.push_back(parse_mem_word(line, width, fmt));
    }
    return words;
}

// pack N index-words (each `word_width` bits) into one bitstring:
// index j at bits [j*word_width +: word_width] (LSB = index 0)
inline Bits pack_words(const std::vector<Bits>& words, int word_width) {
    Bits out(static_cast<int>(words.size()) * word_width);
    for (std::size_t j = 0; j < words.size(); ++j)
        for (int b = 0; b < word_width; ++b)
            out[j * word_width + b] = words[j][b];
    return out;
}

}  // namespace emu
