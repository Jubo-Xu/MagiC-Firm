// signals.hpp — wire value types for the detector-construct design.
//
// `Bits` is a runtime-width bit vector carried on a single sc_signal/port, so an
// `[m-1:0]` bus is one wire of `m` bits rather than `m` separate 1-bit ports.
// Index 0 is bit 0 (LSB / line 0), matching the compiler's LSB=bit-0 convention.
#pragma once

#include <systemc>

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace emu {

struct Bits {
    std::vector<uint8_t> v;  // one byte per bit (0/1); v[i] == bit i

    Bits() = default;
    explicit Bits(std::size_t n, uint8_t init = 0) : v(n, init) {}

    std::size_t size() const { return v.size(); }
    uint8_t&    operator[](std::size_t i) { return v[i]; }
    uint8_t     operator[](std::size_t i) const { return v[i]; }

    bool operator==(const Bits& o) const { return v == o.v; }
    bool operator!=(const Bits& o) const { return v != o.v; }

    bool any() const {
        for (uint8_t x : v)
            if (x) return true;
        return false;
    }
};

// Streamed MSB-first (bit m-1 ... bit 0), e.g. line0=1,line1=1,line2=0 -> "011".
inline std::ostream& operator<<(std::ostream& os, const Bits& b) {
    for (std::size_t i = b.size(); i-- > 0;) os << int(b.v[i]);
    return os;
}

// Required so `Bits` is usable as an sc_signal<> type. Waveform tracing of Bits
// is deferred (M0 skipped trace.hpp); this stub satisfies name lookup.
inline void sc_trace(sc_core::sc_trace_file*, const Bits&, const std::string&) {}

// Read an unsigned integer from bits [off, off+width) of b (bit off = LSB).
// Out-of-range bits read as 0. Used to unpack regfile fields (selector indices,
// packed core words, ...).
inline uint32_t extract(const Bits& b, std::size_t off, std::size_t width) {
    uint32_t v = 0;
    for (std::size_t k = 0; k < width; ++k)
        if (off + k < b.size() && b[off + k]) v |= (1u << k);
    return v;
}

}  // namespace emu
