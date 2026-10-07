// log.hpp — consistent logging over SystemC's report handler.
//
// Design-agnostic. Thin wrappers so every module reports the same way and
// severities route through SystemC (which tracks counts, can escalate warnings
// to errors, and stamps simulation time automatically). Prefer these over raw
// std::cout so simulation-time context and severity handling come for free.
#pragma once

#include <systemc>

#include <string>

namespace emu {

// Message id (the "tag" column in SystemC report output). Pass a module name as
// `who` to scope a message, e.g. emu::log_info("board3", "elaborated").
inline constexpr const char* kMsgId = "emu";

namespace detail {
inline std::string tagged(const std::string& who, const std::string& msg) {
    return who.empty() ? msg : ("[" + who + "] " + msg);
}
}  // namespace detail

inline void log_info(const std::string& who, const std::string& msg) {
    SC_REPORT_INFO(kMsgId, detail::tagged(who, msg).c_str());
}
inline void log_warn(const std::string& who, const std::string& msg) {
    SC_REPORT_WARNING(kMsgId, detail::tagged(who, msg).c_str());
}
inline void log_error(const std::string& who, const std::string& msg) {
    SC_REPORT_ERROR(kMsgId, detail::tagged(who, msg).c_str());
}

}  // namespace emu
