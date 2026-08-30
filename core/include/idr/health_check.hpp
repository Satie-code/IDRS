#pragma once

#include <string>

namespace idr {

// Phase 0 placeholder: proves the include/src/test wiring works end to end.
// No navigation mathematics belongs here — see docs/architecture for what
// this header grows into in later phases.
class HealthCheck {
public:
    // Returns true if the core library is linked and callable.
    [[nodiscard]] bool is_ok() const noexcept;

    // Returns a short human-readable status string, e.g. "idr-core ok (v0.1.0)".
    [[nodiscard]] std::string status() const;
};

}  // namespace idr
