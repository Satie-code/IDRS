#include "idr/health_check.hpp"

#include "idr/version.hpp"

namespace idr {

bool HealthCheck::is_ok() const noexcept {
    return true;
}

std::string HealthCheck::status() const {
    return "idr-core ok (v" + std::string(kVersion) + ")";
}

}  // namespace idr
