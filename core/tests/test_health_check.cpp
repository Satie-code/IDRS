// Minimal Phase 0 test: no test framework dependency, just asserts + exit code.
// CTest reports success purely from the process exit code, which is enough
// for the foundation. A real test framework can be adopted in a later phase
// once there is real navigation logic worth testing.

#include <cstdlib>
#include <iostream>
#include <string>

#include "idr/health_check.hpp"

namespace {

int failures = 0;

void expect(bool condition, const std::string& description) {
    if (!condition) {
        std::cerr << "FAIL: " << description << "\n";
        ++failures;
    } else {
        std::cout << "PASS: " << description << "\n";
    }
}

}  // namespace

int main() {
    idr::HealthCheck health_check;

    expect(health_check.is_ok(), "HealthCheck::is_ok() returns true");
    expect(!health_check.status().empty(), "HealthCheck::status() is non-empty");
    expect(health_check.status().find("idr-core") != std::string::npos,
           "HealthCheck::status() mentions idr-core");

    if (failures > 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All idr_core_tests checks passed\n";
    return EXIT_SUCCESS;
}
