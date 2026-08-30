add_executable(idr_core_tests
    test_health_check.cpp
)

target_link_libraries(idr_core_tests PRIVATE idr_core)

add_test(NAME idr_core_tests COMMAND idr_core_tests)
