add_library(idr_core
    health_check.cpp
)

target_include_directories(idr_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/core/include>
)

target_compile_features(idr_core PUBLIC cxx_std_17)
