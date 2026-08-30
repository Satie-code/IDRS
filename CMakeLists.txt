cmake_minimum_required(VERSION 3.20)

project(idr
    VERSION 0.1.0
    DESCRIPTION "SIH26168 Intelligent Dead Reckoning — C++ navigation core foundation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

enable_testing()

add_subdirectory(core/src)
add_subdirectory(core/tests)
