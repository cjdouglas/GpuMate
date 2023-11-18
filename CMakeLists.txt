cmake_minimum_required (VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Prelude)

project(
  "GpuMate"
  VERSION 0.1.0
  DESCRIPTION "Cross platform, cross vendor GPU code"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(
  GM_VENDOR
  "The vendor to build the libraries for, should be either AMD or NVIDIA."
  "unspecified"
)

option(
  GM_BUILD_TESTS
  "If ON, GpuMate will build all tests."
  OFF
)

# Include sub-projects.
add_subdirectory ("GpuMate")
