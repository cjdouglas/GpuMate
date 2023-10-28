# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.21)

project ("GpuMate")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(
  GM_VENDOR
  "The vendor to build the libraries for, should be either 'amd' or 'nvidia'."
  "unspecified"
)

option(
  GM_BUILD_TESTS
  "If ON, GpuMate will build all tests."
  OFF
)

if (GM_BUILD_TESTS)
  include(FetchGTest)
endif()

# Include sub-projects.
add_subdirectory ("GpuMate")
