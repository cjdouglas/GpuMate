if (NOT GM_VENDOR STREQUAL "AMD" AND NOT GM_VENDOR STREQUAL "NVIDIA")
  message(FATAL_ERROR "please set GM_VENDOR to 'AMD' or 'NVIDIA'")
endif()

set(
  SOURCES
  src/utility/gpu_buffer.cc
)

if (GM_VENDOR STREQUAL "AMD")
  list(
    APPEND
    SOURCES
    src/amd/amd_runtime_impl.cc)
else()
  list(
    APPEND
    SOURCES
    src/nvidia/nvidia_runtime_impl.cc)
endif()

if (GM_EXPORT_SHARED)
  # TODO: add versioning
  add_library(GpuMate SHARED ${SOURCES})
else()
  add_library(GpuMate STATIC ${SOURCES})
endif()

target_include_directories(GpuMate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (GM_VENDOR STREQUAL "AMD")
  find_package(hip REQUIRED)
  find_package(rocblas REQUIRED)
  target_link_libraries(GpuMate PUBLIC hip::host roc::rocblas)
  target_compile_definitions(GpuMate PUBLIC __GM_PLATFORM_AMD__)
elseif (GM_VENDOR STREQUAL "NVIDIA")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(GpuMate PUBLIC CUDA::cudart CUDA::cublas)
  target_compile_definitions(GpuMate PUBLIC __GM_PLATFORM_NVIDIA__)
endif()

if (GM_BUILD_TESTS)
  enable_testing()
  set(
    TEST_SOURCES
    test/test_runtime.cc)
  add_executable(GpuMateTests ${TEST_SOURCES})
  target_link_libraries(GpuMateTests PRIVATE GpuMate GTest::gtest_main)
endif()
