if (NOT GM_VENDOR STREQUAL "amd" AND NOT GM_VENDOR STREQUAL "nvidia")
  message(FATAL_ERROR "please set GM_VENDOR to 'amd' or 'nvidia'")
endif()

if (GM_VENDOR STREQUAL "amd")
  set(
    SOURCES
    src/amd/amd_runtime_impl.cc)
else()
  set(
    SOURCES
    src/nvidia/nvidia_runtime_impl.cc)
endif()

add_library(GpuMate STATIC ${SOURCES})
target_include_directories(GpuMate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (GM_VENDOR STREQUAL "amd")
  find_package(hip REQUIRED)
  find_package(rocblas REQUIRED)
  target_link_libraries(GpuMate PRIVATE hip::host roc::rocblas)
  target_compile_definitions(GpuMate PRIVATE __GM_PLATFORM_AMD__)
elseif (GM_VENDOR STREQUAL "nvidia")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(GpuMate PRIVATE CUDA::cudart CUDA::cublas)
  target_compile_definitions(GpuMate PRIVATE __GM_PLATFORM_NVIDIA__)
endif()

if (GM_BUILD_TESTS)
  # TODO: create test executable here
endif()
