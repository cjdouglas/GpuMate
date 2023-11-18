# GpuMate/CMakeLists.txt

if (NOT GM_VENDOR STREQUAL "AMD" AND NOT GM_VENDOR STREQUAL "NVIDIA")
  message(FATAL_ERROR "please set GM_VENDOR to 'AMD' or 'NVIDIA'")
endif()

# Set sources based on target vendor

set(
  SOURCES
  src/utility/device_buffer.cc
)

if (GM_VENDOR STREQUAL "AMD")
  list(
    APPEND
    SOURCES
    src/amd/amd_runtime_impl.cc
    src/amd/amd_blas_impl.cc
  )
else()
  list(
    APPEND
    SOURCES
    src/nvidia/nvidia_runtime_impl.cc
    src/nvidia/nvidia_blas_impl.cc
  )
endif()

# Create library target

add_library(GpuMate ${SOURCES})

# Define shared library definitions

if (BUILD_SHARED_LIBS)
  target_compile_definitions(GpuMate PRIVATE GM_DLL_EXPORT)
endif()

target_include_directories(
  GpuMate 
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Vendor specific configuration

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

# Install rules

include(InstallTargets)

# Include tests if enabled

if (GM_BUILD_TESTS)
  add_subdirectory(test)
endif()
