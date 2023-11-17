find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
  
if (__GM_PLATFORM_NVIDIA__)
  find_package(CUDAToolkit REQUIRED)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/GpuMateTargets.cmake")

