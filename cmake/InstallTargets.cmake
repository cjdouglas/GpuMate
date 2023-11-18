include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(
  GPU_MATE_INSTALL_LIBDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/GpuMate"
)

install(
  TARGETS GpuMate
  EXPORT GpuMateTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT GpuMateTargets
  FILE GpuMateTargets.cmake
  NAMESPACE GpuMate::
  DESTINATION ${GPU_MATE_INSTALL_LIBDIR}
)

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/GpuMate/include/gpu_mate"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

write_basic_package_version_file(
  "GpuMateConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

if (GM_VENDOR STREQUAL "AMD")
  set(GM_BUILD_DEPENDENCIES "find_package(hip REQUIRED)\nfind_package(rocblas REQUIRED)")
elseif (GM_VENDOR STREQUAL "NVIDIA")
  set(GM_BUILD_DEPENDENCIES "find_package(CUDAToolkit REQUIRED)")
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/InstallConfig.cmake.in" 
  "${CMAKE_CURRENT_BINARY_DIR}/GpuMateConfig.cmake" 
  @ONLY
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/GpuMateConfig.cmake"
  DESTINATION ${GPU_MATE_INSTALL_LIBDIR}
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/GpuMateConfigVersion.cmake"
  DESTINATION ${GPU_MATE_INSTALL_LIBDIR}
)
