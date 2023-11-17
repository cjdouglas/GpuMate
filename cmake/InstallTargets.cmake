install(
  TARGETS GpuMate
  EXPORT GpuMateTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
)

install(
  EXPORT GpuMateTargets
  FILE GpuMateTargets.cmake
  NAMESPACE GpuMate::
  DESTINATION lib/cmake/GpuMate
)

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/GpuMate/include/gpu_mate"
  DESTINATION include
)

install(
  FILES "${PROJECT_SOURCE_DIR}/cmake/InstallConfig.cmake"
  DESTINATION "lib/cmake/GpuMate"
  RENAME "${PROJECT_NAME}Config.cmake"
)
