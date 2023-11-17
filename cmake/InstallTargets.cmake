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
