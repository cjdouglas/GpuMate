# GpuMate/test/CMakeLists.txt

include(FetchGTest)

enable_testing()

set(
  TEST_SOURCES 
  test_blas.cc
  test_device_buffer.cc
  test_runtime_error.cc
  test_runtime_device.cc 
  test_runtime_memory.cc
)

add_executable(GpuMateTests ${TEST_SOURCES})
target_link_libraries(GpuMateTests PRIVATE GpuMate GTest::gtest_main)

add_test(NAME GpuMateTests COMMAND GpuMateTests)
