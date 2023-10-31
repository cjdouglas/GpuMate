#include <gtest/gtest.h>

#include <climits>

#include "gpu_mate/gpu_runtime.h"

using namespace gpu_mate::runtime;

TEST(TestRuntimeError, TestGetLastError) {
  // Invoke an error
  void* ptr;
  gpuError_t error = gpuMalloc(&ptr, ULLONG_MAX);
  gpuError_t query = gpuGetLastError();
  EXPECT_EQ(error, query);
  EXPECT_EQ(gpuGetLastError(), gpuError_t::gpuSuccess);
}

TEST(TestRuntimeError, TestPeekAtLastError) {
  // Invoke an error
  void* ptr;
  gpuError_t error = gpuMalloc(&ptr, ULLONG_MAX);
  gpuError_t peek = gpuPeekAtLastError();
  EXPECT_EQ(error, peek);
  EXPECT_EQ(error, gpuPeekAtLastError());
}

TEST(TestRuntimeError, TestGetErrorName) {
  gpuError_t error = gpuDeviceSynchronize();
  const char* name = gpuGetErrorName(error);
  EXPECT_NE(name, nullptr);
  EXPECT_STRNE(name, "");
}

TEST(TestRuntimeError, TestGetErrorString) {
  gpuError_t error = gpuDeviceSynchronize();
  const char* str = gpuGetErrorString(error);
  EXPECT_NE(str, nullptr);
  EXPECT_STRNE(str, "");
}
