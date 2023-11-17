#include <gtest/gtest.h>

#include <climits>

#include "gpu_mate/gpu_runtime.h"

using namespace gpu_mate::runtime;

TEST(TestRuntimeError, TestGetLastError) {
  // Invoke an error
  void* ptr;
  GpuError error = GpuMalloc(&ptr, ULLONG_MAX);
  GpuError query = GpuGetLastError();
  EXPECT_EQ(error, query);
  EXPECT_EQ(GpuGetLastError(), GpuError::success);
}

TEST(TestRuntimeError, TestPeekAtLastError) {
  // Invoke an error
  void* ptr;
  GpuError error = GpuMalloc(&ptr, ULLONG_MAX);
  GpuError peek = GpuPeekAtLastError();
  EXPECT_EQ(error, peek);
  EXPECT_EQ(error, GpuPeekAtLastError());
}

TEST(TestRuntimeError, TestGetErrorName) {
  GpuError error = GpuDeviceSynchronize();
  const char* name = GpuGetErrorName(error);
  EXPECT_NE(name, nullptr);
  EXPECT_STRNE(name, "");
}

TEST(TestRuntimeError, TestGetErrorString) {
  GpuError error = GpuDeviceSynchronize();
  const char* str = GpuGetErrorString(error);
  EXPECT_NE(str, nullptr);
  EXPECT_STRNE(str, "");
}
