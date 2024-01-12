#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>

#include "gpu_mate/utility/gpu_buffer.h"

using gpu_mate::utility::GpuBuffer;

TEST(TestGpuBuffer, TestCreate) {
  std::optional<GpuBuffer<float>> buffer_a = GpuBuffer<float>::create();
  EXPECT_TRUE(buffer_a.has_value());

  std::optional<GpuBuffer<int>> buffer_b = GpuBuffer<int>::create(0);
  EXPECT_TRUE(buffer_b.has_value());

  std::optional<GpuBuffer<double>> buffer_c = GpuBuffer<double>::create(4);
  EXPECT_TRUE(buffer_c.has_value());

  std::optional<GpuBuffer<char>> buffer_d =
      GpuBuffer<char>::create(static_cast<std::size_t>(-1));
  EXPECT_FALSE(buffer_d.has_value());
}
