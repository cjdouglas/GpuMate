#include <gtest/gtest.h>

#include "gpu_mate/gpu_blas.h"
#include "gpu_mate/utility/gpu_buffer.h"

using namespace gpu_mate::blas;
using gpu_mate::utility::GpuBuffer;

TEST(TestBlas, TestSgemm) {
  GpuBlasHandle handle = GpuBlasHandle::Create();

  const int m = 2;
  const int n = 3;
  const int k = 3;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int lda = m;
  const int ldb = k;
  const int ldc = m;
  const float A[] = {1, 2, 3, 4, 5, 6};
  const float B[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float C[m * n] = {-1, -1, -1, -1, -1, -1};

  GpuBuffer gpuA;
  GpuBuffer gpuB;
  GpuBuffer gpuC;
  gpuA.CopyFrom(A, m * k * sizeof(float), GpuMemcpyKind::hostToDevice);
  gpuB.CopyFrom(B, k * n * sizeof(float), GpuMemcpyKind::hostToDevice);
  gpuC.CopyFrom(C, m * n * sizeof(float), GpuMemcpyKind::hostToDevice);

  const GpuBlasStatus status =
      sgemm(handle, GpuOperation::none, GpuOperation::none, m, n, k, &alpha, A,
            lda, B, ldb, &beta, C, ldc);
  EXPECT_EQ(status, GpuBlasStatus::success);

  const float expected[m * n] = {30, 36, 42, 66, 81, 96};
  for (int i = 0; i < m * n; ++i) {
    EXPECT_FLOAT_EQ(C[i], expected[i]);
  }
}
