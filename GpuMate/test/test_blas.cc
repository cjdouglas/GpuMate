#include <gtest/gtest.h>

#include "gpu_mate/gpu_blas.h"

using namespace gpu_mate::blas;

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
  float C[m * n] = {};

  const GpuBlasStatus status =
      sgemm(handle, GpuOperation::none, GpuOperation::none, m, n, k, &alpha, A,
            lda, B, ldb, &beta, C, ldc);
  EXPECT_EQ(status, GpuBlasStatus::success);
}
