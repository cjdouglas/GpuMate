#ifndef GPU_MATE_GPU_RUNTIME_H
#define GPU_MATE_GPU_RUNTIME_H

#include "gpu_mate/internal/defines.h"

#ifdef __GM_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace gpu_mate {
namespace runtime {

#ifdef __GM_PLATFORM_AMD__
using GpuMateError_t = hipError_t;
#else
using GpuMateError_t = cudaError_t;
#endif

enum class GpuMemcpyKind {
  kHostToHost = 0,
  kHostToDevice = 1,
  kDeviceToHost = 2,
  kDeviceToDevice = 3,
  kDefault = 4,
};

GPU_MATE_API GpuMateError_t GpuMalloc(void** ptr, const size_t size);

GPU_MATE_API GpuMateError_t GpuMemcpy(void* dst, const void* src,
                                      const size_t size,
                                      const GpuMemcpyKind copy_kind);

GPU_MATE_API GpuMateError_t GpuFree(void* ptr);

}  // namespace runtime
}  // namespace gpu_mate

#endif  // GPU_MATE_GPU_RUNTIME_H
