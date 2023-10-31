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

enum class gpuError_t {
  gpuSuccess = 0,
  gpuErrorInvalidValue = 1,
  gpuErrorOutOfMemory = 2,
  gpuErrorNotInitialized = 3,
  gpuErrorDeinitialized = 4,
  gpuErrorProfilerDisabled = 5,
  gpuErrorProfilerNotInitialized = 6,
  gpuErrorProfilerAlreadyStarted = 7,
  gpuErrorProfilerAlreadyStopped = 8,
  gpuErrorInvalidConfiguration = 9,
  gpuErrorInvalidPitchValue = 10,
  gpuErrorInvalidSymbol = 11,
  gpuErrorInvalidDevicePointer = 12,
  gpuErrorInvalidMemcpyDirection = 21,
  gpuErrorInsufficientDriver = 22,
  gpuErrorMissingConfiguration = 23,
  gpuErrorPriorLaunchFailure = 24,
  gpuErrorInvalidDeviceFunction = 25,
  gpuErrorNoDevice = 26,
  gpuErrorInvalidDevice = 27,
  gpuErrorInvalidImage = 28,
  gpuErrorInvalidContext = 29,
  gpuErrorContextAlreadyCurrent = 30,
  gpuErrorMapFailed = 31,
  gpuErrorUnmapFailed = 32,
  gpuErrorArrayIsMapped = 33,
  gpuErrorAlreadyMapped = 34,
  gpuErrorNoBinaryForGpu = 35,
  gpuErrorAlreadyAcquired = 36,
  gpuErrorNotMapped = 37,
  gpuErrorNotMappedAsArray = 38,
  gpuErrorNotMappedAsPointer = 39,
  gpuErrorECCNotCorrectable = 40,
  gpuErrorUnsupportedLimit = 41,
  gpuErrorContextAlreadyInUse = 42,
  gpuErrorPeerAccessUnsupported = 43,
  gpuErrorInvalidKernelFile = 44,
  gpuErrorInvalidGraphicsContext = 45,
  gpuErrorInvalidSource = 46,
  gpuErrorFileNotFound = 47,
  gpuErrorSharedObjectSymbolNotFound = 48,
  gpuErrorSharedObjectInitFailed = 49,
  gpuErrorOperatingSystem = 50,
  gpuErrorInvalidHandle = 51,
  gpuErrorIllegalState = 52,
  gpuErrorNotFound = 53,
  gpuErrorNotReady = 54,
  gpuErrorIllegalAddress = 55,
  gpuErrorLaunchOutOfResources = 56,
  gpuErrorLaunchTimeOut = 57,
  gpuErrorPeerAccessAlreadyEnabled = 58,
  gpuErrorPeerAccessNotEnabled = 59,
  gpuErrorSetOnActiveProcess = 60,
  gpuErrorContextIsDestroyed = 61,
  gpuErrorAssert = 62,
  gpuErrorHostMemoryAlreadyRegistered = 63,
  gpuErrorHostMemoryNotRegistered = 64,
  gpuErrorLaunchFailure = 65,
  gpuErrorCooperativeLaunchTooLarge = 66,
  gpuErrorNotSupported = 67,
  gpuErrorStreamCaptureUnsupported = 68,
  gpuErrorStreamCaptureInvalidated = 69,
  gpuErrorStreamCaptureMerge = 70,
  gpuErrorStreamCaptureUnmatched = 71,
  gpuErrorStreamCaptureUnjoined = 72,
  gpuErrorStreamCaptureIsolation = 73,
  gpuErrorStreamCaptureImplicit = 74,
  gpuErrorCapturedEvent = 75,
  gpuErrorStreamCaptureWrongThread = 76,
  gpuErrorGraphExecUpdateFailure = 77,

  gpuErrorUnknown = 999
};

enum class gpuMemcpyKind {
  gpuMemcpyHostToHost = 0,
  gpuMemcpyHostToDevice = 1,
  gpuMemcpyDeviceToHost = 2,
  gpuMemcpyDeviceToDevice = 3,
  gpuMemcpyDefault = 4,
};

// Device management

GPU_MATE_API gpuError_t gpuGetDevice(int* id);

GPU_MATE_API gpuError_t gpuSetDevice(int id);

GPU_MATE_API gpuError_t gpuGetDeviceCount(int* count);

GPU_MATE_API gpuError_t gpuDeviceSynchronize();

GPU_MATE_API gpuError_t gpuDeviceReset();

// Error handling

GPU_MATE_API gpuError_t gpuGetLastError();

GPU_MATE_API gpuError_t gpuPeekAtLastError();

GPU_MATE_API const char* gpuGetErrorName(gpuError_t error);

GPU_MATE_API const char* gpuGetErrorString(gpuError_t error);

// Memory management

GPU_MATE_API gpuError_t gpuMalloc(void** ptr, const size_t size);

GPU_MATE_API gpuError_t gpuMemcpy(void* dst, const void* src, const size_t size,
                                  const gpuMemcpyKind copy_kind);

GPU_MATE_API gpuError_t gpuFree(void* ptr);

}  // namespace runtime
}  // namespace gpu_mate

#endif  // GPU_MATE_GPU_RUNTIME_H
