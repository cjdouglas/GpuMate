#pragma once

#include "gpu_mate/internal/defines.h"

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
  gpuErrorInvalidMemcpyDirection = 13,
  gpuErrorInsufficientDriver = 14,
  gpuErrorMissingConfiguration = 15,
  gpuErrorPriorLaunchFailure = 16,
  gpuErrorInvalidDeviceFunction = 17,
  gpuErrorNoDevice = 18,
  gpuErrorInvalidDevice = 19,
  gpuErrorInvalidImage = 20,
  gpuErrorInvalidContext = 21,
  gpuErrorContextAlreadyCurrent = 22,
  gpuErrorMapFailed = 23,
  gpuErrorUnmapFailed = 24,
  gpuErrorArrayIsMapped = 25,
  gpuErrorAlreadyMapped = 26,
  gpuErrorNoBinaryForGpu = 27,
  gpuErrorAlreadyAcquired = 28,
  gpuErrorNotMapped = 29,
  gpuErrorNotMappedAsArray = 30,
  gpuErrorNotMappedAsPointer = 31,
  gpuErrorECCNotCorrectable = 32,
  gpuErrorUnsupportedLimit = 33,
  gpuErrorContextAlreadyInUse = 34,
  gpuErrorPeerAccessUnsupported = 35,
  gpuErrorInvalidKernelFile = 36,
  gpuErrorInvalidGraphicsContext = 37,
  gpuErrorInvalidSource = 38,
  gpuErrorFileNotFound = 39,
  gpuErrorSharedObjectSymbolNotFound = 40,
  gpuErrorSharedObjectInitFailed = 41,
  gpuErrorOperatingSystem = 42,
  gpuErrorInvalidHandle = 43,
  gpuErrorIllegalState = 44,
  gpuErrorNotFound = 45,
  gpuErrorNotReady = 46,
  gpuErrorIllegalAddress = 47,
  gpuErrorLaunchOutOfResources = 48,
  gpuErrorLaunchTimeOut = 49,
  gpuErrorPeerAccessAlreadyEnabled = 50,
  gpuErrorPeerAccessNotEnabled = 51,
  gpuErrorSetOnActiveProcess = 52,
  gpuErrorContextIsDestroyed = 53,
  gpuErrorAssert = 54,
  gpuErrorHostMemoryAlreadyRegistered = 55,
  gpuErrorHostMemoryNotRegistered = 56,
  gpuErrorLaunchFailure = 57,
  gpuErrorCooperativeLaunchTooLarge = 58,
  gpuErrorNotSupported = 59,
  gpuErrorStreamCaptureUnsupported = 60,
  gpuErrorStreamCaptureInvalidated = 61,
  gpuErrorStreamCaptureMerge = 62,
  gpuErrorStreamCaptureUnmatched = 63,
  gpuErrorStreamCaptureUnjoined = 64,
  gpuErrorStreamCaptureIsolation = 65,
  gpuErrorStreamCaptureImplicit = 66,
  gpuErrorCapturedEvent = 67,
  gpuErrorStreamCaptureWrongThread = 68,
  gpuErrorGraphExecUpdateFailure = 69,

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
