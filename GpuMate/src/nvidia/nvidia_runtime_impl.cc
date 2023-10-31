#include <cuda_runtime.h>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace runtime {
namespace {

static gpuError_t cudaErrorToGpuError(const cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return gpuError_t::gpuSuccess;
    case cudaErrorInvalidValue:
      return gpuError_t::gpuErrorInvalidValue;
    case cudaErrorMemoryAllocation:
      return gpuError_t::gpuErrorOutOfMemory;
    case cudaErrorInitializationError:
      return gpuError_t::gpuErrorNotInitialized;
    case cudaErrorCudartUnloading:
      return gpuError_t::gpuErrorDeinitialized;
    case cudaErrorProfilerDisabled:
      return gpuError_t::gpuErrorProfilerDisabled;
    case cudaErrorProfilerNotInitialized:
      return gpuError_t::gpuErrorProfilerNotInitialized;
    case cudaErrorProfilerAlreadyStarted:
      return gpuError_t::gpuErrorProfilerAlreadyStarted;
    case cudaErrorProfilerAlreadyStopped:
      return gpuError_t::gpuErrorProfilerAlreadyStopped;
    case cudaErrorInvalidConfiguration:
      return gpuError_t::gpuErrorInvalidConfiguration;
    case cudaErrorInvalidPitchValue:
      return gpuError_t::gpuErrorInvalidPitchValue;
    case cudaErrorInvalidSymbol:
      return gpuError_t::gpuErrorInvalidSymbol;
    case cudaErrorInvalidDevicePointer:
      return gpuError_t::gpuErrorInvalidDevicePointer;
    case cudaErrorInvalidMemcpyDirection:
      return gpuError_t::gpuErrorInvalidMemcpyDirection;
    case cudaErrorInsufficientDriver:
      return gpuError_t::gpuErrorInsufficientDriver;
    case cudaErrorMissingConfiguration:
      return gpuError_t::gpuErrorMissingConfiguration;
    case cudaErrorPriorLaunchFailure:
      return gpuError_t::gpuErrorPriorLaunchFailure;
    case cudaErrorInvalidDeviceFunction:
      return gpuError_t::gpuErrorInvalidDeviceFunction;
    case cudaErrorNoDevice:
      return gpuError_t::gpuErrorNoDevice;
    case cudaErrorInvalidDevice:
      return gpuError_t::gpuErrorInvalidDevice;
    case cudaErrorInvalidKernelImage:
      return gpuError_t::gpuErrorInvalidImage;
    case cudaErrorDeviceUninitialized:
      return gpuError_t::gpuErrorInvalidContext;
    case cudaErrorMapBufferObjectFailed:
      return gpuError_t::gpuErrorMapFailed;
    case cudaErrorUnmapBufferObjectFailed:
      return gpuError_t::gpuErrorUnmapFailed;
    case cudaErrorArrayIsMapped:
      return gpuError_t::gpuErrorArrayIsMapped;
    case cudaErrorAlreadyMapped:
      return gpuError_t::gpuErrorAlreadyMapped;
    case cudaErrorNoKernelImageForDevice:
      return gpuError_t::gpuErrorNoBinaryForGpu;
    case cudaErrorAlreadyAcquired:
      return gpuError_t::gpuErrorAlreadyAcquired;
    case cudaErrorNotMapped:
      return gpuError_t::gpuErrorNotMapped;
    case cudaErrorNotMappedAsArray:
      return gpuError_t::gpuErrorNotMappedAsArray;
    case cudaErrorNotMappedAsPointer:
      return gpuError_t::gpuErrorNotMappedAsPointer;
    case cudaErrorECCUncorrectable:
      return gpuError_t::gpuErrorECCNotCorrectable;
    case cudaErrorUnsupportedLimit:
      return gpuError_t::gpuErrorUnsupportedLimit;
    case cudaErrorDeviceAlreadyInUse:
      return gpuError_t::gpuErrorContextAlreadyInUse;
    case cudaErrorPeerAccessUnsupported:
      return gpuError_t::gpuErrorPeerAccessUnsupported;
    case cudaErrorInvalidPtx:
      return gpuError_t::gpuErrorInvalidKernelFile;
    case cudaErrorInvalidGraphicsContext:
      return gpuError_t::gpuErrorInvalidGraphicsContext;
    case cudaErrorInvalidSource:
      return gpuError_t::gpuErrorInvalidSource;
    case cudaErrorFileNotFound:
      return gpuError_t::gpuErrorFileNotFound;
    case cudaErrorSharedObjectSymbolNotFound:
      return gpuError_t::gpuErrorSharedObjectSymbolNotFound;
    case cudaErrorSharedObjectInitFailed:
      return gpuError_t::gpuErrorSharedObjectInitFailed;
    case cudaErrorOperatingSystem:
      return gpuError_t::gpuErrorOperatingSystem;
    case cudaErrorInvalidResourceHandle:
      return gpuError_t::gpuErrorInvalidHandle;
    case cudaErrorIllegalState:
      return gpuError_t::gpuErrorIllegalState;
    case cudaErrorSymbolNotFound:
      return gpuError_t::gpuErrorNotFound;
    case cudaErrorNotReady:
      return gpuError_t::gpuErrorNotReady;
    case cudaErrorIllegalAddress:
      return gpuError_t::gpuErrorIllegalAddress;
    case cudaErrorLaunchOutOfResources:
      return gpuError_t::gpuErrorLaunchOutOfResources;
    case cudaErrorLaunchTimeout:
      return gpuError_t::gpuErrorLaunchTimeOut;
    case cudaErrorPeerAccessAlreadyEnabled:
      return gpuError_t::gpuErrorPeerAccessAlreadyEnabled;
    case cudaErrorPeerAccessNotEnabled:
      return gpuError_t::gpuErrorPeerAccessNotEnabled;
    case cudaErrorSetOnActiveProcess:
      return gpuError_t::gpuErrorSetOnActiveProcess;
    case cudaErrorContextIsDestroyed:
      return gpuError_t::gpuErrorContextIsDestroyed;
    case cudaErrorAssert:
      return gpuError_t::gpuErrorAssert;
    case cudaErrorHostMemoryAlreadyRegistered:
      return gpuError_t::gpuErrorHostMemoryAlreadyRegistered;
    case cudaErrorHostMemoryNotRegistered:
      return gpuError_t::gpuErrorHostMemoryNotRegistered;
    case cudaErrorLaunchFailure:
      return gpuError_t::gpuErrorLaunchFailure;
    case cudaErrorCooperativeLaunchTooLarge:
      return gpuError_t::gpuErrorCooperativeLaunchTooLarge;
    case cudaErrorNotSupported:
      return gpuError_t::gpuErrorNotSupported;
    case cudaErrorStreamCaptureUnsupported:
      return gpuError_t::gpuErrorStreamCaptureUnsupported;
    case cudaErrorStreamCaptureInvalidated:
      return gpuError_t::gpuErrorStreamCaptureInvalidated;
    case cudaErrorStreamCaptureMerge:
      return gpuError_t::gpuErrorStreamCaptureMerge;
    case cudaErrorStreamCaptureUnmatched:
      return gpuError_t::gpuErrorStreamCaptureUnmatched;
    case cudaErrorStreamCaptureUnjoined:
      return gpuError_t::gpuErrorStreamCaptureUnjoined;
    case cudaErrorStreamCaptureIsolation:
      return gpuError_t::gpuErrorStreamCaptureIsolation;
    case cudaErrorStreamCaptureImplicit:
      return gpuError_t::gpuErrorStreamCaptureImplicit;
    case cudaErrorCapturedEvent:
      return gpuError_t::gpuErrorCapturedEvent;
    case cudaErrorStreamCaptureWrongThread:
      return gpuError_t::gpuErrorStreamCaptureWrongThread;
    case cudaErrorGraphExecUpdateFailure:
      return gpuError_t::gpuErrorGraphExecUpdateFailure;
    case cudaErrorUnknown:
    default:
      return gpuError_t::gpuErrorUnknown;
  }
}

static cudaError_t gpuErrorToCudaError(const gpuError_t error) {
  switch (error) {
    case gpuError_t::gpuSuccess:
      return cudaSuccess;
    case gpuError_t::gpuErrorInvalidValue:
      return cudaErrorInvalidValue;
    case gpuError_t::gpuErrorOutOfMemory:
      return cudaErrorMemoryAllocation;
    case gpuError_t::gpuErrorNotInitialized:
      return cudaErrorInitializationError;
    case gpuError_t::gpuErrorDeinitialized:
      return cudaErrorCudartUnloading;
    case gpuError_t::gpuErrorProfilerDisabled:
      return cudaErrorProfilerDisabled;
    case gpuError_t::gpuErrorProfilerNotInitialized:
      return cudaErrorProfilerNotInitialized;
    case gpuError_t::gpuErrorProfilerAlreadyStarted:
      return cudaErrorProfilerAlreadyStarted;
    case gpuError_t::gpuErrorProfilerAlreadyStopped:
      return cudaErrorProfilerAlreadyStopped;
    case gpuError_t::gpuErrorInvalidConfiguration:
      return cudaErrorInvalidConfiguration;
    case gpuError_t::gpuErrorInvalidPitchValue:
      return cudaErrorInvalidPitchValue;
    case gpuError_t::gpuErrorInvalidSymbol:
      return cudaErrorInvalidSymbol;
    case gpuError_t::gpuErrorInvalidDevicePointer:
      return cudaErrorInvalidDevicePointer;
    case gpuError_t::gpuErrorInvalidMemcpyDirection:
      return cudaErrorInvalidMemcpyDirection;
    case gpuError_t::gpuErrorInsufficientDriver:
      return cudaErrorInsufficientDriver;
    case gpuError_t::gpuErrorMissingConfiguration:
      return cudaErrorMissingConfiguration;
    case gpuError_t::gpuErrorPriorLaunchFailure:
      return cudaErrorPriorLaunchFailure;
    case gpuError_t::gpuErrorInvalidDeviceFunction:
      return cudaErrorInvalidDeviceFunction;
    case gpuError_t::gpuErrorNoDevice:
      return cudaErrorNoDevice;
    case gpuError_t::gpuErrorInvalidDevice:
      return cudaErrorInvalidDevice;
    case gpuError_t::gpuErrorInvalidImage:
      return cudaErrorInvalidKernelImage;
    case gpuError_t::gpuErrorInvalidContext:
      return cudaErrorDeviceUninitialized;
    case gpuError_t::gpuErrorContextAlreadyCurrent:
      return cudaErrorDeviceAlreadyInUse;
    case gpuError_t::gpuErrorMapFailed:
      return cudaErrorMapBufferObjectFailed;
    case gpuError_t::gpuErrorUnmapFailed:
      return cudaErrorUnmapBufferObjectFailed;
    case gpuError_t::gpuErrorArrayIsMapped:
      return cudaErrorArrayIsMapped;
    case gpuError_t::gpuErrorAlreadyMapped:
      return cudaErrorAlreadyMapped;
    case gpuError_t::gpuErrorNoBinaryForGpu:
      return cudaErrorNoKernelImageForDevice;
    case gpuError_t::gpuErrorAlreadyAcquired:
      return cudaErrorAlreadyAcquired;
    case gpuError_t::gpuErrorNotMapped:
      return cudaErrorNotMapped;
    case gpuError_t::gpuErrorNotMappedAsArray:
      return cudaErrorNotMappedAsArray;
    case gpuError_t::gpuErrorNotMappedAsPointer:
      return cudaErrorNotMappedAsPointer;
    case gpuError_t::gpuErrorECCNotCorrectable:
      return cudaErrorECCUncorrectable;
    case gpuError_t::gpuErrorUnsupportedLimit:
      return cudaErrorUnsupportedLimit;
    case gpuError_t::gpuErrorContextAlreadyInUse:
      return cudaErrorDeviceAlreadyInUse;
    case gpuError_t::gpuErrorPeerAccessUnsupported:
      return cudaErrorPeerAccessUnsupported;
    case gpuError_t::gpuErrorInvalidKernelFile:
      return cudaErrorInvalidPtx;
    case gpuError_t::gpuErrorInvalidGraphicsContext:
      return cudaErrorInvalidGraphicsContext;
    case gpuError_t::gpuErrorInvalidSource:
      return cudaErrorInvalidSource;
    case gpuError_t::gpuErrorFileNotFound:
      return cudaErrorFileNotFound;
    case gpuError_t::gpuErrorSharedObjectSymbolNotFound:
      return cudaErrorSharedObjectSymbolNotFound;
    case gpuError_t::gpuErrorSharedObjectInitFailed:
      return cudaErrorSharedObjectInitFailed;
    case gpuError_t::gpuErrorOperatingSystem:
      return cudaErrorOperatingSystem;
    case gpuError_t::gpuErrorInvalidHandle:
      return cudaErrorInvalidResourceHandle;
    case gpuError_t::gpuErrorIllegalState:
      return cudaErrorIllegalState;
    case gpuError_t::gpuErrorNotFound:
      return cudaErrorSymbolNotFound;
    case gpuError_t::gpuErrorNotReady:
      return cudaErrorNotReady;
    case gpuError_t::gpuErrorIllegalAddress:
      return cudaErrorIllegalAddress;
    case gpuError_t::gpuErrorLaunchOutOfResources:
      return cudaErrorLaunchOutOfResources;
    case gpuError_t::gpuErrorLaunchTimeOut:
      return cudaErrorLaunchTimeout;
    case gpuError_t::gpuErrorPeerAccessAlreadyEnabled:
      return cudaErrorPeerAccessAlreadyEnabled;
    case gpuError_t::gpuErrorPeerAccessNotEnabled:
      return cudaErrorPeerAccessNotEnabled;
    case gpuError_t::gpuErrorSetOnActiveProcess:
      return cudaErrorSetOnActiveProcess;
    case gpuError_t::gpuErrorContextIsDestroyed:
      return cudaErrorContextIsDestroyed;
    case gpuError_t::gpuErrorAssert:
      return cudaErrorAssert;
    case gpuError_t::gpuErrorHostMemoryAlreadyRegistered:
      return cudaErrorHostMemoryAlreadyRegistered;
    case gpuError_t::gpuErrorHostMemoryNotRegistered:
      return cudaErrorHostMemoryNotRegistered;
    case gpuError_t::gpuErrorLaunchFailure:
      return cudaErrorLaunchFailure;
    case gpuError_t::gpuErrorCooperativeLaunchTooLarge:
      return cudaErrorCooperativeLaunchTooLarge;
    case gpuError_t::gpuErrorNotSupported:
      return cudaErrorNotSupported;
    case gpuError_t::gpuErrorStreamCaptureUnsupported:
      return cudaErrorStreamCaptureUnsupported;
    case gpuError_t::gpuErrorStreamCaptureInvalidated:
      return cudaErrorStreamCaptureInvalidated;
    case gpuError_t::gpuErrorStreamCaptureMerge:
      return cudaErrorStreamCaptureMerge;
    case gpuError_t::gpuErrorStreamCaptureUnmatched:
      return cudaErrorStreamCaptureUnmatched;
    case gpuError_t::gpuErrorStreamCaptureUnjoined:
      return cudaErrorStreamCaptureUnjoined;
    case gpuError_t::gpuErrorStreamCaptureIsolation:
      return cudaErrorStreamCaptureIsolation;
    case gpuError_t::gpuErrorStreamCaptureImplicit:
      return cudaErrorStreamCaptureImplicit;
    case gpuError_t::gpuErrorCapturedEvent:
      return cudaErrorCapturedEvent;
    case gpuError_t::gpuErrorStreamCaptureWrongThread:
      return cudaErrorStreamCaptureWrongThread;
    case gpuError_t::gpuErrorGraphExecUpdateFailure:
      return cudaErrorGraphExecUpdateFailure;
    case gpuError_t::gpuErrorUnknown:
    default:
      return cudaErrorUnknown;
  }
}

static cudaMemcpyKind mapMemcpyKind(const gpuMemcpyKind copy_kind) {
  switch (copy_kind) {
    case gpuMemcpyKind::gpuMemcpyHostToHost:
      return cudaMemcpyHostToHost;
    case gpuMemcpyKind::gpuMemcpyHostToDevice:
      return cudaMemcpyHostToDevice;
    case gpuMemcpyKind::gpuMemcpyDeviceToHost:
      return cudaMemcpyDeviceToHost;
    case gpuMemcpyKind::gpuMemcpyDeviceToDevice:
      return cudaMemcpyDeviceToDevice;
    case gpuMemcpyKind::gpuMemcpyDefault:
    default:
      return cudaMemcpyDefault;
  }
}
}  // namespace

// Device management

GPU_MATE_API gpuError_t gpuGetDevice(int* id) {
  return cudaErrorToGpuError(cudaGetDevice(id));
}

GPU_MATE_API gpuError_t gpuSetDevice(int id) {
  return cudaErrorToGpuError(cudaSetDevice(id));
}

GPU_MATE_API gpuError_t gpuGetDeviceCount(int* count) {
  return cudaErrorToGpuError(cudaGetDeviceCount(count));
}

GPU_MATE_API gpuError_t gpuDeviceSynchronize() {
  return cudaErrorToGpuError(cudaDeviceSynchronize());
}

GPU_MATE_API gpuError_t gpuDeviceReset() {
  return cudaErrorToGpuError(cudaDeviceReset());
}

// Error handling

gpuError_t gpuGetLastError() {
  return cudaErrorToGpuError(cudaGetLastError());
};

gpuError_t gpuPeekAtLastError() {
  return cudaErrorToGpuError(cudaPeekAtLastError());
}

const char* gpuGetErrorName(gpuError_t error) {
  return cudaGetErrorName(gpuErrorToCudaError(error));
}

const char* gpuGetErrorString(gpuError_t error) {
  return cudaGetErrorString(gpuErrorToCudaError(error));
}

// Memory management

gpuError_t gpuMalloc(void** ptr, const size_t size) {
  return cudaErrorToGpuError(cudaMalloc(ptr, size));
}

gpuError_t gpuMemcpy(void* dst, const void* src, const size_t size,
                     const gpuMemcpyKind copy_kind) {
  return cudaErrorToGpuError(
      cudaMemcpy(dst, src, size, mapMemcpyKind(copy_kind)));
}

gpuError_t gpuFree(void* ptr) { return cudaErrorToGpuError(cudaFree(ptr)); }

}  // namespace runtime
}  // namespace gpu_mate
