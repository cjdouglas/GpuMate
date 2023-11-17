#include <cuda_runtime.h>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace runtime {
namespace {

static GpuError CudaToGpuError(const cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return GpuError::success;
    case cudaErrorInvalidValue:
      return GpuError::invalidValue;
    case cudaErrorMemoryAllocation:
      return GpuError::outOfMemory;
    case cudaErrorInitializationError:
      return GpuError::notInitialized;
    case cudaErrorCudartUnloading:
      return GpuError::deinitialized;
    case cudaErrorProfilerDisabled:
      return GpuError::profilerDisabled;
    case cudaErrorProfilerNotInitialized:
      return GpuError::profilerNotInitialized;
    case cudaErrorProfilerAlreadyStarted:
      return GpuError::profilerAlreadyStarted;
    case cudaErrorProfilerAlreadyStopped:
      return GpuError::profilerAlreadyStopped;
    case cudaErrorInvalidConfiguration:
      return GpuError::invalidConfiguration;
    case cudaErrorInvalidPitchValue:
      return GpuError::invalidPitchValue;
    case cudaErrorInvalidSymbol:
      return GpuError::invalidSymbol;
    case cudaErrorInvalidDevicePointer:
      return GpuError::invalidDevicePointer;
    case cudaErrorInvalidMemcpyDirection:
      return GpuError::invalidMemcpyDirection;
    case cudaErrorInsufficientDriver:
      return GpuError::insufficientDriver;
    case cudaErrorMissingConfiguration:
      return GpuError::missingConfiguration;
    case cudaErrorPriorLaunchFailure:
      return GpuError::priorLaunchFailure;
    case cudaErrorInvalidDeviceFunction:
      return GpuError::invalidDeviceFunction;
    case cudaErrorNoDevice:
      return GpuError::noDevice;
    case cudaErrorInvalidDevice:
      return GpuError::invalidDevice;
    case cudaErrorInvalidKernelImage:
      return GpuError::invalidImage;
    case cudaErrorDeviceUninitialized:
      return GpuError::invalidContext;
    case cudaErrorMapBufferObjectFailed:
      return GpuError::mapFailed;
    case cudaErrorUnmapBufferObjectFailed:
      return GpuError::unmapFailed;
    case cudaErrorArrayIsMapped:
      return GpuError::arrayIsMapped;
    case cudaErrorAlreadyMapped:
      return GpuError::alreadyMapped;
    case cudaErrorNoKernelImageForDevice:
      return GpuError::noBinaryForGpu;
    case cudaErrorAlreadyAcquired:
      return GpuError::alreadyAcquired;
    case cudaErrorNotMapped:
      return GpuError::notMapped;
    case cudaErrorNotMappedAsArray:
      return GpuError::notMappedAsArray;
    case cudaErrorNotMappedAsPointer:
      return GpuError::notMappedAsPointer;
    case cudaErrorECCUncorrectable:
      return GpuError::eccNotCorrectable;
    case cudaErrorUnsupportedLimit:
      return GpuError::unsupportedLimit;
    case cudaErrorDeviceAlreadyInUse:
      return GpuError::contextAlreadyInUse;
    case cudaErrorPeerAccessUnsupported:
      return GpuError::peerAccessUnsupported;
    case cudaErrorInvalidPtx:
      return GpuError::invalidKernelFile;
    case cudaErrorInvalidGraphicsContext:
      return GpuError::invalidGraphicsContext;
    case cudaErrorInvalidSource:
      return GpuError::invalidSource;
    case cudaErrorFileNotFound:
      return GpuError::fileNotFound;
    case cudaErrorSharedObjectSymbolNotFound:
      return GpuError::sharedObjectSymbolNotFound;
    case cudaErrorSharedObjectInitFailed:
      return GpuError::sharedObjectInitFailed;
    case cudaErrorOperatingSystem:
      return GpuError::operatingSystem;
    case cudaErrorInvalidResourceHandle:
      return GpuError::invalidHandle;
    case cudaErrorIllegalState:
      return GpuError::illegalState;
    case cudaErrorSymbolNotFound:
      return GpuError::notFound;
    case cudaErrorNotReady:
      return GpuError::notReady;
    case cudaErrorIllegalAddress:
      return GpuError::illegalAddress;
    case cudaErrorLaunchOutOfResources:
      return GpuError::launchOutOfResources;
    case cudaErrorLaunchTimeout:
      return GpuError::launchTimeOut;
    case cudaErrorPeerAccessAlreadyEnabled:
      return GpuError::peerAccessAlreadyEnabled;
    case cudaErrorPeerAccessNotEnabled:
      return GpuError::peerAccessNotEnabled;
    case cudaErrorSetOnActiveProcess:
      return GpuError::setOnActiveProcess;
    case cudaErrorContextIsDestroyed:
      return GpuError::contextIsDestroyed;
    case cudaErrorAssert:
      return GpuError::assert;
    case cudaErrorHostMemoryAlreadyRegistered:
      return GpuError::hostMemoryAlreadyRegistered;
    case cudaErrorHostMemoryNotRegistered:
      return GpuError::hostMemoryNotRegistered;
    case cudaErrorLaunchFailure:
      return GpuError::launchFailure;
    case cudaErrorCooperativeLaunchTooLarge:
      return GpuError::cooperativeLaunchTooLarge;
    case cudaErrorNotSupported:
      return GpuError::notSupported;
    case cudaErrorStreamCaptureUnsupported:
      return GpuError::streamCaptureUnsupported;
    case cudaErrorStreamCaptureInvalidated:
      return GpuError::streamCaptureInvalidated;
    case cudaErrorStreamCaptureMerge:
      return GpuError::streamCaptureMerge;
    case cudaErrorStreamCaptureUnmatched:
      return GpuError::streamCaptureUnmatched;
    case cudaErrorStreamCaptureUnjoined:
      return GpuError::streamCaptureUnjoined;
    case cudaErrorStreamCaptureIsolation:
      return GpuError::streamCaptureIsolation;
    case cudaErrorStreamCaptureImplicit:
      return GpuError::streamCaptureImplicit;
    case cudaErrorCapturedEvent:
      return GpuError::capturedEvent;
    case cudaErrorStreamCaptureWrongThread:
      return GpuError::streamCaptureWrongThread;
    case cudaErrorGraphExecUpdateFailure:
      return GpuError::graphExecUpdateFailure;
    case cudaErrorUnknown:
    default:
      return GpuError::gpuErrorUnknown;
  }
}

static cudaError_t GpuToCudaError(const GpuError error) {
  switch (error) {
    case GpuError::success:
      return cudaSuccess;
    case GpuError::invalidValue:
      return cudaErrorInvalidValue;
    case GpuError::outOfMemory:
      return cudaErrorMemoryAllocation;
    case GpuError::notInitialized:
      return cudaErrorInitializationError;
    case GpuError::deinitialized:
      return cudaErrorCudartUnloading;
    case GpuError::profilerDisabled:
      return cudaErrorProfilerDisabled;
    case GpuError::profilerNotInitialized:
      return cudaErrorProfilerNotInitialized;
    case GpuError::profilerAlreadyStarted:
      return cudaErrorProfilerAlreadyStarted;
    case GpuError::profilerAlreadyStopped:
      return cudaErrorProfilerAlreadyStopped;
    case GpuError::invalidConfiguration:
      return cudaErrorInvalidConfiguration;
    case GpuError::invalidPitchValue:
      return cudaErrorInvalidPitchValue;
    case GpuError::invalidSymbol:
      return cudaErrorInvalidSymbol;
    case GpuError::invalidDevicePointer:
      return cudaErrorInvalidDevicePointer;
    case GpuError::invalidMemcpyDirection:
      return cudaErrorInvalidMemcpyDirection;
    case GpuError::insufficientDriver:
      return cudaErrorInsufficientDriver;
    case GpuError::missingConfiguration:
      return cudaErrorMissingConfiguration;
    case GpuError::priorLaunchFailure:
      return cudaErrorPriorLaunchFailure;
    case GpuError::invalidDeviceFunction:
      return cudaErrorInvalidDeviceFunction;
    case GpuError::noDevice:
      return cudaErrorNoDevice;
    case GpuError::invalidDevice:
      return cudaErrorInvalidDevice;
    case GpuError::invalidImage:
      return cudaErrorInvalidKernelImage;
    case GpuError::invalidContext:
      return cudaErrorDeviceUninitialized;
    case GpuError::contextAlreadyCurrent:
      return cudaErrorDeviceAlreadyInUse;
    case GpuError::mapFailed:
      return cudaErrorMapBufferObjectFailed;
    case GpuError::unmapFailed:
      return cudaErrorUnmapBufferObjectFailed;
    case GpuError::arrayIsMapped:
      return cudaErrorArrayIsMapped;
    case GpuError::alreadyMapped:
      return cudaErrorAlreadyMapped;
    case GpuError::noBinaryForGpu:
      return cudaErrorNoKernelImageForDevice;
    case GpuError::alreadyAcquired:
      return cudaErrorAlreadyAcquired;
    case GpuError::notMapped:
      return cudaErrorNotMapped;
    case GpuError::notMappedAsArray:
      return cudaErrorNotMappedAsArray;
    case GpuError::notMappedAsPointer:
      return cudaErrorNotMappedAsPointer;
    case GpuError::eccNotCorrectable:
      return cudaErrorECCUncorrectable;
    case GpuError::unsupportedLimit:
      return cudaErrorUnsupportedLimit;
    case GpuError::contextAlreadyInUse:
      return cudaErrorDeviceAlreadyInUse;
    case GpuError::peerAccessUnsupported:
      return cudaErrorPeerAccessUnsupported;
    case GpuError::invalidKernelFile:
      return cudaErrorInvalidPtx;
    case GpuError::invalidGraphicsContext:
      return cudaErrorInvalidGraphicsContext;
    case GpuError::invalidSource:
      return cudaErrorInvalidSource;
    case GpuError::fileNotFound:
      return cudaErrorFileNotFound;
    case GpuError::sharedObjectSymbolNotFound:
      return cudaErrorSharedObjectSymbolNotFound;
    case GpuError::sharedObjectInitFailed:
      return cudaErrorSharedObjectInitFailed;
    case GpuError::operatingSystem:
      return cudaErrorOperatingSystem;
    case GpuError::invalidHandle:
      return cudaErrorInvalidResourceHandle;
    case GpuError::illegalState:
      return cudaErrorIllegalState;
    case GpuError::notFound:
      return cudaErrorSymbolNotFound;
    case GpuError::notReady:
      return cudaErrorNotReady;
    case GpuError::illegalAddress:
      return cudaErrorIllegalAddress;
    case GpuError::launchOutOfResources:
      return cudaErrorLaunchOutOfResources;
    case GpuError::launchTimeOut:
      return cudaErrorLaunchTimeout;
    case GpuError::peerAccessAlreadyEnabled:
      return cudaErrorPeerAccessAlreadyEnabled;
    case GpuError::peerAccessNotEnabled:
      return cudaErrorPeerAccessNotEnabled;
    case GpuError::setOnActiveProcess:
      return cudaErrorSetOnActiveProcess;
    case GpuError::contextIsDestroyed:
      return cudaErrorContextIsDestroyed;
    case GpuError::assert:
      return cudaErrorAssert;
    case GpuError::hostMemoryAlreadyRegistered:
      return cudaErrorHostMemoryAlreadyRegistered;
    case GpuError::hostMemoryNotRegistered:
      return cudaErrorHostMemoryNotRegistered;
    case GpuError::launchFailure:
      return cudaErrorLaunchFailure;
    case GpuError::cooperativeLaunchTooLarge:
      return cudaErrorCooperativeLaunchTooLarge;
    case GpuError::notSupported:
      return cudaErrorNotSupported;
    case GpuError::streamCaptureUnsupported:
      return cudaErrorStreamCaptureUnsupported;
    case GpuError::streamCaptureInvalidated:
      return cudaErrorStreamCaptureInvalidated;
    case GpuError::streamCaptureMerge:
      return cudaErrorStreamCaptureMerge;
    case GpuError::streamCaptureUnmatched:
      return cudaErrorStreamCaptureUnmatched;
    case GpuError::streamCaptureUnjoined:
      return cudaErrorStreamCaptureUnjoined;
    case GpuError::streamCaptureIsolation:
      return cudaErrorStreamCaptureIsolation;
    case GpuError::streamCaptureImplicit:
      return cudaErrorStreamCaptureImplicit;
    case GpuError::capturedEvent:
      return cudaErrorCapturedEvent;
    case GpuError::streamCaptureWrongThread:
      return cudaErrorStreamCaptureWrongThread;
    case GpuError::graphExecUpdateFailure:
      return cudaErrorGraphExecUpdateFailure;
    case GpuError::gpuErrorUnknown:
    default:
      return cudaErrorUnknown;
  }
}

static cudaMemcpyKind MapMemcpyKind(const GpuMemcpyKind copy_kind) {
  switch (copy_kind) {
    case GpuMemcpyKind::hostToHost:
      return cudaMemcpyHostToHost;
    case GpuMemcpyKind::hostToDevice:
      return cudaMemcpyHostToDevice;
    case GpuMemcpyKind::deviceToHost:
      return cudaMemcpyDeviceToHost;
    case GpuMemcpyKind::deviceToDevice:
      return cudaMemcpyDeviceToDevice;
    case GpuMemcpyKind::memcpyDefault:
    default:
      return cudaMemcpyDefault;
  }
}
}  // namespace

// GpuStream implementation

GpuStream::GpuStream() {
  cudaStream_t handle;
  GPU_CHECK(CudaToGpuError(cudaStreamCreate(&handle)));
  handle_ = static_cast<void*>(handle);
}

GpuStream::~GpuStream() {
  cudaStream_t handle = static_cast<cudaStream_t>(handle_);
  GPU_CHECK(CudaToGpuError(cudaStreamDestroy(handle)));
}

// Device management

GPU_MATE_API GpuError GpuGetDevice(int* id) {
  return CudaToGpuError(cudaGetDevice(id));
}

GPU_MATE_API GpuError GpuSetDevice(int id) {
  return CudaToGpuError(cudaSetDevice(id));
}

GPU_MATE_API GpuError GpuGetDeviceCount(int* count) {
  return CudaToGpuError(cudaGetDeviceCount(count));
}

GPU_MATE_API GpuError GpuDeviceSynchronize() {
  return CudaToGpuError(cudaDeviceSynchronize());
}

GPU_MATE_API GpuError GpuDeviceReset() {
  return CudaToGpuError(cudaDeviceReset());
}

// Error handling

GpuError GpuGetLastError() { return CudaToGpuError(cudaGetLastError()); };

GpuError GpuPeekAtLastError() { return CudaToGpuError(cudaPeekAtLastError()); }

const char* GpuGetErrorName(GpuError error) {
  return cudaGetErrorName(GpuToCudaError(error));
}

const char* GpuGetErrorString(GpuError error) {
  return cudaGetErrorString(GpuToCudaError(error));
}

// Memory management

GpuError GpuMalloc(void** ptr, const size_t size) {
  return CudaToGpuError(cudaMalloc(ptr, size));
}

GpuError GpuMemcpy(void* dst, const void* src, const size_t size,
                   const GpuMemcpyKind copy_kind) {
  return CudaToGpuError(cudaMemcpy(dst, src, size, MapMemcpyKind(copy_kind)));
}

GpuError GpuFree(void* ptr) { return CudaToGpuError(cudaFree(ptr)); }

}  // namespace runtime
}  // namespace gpu_mate
