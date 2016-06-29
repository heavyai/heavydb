#include "GpuCudaBuffer.h"
#include "../../../CudaMgr/CudaMgr.h"

#include <assert.h>

namespace Buffer_Namespace {

GpuCudaBuffer::GpuCudaBuffer(BufferMgr* bm,
                             BufferList::iterator segIt,
                             const int deviceId,
                             CudaMgr_Namespace::CudaMgr* cudaMgr,
                             const size_t pageSize,
                             const size_t numBytes)
    : Buffer(bm, segIt, deviceId, pageSize, numBytes), cudaMgr_(cudaMgr) {
}

void GpuCudaBuffer::readData(int8_t* const dst,
                             const size_t numBytes,
                             const size_t offset,
                             const MemoryLevel dstBufferType,
                             const int dstDeviceId) {
  if (dstBufferType == CPU_LEVEL) {
    cudaMgr_->copyDeviceToHost(dst, mem_ + offset, numBytes, deviceId_);  // need to replace 0 with gpu num
  } else if (dstBufferType == GPU_LEVEL) {
    //@todo fill this in
    // CudaUtils::copyGpuToGpu(dst, mem_ + offset, numBytes, 1, dst->getDeviceId());
    //@todo, populate device id
    // CudaUtils::copyGpuToGpu(dst, mem_ + offset, numBytes, 1, 0);
    cudaMgr_->copyDeviceToDevice(dst, mem_ + offset, numBytes, dstDeviceId, deviceId_);

  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

void GpuCudaBuffer::writeData(int8_t* const src,
                              const size_t numBytes,
                              const size_t offset,
                              const MemoryLevel srcBufferType,
                              const int srcDeviceId) {
  if (srcBufferType == CPU_LEVEL) {
    // std::cout << "Writing to GPU from source CPU" << std::endl;

    cudaMgr_->copyHostToDevice(mem_ + offset, src, numBytes, deviceId_);  // need to replace 0 with gpu num

  } else if (srcBufferType == GPU_LEVEL) {
    // std::cout << "Writing to GPU from source GPU" << std::endl;
    assert(srcDeviceId >= 0);
    cudaMgr_->copyDeviceToDevice(mem_ + offset, src, numBytes, deviceId_, srcDeviceId);
    // CudaUtils::copyGpuToGpu(mem_ + offset, src, numBytes, 1, deviceId_);
    //@todo fill this in
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

}  // Buffer_Namespace
