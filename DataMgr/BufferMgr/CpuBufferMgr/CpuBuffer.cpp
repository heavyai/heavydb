#include "CpuBuffer.h"
#include "../../../CudaMgr/CudaMgr.h"
#include <cstring>
#include <assert.h>

namespace Buffer_Namespace {

CpuBuffer::CpuBuffer(BufferMgr* bm,
                     BufferList::iterator segIt,
                     const int deviceId,
                     CudaMgr_Namespace::CudaMgr* cudaMgr,
                     const size_t pageSize,
                     const size_t numBytes)
    : Buffer(bm, segIt, deviceId, pageSize, numBytes), cudaMgr_(cudaMgr) {
}

void CpuBuffer::readData(int8_t* const dst,
                         const size_t numBytes,
                         const size_t offset,
                         const MemoryLevel dstMemoryLevel,
                         const int dstDeviceId) {
  if (dstMemoryLevel == CPU_LEVEL) {
    memcpy(dst, mem_ + offset, numBytes);
  } else if (dstMemoryLevel == GPU_LEVEL) {
    //@todo: use actual device id in next call
    assert(dstDeviceId >= 0);
    cudaMgr_->copyHostToDevice(dst, mem_ + offset, numBytes, dstDeviceId);  // need to replace 0 with gpu num
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

void CpuBuffer::writeData(int8_t* const src,
                          const size_t numBytes,
                          const size_t offset,
                          const MemoryLevel srcMemoryLevel,
                          const int srcDeviceId) {
  if (srcMemoryLevel == CPU_LEVEL) {
    // std::cout << "Writing to CPU from source CPU" << std::endl;
    memcpy(mem_ + offset, src, numBytes);
  } else if (srcMemoryLevel == GPU_LEVEL) {
    // std::cout << "Writing to CPU from source GPU" << std::endl;
    //@todo: use actual device id in next call
    assert(srcDeviceId >= 0);
    cudaMgr_->copyDeviceToHost(mem_ + offset, src, numBytes, srcDeviceId);  // need to replace 0 with gpu num
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

}  // Buffer_Namespace
