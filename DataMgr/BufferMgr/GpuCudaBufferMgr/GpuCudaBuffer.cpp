#include "GpuCudaBuffer.h"
#include "../CudaUtils.h"

namespace Buffer_Namespace {

    GpuCudaBuffer::GpuCudaBuffer(BufferMgr *bm, BufferList::iterator segIt, const int gpuNum,  const size_t pageSize, const size_t numBytes): Buffer(bm, segIt, pageSize, numBytes), gpuNum_(gpuNum) {} 

    void GpuCudaBuffer::readData(int8_t * const dst, const size_t numBytes, const BufferType dstBufferType, const size_t offset) {
        if (dstBufferType == CPU_BUFFER) {
            CudaUtils::copyToHost(dst, mem_ + offset, numBytes, 1, gpuNum_);
        }
        else if (dstBufferType == GPU_BUFFER) {
            //CudaUtils::copyGpuToGpu(dst, mem_ + offset, numBytes, 1, dst->getDeviceId());
            //@todo, populate device id
            CudaUtils::copyGpuToGpu(dst, mem_ + offset, numBytes, 1, 0);
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }

    void GpuCudaBuffer::writeData(int8_t * const src, const size_t numBytes, const BufferType srcBufferType, const size_t offset) {
        if (srcBufferType == CPU_BUFFER) {
            CudaUtils::copyToGpu(mem_ + offset, src, numBytes, 1, gpuNum_);
        }
        else if (srcBufferType == GPU_BUFFER) {
            CudaUtils::copyGpuToGpu(mem_ + offset, src, numBytes, 1, gpuNum_);
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }

} // Buffer_Namespace
