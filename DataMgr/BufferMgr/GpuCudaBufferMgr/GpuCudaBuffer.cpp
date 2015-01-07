#include "GpuCudaBuffer.h"
#include "../CudaUtils.h"

namespace Buffer_Namespace {

    GpuCudaBuffer::GpuCudaBuffer(BufferMgr *bm, BufferList::iterator segIt, const int gpuNum,  const mapd_size_t pageSize, const mapd_size_t numBytes): Buffer(bm, segIt, pageSize, numBytes), gpuNum_(gpuNum) {} 

    void GpuCudaBuffer::readData(mapd_addr_t const dst, const mapd_size_t numBytes, const BufferType dstBufferType, const mapd_size_t offset) {
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

    void GpuCudaBuffer::writeData(mapd_addr_t const src, const mapd_size_t numBytes, const BufferType srcBufferType, const mapd_size_t offset) {
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
