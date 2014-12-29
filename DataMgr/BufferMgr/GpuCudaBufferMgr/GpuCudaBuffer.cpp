#include "GpuCudaBuffer.h"
#include "../CudaUtils.h"

namespace Buffer_Namespace {

    GpuCudaBuffer::GpuCudaBuffer(BufferMgr *bm, BufferList::iterator segIt, const int gpuNum,  const mapd_size_t pageSize, const mapd_size_t numBytes): Buffer(bm, segIt, pageSize, numBytes), gpuNum_(gpuNum) {} 

    void GpuCudaBuffer::readData(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset) {
        CudaUtils::copyToHost(dst, mem_ + offset, numBytes, 1, gpuNum_);

    }

    void GpuCudaBuffer::writeData(mapd_addr_t const src, const mapd_size_t numBytes, const mapd_size_t offset) {
        CudaUtils::copyToGpu(src, mem_ + offset, numBytes, 1, gpuNum_);
    }

} // Buffer_Namespace
