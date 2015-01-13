#include "CpuBuffer.h"
#include "../CudaUtils.h"

#include <cstring>

namespace Buffer_Namespace {

    CpuBuffer::CpuBuffer(BufferMgr *bm, BufferList::iterator segIt,  const mapd_size_t pageSize, const mapd_size_t numBytes): Buffer(bm, segIt, pageSize, numBytes) {}


    void CpuBuffer::readData(mapd_addr_t const dst, const mapd_size_t numBytes, const BufferType dstBufferType, const mapd_size_t offset) {
        if (dstBufferType == CPU_BUFFER) {
            memcpy(dst, mem_ + offset, numBytes);
        }
        else if (dstBufferType == GPU_BUFFER) {
            //CudaUtils::copyToGpu(dst,mem_+offset,numBytes,1,dst->getDeviceId());
            //@todo: use actual device id in next call
            CudaUtils::copyToGpu(dst,mem_+offset,numBytes,1,0);
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }

    void CpuBuffer::writeData(mapd_addr_t const src, const mapd_size_t numBytes, const BufferType srcBufferType, const mapd_size_t offset) {
        if (srcBufferType == CPU_BUFFER) {
            std::cout << "At Cpu_Buffer Writedata" << std::endl;
            memcpy(mem_+offset, src, numBytes);
        }
        else if (srcBufferType == GPU_BUFFER) {
            std::cout << "At Gpu_Buffer Writedata" << std::endl;
            //CudaUtils::copyToHost(mem_+offset, src, numBytes,1,src->getDeviceId());
            //@todo: use actual device id in next call
            CudaUtils::copyToHost(mem_+offset, src, numBytes,1,0);
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }


} // Buffer_Namespace
