#include "CpuBuffer.h"
#include "../CudaUtils.h"

#include <cstring>

namespace Buffer_Namespace {

    CpuBuffer::CpuBuffer(BufferMgr *bm, BufferList::iterator segIt,  const size_t pageSize, const size_t numBytes): Buffer(bm, segIt, pageSize, numBytes) {}


    void CpuBuffer::readData(int8_t * const dst, const size_t numBytes, const BufferType dstBufferType, const size_t offset) {
        if (dstBufferType == CPU_BUFFER) {
            memcpy(dst, mem_ + offset, numBytes);
        }
        else if (dstBufferType == GPU_BUFFER) {
            //CudaUtils::copyToGpu(dst,mem_+offset,numBytes,1,dst->getDeviceId());
            //@todo: use actual device id in next call
            #ifdef USE_GPU
            CudaUtils::copyToGpu(dst,mem_+offset,numBytes,1,0);
            #endif
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }

    void CpuBuffer::writeData(int8_t * const src, const size_t numBytes, const BufferType srcBufferType, const size_t offset) {
        if (srcBufferType == CPU_BUFFER) {
            std::cout << "At Cpu_Buffer Writedata" << std::endl;
            memcpy(mem_+offset, src, numBytes);
        }
        else if (srcBufferType == GPU_BUFFER) {
            std::cout << "At Gpu_Buffer Writedata" << std::endl;
            //CudaUtils::copyToHost(mem_+offset, src, numBytes,1,src->getDeviceId());
            //@todo: use actual device id in next call
            #ifdef USE_GPU
            CudaUtils::copyToHost(mem_+offset, src, numBytes,1,0);
            #endif
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }


} // Buffer_Namespace
