#include "CpuBuffer.h"
#include "../../../CudaMgr/CudaMgr.h"
#include <cstring>

namespace Buffer_Namespace {

    CpuBuffer::CpuBuffer(BufferMgr *bm, BufferList::iterator segIt, CudaMgr_Namespace::CudaMgr * cudaMgr, const size_t pageSize, const size_t numBytes): Buffer(bm, segIt, pageSize, numBytes), cudaMgr_(cudaMgr) {}


    void CpuBuffer::readData(int8_t * const dst, const size_t numBytes, const BufferType dstBufferType, const size_t offset) {
        if (dstBufferType == CPU_BUFFER) {
            memcpy(dst, mem_ + offset, numBytes);
        }
        else if (dstBufferType == GPU_BUFFER) {
            //@todo: use actual device id in next call
            cudaMgr_->copyHostToDevice(dst,mem_+offset,numBytes,0); // need to replace 0 with gpu num 
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
            //@todo: use actual device id in next call
            cudaMgr_->copyDeviceToHost(mem_+offset,src,numBytes,0); // need to replace 0 with gpu num 
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }


} // Buffer_Namespace
