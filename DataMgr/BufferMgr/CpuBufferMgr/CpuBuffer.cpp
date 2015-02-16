#include "CpuBuffer.h"
#include "../../../CudaMgr/CudaMgr.h"
#include <cstring>

namespace Buffer_Namespace {

    CpuBuffer::CpuBuffer(BufferMgr *bm, BufferList::iterator segIt, const int deviceId, CudaMgr_Namespace::CudaMgr * cudaMgr, const size_t pageSize, const size_t numBytes): Buffer(bm, segIt, deviceId, pageSize, numBytes), cudaMgr_(cudaMgr) {}


    void CpuBuffer::readData(int8_t * const dst, const size_t numBytes, const MemoryLevel dstMemoryLevel, const size_t offset) {
        if (dstMemoryLevel == CPU_LEVEL) {
            memcpy(dst, mem_ + offset, numBytes);
        }
        else if (dstMemoryLevel == GPU_LEVEL) {
            //@todo: use actual device id in next call
            cudaMgr_->copyHostToDevice(dst,mem_+offset,numBytes,0); // need to replace 0 with gpu num 
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }

    void CpuBuffer::writeData(int8_t * const src, const size_t numBytes, const MemoryLevel srcMemoryLevel, const size_t offset) {
        if (srcMemoryLevel == CPU_LEVEL) {
            std::cout << "At Cpu_Buffer Writedata" << std::endl;
            memcpy(mem_+offset, src, numBytes);
        }
        else if (srcMemoryLevel == GPU_LEVEL) {
            std::cout << "At Gpu_Buffer Writedata" << std::endl;
            //@todo: use actual device id in next call
            cudaMgr_->copyDeviceToHost(mem_+offset,src,numBytes,0); // need to replace 0 with gpu num 
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }


} // Buffer_Namespace
