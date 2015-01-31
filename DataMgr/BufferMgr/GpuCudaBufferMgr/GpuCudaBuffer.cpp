#include "GpuCudaBuffer.h"
#include "../../../CudaMgr/CudaMgr.h"

namespace Buffer_Namespace {

    GpuCudaBuffer::GpuCudaBuffer(BufferMgr *bm, BufferList::iterator segIt, const int gpuNum, CudaMgr_Namespace::CudaMgr *cudaMgr,const size_t pageSize, const size_t numBytes): Buffer(bm, segIt, pageSize, numBytes), gpuNum_(gpuNum), cudaMgr_(cudaMgr) {} 

    void GpuCudaBuffer::readData(int8_t * const dst, const size_t numBytes, const BufferType dstBufferType, const size_t offset) {
        if (dstBufferType == CPU_BUFFER) {
            cudaMgr_->copyDeviceToHost(dst,mem_+offset,numBytes,gpuNum_); // need to replace 0 with gpu num 
        }
        else if (dstBufferType == GPU_BUFFER) {
            //@todo fill this in
            //CudaUtils::copyGpuToGpu(dst, mem_ + offset, numBytes, 1, dst->getDeviceId());
            //@todo, populate device id
            //CudaUtils::copyGpuToGpu(dst, mem_ + offset, numBytes, 1, 0);
            cudaMgr_->copyDeviceToDevice(dst,mem_+offset,numBytes,gpuNum_,gpuNum_);

        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }

    void GpuCudaBuffer::writeData(int8_t * const src, const size_t numBytes, const BufferType srcBufferType, const size_t offset) {
        if (srcBufferType == CPU_BUFFER) {
            cudaMgr_->copyHostToDevice(mem_+offset,src,numBytes,gpuNum_); // need to replace 0 with gpu num 

        }
        else if (srcBufferType == GPU_BUFFER) {
            cudaMgr_->copyDeviceToDevice(mem_+offset,src,numBytes,gpuNum_,gpuNum_);
            //CudaUtils::copyGpuToGpu(mem_ + offset, src, numBytes, 1, gpuNum_);
            //@todo fill this in
        }
        else {
            throw std::runtime_error("Unsupported buffer type");
        }
    }

} // Buffer_Namespace
