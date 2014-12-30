#include "GpuCudaBufferMgr.h"
#include "GpuCudaBuffer.h"
#include "../CudaUtils.h"

namespace Buffer_Namespace {

    GpuCudaBufferMgr::GpuCudaBufferMgr(const size_t maxBufferSize, const int gpuNum, const size_t bufferAllocIncrement,  const size_t pageSize, File_Namespace::FileMgr *fileMgr) : BufferMgr(maxBufferSize, bufferAllocIncrement, pageSize, fileMgr), gpuNum_(gpuNum) {}

    GpuCudaBufferMgr::~GpuCudaBufferMgr() {
        freeAllMem();
    }

    void GpuCudaBufferMgr::addSlab(const size_t slabSize) {
        slabs_.resize(slabs_.size()+1);
        CudaUtils::allocGpuMem(slabs_.back(),slabSize,1,gpuNum_);
        slabSegments_.resize(slabSegments_.size()+1);
        slabSegments_[slabSegments_.size()-1].push_back(BufferSeg(0,numPagesPerSlab_));
    }

    void GpuCudaBufferMgr::freeAllMem() {
        for (auto bufIt = slabs_.begin(); bufIt != slabs_.end(); ++bufIt) { 
            CudaUtils::gpuFree(*bufIt);
        }
    }

    void GpuCudaBufferMgr::createBuffer(BufferList::iterator segIt, const mapd_size_t pageSize, const mapd_size_t initialSize) {
        new GpuCudaBuffer(this, segIt, gpuNum_, pageSize, initialSize); // this line is admittedly a bit weird but the segment iterator passed into buffer takes the address of the new Buffer in its buffer member
    }


} // Buffer_Namespace
