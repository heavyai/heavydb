#include "CpuBufferMgr.h"
#include "CpuBuffer.h"
#ifdef USE_GPU
#include "../CudaUtils.h"
#endif

namespace Buffer_Namespace {

    CpuBufferMgr::CpuBufferMgr(const size_t maxBufferSize, const CpuBufferMgrMemType cpuBufferMgrMemType, const size_t bufferAllocIncrement,  const size_t pageSize, AbstractBufferMgr *parentMgr) : BufferMgr(maxBufferSize, bufferAllocIncrement, pageSize, parentMgr), cpuBufferMgrMemType_(cpuBufferMgrMemType) {}

    CpuBufferMgr::~CpuBufferMgr() {
        freeAllMem();
    }

    void CpuBufferMgr::addSlab(const size_t slabSize) {
        slabs_.resize(slabs_.size()+1);
        if (cpuBufferMgrMemType_ == CUDA_HOST) {
            #ifdef USE_GPU
            CudaUtils::allocPinnedHostMem(slabs_.back(),slabSize,1);
            #endif 
        }
        else {

            slabs_.back() = new int8_t[slabSize];
        }
        //slabs_.push_back((int8_t *) malloc(slabSize));
        slabSegments_.resize(slabSegments_.size()+1);
        slabSegments_[slabSegments_.size()-1].push_back(BufferSeg(0,numPagesPerSlab_));
    }

    void CpuBufferMgr::freeAllMem() {
        for (auto bufIt = slabs_.begin(); bufIt != slabs_.end(); ++bufIt) { 
            if (cpuBufferMgrMemType_ == CUDA_HOST) {
                #ifdef USE_GPU
                CudaUtils::hostFree(*bufIt);
                #endif
            }
            else {
                delete [] *bufIt;
            }
        }
    }
    
    //void BufferMgr::fetchChunk(const ChunkKey &key, AbstractBuffer *destBuffer, const size_t numBytes) {
    //    auto chunkIt = chunkIndex_.find(key);
    //    AbstractBuffer * buffer;
    //    if (chunkIt == chunkIndex_.end()) {
    //        if (parentMgr_ == 0) {
    //            throw std::runtime_error("Chunk does not exist");
    //        }
    //        buffer = createChunk(key,pageSize_,numBytes);
    //        parentMgr -> fetchChunk(key, buffer, numBytes);
    //    }
    //    else {
    //        buffer = chunkIt -> second -> buffer;
    //    }
    //    size_t chunkSize = numBytes == 0 ? buffer -> size() : numBytes;
    //    destBuffer->reserve(chunkSize);
    //    std::cout << "After reserve chunksize: " << chunkSize << std::endl;
    //    if (chunk->isUpdated()) {
    //        chunk->read(destBuffer->getMemoryPtr(),chunkSize,0);
    //    }
    //    else {
    //        chunk->read(destBuffer->getMemoryPtr()+destBuffer->size(),chunkSize-destBuffer->size(),destBuffer->size());
    //    }
    //    destBuffer->setSize(chunkSize);
    //}



    void CpuBufferMgr::allocateBuffer(BufferList::iterator segIt, const size_t pageSize, const size_t initialSize) {
        new CpuBuffer(this, segIt, pageSize, initialSize); // this line is admittedly a bit weird but the segment iterator passed into buffer takes the address of the new Buffer in its buffer member
    }


} // Buffer_Namespace
