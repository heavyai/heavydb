#ifndef GPUCUDABUFFERMGR_H
#define GPUCUDABUFFERMGR_H

#include "../BufferMgr.h"

namespace Buffer_Namespace {

    class GpuCudaBufferMgr :  public BufferMgr {

        public:
            GpuCudaBufferMgr(const size_t maxBufferSize, const int gpuNum, const size_t bufferAllocIncrement = 1073741824,  const size_t pageSize = 512, AbstractBufferMgr *parentMgr = 0);
            virtual inline MgrType getMgrType () {return GPU_MGR;}
            ~GpuCudaBufferMgr();


        private:
            virtual void addSlab(const size_t slabSize);
            virtual void freeAllMem();
            virtual void allocateBuffer(BufferList::iterator segIt, const size_t pageSize, const size_t initialSize);
            int gpuNum_;

    };

} // Buffer_Namespace

#endif // CPUBUFFERMGR_H
