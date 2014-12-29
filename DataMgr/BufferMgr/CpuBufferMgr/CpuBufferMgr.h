#ifndef CPUBUFFERMGR_H
#define CPUBUFFERMGR_H

#include "../BufferMgr.h"

namespace Buffer_Namespace {

    enum CpuBufferMgrMemType {CPU_HOST,CUDA_HOST};
    class CpuBufferMgr :  public BufferMgr {

        public:
            CpuBufferMgr(const size_t maxBufferSize,CpuBufferMgrMemType cpuBufferMgrMemType, const size_t bufferAllocIncrement = 2147483648,  const size_t pageSize = 512, File_Namespace::FileMgr *fileMgr = 0);
            ~CpuBufferMgr();
        private:
            virtual void addSlab(const size_t slabSize);
            virtual void freeAllMem();
            virtual void createBuffer(BufferList::iterator segIt, const mapd_size_t pageSize, const mapd_size_t initialSize);
            CpuBufferMgrMemType cpuBufferMgrMemType_;

    };

} // Buffer_Namespace

#endif // CPUBUFFERMGR_H
