#ifndef CPUBUFFER_H
#define CPUBUFFER_H

#include "../Buffer.h"

namespace CudaMgr_Namespace {
    class CudaMgr;
}

namespace Buffer_Namespace {
    class CpuBuffer: public Buffer {

        public:
            CpuBuffer(BufferMgr *bm, BufferList::iterator segIt, const int deviceId, CudaMgr_Namespace::CudaMgr * cudaMgr, const size_t pageSize = 512, const size_t numBytes = 0);

            virtual inline Data_Namespace::MemoryLevel getType() const {return CPU_LEVEL;}


        private:
            void readData(int8_t * const dst, const size_t numBytes, const MemoryLevel dstMemoryLevel, const size_t offset = 0);
            void writeData(int8_t * const src, const size_t numBytes, const MemoryLevel srcMemoryLevel, const size_t offset = 0);
            CudaMgr_Namespace::CudaMgr * cudaMgr_;

    };
} // Buffer_Namespace


#endif // CPUBUFFER_H
