#ifndef CPUBUFFER_H
#define CPUBUFFER_H

#include "../Buffer.h"

namespace Buffer_Namespace {
    class CpuBuffer: public Buffer {

        public:
            CpuBuffer(BufferMgr *bm, BufferList::iterator segIt,  const size_t pageSize = 512, const size_t numBytes = 0);

            virtual inline Data_Namespace::BufferType getType() const {return CPU_BUFFER;}


        private:
            void readData(int8_t * const dst, const size_t numBytes, const BufferType dstBufferType, const size_t offset = 0);
            void writeData(int8_t * const src, const size_t numBytes, const BufferType srcBufferType, const size_t offset = 0);

    };
} // Buffer_Namespace


#endif // CPUBUFFER_H
