#ifndef CPUBUFFER_H
#define CPUBUFFER_H

#include "../Buffer.h"

namespace Buffer_Namespace {
    class CpuBuffer: public Buffer {

        public:
            CpuBuffer(BufferMgr *bm, BufferList::iterator segIt,  const mapd_size_t pageSize = 512, const mapd_size_t numBytes = 0);

        private:
            void readData(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset);
            void writeData(mapd_addr_t const src, const mapd_size_t numBytes, const mapd_size_t offset);

    };
} // Buffer_Namespace


#endif // CPUBUFFER_H
