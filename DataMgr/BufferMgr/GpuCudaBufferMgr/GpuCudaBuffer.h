#ifndef GPUCUDABUFFER_H
#define GPUCUDABUFFER_H

#include "../Buffer.h"

namespace Buffer_Namespace {
    class GpuCudaBuffer: public Buffer {

        public:
            GpuCudaBuffer(BufferMgr *bm, BufferList::iterator segIt, const int gpuNum, const mapd_size_t pageSize = 512, const mapd_size_t numBytes = 0);

        private:
            void readData(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset);
            void writeData(mapd_addr_t const src, const mapd_size_t numBytes, const mapd_size_t offset);

            int gpuNum_;

    };
} // Buffer_Namespace


#endif // GPUCUDABUFFER_H
