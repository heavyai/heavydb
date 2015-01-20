#ifndef GPUCUDABUFFER_H
#define GPUCUDABUFFER_H

#include "../Buffer.h"

namespace Buffer_Namespace {
    class GpuCudaBuffer: public Buffer {

        public:
            GpuCudaBuffer(BufferMgr *bm, BufferList::iterator segIt, const int gpuNum, const mapd_size_t pageSize = 512, const mapd_size_t numBytes = 0);
            virtual inline Data_Namespace::BufferType getType() const {return GPU_BUFFER;}
            virtual inline int getDeviceId() const { return gpuNum_; }

        private:
            void readData(mapd_addr_t const dst, const mapd_size_t numBytes, const BufferType dstBufferType = CPU_BUFFER, const mapd_size_t offset = 0);
            void writeData(mapd_addr_t const src, const mapd_size_t numBytes, const BufferType srcBufferType = CPU_BUFFER, const mapd_size_t offset = 0);
            int gpuNum_;


    };
} // Buffer_Namespace


#endif // GPUCUDABUFFER_H
