#ifndef GPUCUDABUFFER_H
#define GPUCUDABUFFER_H

#include "../Buffer.h"

namespace CudaMgr_Namespace {
    class CudaMgr;
}
namespace Buffer_Namespace {
    
    class GpuCudaBuffer: public Buffer {

        public:
            GpuCudaBuffer(BufferMgr *bm, BufferList::iterator segIt, const int gpuNum, CudaMgr_Namespace::CudaMgr *cudaMgr, const size_t pageSize = 512, const size_t numBytes = 0);
            virtual inline Data_Namespace::BufferType getType() const {return GPU_BUFFER;}
            virtual inline int getDeviceId() const { return gpuNum_; }

        private:
            void readData(int8_t * const dst, const size_t numBytes, const BufferType dstBufferType = CPU_BUFFER, const size_t offset = 0);
            void writeData(int8_t * const src, const size_t numBytes, const BufferType srcBufferType = CPU_BUFFER, const size_t offset = 0);
            int gpuNum_;
            CudaMgr_Namespace::CudaMgr *cudaMgr_;


    };
} // Buffer_Namespace


#endif // GPUCUDABUFFER_H
