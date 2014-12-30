#include "CpuBuffer.h"

namespace Buffer_Namespace {

    CpuBuffer::CpuBuffer(BufferMgr *bm, BufferList::iterator segIt,  const mapd_size_t pageSize, const mapd_size_t numBytes): Buffer(bm, segIt, pageSize, numBytes) {}


    void CpuBuffer::readData(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset) {
        memcpy(dst, mem_ + offset, numBytes);
    }

    void CpuBuffer::writeData(mapd_addr_t const src, const mapd_size_t numBytes, const mapd_size_t offset) {
        memcpy(mem_ + offset, src, numBytes);
    }


} // Buffer_Namespace
