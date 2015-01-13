#ifndef BUFFERSEG_H
#define BUFFERSEG_H

#include <list>

namespace Buffer_Namespace {

    class Buffer; //forward declaration

    // Memory Pages types in buffer pool
    enum MemStatus {FREE, USED};

    struct BufferSeg {
        int startPage;
        size_t numPages;
        MemStatus memStatus;
        Buffer * buffer;
        ChunkKey chunkKey;
        unsigned int lastTouched;
        unsigned int pinCount;
        int slabNum;

        BufferSeg(): memStatus (FREE), buffer(0),pinCount(0),lastTouched(0),slabNum(-1) {}
        BufferSeg(const int startPage, const size_t numPages): startPage(startPage), numPages(numPages),  memStatus (FREE), buffer(0),pinCount(0), lastTouched(0),slabNum(-1) {}
        BufferSeg(const int startPage, const size_t numPages, const MemStatus memStatus): startPage(startPage), numPages(numPages),  memStatus (memStatus), buffer(0),pinCount(0),lastTouched(0),slabNum(-1) {}
        BufferSeg(const int startPage, const size_t numPages, const MemStatus memStatus, const int lastTouched): startPage(startPage), numPages(numPages),  memStatus (memStatus), lastTouched(lastTouched), buffer(0),pinCount(0),slabNum(-1) {}
    };

    typedef std::list<BufferSeg> BufferList;
}

#endif 
