/**
 * @file	BufferMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 */
#include <iostream>
#include <cassert>
#include <cstring>
#include "BufferMgr.h"
#include "../File/FileMgr.h"

using std::cout;
using std::endl;
using File_Namespace::Chunk;

namespace Buffer_Namespace {

BufferMgr::BufferMgr(mapd_size_t numPages, mapd_size_t pageSize, FileMgr *fm = NULL) {
    assert(numPages > 0 && pageSize > 0);
    pageSize_ = pageSize;
    fm_ = fm;
    nextPage_ = 0;

    // Allocate host buffer pool
    hostMemSize_ = numPages * pageSize;
    hostMem_ = new mapd_byte_t[hostMemSize_];

#ifdef DEBUG_VERBOSE
    printMemAlloc();
#endif
}

BufferMgr::~BufferMgr() {
    // Delete host memory
    delete[] hostMem_;
}

Buffer* BufferMgr::createBuffer(mapd_size_t n) {
    // compute number of pages needed
    mapd_size_t numPages = (n + pageSize_ - 1) / pageSize_;

    // allocate new buffer and insert its pages
    Buffer *b = new Buffer();
    b->begin = nextPage_;
    for (mapd_size_t i = 0; i < numPages; ++i) {
        b->pages.push_back(new Page(nextPage_));
        nextPage_ += pageSize_; 
    }
    b->end = nextPage_;

    // insert into BufferMgr's vector of buffers and pin it
    buffers_.push_back(b);
    b->pins++;

    return b;
}

mapd_size_t BufferMgr::updateBuffer(Buffer &b, mapd_addr_t offset, mapd_size_t n, mapd_addr_t *src) {
    assert(n > 0);
    mapd_addr_t dest = b.begin + offset;

    // perform update
    if (n > 0) {
        if (dest < b.end) {
            memcpy(hostMem_ + dest, src, n);
            b.end = dest + n;
        }
        else {
            // @todo create buffer of new size pointed to by b
            fprintf(stderr, "[%s:%d] Dynamic reallocation of buffers is currently unsupported.\n", __func__, __LINE__);
            return 0;
        }
    }

    // flag pages and buffer as dirty
    setDirtyPages(b, dest, b.end);
    
    return n;
}

void BufferMgr::deleteBuffer(Buffer *b) {
    // free pages
    // @todo free pages of buffer

    // free buffer and remove from BufferMgr's list of buffers
    delete b;
    buffers_.remove(b); // @todo thread safe needed?
}

mapd_size_t BufferMgr::copyBuffer(Buffer &bSrc, Buffer &bDest, mapd_size_t n) {
    mapd_size_t bSrcSz = bSrc.size();
    mapd_size_t bDestSz = bDest.size();

    // copy max possible number of bytes up to size n
    n = (bSrcSz < n) ? bSrcSz : n;
    n = (bDestSz < n) ? bDestSz : n;
    memcpy(hostMem_ + bSrc.begin, hostMem_ + bDest.begin, n);

    // flag destination buffer and its affected pages as dirty
    setDirtyPages(bDest, 0, n);

    return n;
}

bool BufferMgr::loadChunk(const ChunkKey &key, Buffer &b, bool fast) {
    // Returns without verifying chunk will fit in buffer
    if (fast) { 
        Chunk *c = fm_->getChunk(key, hostMem_ + b.begin);
        if (c) {
            b.pin();
            return true;
        }
        return false;
    }

    // Otherwise, checks that chunk will fit in buffer and attempts to
    // reallocate buffer if it needs to
    mapd_size_t *size;
    fm_->getChunkActualSize(key, size);
    if (*size > b.size()) {
        // @todo create buffer of new size pointed to by b
        fprintf(stderr, "[%s:%d] Dynamic reallocation of buffers is currently unsupported.\n", __func__, __LINE__);
        return 0;
    }
    Chunk *c = fm_->getChunk(key, hostMem_ + b.begin);
    if (c) {
        b.pin();
        return true;
    }
    return false;
}

void flushChunk(Buffer &b, bool all = false, bool force = false) {
    if (!force && !b.dirty)
        return;

    
}

void BufferMgr::printMemAlloc() {
    printf("Host memory = %u bytes\n", hostMemSize_);
    printf("# of pages  = %u\n", hostMemSize_ / pageSize_);
    printf("Page size   = %u\n", pageSize_);
}

} // Buffer_Namespace