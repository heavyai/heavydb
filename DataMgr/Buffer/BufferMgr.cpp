/**
* @file	BufferMgr.cpp
* @author	Steven Stewart <steve@map-d.com>
*
*/

#include <iostream>
#include <map>
#include <cstring>
#include <sys/statvfs.h>
#include "../File/FileMgr.h"
#include "BufferMgr.h"

using std::cout;
using std::endl;
using std::pair;
using std::map;

namespace Buffer_Namespace {

BufferMgr::BufferMgr(mapd_size_t hostMemorySize, FileMgr *fm) : fm_(fm) {
    // Allocate host memory
    hostMem_ = new mapd_byte_t[hostMemorySize];

    // Retrieve file system information
    struct statvfs buf;
    if (statvfs(".", &buf) == -1) {
        fprintf(stderr, "[%s:%d] Error: statvfs() failed.\n", __func__, __LINE__);
        exit(EXIT_FAILURE);
    }
    frameSize_ = buf.f_frsize; // fundamental physical block size
    
    // initialize variables
    numHitsHost_ = 0;
    numMissHost_ = 0;

    // Print summary
    printf("Host memory size = %u bytes\n", hostMemorySize);
    printf("Frame size = %u bytes\n", frameSize_);
    printf("# of frames = %lu\n", frames_.size());
}

BufferMgr::BufferMgr(mapd_size_t hostMemorySize, mapd_size_t frameSize, FileMgr *fm) : fm_(fm) {
    // Allocate host memory
    hostMem_ = new mapd_byte_t[hostMemorySize];
    frameSize_ = frameSize;
    
    // initialize variables
    numHitsHost_ = 0;
    numMissHost_ = 0;

    // Print summary
    printf("Host memory size = %u bytes\n", hostMemorySize);
    printf("Frame size = %u bytes\n", frameSize_);
    printf("# of frames = %lu\n", frames_.size());
}

BufferMgr::~BufferMgr() {
    delete[] hostMem_;
}

std::pair<bool, bool> BufferMgr::chunkStatus(const ChunkKey &key) {
    std::pair<bool, bool> status;
    auto it = chunkIndex_.find(key);
    
    // is chunk in chunkIndex_?
    status.first = (it != chunkIndex_.end());
    
    // is chunk cached in buffer pool?
    status.second = status.first ? it->second->isCached : false;
    
    return status;
}

bool BufferMgr::insertIntoIndex(pair<ChunkKey, PageInfo*> e) {
    pair<map<ChunkKey,PageInfo*>::iterator, bool> ret;
    ret = chunkIndex_.insert(e);
    return ret.second;
}

PageInfo* BufferMgr::getChunk(const ChunkKey &key, bool pin) {
    return getChunk(key, 0, pin);
}

PageInfo* BufferMgr::getChunk(const ChunkKey &key, mapd_size_t pad, bool pin) {
    // find chunk's page
    PageInfo *pInfo = findChunkPage(key);

    // found
    if (pInfo) {
        if (pin) pInfo->pin();
        return pInfo;
    }

    // not found -- request actual chunk size from file manager
    mapd_size_t actualSize;
    fm_->getChunkActualSize(key, &actualSize);
    if (actualSize <= 0)
        return NULL; // chunk doesn't exist

    // @todo Find enough frames for the chunk in the buffer pool

    // @todo if not enough frames, invoke eviction strategy

    return pInfo;
}

// @todo ability to extend page bounds 
// @todo fix bugs: should be using bounds as frame IDs, not memory addresses!
/*bool BufferMgr::updateChunk(const ChunkKey &key, mapd_size_t offset, mapd_size_t size, mapd_byte_t *src) {
    // find chunk's page
    PageInfo *pInfo = findChunkPage(key);
    if (!pInfo)
        return false;

    // if not within page's bounds, return false
    mapd_size_t begin = pInfo->bounds.first + offset;
    if (begin + size > pInfo->bounds.second)
        return false;

    // Copy source data into chunk
    mapd_byte_t *dest = hostMem_ + begin;
    memcpy(dest, src, size);

    // Update last address written to for the page
    pInfo->lastAddr = begin + size;

    return true;
}*/

bool BufferMgr::removeChunk(const ChunkKey &key) {
    // find chunk's page
    PageInfo *pInfo = findChunkPage(key);

    // not found, or if page is pinned, return false
    if (!pInfo || pInfo->isPinned())
        return false;

    // free page and remove entry in chunkIndex_
    delete pInfo;
    chunkIndex_.erase(key);

    return true;
}

/*
void appendChunk(const ChunkKey &key, mapd_size_t size, mapd_byte_t *src) {

}
*/

/*bool flushChunk(const ChunkKey &key, unsigned int epoch) {
    PageInfo* pInfo = findChunkPage(key);

    // not found
    if (!pInfo)
        return false;

    // gather frame information for the page
    Frame *fr = pInfo->bounds.first / frameSize_;
    mapd_size_t numFrames = pInfo
}*/

/*
void BufferMgr::printFramesHost() {
    
}

void BufferMgr::printPagesHost() {
    
}

void BufferMgr::printChunksHost() {
    
}
*/

PageInfo* BufferMgr::findChunkPage(const ChunkKey key) {
    auto iter = chunkIndex_.find(key);
    if (iter == chunkIndex_.end()) // not found
        return NULL;
    return iter->second;
}

} // Buffer_Namespace