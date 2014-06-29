/**
* @file	BufferMgr.cpp
* @author	Steven Stewart <steve@map-d.com>
*
*/

#include <iostream>
#include <map>
#include <sys/statvfs.h>
#include "BufferMgr.h"
#include "../File/FileMgr.h"

using std::cout;
using std::endl;

BufferMgr::BufferMgr(mapd_size_t hostMemorySize, FileMgr &fm) : fm_(fm) {
    // Allocate host memory
    hostMem_ = new mapd_byte_t[hostMemorySize];
    
    // Retrieve file system information
    struct statvfs buf;
    if (statvfs(".", &buf) == -1) {
        fprintf(stderr, "[%s:%d] Error: statvfs() failed.\n", __func__, __LINE__);
        exit(EXIT_FAILURE);
    }
    frameSize_ = buf.f_frsize; // fundamental physical block size
    numFrames_ = hostMemorySize / frameSize_;
    
    // Print summary
    printf("Host memory size = %u bytes\n", hostMemorySize);
    printf("Frame size = %u bytes\n", frameSize_);
    printf("# of frames = %u\n", numFrames_);
}

BufferMgr::BufferMgr(mapd_size_t hostMemorySize, mapd_size_t frameSize, FileMgr &fm) : fm_(fm) {
    // Allocate host memory
    hostMem_ = new mapd_byte_t[hostMemorySize];

    frameSize_ = frameSize;
    numFrames_ = hostMemorySize / frameSize_;
    
    // Print summary
    printf("Host memory size = %u bytes\n", hostMemorySize);
    printf("Frame size = %u bytes\n", frameSize_);
    printf("# of frames = %u\n", numFrames_);
}

BufferMgr::~BufferMgr() {
    delete[] hostMem_;
}

/**
 *
 *
 */
mapd_err_t BufferMgr::getChunkHost(const ChunkKey &key, PageInfo &page, bool pin) {

    // Search for the page using the key
    ChunkToPageMap::iterator iter = chunkIndex.find(key);

    // found
    if (iter != chunkIndex.end()) {
        page = iter->second;
        if (pin) page.pin();
        return MAPD_SUCCESS;
    }
    
    // not found -- request actual chunk size from file manager
    mapd_size_t actualSize;
    fm_.getChunkActualSize(key, &actualSize);
    if (actualSize <= 0)
        return MAPD_ERR_BUFFER; // chunk doesn't exist
    
    // @todo Find enough frames for the chunk in the buffer pool
    
    // @todo if not enough frames, invoke eviction strategy
    
    return MAPD_ERR_BUFFER;
}

/**
 * Returns whether or not the chunk is in the host buffer pool.
 *
 */
bool BufferMgr::isCachedHost(const ChunkKey &key) {
    return chunkIndex.find(key) == chunkIndex.end();
}

bool BufferMgr::insertIntoIndex(std::pair<ChunkKey, PageInfo> e) {
    std::pair<std::map<ChunkKey,PageInfo>::iterator, bool> ret;
    ret = chunkIndex.insert(e);
    return ret.second;
}

void BufferMgr::printFramesHost() {
    
}

void BufferMgr::printPagesHost() {
    
}

void BufferMgr::printChunksHost() {
    
}

