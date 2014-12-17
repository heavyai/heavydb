//
//  Buffer.cpp
//  mapd2
//
//  Created by Steven Stewart on 8/27/14.
//  Copyright (c) 2014 Map-D Technologies, Inc. All rights reserved.
//
#include <cassert>
#include <stdexcept>

#include "Buffer.h"
#include "BufferMgr.h"


namespace Buffer_Namespace {

    Buffer::Buffer(BufferMgr *bm, const ChunkKey &chunkKey, BufferSeg::iterator &segIt,  const mapd_size_t pageSize, const mapd_size_t numBytes): bm_(bm), chunkKey_(chunkKey), pageSize_(pageSize), dirty_(false), mem_(0) {
        if (numBytes > 0) {
            numPages_ = (numBytes + pageSize_ -1 ) / pageSize_;
            numBytes_ = numPages_ * pageSize_;
            bm_ -> reserveBuffer(segIt,numBytes_);
        }
        else {
            numBytes_ = 0;
            numPages_ = 0;
        }
    }


    
    Buffer::Buffer(const mapd_addr_t mem, const mapd_size_t numPages, const mapd_size_t pageSize, const int epoch)
    : mem_(mem), nbytes_(numPages * pageSize), pageSize_(pageSize), used_(0), epoch_(epoch), dirty_(false)
    {
        assert(pageSize_ > 0);
        for (mapd_size_t i = 0; i < numPages; ++i)
            pages_.push_back(Page(mem + (i * pageSize), false));
    
    }
    
    Buffer::~Buffer() {
        
    }
    
    void Buffer::read(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset) {
        assert(dst && mem_);
        if (numBytes offset > numBytes_) {
            throw std::runtime_error("Buffer: Out of bounds read error");
        }
        memcpy(dst, mem_ + offset, numBytes);
    }
    
    void Buffer::write(mapd_addr_t src, const mapd_size_t numBytes, const mapd_size_t offset) {
        assert(numBytes > 0); // cannot write 0 bytes
        if (numBytes + offset > numBytes_) {

        
        // check for buffer overflow
        if ((used_ + nbytes) > size())
            throw std::runtime_error("Attempted write exceeds boundaries of Buffer.");
        
        // write source contents to buffer
        assert(mem_ && src);
        memcpy(mem_ + offset, src, nbytes);
        used_ = std::max(used_, offset + nbytes);
        
        // update dirty flags for buffer and each affected page
        dirty_ = true;
        mapd_size_t firstPage = offset / pageSize_;
        mapd_size_t lastPage = (offset + nbytes - 1) / pageSize_;
        
        for (mapd_size_t i = firstPage; i <= lastPage; ++i)
            pages_[i].dirty = true;
    }
    
    void Buffer::append(mapd_addr_t src, const mapd_size_t nbytes) {
        write(src, used_, nbytes);
    }
    
    const mapd_byte_t* Buffer::getMemoryPtr() const {
        return mem_;
    }
    
    mapd_size_t Buffer::pageCount() const {
        assert((nbytes_ % pageSize_) == 0);
        return nbytes_ / pageSize_;
    }
    
    mapd_size_t Buffer::pageSize() const {
        return pageSize_;
    }
    
    mapd_size_t Buffer::size() const {
        return nbytes_;
    }
    
    mapd_size_t Buffer::used() const {
        return used_;
    }
    
    bool Buffer::isDirty() const {
        return dirty_;
    }
    
}
