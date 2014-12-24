//
//  Buffer.cpp
//  mapd2
//
//  Created by Steven Stewart on 8/27/14.
//  Copyright (c) 2014 MapD Technologies, Inc. All rights reserved.
//
#include <cassert>
#include <stdexcept>

#include "Buffer.h"
#include "BufferMgr.h"


namespace Buffer_Namespace {

    Buffer::Buffer(BufferMgr *bm, BufferList::iterator segIt,  const mapd_size_t pageSize, const mapd_size_t numBytes): bm_(bm), segIt_(segIt), pageSize_(pageSize), dirty_(false), mem_(0), numPages_(0) {
        // so that the pointer value of this Buffer is stored
        segIt_ -> buffer = this;
        if (numBytes > 0) {
            reserve(numBytes);
        }
    }


   /* 
    Buffer::Buffer(const mapd_addr_t mem, const mapd_size_t numPages, const mapd_size_t pageSize, const int epoch):
     mem_(mem), pageSize_(pageSize), used_(0), epoch_(epoch), dirty_(false)
    {
        assert(pageSize_ > 0);
        pageDirtyFlags_.resize(numPages);
        for (mapd_size_t i = 0; i < numPages; ++i)
            pages_.push_back(Page(mem + (i * pageSize), false));
    
    }
    */
    
    Buffer::~Buffer() {
        
    }

    void Buffer::reserve(const mapd_size_t numBytes) {
        mapd_size_t numPages = (numBytes + pageSize_ -1 ) / pageSize_;
        if (numPages > numPages_) {
            pageDirtyFlags_.resize(numPages);
            numPages_ = numPages;
            segIt_ = bm_ -> reserveBuffer(segIt_,size());
        }
    }
    
    void Buffer::read(mapd_addr_t const dst, const mapd_size_t numBytes, const mapd_size_t offset) {
        assert(dst && mem_);
        if (numBytes + offset > size()) {
            throw std::runtime_error("Buffer: Out of bounds read error");
        }
        memcpy(dst, mem_ + offset, numBytes);
    }
    
    void Buffer::write(mapd_addr_t src, const mapd_size_t numBytes, const mapd_size_t offset) {
        assert(numBytes > 0); // cannot write 0 bytes
        if (numBytes + offset > size()) {
            reserve(numBytes+offset);
            //bm_ -> reserveBuffer(segIt_,numBytes + offset);
        }
        // write source contents to buffer
        assert(mem_ && src);
        memcpy(mem_ + offset, src, numBytes);
        
        // update dirty flags for buffer and each affected page
        dirty_ = true;
        mapd_size_t firstDirtyPage = offset / pageSize_;
        mapd_size_t lastDirtyPage = (offset + numBytes - 1) / pageSize_;
        for (mapd_size_t i = firstDirtyPage; i <= lastDirtyPage; ++i) {
            pageDirtyFlags_[i] = true;
        }
    }
    
    
    mapd_byte_t* Buffer::getMemoryPtr() {
        return mem_;
    }
    
}
