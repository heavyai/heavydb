/**
 * @file        Buffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 */
#include <cstring>
#include <cassert>
#include <iostream>
#include "Buffer.h"

namespace Buffer_Namespace {
    
    Buffer::Buffer(mapd_addr_t host_ptr, mapd_size_t numPages, mapd_size_t pageSize, mapd_size_t lastUsedTime) :
    host_ptr_(host_ptr),
    pageSize_(pageSize),
    lastUsedTime_(lastUsedTime),
    length_(0),
    pins_(1),
    dirty_(false)
    {
        assert(pageSize_ > 0);
        for (mapd_size_t i = 0; i < numPages; ++i)
            pages_.push_back(new Page(host_ptr + (i * pageSize), false));
    }
    
    Buffer::~Buffer() {
        while (pages_.size() > 0) {
            delete pages_.back();
            pages_.pop_back();
        }
    }

    /// this method returns 0 if it cannot write the full n bytes
    size_t Buffer::write(mapd_size_t offset, mapd_size_t n, mapd_addr_t src) {
        assert(n > 0); // cannot write 0 bytes
        
        // check for buffer overflow
        if ((length_ + n) > size())
            return 0;
        
        // write source contents to buffer
        assert(host_ptr_ && src);
        memcpy(host_ptr_ + offset, src, n);
        length_ = std::max(length_, offset + n);
        
        // update dirty flags for buffer and each affected page
        dirty_ = true;
        mapd_size_t firstPage = offset / pageSize_;
        mapd_size_t lastPage = (offset + n - 1) / pageSize_;

        for (mapd_size_t i = firstPage; i <= lastPage; ++i)
            pages_[i]->dirty = true;
        
        return n;
    }
    
    /// this method returns 0 if it cannot append the full n bytes
    size_t Buffer::append(mapd_size_t n, mapd_addr_t src) {
        assert(n > 0 && src);
        
        // Cannot append beyond the total allocated buffer size
        if ((length_ + n) > size())
            return 0;
        
        // writes n bytes to src using the current buffer length
        // as the beginning offset
        return write(length_, n, src);
    }
    
    /// this method returns 0 if it cannot copy the full n bytes
    size_t Buffer::copy(mapd_size_t offset, mapd_size_t n, mapd_addr_t dest) {
        assert(n > 0 && dest);
        if ((n + offset) > length_)
            return 0;
        memcpy(dest, host_ptr_ + offset, n);
        return n;
    }
    
    std::vector<bool> Buffer::getDirty() {
        std::vector<bool> dirtyFlags;
        for (int i = 0; i < pages_.size(); ++i) {
            if (pages_[i]->dirty)
                dirtyFlags.push_back(true);
            else
                dirtyFlags.push_back(false);
        }
        return dirtyFlags;
    }
    
    void Buffer::print() {
        printf("host pointer = %p\n", host_ptr_);
        printf("page size    = %lu\n", pageSize_);
        printf("# of pages   = %lu\n", pages_.size());
        printf("length       = %lu\n", length_);
        printf("size         = %lu\n", size());
        printf("pin count    = %d\n", pins_);
        printf("dirty        = %s\n", dirty_ ? "true" : "false");
    }
    
} // Buffer_Namespace
