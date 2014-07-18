/**
 * @file		Buffer.h
 * @author		Steven Stewart <steve@map-d.com>
 */
#ifndef _BUFFER_H_
#define _BUFFER_H_

#include <vector>
#include "../../Shared/types.h"

namespace Buffer_Namespace {

/**
 * @struct Page
 */
struct Page {
    mapd_addr_t addr = NULL;
    bool dirty = false;

    Page(mapd_addr_t addrIn, bool dirtyIn = false)
        : addr(addrIn), dirty(dirtyIn) {}
};

/**
 * @class Buffer
 */
class Buffer {
public:
    /// Constructor
    Buffer(mapd_addr_t host_ptr, mapd_size_t numPages, mapd_size_t pageSize);

    /// Destructor
    ~Buffer();

    // Buffer API
    bool write(mapd_size_t offset, mapd_size_t n, mapd_addr_t src);
    bool append(mapd_size_t n, mapd_addr_t src);
    void copy(mapd_size_t offset, mapd_size_t n, mapd_addr_t dest);
    std::vector<bool> getDirty();
    void print();

    // Mutators
    inline void pin() { pins_++; }
    inline void unpin() { pins_--; }

    // Accessors
    inline mapd_addr_t  host_ptr() { return host_ptr_; }
    inline bool pinned() { return pins_ > 0; }
    inline bool dirty() { return dirty_; }
    inline mapd_size_t numPages() { return pages_.size(); }
    inline mapd_size_t pageSize() { return pageSize_; }
    inline mapd_size_t length() { return length_; }
    inline mapd_size_t size() { return pageSize_ * pages_.size(); }

private:
    Buffer(const Buffer&);
    Buffer& operator =(const Buffer&);

    mapd_addr_t host_ptr_;
    mapd_size_t length_;
    mapd_size_t pageSize_;
    int pins_;
    bool dirty_;
    std::vector<Page*> pages_;
};

} // Buffer_Namespace

#endif // _BUFFER_H_