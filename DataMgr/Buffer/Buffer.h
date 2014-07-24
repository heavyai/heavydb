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
    mapd_addr_t addr = NULL;    /// memory address for beginning of page
    bool dirty = false;         /// indicates the page has been modified

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

    /**
     * @brief Writes n bytes from src to the buffer at the specified offset.
     *
     * This method will write n bytes from src to the buffer at the specified
     * offset from the beginning of the buffer. Note that this method returns
     * false if the buffer's boundaries are exceeded by the request.
     *
     * @return bool Returns false if the write fails.
     */
    bool write(mapd_size_t offset, mapd_size_t n, mapd_addr_t src);
    
    /**
     * @brief Appends the data at src to the end of the buffer.
     *
     * This method will copy n bytes from src to end of the buffer. This method
     * will return false if the copy were to exceed the buffer's boundaries.
     *
     * @return bool Returns false if appending beyond buffer's total size.
     */
    bool append(mapd_size_t n, mapd_addr_t src);
    
    /**
     * @brief Copies content from the buffer to the destination memory address.
     *
     * This method will copy n bytes beginning at the specified offset to the
     * destination memory addres (called dest). Note that this method returns
     * false if the buffer's boundaries are exceeded by the request.
     *
     * @return bool Returns false if the copy fails.
     */
    bool copy(mapd_size_t offset, mapd_size_t n, mapd_addr_t dest);
    
    /**
     * @brief Returns the "dirty" status of the pages in the buffer.
     * @return vector<bool> For each entry, "true" indicates a dirty page.
     */
    std::vector<bool> getDirty();
    
    /**
     * @brief Prints a representation of the Buffer object to stdout
     */
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

    mapd_addr_t host_ptr_;      /// memory address of first page of buffer in host memory
    mapd_size_t length_;        /// the length (used bytes) of the buffer
    mapd_size_t pageSize_;      /// the size of each page within the buffer
    int pins_;                  /// the number of pins (i.e., resources using the buffer)
    bool dirty_;                /// indicates that some page within the buffer has been modified
    std::vector<Page*> pages_;  /// a vector of pages that form the content of the buffer
};

} // Buffer_Namespace

#endif // _BUFFER_H_
