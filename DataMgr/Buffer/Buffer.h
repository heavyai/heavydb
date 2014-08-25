/**
 * @file		Buffer.h
 * @author		Steven Stewart <steve@map-d.com>
 */
#ifndef DATAMGR_BUFFER_BUFFER_H
#define DATAMGR_BUFFER_BUFFER_H

#include <vector>
#include <cassert>
#include "../../Shared/types.h"

namespace Buffer_Namespace {
    
    /**
     * @struct  Page
     * @brief   A page holds a memory location and a "dirty" flag to indicate whether its been modified.
     */
    struct Page {
        mapd_addr_t addr = NULL;    /// memory address for beginning of page
        bool dirty = false;         /// indicates the page has been modified
        
        /// Constructor
        Page(mapd_addr_t addrIn, bool dirtyIn = false) : addr(addrIn), dirty(dirtyIn) {}
    };
    
    /**
     * @class Buffer
     * @brief A buffer is an area of memory in the buffer pool consisting of pages.
     *
     * A Buffer object refers to a region of host memory whose address begins at that
     * pointed to by a host pointer (host_ptr), and which consists of a set of
     * equally-sized pages established upon instantiation.
     *
     * The API provides basic I/O operations
     *
     *
     * Patterns:
     *      Forbid Copying Idiom 4.1
     */
    class Buffer {
    public:
        /// Constructor
        Buffer(mapd_addr_t host_ptr, mapd_size_t numPages, mapd_size_t pageSize, mapd_size_t lastTouchedTime);
        
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
        size_t write(mapd_size_t offset, mapd_size_t n, mapd_addr_t src);
        
        /**
         * @brief Appends the data at src to the end of the buffer.
         *
         * This method will copy n bytes from src to end of the buffer. This method
         * will return false if the copy were to exceed the buffer's boundaries.
         *
         * @return bool Returns false if appending beyond buffer's total size.
         */
        size_t append(mapd_size_t n, mapd_addr_t src);
        
        /**
         * @brief Copies content from the buffer to the destination memory address.
         *
         * This method will copy n bytes beginning at the specified offset to the
         * destination memory addres (called dest). Note that this method returns
         * false if the buffer's boundaries are exceeded by the request.
         *
         * @return bool Returns false if the copy fails.
         */
        size_t copy(mapd_size_t offset, mapd_size_t n, mapd_addr_t dest);
        
        /**
         * @brief Returns the "dirty" status of the pages in the buffer.
         * @return vector<bool> For each entry, "true" indicates a dirty page.
         */
        std::vector<bool> getDirty();
        
        /**
         * @brief Prints a representation of the Buffer object to stdout
         */
        void print();
        
        /**
         * @brief Prints a representation of the contents of the Buffer as the specified type
         */
        void print(mapd_data_t type);
        
        /// Increments the pin count
        inline void pin() { pins_++; }
        
        /// Incremenets the pin count while setting lastUsedTime
        inline void pin(mapd_size_t lastUsedTime) {
            lastUsedTime_ = lastUsedTime;
            pins_++;
        }
        
        /// Decrements the pin count
        inline void unpin() { pins_--; }
        
        /// Sets the length of the Buffer (number of used bytes)
        inline void length(mapd_size_t length) { assert(length <= size()); this->length_ = length; }
        
        /// Returns a pointer to host memory for the Buffer
        inline mapd_addr_t  host_ptr() { return host_ptr_; }
        
        /// Returns true if the Buffer is pinned
        inline bool pinned() { return pins_ > 0; }
        
        /// Returns true if the Buffer is dirty (has been modified)
        inline bool dirty() { return dirty_; }
        
        /// Returns the number of pages in the Buffer
        inline mapd_size_t numPages() { return pages_.size(); }
        
        /// Returns the page size of the pages in the Buffer
        inline mapd_size_t pageSize() { return pageSize_; }
        
        /// Returns the length (number of used bytes) in the Buffer
        inline mapd_size_t length() { return length_; }
        
        /// Returns the total size of the Buffer in bytes
        inline mapd_size_t size() { return pageSize_ * pages_.size(); }
        
    private:
        Buffer(const Buffer&); /// cannot copy via copy constructor
        Buffer& operator =(const Buffer&); /// cannot copy via assignment operator
        
        mapd_addr_t host_ptr_;      /// memory address of first page of buffer in host memory
        mapd_size_t length_;        /// the length (used bytes) of the buffer
        mapd_size_t pageSize_;      /// the size of each page within the buffer
        int pins_;                  /// the number of pins (i.e., resources using the buffer)
        bool dirty_;                /// indicates that some page within the buffer has been modified
        std::vector<Page*> pages_;  /// a vector of pages that form the content of the buffer

        mapd_size_t lastUsedTime_;  /// Set to when buffer was last pinned
    };
    
} // Buffer_Namespace

#endif // DATAMGR_BUFFER_BUFFER_H
