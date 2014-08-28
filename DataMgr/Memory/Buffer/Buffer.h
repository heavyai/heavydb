/**
 * @file		Buffer.h
 * @author		Steven Stewart <steve@map-d.com>
 */
#ifndef DATAMGR_MEMORY_BUFFER_BUFFER_H
#define DATAMGR_MEMORY_BUFFER_BUFFER_H

#include <iostream>
#include "../AbstractDatum.h"

using namespace Memory_Namespace;

namespace Buffer_Namespace {
    
    /**
     * @struct  Page
     * @brief   A page holds a memory location and a "dirty" flag.
     */
    struct Page {
        mapd_addr_t addr = NULL;    /// memory address for beginning of page
        bool dirty = false;         /// indicates the page has been modified
        
        /// Constructor
        Page(mapd_addr_t addrIn, bool dirtyIn = false) : addr(addrIn), dirty(dirtyIn) {}
    };
    
    /**
     * @class   Buffer
     * @brief
     */
    class Buffer : public AbstractDatum {
        friend class BufferMgr;
        
    public:
        Buffer(mapd_addr_t mem, mapd_size_t numPages, mapd_size_t pageSize, int epoch);
        virtual ~Buffer();
        
        virtual void read(mapd_addr_t const dst, const mapd_size_t offset, const mapd_size_t nbytes = 0);
        virtual void write(mapd_addr_t src, const mapd_size_t offset, const mapd_size_t nbytes);
        virtual void append(mapd_addr_t src, const mapd_size_t nbytes);
        
        virtual mapd_size_t pageCount() const;
        virtual mapd_size_t pageSize() const;
        virtual mapd_size_t size() const;
        virtual mapd_size_t used() const;
        virtual bool isDirty() const;
        
    private:
        mapd_addr_t mem_;           /// pointer to beginning of datum's memory
        mapd_size_t nbytes_;        /// total number of bytes allocated for head pointer
        mapd_size_t used_;          /// total number of used bytes in the datum
        mapd_size_t pageSize_;      /// the size of each page in the datum buffer
        int epoch_;                 /// indicates when the datum was last flushed
        bool dirty_;                /// true if buffer has been modified
        std::vector<Page> pages_;   /// a vector of pages that form the content of the buffer
    };
    
} // Buffer_Namespace

#endif // DATAMGR_MEMORY_BUFFER_BUFFER_H