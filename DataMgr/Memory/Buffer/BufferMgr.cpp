/**
 * @file    BufferMgr.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include <cassert>
#include "BufferMgr.h"

namespace Buffer_Namespace {

    /// Allocates memSize bytes for the buffer pool and initializes the free memory map.
    BufferMgr::BufferMgr(mapd_size_t memSize) {
        assert(memSize > 0);
        memSize_ = memSize;
        mem_ = (mapd_addr_t) new mapd_byte_t[memSize_];
        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(memSize_, mem_));
    }

    /// Frees the heap-allocated buffer pool memory
    BufferMgr::~BufferMgr() {
        delete[] mem_;
    }
    
    /// Throws a runtime_error if the Chunk already exists
    void BufferMgr::createChunk(const ChunkKey &key, mapd_size_t pageSize) {
        if (chunkPageSize_.find(key) != chunkPageSize_.end())
            throw std::runtime_error("Chunk already exists.");
        chunkPageSize_.insert(std::pair<ChunkKey, mapd_size_t>(key, pageSize));
    }
    
    /// This method throws a runtime_error when deleting a Chunk that does not exist.
    void BufferMgr::deleteChunk(const ChunkKey &key) {
        auto chunkPageSizeIt = chunkPageSize_.find(key);
        if (chunkPageSizeIt == chunkPageSize_.end()) {
            assert(chunkIndex_.find(key) == chunkIndex_.end());
            throw std::runtime_error("Chunk does not exist");
        }
        
        // lookup the buffer for the Chunk in chunkIndex_
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIndex_.find(key) == chunkIndex_.end()) {
            chunkPageSize_.erase(chunkPageSizeIt);
            return;
        }

        // return the free memory used by the Chunk back to the free memory pool
        // by inserting the buffer's size mapped to the buffer's address
        Buffer *b = chunkIt->second;
        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(b->size(), b->mem_));
        // @todo some work still to do on free-space mgmt.
        
        // erase the Chunk's index entries
        chunkPageSize_.erase(chunkPageSizeIt);
        chunkIndex_.erase(chunkIt);
    }
    
    /// Frees the buffer/memory used by the Chunk, but keeps its chunkPageSize_ entry
    void BufferMgr::releaseChunk(const ChunkKey &key) {
        // lookup the buffer for the Chunk in chunkIndex_
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIndex_.find(key) == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        assert(chunkPageSize_.find(key) != chunkPageSize_.end());
        
        // return the free memory used by the Chunk back to the free memory pool
        Buffer *b = chunkIt->second;
        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(b->size(), b->mem_));
        // @todo some work still to do on free-space mgmt.
        
        // remove the chunkIndex_ entry (but keep the chunkPageSize_ entry)
        chunkIndex_.erase(chunkIt);
    }
    
    /// Returns a pointer to the Buffer holding the chunk, if it exists; otherwise,
    /// throws a runtime_error.
    AbstractDatum* BufferMgr::getChunk(ChunkKey &key) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIndex_.find(key) == chunkIndex_.end())
            throw std::runtime_error("Chunk does not exist.");
        return chunkIt->second;
    }
    
    AbstractDatum* BufferMgr::putChunk(const ChunkKey &key, AbstractDatum *d) {
        assert(d->size() > 0);
        mapd_size_t nbytes = d->size();
        
        // create Chunk if it doesn't exist
        auto chunkPageSizeIt = chunkPageSize_.find(key);
        if (chunkPageSizeIt == chunkPageSize_.end())
            createChunk(key, d->pageSize());
        
        // check if Chunk's Buffer exists
        Buffer *b = nullptr;
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIndex_.find(key) == chunkIndex_.end()) {
            b = new Buffer(nullptr, d->pageCount(), d->pageSize(), -1);
            chunkIndex_.insert(std::pair<ChunkKey, Buffer*>(key, b));
        }
        else
            b = chunkIt->second;
        
        // should be a consistent page size for a given ChunkKey
        assert(b->pageSize() == d->pageSize());
        
        // if necessary, reserve memory for b
        if (b->mem_ == nullptr) {

            // Find n bytes of free memory in the buffer pool
            auto freeMemIt = freeMem_.lower_bound(nbytes);
            if (freeMemIt == freeMem_.end()) {
                delete b;
                chunkIndex_.erase(chunkIt);
                throw std::runtime_error("Out of memory");
                // @todo eviction strategies
            }
            
            // Save free memory information
            mapd_size_t freeMemSize = freeMemIt->first;
            mapd_addr_t bufAddr = freeMemIt->second;
            
            // update Buffer's pointer
            b->mem_ = bufAddr;
            
            // Remove entry from map, and insert new entry
            freeMem_.erase(freeMemIt);
            if (freeMemSize - nbytes > 0)
                freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(freeMemSize - nbytes, bufAddr + nbytes));
        }
        
        // b and d should be the same size
        if (b->size() != d->size())
            throw std::runtime_error("Size mismatch between source and destination buffers.");

        // read the contents of d into b
        d->read(b->mem_, 0);
        
        return b;
    }
    
    /// client is responsible for deleting memory allocated for b->mem_
    AbstractDatum* BufferMgr::createDatum(mapd_size_t pageSize, mapd_size_t nbytes) {
        assert(pageSize > 0 && nbytes > 0);
        mapd_size_t numPages = (pageSize + nbytes - 1) / pageSize;
        Buffer *b = new Buffer(nullptr, numPages, pageSize, -1);
        
        // Find nbytes of free memory in the buffer pool
        auto freeMemIt = freeMem_.lower_bound(nbytes);
        if (freeMemIt == freeMem_.end()) {
            delete b;
            throw std::runtime_error("Out of memory");
            // @todo eviction strategies
        }
        
        // Save free memory information
        mapd_size_t freeMemSize = freeMemIt->first;
        mapd_addr_t bufAddr = freeMemIt->second;

        // update Buffer's pointer
        b->mem_ = bufAddr;
        
        // Remove entry from map, and insert new entry
        freeMem_.erase(freeMemIt);
        if (freeMemSize - nbytes > 0)
            freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(freeMemSize - nbytes, bufAddr + nbytes));
        
        return b;
    }
    
    void BufferMgr::deleteDatum(AbstractDatum *d) {
        assert(d);
        Buffer *b = (Buffer*)d;
        
        // return the free memory used by the Chunk back to the free memory pool
        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(b->size(), b->mem_));
        delete[] b->mem_;
        delete b;
    }
    
    AbstractDatum* BufferMgr::putDatum(AbstractDatum *d) {
        return nullptr;
    }
    
    mapd_size_t BufferMgr::size() {
        return memSize_;
    }
    
}