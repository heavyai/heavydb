/**
 * @file    BufferMgr.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "BufferMgr.h"

#include <cassert>
#include <limits>

namespace Buffer_Namespace {

    /// Allocates memSize bytes for the buffer pool and initializes the free memory map.
    BufferMgr::BufferMgr(const size_t bufferSize, const size_t pageSize): bufferSize_(bufferSize),pageSize_(pageSize) {
        assert(bufferSize_ > 0 && pageSize_ > 0);
        numPages_ = bufferSize_ / pageSize_; 
        bufferPool_ = (mapd_addr_t) new mapd_byte_t[bufferSize_];
        bufferSegments_.push_back(BufferSeg(0,numPages_));

        //freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(numPages_, bufferPool_));
    }

    /// Frees the heap-allocated buffer pool memory
    BufferMgr::~BufferMgr() {
        delete[] buffer_;
    
    /// Throws a runtime_error if the Chunk already exists
    void AbstractDatum * BufferMgr::createChunk(const ChunkKey &key, const mapd_size_t chunkPageSize, const size_t initialSize) {
        assert (chunkPageSize_ % pageSize_ == 0);
        // ChunkPageSize here is just for recording dirty pages
        if (chunkIndex_.find(key) != chunkIndex_.end()) {
            throw std::runtime_error("Chunk already exists.");
        }
        chunkIndex_[key] = new Buffer (this,pageSize,key);
        Buffer *buffer = chunkIndex_[key];
        /*
        if (initialSize > 0) {
            size_t chunkNumPages = (initialSize + pageSize_ - 1) / pageSize_;
            buffer -> reserve(chunkNumPages);
        */
        return (buffer);
    }

    void BufferMgr::findFreeSpace(size_t numBytes) {
        size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
        for (auto bufferIt = bufferSegments_.begin(); bufferIt != bufferSegments_.end(); ++bufferIt) {
            if (bufferIt -> memStatus == FREE && bufferIt -> numPages >= numPagesRequsted) {
                return bufferIt;
            }
        }
        // If we're here then we didn't find a free segment of sufficient size
        // - need to evict
        
        BufferList::iterator bestEvictionStart = bufferList_.end();
        unsigned int minScore = std::numeric_limits<unsigned int>::max(); 
        // We're going for lowest score here, like golf
        // This is because score is the sum of the lastTouched score for all
        // pages evicted. Evicting less pages and older pages will lower the
        // score

        for (auto bufferIt = bufferSegments_.begin(); bufferIt != bufferSegments_.end(); ++bufferIt) {
            // We can't evict pinned  buffers - only normal used
            // buffers
            if (bufferIt -> memStatus != PINNED) {
                size_t pageCount = 0;
                size_t score = 0;
                bool solutionFound = false;
                for (auto evictIt = bufferIt; evictIt != bufferSegments_.end(); ++evictIt) {
                   if (evictIt -> memStatus == PINNED) { // If pinned then we're at a dead end
                       break;
                    }
                    pageCount += evictIt -> numPages;
                    if (evictIt -> memStatus == USED) {
                        score += evictIt -> lastTouched;
                    }
                    if (pageCount >= numPagesRequsted) {
                        solutionFound = true;
                        break;
                    }
                }
                if (solutionFound && score < minScore) {
                    minScore = score;
                    bestEvictionStart = bufferIt;
                }
                else if (evictIt == bufferSegments_.end()) {
                    // this means that every segment after this will fail as
                    // well, so our search has proven futile
                    throw std::runtime_error ("Couldn't evict chunks to get free space");
                    break;
                    // in reality we should try to rearrange the buffer to get
                    // more contiguous free space
                }
                // other possibility is ending at PINNED - do nothing in this
                // case

            }
        }
        return bestEvictionStart;
    }



    
    /// This method throws a runtime_error when deleting a Chunk that does not exist.
    void BufferMgr::deleteChunk(const ChunkKey &key) {
        // lookup the buffer for the Chunk in chunkIndex_
        auto chunkIt = chunkIndex_.find(key);
        Buffer *buffer = chunkIt -> second;



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
        // @todo write this putDatum() method
        return nullptr;
    }
    
    mapd_size_t BufferMgr::size() {
        return memSize_;
    }
    
}
