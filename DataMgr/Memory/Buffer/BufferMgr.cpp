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
    }
    
    /// Throws a runtime_error if the Chunk already exists
    void AbstractDatum * BufferMgr::createChunk(const ChunkKey &chunkKey, const mapd_size_t chunkPageSize, const size_t initialSize) {
        assert (chunkPageSize % pageSize_ == 0);
        // ChunkPageSize here is just for recording dirty pages
        if (chunkIndex_.find(chunkKey) != chunkIndex_.end()) {
            throw std::runtime_error("Chunk already exists.");
        }

        bufferSegments_.push_back(BufferSeg(-1,0,PINNED);
        bufferSegments_.back().buffer =  new Buffer(this, chunkKey, std::prev(bufferSegments_.end(),1) chunkPageSize,  initialSize)); 

        chunkIndex_[chunkKey] = bufferSegments_.back();
        /*
        if (initialSize > 0) {
            size_t chunkNumPages = (initialSize + pageSize_ - 1) / pageSize_;
            buffer -> reserve(chunkNumPages);
        }
        */
        return buffer;
    }

    BufferList::iterator BufferMgr::evict(BufferList::iterator &evictStart, const size_t numPagesRequested) {
        // We can assume here that buffer for evictStart either doesn't exist
        // (evictStart is first buffer) or was not free, so don't need ot merge
        // it
        auto evictIt = evictStart;
        size_t numPages = 0;
        size_t startPage = evictStart -> startPage;
        while (numPages < numPagesRequsted) {
            assert (evictIt -> pinCount == 0):
            numPages += evictIt -> numPages;
            evictIt = bufferList_.erase(evictIt); // erase operations returns next iterator - safe if we ever move to a vector (as opposed to erase(evictIt++)
        }
        BufferSeg dataSeg(startPage,numPagesRequested,USED,bufferEpoch_++); // until we can 
        dataSeg.pinCount++;
        bufferList_.insert(evictIt,dataSeg); // Will insert before evictIt
        if (numPagesRequsted < numPages) {
            size_t excessPages = numPages - numPagesRequsted;
            if (evictIt != bufferList_.end() && evictIt -> memStatus == FREE) { // need to merge with current page
                evictIt -> startPage = startPage + numPagesRequsted;
                evictIt -> numPages += excessPages;
            }
            else { // need to insert a free seg before evictIt for excessPages
                BufferSeg freeSeg(startPage + numPagesRequsted,excessPages,FREE);
                bufferList_.insert(evictIt,freeSeg);
            }
        }
        return dataSeg;
    }

    BufferList::iterator BufferMgr::reserveBuffer(BufferSeg::iterator &segIt, size_t numBytes) {
        // doesn't resize to be smaller - so more like reserve

        size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
        size_t numPagesExtraNeeded = numPagesRequested -  segIt -> numPages;
        if (numPagesExtraNeeded < segIt -> numPages) { // We already have enough pages in existing segment
            return segIt;
        }
        // First check for freeSeg after segIt 
        if (segIt -> startPage >= 0) { //not dummy page
            BufferSeg::iterator nextIt = std::next(segIt);
            if (nextIt != bufferList_.end() && nextIt -> memStatus == FREE && nextIt -> numPages >= numPagesExtraNeeded) { // Then we can just use the next BufferSeg which happens to be free
                size_t leftoverPages = nextIt -> numPages - numPagesExtraNeeded;
                nextIt -> numPages = leftoverPages;
                segIt -> numPages = numPagesRequested;
                return segIt;
            }
        }
        /* If we're here then we couldn't keep
         * buffer in existing slot - need to find
         * new segment, copy data over, and then
         * delete old
         */
        
        segIt -> pinCount++; // so we don't evict this while trying to find a new segment for it 
        auto newSegIt = getBuffer(numBytes);
        mapd_addr_t oldMem_ = segIt -> mem_;
        newSegIt -> buffer = segIt -> buffer;
        newSegIt -> buffer -> mem_ = bufferPool_ + newSegIt -> startPage * pageSize_;
        // now need to copy over memory
        // only do this if the old segment is valid (i.e. not new w/
        // unallocated buffer
        if (segIt -> startPage >= 0)  {
            memcpy(newSegIt -> buffer -> mem_, oldMem_, newSegIt->numBytes_);
        }
        // Deincrement pin count to reverse effect above above
        bufferList_.erase(segIt);
        chunkIndex_[newSegIt -> buffer -> chunkKey] = newSegIt; 
    }

    BufferList::iterator BufferMgr::getBuffer(size_t numBytes) {

        for (auto bufferIt = bufferSegments_.begin(); bufferIt != bufferSegments_.end(); ++bufferIt) {
            if (bufferIt -> memStatus == FREE && bufferIt -> numPages >= numPagesRequsted) {
                // startPage doesn't change
                size_t excessPages = bufferIt -> numPages - numPagesRequsted;
                bufferIt -> numPages = numPagesRequested;
                bufferIt -> memStatus = USED;
                bufferIt -> pinCount = 1;
                bufferIt -> lastTouched  = bufferEpoch_++;
                if (excessPages > 0) {
                    BufferSeg freeSeg(bufferIt->startPage+numPagesRequsted,excessPages,FREE);
                    auto tempIt = bufferIt; // this should make a copy and not be a reference
                    // - as we do not want to increment bufferIt
                    tempIt++;
                    bufferList_.insert(tempIt,freeSeg);
                }
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
            /* Note there are some shortcuts we could take here - like we
             * should never consider a USED buffer coming after a free buffer
             * as we would have used the FREE buffer, but we won't worry about
             * this for now
             */

            // We can't evict pinned  buffers - only normal used
            // buffers

            if (bufferIt -> pinCount == 0) {
                size_t pageCount = 0;
                size_t score = 0;
                bool solutionFound = false;
                for (auto evictIt = bufferIt; evictIt != bufferSegments_.end(); ++evictIt) {
                   if (evictIt -> pinCount > 0) { // If pinned then we're at a dead end
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
        bestEvictionStart = evict(bestEvictionStart,numPagesRequested);
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
    AbstractDatum* BufferMgr::getChunk(ChunkKey &key,size_t numBytes) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt != chunkIndex_.end()) {
            chunkIt -> second -> memStatus = USED; // necessary?
            chunkIt -> second -> pinCount++; 
            chunkIt -> second -> lastTouched = bufferEpoch++; // necessary?
            return chunkIt -> second -> buffer; 
        }
        else { // If wasn't in pool then we need to fetch it
            Buffer *buffer;
            fileMgr_ -> fetchChunk(key,buffer,numBytes); // this should put buffer in a BufferSegment
            return buffer;
        }
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
