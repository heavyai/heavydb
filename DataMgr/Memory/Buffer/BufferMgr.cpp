/**
 * @file    BufferMgr.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "BufferMgr.h"
#include "Buffer.h"

#include <cassert>
#include <limits>

namespace Buffer_Namespace {

    /// Allocates memSize bytes for the buffer pool and initializes the free memory map.
    BufferMgr::BufferMgr(const size_t bufferSize, const size_t pageSize, File_Namespace::FileMgr *fileMgr): bufferSize_(bufferSize),pageSize_(pageSize), fileMgr_(fileMgr), bufferEpoch_(0) {
        assert(bufferSize_ > 0 && pageSize_ > 0 && bufferSize_ % pageSize_ == 0);
        numPages_ = bufferSize_ / pageSize_; 
        bufferPool_ = (mapd_addr_t) new mapd_byte_t[bufferSize_];
        bufferSegments_.push_back(BufferSeg(0,numPages_));
    }

    /// Frees the heap-allocated buffer pool memory
    BufferMgr::~BufferMgr() {
        delete[] bufferPool_;
    }
    
    /// Throws a runtime_error if the Chunk already exists
    AbstractBuffer * BufferMgr::createChunk(const ChunkKey &chunkKey, const mapd_size_t chunkPageSize, const size_t initialSize) {
        assert (chunkPageSize == pageSize_);
        // ChunkPageSize here is just for recording dirty pages
        if (chunkIndex_.find(chunkKey) != chunkIndex_.end()) {
            throw std::runtime_error("Chunk already exists");
        }

        bufferSegments_.push_back(BufferSeg(-1,0,USED));
        bufferSegments_.back().buffer =  new Buffer(this, chunkKey, std::prev(bufferSegments_.end(),1), chunkPageSize, initialSize); 
        chunkIndex_[chunkKey] = std::prev(bufferSegments_.end(),1);
        /*
        if (initialSize > 0) {
            size_t chunkNumPages = (initialSize + pageSize_ - 1) / pageSize_;
            buffer -> reserve(chunkNumPages);
        }
        */
        return bufferSegments_.back().buffer;
    }

    BufferList::iterator BufferMgr::evict(BufferList::iterator &evictStart, const size_t numPagesRequested) {
        // We can assume here that buffer for evictStart either doesn't exist
        // (evictStart is first buffer) or was not free, so don't need ot merge
        // it
        auto evictIt = evictStart;
        size_t numPages = 0;
        size_t startPage = evictStart -> startPage;
        while (numPages < numPagesRequested) {
            assert (evictIt -> pinCount == 0);
            numPages += evictIt -> numPages;
            evictIt = bufferSegments_.erase(evictIt); // erase operations returns next iterator - safe if we ever move to a vector (as opposed to erase(evictIt++)
        }
        BufferSeg dataSeg(startPage,numPagesRequested,USED,bufferEpoch_++); // until we can 
        dataSeg.pinCount++;
        auto dataSegIt = bufferSegments_.insert(evictIt,dataSeg); // Will insert before evictIt
        if (numPagesRequested < numPages) {
            size_t excessPages = numPages - numPagesRequested;
            if (evictIt != bufferSegments_.end() && evictIt -> memStatus == FREE) { // need to merge with current page
                evictIt -> startPage = startPage + numPagesRequested;
                evictIt -> numPages += excessPages;
            }
            else { // need to insert a free seg before evictIt for excessPages
                BufferSeg freeSeg(startPage + numPagesRequested,excessPages,FREE);
                bufferSegments_.insert(evictIt,freeSeg);
            }
        }
        return dataSegIt;
    }

    BufferList::iterator BufferMgr::reserveBuffer(BufferList::iterator &segIt, size_t numBytes) {
        // doesn't resize to be smaller - like std::reserve

        size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;
        size_t numPagesExtraNeeded = numPagesRequested -  segIt -> numPages;
        if (numPagesExtraNeeded < segIt -> numPages) { // We already have enough pages in existing segment
            return segIt;
        }
        // First check for freeSeg after segIt 
        if (segIt -> startPage >= 0) { //not dummy page
            BufferList::iterator nextIt = std::next(segIt);
            if (nextIt != bufferSegments_.end() && nextIt -> memStatus == FREE && nextIt -> numPages >= numPagesExtraNeeded) { // Then we can just use the next BufferSeg which happens to be free
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
        auto newSegIt = findFreeBuffer(numBytes);
        mapd_addr_t oldMem_ = segIt -> buffer ->  mem_;
        newSegIt -> buffer = segIt -> buffer;
        newSegIt -> buffer -> mem_ = bufferPool_ + newSegIt -> startPage * pageSize_;
        // now need to copy over memory
        // only do this if the old segment is valid (i.e. not new w/
        // unallocated buffer
        if (segIt -> startPage >= 0)  {
            memcpy(newSegIt -> buffer -> mem_, oldMem_, newSegIt->buffer->size());
        }
        // Deincrement pin count to reverse effect above above
        bufferSegments_.erase(segIt);
        chunkIndex_[newSegIt -> buffer -> chunkKey_] = newSegIt; 
        return newSegIt;
    }

    BufferList::iterator BufferMgr::findFreeBuffer(size_t numBytes) {
        size_t numPagesRequested = (numBytes + pageSize_ - 1) / pageSize_;

        for (auto bufferIt = bufferSegments_.begin(); bufferIt != bufferSegments_.end(); ++bufferIt) {
            if (bufferIt -> memStatus == FREE && bufferIt -> numPages >= numPagesRequested) {
                // startPage doesn't change
                size_t excessPages = bufferIt -> numPages - numPagesRequested;
                bufferIt -> numPages = numPagesRequested;
                bufferIt -> memStatus = USED;
                bufferIt -> pinCount = 1;
                bufferIt -> lastTouched  = bufferEpoch_++;
                if (excessPages > 0) {
                    BufferSeg freeSeg(bufferIt->startPage+numPagesRequested,excessPages,FREE);
                    auto tempIt = bufferIt; // this should make a copy and not be a reference
                    // - as we do not want to increment bufferIt
                    tempIt++;
                    bufferSegments_.insert(tempIt,freeSeg);
                }
                return bufferIt;
            }
        }
        // If we're here then we didn't find a free segment of sufficient size
        // - need to evict
        
        BufferList::iterator bestEvictionStart = bufferSegments_.end();
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
                auto evictIt = bufferIt;
                for (; evictIt != bufferSegments_.end(); ++evictIt) {
                   if (evictIt -> pinCount > 0) { // If pinned then we're at a dead end
                       break;
                    }
                    pageCount += evictIt -> numPages;
                    if (evictIt -> memStatus == USED) {
                        score += evictIt -> lastTouched;
                    }
                    if (pageCount >= numPagesRequested) {
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
        //Buffer *buffer = chunkIt -> second -> buffer;

        if (chunkIndex_.find(key) == chunkIndex_.end()) {
            throw std::runtime_error("Chunk does not exist");
        }
        auto  segIt = chunkIt->second;
        delete segIt->buffer; // Delete Buffer for segment
        if (segIt != bufferSegments_.begin()) {
            auto prevIt = std::prev(segIt);
            if (prevIt -> memStatus == FREE) { 
                segIt -> startPage = prevIt -> startPage;
                segIt -> numPages += prevIt -> numPages; 
                bufferSegments_.erase(prevIt);
            }
        }
        auto nextIt = std::next(segIt);
        if (nextIt != bufferSegments_.end()) {
            if (nextIt -> memStatus == FREE) { 
                segIt -> numPages += nextIt -> numPages;
                bufferSegments_.erase(nextIt);
            }
        }
        segIt -> memStatus = FREE;
        segIt -> pinCount = 0;
        segIt -> buffer = 0;

        /*Below is the other, more complicted algorithm originally chosen*/

        //size_t numPageInSeg = segIt -> numPages;
        //bool prevFree = false; 
        //bool nextFree = false; 
        //if (segIt != bufferSegments_.begin()) {
        //    auto prevIt = std::prev(segIt);
        //    if (prevIt -> memStatus == FREE) { 
        //        prevFree = true;
        //        prevIt -> numPages += numPagesInSeg;
        //        segIt = prevIt;
        //    }
        //}
        //auto nextIt = std::next(segIt);
        //if (nextIt != bufferSegments_.end()) {
        //    if (nextIt -> memStatus == FREE) {
        //        nextFree = true;
        //        if (prevFree) {
        //            prevIt -> numPages += nextIt -> numPages; 
        //            bufferSegments_.erase(nextIt);
        //        }
        //        else {
        //            nextIt -> numPages += numPagesInSeg;
        //        }
        //    }
        //}
        //if (!prevFree && !nextFree) {
        //    segIt -> memStatus = FREE;
        //    segIt -> pinCount = 0;
        //    segIt -> buffer = 0;
        //}
        //else {
        //    bufferSegments_.erase(segIt);
        //}
    }

    
    /// Returns a pointer to the Buffer holding the chunk, if it exists; otherwise,
    /// throws a runtime_error.
    AbstractBuffer* BufferMgr::getChunk(ChunkKey &key, const mapd_size_t numBytes) {
        auto chunkIt = chunkIndex_.find(key);
        if (chunkIt != chunkIndex_.end()) {
            chunkIt -> second -> pinCount++; 
            chunkIt -> second -> lastTouched = bufferEpoch_++;
            return chunkIt -> second -> buffer; 
        }
        else { // If wasn't in pool then we need to fetch it
            Buffer *buffer;
            fileMgr_ -> fetchChunk(key,buffer,numBytes); // this should put buffer in a BufferSegment
            return buffer;
        }
    }

    void BufferMgr::fetchChunk(const ChunkKey &key, AbstractBuffer *destBuffer, const mapd_size_t numBytes) {

    }
    
    AbstractBuffer* BufferMgr::putChunk(const ChunkKey &key, AbstractBuffer *d, const mapd_size_t numBytes) {
        //assert(d->size() > 0);
        //mapd_size_t nbytes = d->size();
        //// create Chunk if it doesn't exist
        //auto chunkPageSizeIt = chunkPageSize_.find(key);
        //if (chunkPageSizeIt == chunkPageSize_.end())
        //    createChunk(key, d->pageSize());
        //
        //// check if Chunk's Buffer exists
        //Buffer *b = nullptr;
        //auto chunkIt = chunkIndex_.find(key);
        //if (chunkIndex_.find(key) == chunkIndex_.end()) {
        //    b = new Buffer(nullptr, d->pageCount(), d->pageSize(), -1);
        //    chunkIndex_.insert(std::pair<ChunkKey, Buffer*>(key, b));
        //}
        //else {
        //    b = chunkIt->second;
        //}
        //
        //// should be a consistent page size for a given ChunkKey
        //assert(b->pageSize() == d->pageSize());
        //
        //// if necessary, reserve memory for b
        //if (b->mem_ == nullptr) {

        //    // Find n bytes of free memory in the buffer pool
        //    auto freeMemIt = freeMem_.lower_bound(nbytes);
        //    if (freeMemIt == freeMem_.end()) {
        //        delete b;
        //        chunkIndex_.erase(chunkIt);
        //        throw std::runtime_error("Out of memory");
        //        // @todo eviction strategies
        //    }
        //    
        //    // Save free memory information
        //    mapd_size_t freeMemSize = freeMemIt->first;
        //    mapd_addr_t bufAddr = freeMemIt->second;
        //    
        //    // update Buffer's pointer
        //    b->mem_ = bufAddr;
        //    
        //    // Remove entry from map, and insert new entry
        //    freeMem_.erase(freeMemIt);
        //    if (freeMemSize - nbytes > 0)
        //        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(freeMemSize - nbytes, bufAddr + nbytes));
        //}
        //
        //// b and d should be the same size
        //if (b->size() != d->size())
        //    throw std::runtime_error("Size mismatch between source and destination buffers.");

        //// read the contents of d into b
        //d->read(b->mem_, 0);
        //
        //return b;
    }
    
    /// client is responsible for deleting memory allocated for b->mem_
    AbstractBuffer* BufferMgr::createBuffer(mapd_size_t pageSize, mapd_size_t nbytes) {
        //assert(pageSize > 0 && nbytes > 0);
        //mapd_size_t numPages = (pageSize + nbytes - 1) / pageSize;
        //Buffer *b = new Buffer(nullptr, numPages, pageSize, -1);
        //
        //// Find nbytes of free memory in the buffer pool
        //auto freeMemIt = freeMem_.lower_bound(nbytes);
        //if (freeMemIt == freeMem_.end()) {
        //    delete b;
        //    throw std::runtime_error("Out of memory");
        //    // @todo eviction strategies
        //}
        //
        //// Save free memory information
        //mapd_size_t freeMemSize = freeMemIt->first;
        //mapd_addr_t bufAddr = freeMemIt->second;

        //// update Buffer's pointer
        //b->mem_ = bufAddr;
        //
        //// Remove entry from map, and insert new entry
        //freeMem_.erase(freeMemIt);
        //if (freeMemSize - nbytes > 0)
        //    freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(freeMemSize - nbytes, bufAddr + nbytes));
        //
        //return b;
    }
    
    void BufferMgr::deleteBuffer(AbstractBuffer *d) {
        //assert(d);
        //Buffer *b = (Buffer*)d;
        //
        //// return the free memory used by the Chunk back to the free memory pool
        //freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(b->size(), b->mem_));
        //delete[] b->mem_;
        //delete b;
    }
    
    AbstractBuffer* BufferMgr::putBuffer(AbstractBuffer *d) {
        // @todo write this putBuffer() method
        return nullptr;
    }
    
    mapd_size_t BufferMgr::size() {
        return pageSize_*numPages_;
    }
    
}
