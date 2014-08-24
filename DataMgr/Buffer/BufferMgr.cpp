/**
 * @file	BufferMgr.cpp
 * @author	Steven Stewart <steve@map-d.com>
 */
#include <iostream>
#include <cassert>
#include <cstring>
#include <exception>
#include "BufferMgr.h"
#include "Buffer.h"
#include "../File/FileMgr.h"

using std::cout;
using std::endl;
using File_Namespace::Chunk;

namespace Buffer_Namespace {
    
    /// Prints a warning if fm_ is NULL. It is possible to use BufferMgr without a
    /// file manager, hence why a warning is issued to stderr.
    BufferMgr::BufferMgr(mapd_size_t hostMemSize, FileMgr *fm): opCounter_(0) {
        assert(hostMemSize > 0);
        if (fm == NULL)
            fprintf(stderr, "[%s:%d] Warning: null reference to file manager.\n", __func__, __LINE__);
        fm_ = fm;
        hostMemSize_ = hostMemSize;
        hostMem_ = (mapd_addr_t) new mapd_byte_t[hostMemSize];
        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(hostMemSize, hostMem_));
    }
    
    BufferMgr::~BufferMgr() {
        // Delete buffers
        while (buffers_.size() > 0) {
            delete buffers_.back();
            buffers_.pop_back();
        }
        
        // Delete host memory
        delete[] hostMem_;
    }
    
    /// Finds contiguous free memory in order to create a new Buffer with the requested pages
    Buffer* BufferMgr::createBuffer(mapd_size_t numPages, mapd_size_t pageSize) {
        // Compute total bytes needed
        mapd_size_t n = numPages * pageSize;
        assert(n > 0);
        
        // Find n bytes of free memory in the buffer pool
        auto it = freeMem_.lower_bound(n);
        if (it == freeMem_.end()) {
            fprintf(stderr, "[%s:%d] Error: unable to find %lu bytes available in the buffer pool.\n", __func__, __LINE__, n);
            // @todo eviction strategies
            return NULL;
        }
        
        // Save free memory information
        mapd_size_t freeMemSize = it->first;
        mapd_addr_t bufAddr = it->second;
        
        // Remove entry from map, and insert new entry
        freeMem_.erase(it);
        if (freeMemSize - n > 0)
            freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(freeMemSize - n, bufAddr + n));
        // @todo Defragmentation
        
        // Create Buffer object and add to the BufferMgr
        Buffer *b = new Buffer(bufAddr, numPages, pageSize, opCounter_++);
        buffers_.push_back(b);
        return b;
    }
    
    void BufferMgr::deleteBuffer(Buffer *b) {
        // save buffer information
        mapd_size_t bufferSize = b->size();
        mapd_addr_t bufferAddr = b->host_ptr();
        
        // free buffer and remove from BufferMgr's list of buffers
        delete b;
        buffers_.remove(b); // @todo thread safe needed?
        
        // Add free memory back to buffer pool
        freeMem_.insert(std::pair<mapd_size_t, mapd_addr_t>(bufferSize, bufferAddr));
        // @todo merge with other contiguous free memory entries
    }
    
    Buffer* BufferMgr::createChunk(const ChunkKey &key, mapd_size_t numPages, mapd_size_t pageSize) {
        Buffer *b = NULL;
        
        // First, check if chunk already exists
        if ((b = getChunkBuffer(key)) != NULL) {
            if (b->numPages() != numPages || b->pageSize() != pageSize)
                fprintf(stderr, "[%s:%d] Warning: Chunk already exists; ignoring requested number of pages and page size.\n", __func__, __LINE__);
            return b;
        }
        
        // Ask the file manager to create the new Chunk
        fm_->createChunk(key, numPages * pageSize, 4, NULL, -1);
        
        // Create a new buffer for the new chunk
        b = createBuffer(numPages, pageSize);
        if (b == NULL) {
            return NULL; // unable to create the new buffer
        }
        
        // Insert an entry in chunkIndex_ for the new chunk. Just do it.
        chunkIndex_[key] = b;
        return b;
    }
    
    std::pair<bool, bool> BufferMgr::deleteChunk(const ChunkKey &key) {
        // Return values
        bool bool1 = false;
        bool bool2 = false;
        
        // Delete the Chunk's buffer if it is currently cached
        Buffer *b = findChunkBuffer(key);
        if (b != NULL) {
            deleteBuffer(b);
            b = NULL; // to prevent accessing newly unallocated memory
        }
        bool1 = true;
        
        // Ask file manager to delete the chunk from the file system
        if (fm_->deleteChunk(key) == MAPD_FAILURE)
            bool2 = false;
        else
            bool2 = true;
        
        return std::pair<bool, bool>(bool1, bool2);
    }
    
    /// Presently, only returns the Buffer if it is not currently pinned
    Buffer* BufferMgr::getChunkBuffer(const ChunkKey &key) {
        Buffer *b = NULL;
        
        // Check if buffer is already cached
        b = findChunkBuffer(key);
       
        //@todo create read pins and write pins
        if (b != NULL) {
            if (b->pinned())
                return NULL;
            else {
                b->pin(opCounter_++);
                return b;
            }
        }
        
        // Determine number of pages and page size for chunk
        mapd_size_t numPages;
        mapd_size_t size;
        if ((fm_->getChunkSize(key, &numPages, &size)) != MAPD_SUCCESS)
            return NULL; // Chunk does not exist in file system
        
        //    printf("size: %d numPages: %d\n", size, numPages);
        if (size == 0) {
            //      printf("cmon\n");
            //@todo: proper error handling needed
            return NULL;
        }
        assert((size % numPages) == 0);
        
        // Create buffer and load chunk
        b = createBuffer(numPages, size / numPages);
        if ((fm_->getChunk(key, b->host_ptr())) == NULL) {
            deleteBuffer(b);
            return NULL;
        }
        else {
            // Insert an entry in chunkIndex_ for the new chunk. Just do it.
            chunkIndex_[key] = b;
            mapd_size_t actualSize = 0;
            if ((fm_->getChunkActualSize(key, &actualSize)) != MAPD_SUCCESS)
                return NULL;
            b->length(actualSize);
        }
        return b;
    }
    
    /// Presently, only returns the pointer if the buffer is not currently pinned
    mapd_addr_t BufferMgr::getChunkAddr(const ChunkKey &key, mapd_size_t *length) {
        Buffer *b = findChunkBuffer(key);
        if (b && b->pinned())
            return NULL;
        else if (!b)
            return NULL;
        
        if (length) *length = b->length();
        b->pin(opCounter_++);
        return b->host_ptr();
    }
    
    bool BufferMgr::chunkIsCached(const ChunkKey &key) {
        auto it = chunkIndex_.find(key);
        return (it != chunkIndex_.end());
    }
    
    /// Presently, only flushes a chunk if its buffer is unpinned, and flushes it right away (no queue)
    bool BufferMgr::flushChunk(const ChunkKey &key) {
        Buffer *b = findChunkBuffer(key);
        if (b == NULL) {
            return false;
        }
        else if (b && b->pinned()) {
            return false;
        }
        
        // @todo temporarly using 0 for epoch: update this
        int epoch = 0;
        if ((fm_->putChunk(key, b->length(), b->host_ptr(), epoch)) != MAPD_SUCCESS)
            return false;
        return true;
    }
    
    void BufferMgr::printMemAlloc() {
        mapd_size_t freeMemSize = 0;
        auto it = freeMem_.begin();
        for (; it != freeMem_.end(); ++it)
            freeMemSize += it->first;
        
        printf("Total memory  = %lu bytes\n", hostMemSize_);
        printf("Used memory   = %lu bytes\n", hostMemSize_ - freeMemSize);
        printf("Free memory   = %lu bytes\n", freeMemSize);
        printf("# of buffers  = %lu\n", buffers_.size());
    }
    
    void BufferMgr::printChunkIndex() {
        //auto it = chunkIndex_.begin();
        
    }
    
    Buffer* BufferMgr::findChunkBuffer(const ChunkKey &key) {
        auto it = chunkIndex_.find(key);
        if (it == chunkIndex_.end()) // not found
            return NULL;
        return it->second;
    }

    //void BufferMgr::evictLRU () {




    
} // Buffer_Namespace
