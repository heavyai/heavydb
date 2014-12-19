/**
 * @file	BufferMgr.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and related
 * data structures and types.
 */
#ifndef DATAMGR_MEMORY_BUFFER_BUFFERMGR_H
#define DATAMGR_MEMORY_BUFFER_BUFFERMGR_H

#include <iostream>
#include <map>
#include <list>
#include "../AbstractDatum.h"
#include "../AbstractDataMgr.h"
#include "BufferSeg.h"

using namespace Memory_Namespace;

namespace Buffer_Namespace {

    /**
     * @class   BufferMgr
     * @brief
     *
     * Note(s): Forbid Copying Idiom 4.1
     */
    class BufferMgr : public AbstractDataMgr { // implements
        
    public:
        
        /// Constructs a BufferMgr object that allocates memSize bytes.
        //@todo change this to size_t
        explicit BufferMgr(const size_t bufferSize, const mapd_size_t pageSize);
        
        /// Destructor
        virtual ~BufferMgr();
        
        /// Creates a chunk with the specified key and page size.
        virtual AbstractDatum * createChunk(const ChunkKey &key, const mapd_size_t pageSize, const mapd_size_t numBytes = 0);
        
        /// Deletes the chunk with the specified key
        virtual void deleteChunk(const ChunkKey &key);
        
        /// Returns the a pointer to the chunk with the specified key.
        virtual AbstractDatum* getChunk(ChunkKey &key);
        
        /**
         * @brief Puts the contents of d into the Buffer with ChunkKey key.
         * @param key - Unique identifier for a Chunk.
         * @param d - An object representing the source data for the Chunk.
         * @return AbstractDatum*
         */
        virtual AbstractDatum* putChunk(const ChunkKey &key, AbstractDatum *d, mapd_size_t numBytes = 0);

        // Datum API
        virtual AbstractDatum* createDatum(mapd_size_t pageSize, mapd_size_t numBytes = 0);
        virtual void deleteDatum(AbstractDatum *d);
        virtual AbstractDatum* putDatum(AbstractDatum *d);
        
        /// Returns the total number of bytes allocated.
        mapd_size_t size();

        inline MgrType getMgrType () {return CPU_MGR;};

        BufferList::iterator reserveBuffer(BufferList::iterator & segIt, size_t numBytes);
        
    private:
        BufferMgr(const BufferMgr&); // private copy constructor
        BufferMgr& operator=(const BufferMgr&); // private assignment
        


        //std::map<ChunkKey, Buffer*> chunkIndex_;
        std::map<ChunkKey, BufferList::iterator> chunkIndex_;
        size_t bufferSize_;   /// number of bytes allocated for the buffer pool
        size_t pageSize_;
        size_t numPages_;
        mapd_addr_t bufferPool_;       /// beginning memory address of the buffer pool
        //FileMgr *fileMgr_;

        /// Maps sizes of free memory areas to host buffer pool memory addresses
        //@todo change this to multimap
        //std::multimap<mapd_size_t, mapd_addr_t> freeMem_;
        BufferList bufferList_;
        //std::map<mapd_size_t, mapd_addr_t> freeMem_;

        BufferList::iterator evict(BufferList::iterator &evictStart, const size_t numPagesRequested);
        BufferList::iterator findFreeBuffer(size_t numBytes);

        /**
         * @brief Gets a buffer of required size and returns an iterator to it
         *
         * If possible, this function will just select a free buffer of
         * sufficient size and use that. If not, it will evict as many
         * non-pinned but used buffers as needed to have enough space for the
         * buffer
         *
         * @return An iterator to the reserved buffer. We guarantee that this
         * buffer won't be evicted by PINNING it - caller should change this to
         * USED if applicable
         *
         */


    };

} // Buffer_Namespace

#endif // DATAMGR_MEMORY_BUFFER_BUFFERMGR_H
