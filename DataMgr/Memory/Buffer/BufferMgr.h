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
#include "../AbstractDatum.h"
#include "../AbstractDataMgr.h"
#include "Buffer.h"

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
        explicit BufferMgr(mapd_size_t memSize);
        
        /// Destructor
        virtual ~BufferMgr();
        
        /// Creates a chunk with the specified key and page size.
        virtual void createChunk(const ChunkKey &key, mapd_size_t pageSize);
        
        /// Deletes the chunk with the specified key
        virtual void deleteChunk(const ChunkKey &key);
        
        /// Releases (frees) the memory used by the chunk with the specified key
        virtual void releaseChunk(const ChunkKey &key);
        
        /// Returns the a pointer to the chunk with the specified key.
        virtual AbstractDatum* getChunk(ChunkKey &key);
        
        /**
         * @brief Puts the contents of d into the Buffer with ChunkKey key.
         * @param key - Unique identifier for a Chunk.
         * @param d - An object representing the source data for the Chunk.
         * @return AbstractDatum*
         */
        virtual AbstractDatum* putChunk(const ChunkKey &key, AbstractDatum *d);

        // Datum API
        virtual AbstractDatum* createDatum(mapd_size_t pageSize, mapd_size_t nbytes = 0);
        virtual void deleteDatum(AbstractDatum *d);
        virtual AbstractDatum* putDatum(AbstractDatum *d);
        
        /// Returns the total number of bytes allocated.
        mapd_size_t size();
        
    private:
        BufferMgr(const BufferMgr&); // private copy constructor
        BufferMgr& operator=(const BufferMgr&); // private assignment
        std::map<ChunkKey, Buffer*> chunkIndex_;
        std::map<ChunkKey, mapd_size_t> chunkPageSize_;
        mapd_size_t memSize_;   /// number of bytes allocated for the buffer pool
        mapd_addr_t mem_;       /// beginning memory address of the buffer pool

        /// Maps sizes of free memory areas to host buffer pool memory addresses
        std::multimap<mapd_size_t, mapd_addr_t> freeMem_;
    };

} // Buffer_Namespace

#endif // DATAMGR_MEMORY_BUFFER_BUFFERMGR_H