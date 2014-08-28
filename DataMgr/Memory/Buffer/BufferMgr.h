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
     */
    class BufferMgr : public AbstractDataMgr { // implements
        
    public:
        BufferMgr(mapd_size_t memSize);
        
        virtual ~BufferMgr();
        
        // Chunk API
        virtual void createChunk(const ChunkKey &key, mapd_size_t pageSize);
        virtual void deleteChunk(const ChunkKey &key);
        virtual void releaseChunk(const ChunkKey &key);
        virtual AbstractDatum* getChunk(ChunkKey &key);
        virtual AbstractDatum* putChunk(const ChunkKey &key, AbstractDatum *d);

        // Datum API
        virtual AbstractDatum* createDatum(mapd_size_t pageSize, mapd_size_t nbytes = 0);
        virtual void deleteDatum(AbstractDatum *d);
        virtual AbstractDatum* putDatum(AbstractDatum *d);
        
        // Other
        mapd_size_t size();
        
    private:
        std::map<ChunkKey, Buffer*> chunkIndex_;
        std::map<ChunkKey, mapd_size_t> chunkPageSize_;
        mapd_size_t memSize_;   /// number of bytes allocated for host buffer pool
        mapd_addr_t mem_;       /// beginning memory address of host buffer pool

        /// Maps sizes of free memory areas to host buffer pool memory addresses
        std::multimap<mapd_size_t, mapd_addr_t> freeMem_;
    };

} // Buffer_Namespace

#endif // DATAMGR_MEMORY_BUFFER_BUFFERMGR_H