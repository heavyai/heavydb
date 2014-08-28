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
        BufferMgr(mapd_size_t pageSize, mapd_size_t numPages);
        
        virtual ~BufferMgr();
        
        virtual AbstractDatum* createChunk(const ChunkKey &key, mapd_size_t pageSize, mapd_size_t nbytes = 0, mapd_addr_t buf = nullptr);
        virtual void deleteChunk(const ChunkKey &key);
        virtual void releaseChunk(const ChunkKey &key);
        virtual void getChunk(ChunkKey &key);
        virtual AbstractDatum* putChunk(const ChunkKey &key, AbstractDatum *d);
        
        // Datum API
        virtual void createDatum(mapd_size_t pageSize, mapd_size_t nbytes = 0);
        virtual void deleteDatum(int id);
        virtual AbstractDatum* putDatum(AbstractDatum *d);
        
    private:
        std::map<ChunkKey, Buffer> chunkIndex_;
        mapd_size_t memSize_;   /// number of bytes allocated for host buffer pool
        mapd_addr_t mem_;       /// beginning memory address of host buffer pool

        /// Maps sizes of free memory areas to host buffer pool memory addresses
        std::multimap<mapd_size_t, mapd_addr_t> freeMem_;
    };

} // Buffer_Namespace

#endif // DATAMGR_MEMORY_BUFFER_BUFFERMGR_H