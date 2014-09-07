/**
 * @file    MemoryMgr.h
 */
#ifndef DATAMGR_MEMORY_MEMORYMGR_H
#define DATAMGR_MEMORY_MEMORYMGR_H

#include <iostream>
#include <map>
#include "../../Shared/types.h"
#include "AbstractDataMgr.h"
#include "AbstractDatum.h"

namespace Memory_Namespace {
    
    typedef std::multimap<ChunkKey, AbstractDataMgr*> ChunkKeyToDataMgrMMap;
    
    /**
     * @class   MemoryMgr
     * @brief   Managing memory is fun.
     */
    class MemoryMgr : public AbstractDataMgr {
        
    public:
        MemoryMgr();
        ~MemoryMgr();
        
        // Chunk API
        virtual AbstractDatum* createChunk(const ChunkKey &key, mapd_size_t pageSize, mapd_size_t nbytes = 0, mapd_addr_t buf = nullptr);
        
        virtual void deleteChunk(const ChunkKey &key);
        virtual void releaseChunk(const ChunkKey &key);
        virtual void copyChunkToDatum(const ChunkKey &key, AbstractDatum *datum);
        
        // Datum API
        virtual AbstractDatum* createDatum(mapd_size_t pageSize, mapd_size_t nbytes = 0);
        virtual void deleteDatum(AbstractDatum *d);
        
        // MemoryMgr-specific API
        
        
    private:
        ChunkKeyToDataMgrMMap chunkIndex_;
        
    };

} // Memory_Namespace
    
#endif // DATAMGR_MEMORY_MEMORYMGR_H
