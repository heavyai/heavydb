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
   
    typedef vector <vector <AbstractDataMgr *> > AbstractDataMgrVec; // one vector for each level 
    typedef std::map<int,std::vector<int> > FirstLevelPartitionToDevicesMap;
    typedef std::map<ChunkKey,std::vector<bool> > ChunkKeyLocationMap; // maybe should be 

    /**
     * @class   MemoryMgr
     * @brief   Managing memory is fun.
     */
    class MemoryMgr {
        
    public:
        MemoryMgr(const size_t cpuBufferSize);
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
        AbstractDataMgrVec abstractDataMgrVec_;
        FirstLevelPartitionToDevicesMap firstLevelPartitionToDevicesMap_;
        ChunkKeyLocationMap chunkKeyLocationMap_;
        // needs to store page size
    
        // need a map of partition id to device at each level 
        // need a map of chunk key to level that its on

        
    };

} // Memory_Namespace
    
#endif // DATAMGR_MEMORY_MEMORYMGR_H
