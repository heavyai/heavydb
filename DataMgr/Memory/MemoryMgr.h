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

    enum Level {DISK_LEVEL = 0, CPU_LEVEL = 1, GPU_LEVEL = 2};

   
    typedef vector <vector <AbstractDataMgr *> > AbstractDataMgrVec; // one vector for each level 
    //typedef std::map<int,std::vector<int> > FirstLevelPartitionToDevicesMap; // this probably needs to be different for each table
    /*
     * @type FirstLevelPartitionToDevicesMap
     * @brief Maps a pair of ints signifying table and
     * first-level/inter-node partion id to the devices
     * for that table
     */

    typedef std::map<poir<int,int>,std::vector<int> > FirstLevelPartitionToDevicesMap; // this probably needs to be different for each table
    typedef std::map<ChunkKey,std::vector<bool> > ChunkKeyLocationMap; // maybe should be 

    /**
     * @class   MemoryMgr
     * @brief   Manages Memory
     */
    class MemoryMgr {
        
    public:
        MemoryMgr(const size_t cpuBufferSize);
        ~MemoryMgr();
        
        // Chunk API
        virtual AbstractDatum* createChunk(const ChunkKey &key, mapd_size_t pageSize, mapd_size_t numBytes = 0, enum level,  mapd_addr_t buf = nullptr);
        
        virtual void deleteChunk(const ChunkKey &key);
        virtual void copyChunkToDatum(const ChunkKey &key, AbstractDatum *datum);
        
        // Datum API
        virtual AbstractDatum* createDatum(mapd_size_t pageSize, mapd_size_t nbytes = 0);
        virtual void deleteDatum(AbstractDatum *d);
        
        // MemoryMgr-specific API
        
        
    private:
        AbstractDataMgrVec abstractDataMgrVec_;
        FirstLevelPartitionToDevicesMap firstLevelPartitionToDevicesMap_;
        ChunkKeyLocationMap chunkKeyLocationMap_;
        int partIdIndex_; // which slot of the ChunkKey are we partitioning between devices on
        // needs to store page size
    
        // need a map of partition id to device at each level 
        // need a map of chunk key to level that its on

        
    };

} // Memory_Namespace
    
#endif // DATAMGR_MEMORY_MEMORYMGR_H
