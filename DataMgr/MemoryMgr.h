/**
 * @file    MemoryMgr.h
 * @author Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_MEMORYMGR_H
#define DATAMGR_MEMORY_MEMORYMGR_H

#include <map>
#include <vector>
#include <string>
#include "AbstractBufferMgr.h"
#include "AbstractBuffer.h"

namespace Memory_Namespace {

    enum MemoryLevel {DISK_LEVEL = 0, CPU_LEVEL = 1, GPU_LEVEL = 2};

    class MemoryMgr { 

        public:
            MemoryMgr(const int partitionKeyIndex, const std::string &dataDir);
            AbstractBuffer * createChunk(const MemoryLevel memoryLevel, ChunkKey &key);
            AbstractBuffer * getChunk(const MemoryLevel memoryLevel, ChunkKey &key, const mapd_size_t numBytes = 0);

            void checkpoint();

            // database_id, table_id, partitioner_id, column_id, fragment_id

        private:
            void populateMgrs();
            std::vector <std::vector <AbstractBufferMgr *> > bufferMgrs_;
            std::vector <int> levelSizes_;
            std::string dataDir_;
            int partitionKeyIndex_;
    };
} // Memory_Namespace


#endif






