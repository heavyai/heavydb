/**
 * @file    MemoryMgr.h
 * @author Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_MEMORYMGR_H
#define DATAMGR_MEMORY_MEMORYMGR_H

#include "AbstractBufferMgr.h"
#include "AbstractBuffer.h"

#include <map>
#include <vector>
#include <string>
#include <gtest/gtest_prod.h>

namespace Memory_Namespace {

    enum MemoryLevel {DISK_LEVEL = 0, CPU_LEVEL = 1, GPU_LEVEL = 2};

    class MemoryMgr { 

        public:
            MemoryMgr(const int partitionKeyIndex, const std::string &dataDir);
            AbstractBuffer * createChunk(const MemoryLevel memoryLevel, const ChunkKey &key);
            AbstractBuffer * getChunk(const MemoryLevel memoryLevel, const ChunkKey &key, const mapd_size_t numBytes = 0);
            void deleteChunk(const ChunkKey &key);

            AbstractBuffer * createBuffer(const MemoryLevel memoryLevel, const int deviceId, const mapd_size_t numBytes);
            void deleteBuffer(const MemoryLevel memoryLevel, const int deviceId, AbstractBuffer *buffer);
            AbstractBuffer * copyBuffer(const MemoryLevel memoryLevel, const int deviceId, const AbstractBuffer * srcBuffer);


            void checkpoint();

            // database_id, table_id, partitioner_id, column_id, fragment_id

        private:
            FRIEND_TEST(MemoryMgrTest,buffer);
            void populateMgrs();
            std::vector <std::vector <AbstractBufferMgr *> > bufferMgrs_;
            std::vector <int> levelSizes_;
            std::string dataDir_;
            int partitionKeyIndex_;
    };
} // Memory_Namespace


#endif






