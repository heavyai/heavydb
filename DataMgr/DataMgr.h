/**
 * @file    DataMgr.h
 * @author Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_H
#define DATAMGR_H

#include "AbstractBufferMgr.h"
#include "AbstractBuffer.h"
#include "MemoryLevel.h"

#include <map>
#include <vector>
#include <string>
#include <gtest/gtest_prod.h>



namespace File_Namespace {
    class FileBuffer;
}

namespace Data_Namespace {

    class DataMgr { 

        friend class FileMgr; 

        public:
            DataMgr(const int partitionKeyIndex, const std::string &dataDir);
            AbstractBuffer * createChunk(const MemoryLevel memoryLevel, const ChunkKey &key);
            AbstractBuffer * getChunk(const MemoryLevel memoryLevel, const ChunkKey &key, const size_t numBytes = 0);
            void deleteChunk(const ChunkKey &key);
            void deleteChunksWithPrefix(const ChunkKey &keyPrefix);
            AbstractBuffer * createBuffer(const MemoryLevel memoryLevel, const int deviceId, const size_t numBytes);
            void deleteBuffer(const MemoryLevel memoryLevel, const int deviceId, AbstractBuffer *buffer);
            AbstractBuffer * copyBuffer(const MemoryLevel memoryLevel, const int deviceId, const AbstractBuffer * srcBuffer);
            //const std::map<ChunkKey, File_Namespace::FileBuffer *> & getChunkMap();
            const std::map<ChunkKey, File_Namespace::FileBuffer *> & getChunkMap();
            void checkpoint();
            void getChunkMetadataVec(std::vector<std::pair<ChunkKey,ChunkMetadata> > &chunkMetadataVec);
            void getChunkMetadataVecForKeyPrefix(std::vector<std::pair<ChunkKey,ChunkMetadata> > &chunkMetadataVec, const ChunkKey &keyPrefix);

            // database_id, table_id, partitioner_id, column_id, fragment_id

        private:
            FRIEND_TEST(DataMgrTest,buffer);
            FRIEND_TEST(DataMgrTest,deletePrefix);
            void populateMgrs();
            std::vector <std::vector <AbstractBufferMgr *> > bufferMgrs_;
            std::vector <int> levelSizes_;
            int partitionKeyIndex_;
            std::string dataDir_;
    };
} // Data_Namespace


#endif // DATAMGR_H






