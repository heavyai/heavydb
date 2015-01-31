/**
 * @file    DataMgr.cpp
 * @author Todd Mostak <todd@mapd.com>
 */

#include "DataMgr.h"
#include "FileMgr/FileMgr.h"
#include "../CudaMgr/CudaMgr.h"
#include "BufferMgr/CpuBufferMgr/CpuBufferMgr.h"
#include "BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.h"

#include <limits>


using namespace std;
using namespace Buffer_Namespace;
using namespace File_Namespace;

namespace Data_Namespace {

    DataMgr::DataMgr(const int partitionKeyIndex, const string &dataDir): partitionKeyIndex_(partitionKeyIndex), dataDir_(dataDir) {
        try {
            cudaMgr_ = new CudaMgr_Namespace::CudaMgr;
            hasGpus_ = true;
        }
        catch (std::runtime_error &error) {
            hasGpus_ = false;
            cudaMgr_ = 0;
        }

        populateMgrs();
    }

    DataMgr::~DataMgr() {
        int numLevels = bufferMgrs_.size();
        for (int level = numLevels - 1; level >= 0; --level) {
            for (int device = 0; device < bufferMgrs_[level].size(); device++) {
                delete bufferMgrs_[level][device];
            }
        }
        if (hasGpus_) {
            delete cudaMgr_;
        }
    }



    void DataMgr::populateMgrs() {
        bufferMgrs_.resize(2);
        bufferMgrs_[0].push_back(new FileMgr (dataDir_)); 
        levelSizes_.push_back(1);
        if (hasGpus_) {
            bufferMgrs_.resize(3);
            bufferMgrs_[1].push_back(new CpuBufferMgr(std::numeric_limits<unsigned int>::max(), CUDA_HOST, cudaMgr_, 1 << 30,512,bufferMgrs_[0][0]));  // allocate 4GB for now
            levelSizes_.push_back(1);
            int numGpus = cudaMgr_->getDeviceCount();
            for (int gpuNum = 0; gpuNum < numGpus; ++gpuNum) {
                size_t gpuMaxMemSize = (cudaMgr_->deviceProperties[gpuNum].globalMem) - (1<<29); // set max mem size to be size of global mem - 512MB
                bufferMgrs_[2].push_back(new GpuCudaBufferMgr(gpuMaxMemSize, gpuNum, cudaMgr_, 1 << 29,512,bufferMgrs_[1][0]));
            }
            levelSizes_.push_back(numGpus);
        }
        else {
            bufferMgrs_[1].push_back(new CpuBufferMgr(std::numeric_limits<unsigned int>::max(), CPU_HOST, cudaMgr_, 1 << 30,512,bufferMgrs_[0][0]));  // allocate 4GB for now
            levelSizes_.push_back(1);
        }
    }
    /*
    DataMgr::getAllChunkMetaInfo(std::vector<std::pair<ChunkKey,int64_t> > &metadata)  {
        // needed by TablePartitionerMgr
        bufferMgrs_[0]->getAllChunkMetaInfo(metadata);
    }
    */
    /*
    const std::map<ChunkKey, File_Namespace::FileBuffer *> & DataMgr::getChunkMap()  {
        return reinterpret_cast <File_Namespace::FileMgr *> (bufferMgrs_[0][0])->chunkIndex_;
    }
    */
    
    void DataMgr::getChunkMetadataVec(std::vector<std::pair <ChunkKey,ChunkMetadata> > &chunkMetadataVec) {
        // Can we always assume this will just be at the disklevel bc we just
        // started?
        bufferMgrs_[0][0]->getChunkMetadataVec(chunkMetadataVec);
    }

    void DataMgr::getChunkMetadataVecForKeyPrefix(std::vector<std::pair <ChunkKey,ChunkMetadata> > &chunkMetadataVec, const ChunkKey &keyPrefix) {
        bufferMgrs_[0][0]->getChunkMetadataVecForKeyPrefix(chunkMetadataVec,keyPrefix);
    }

    AbstractBuffer * DataMgr::createChunk(const ChunkKey &key, const MemoryLevel memoryLevel, const int deviceId) {
        int level = static_cast <int> (memoryLevel);
        //int device = key[partitionKeyIndex_] % levelSizes_[level];
        return bufferMgrs_[level][deviceId]->createChunk(key);
    }

    AbstractBuffer * DataMgr::getChunk(const ChunkKey &key, const MemoryLevel memoryLevel, const int deviceId, const size_t numBytes) {
        int level = static_cast <int> (memoryLevel);
        //int device = key[partitionKeyIndex_] % levelSizes_[level];
        return bufferMgrs_[level][deviceId]->getChunk(key, numBytes);
    }

    void DataMgr::deleteChunk(const ChunkKey &key) {
        // We don't know whether a given manager (of
        // correct partition key) actually has a chunk at
        // a given point. So try-except block a delete to
        // all of them.  Will change if we have DataMgr
        // keep track of this state
        int numLevels = bufferMgrs_.size();
        for (int level = numLevels - 1; level >= 0; --level) {
            int device = key[partitionKeyIndex_] % levelSizes_[level];
            try {
                bufferMgrs_[level][device]->deleteChunk(key);
            }
            catch (std::runtime_error &error) {
                std::cout << "Chunk did not exist at level " <<level <<  std::endl;
            }
        }
    }

    void DataMgr::deleteChunksWithPrefix(const ChunkKey &keyPrefix) {
        int numLevels = bufferMgrs_.size();
        for (int level = numLevels - 1; level >= 0; --level) {
            for (int device = 0; device < levelSizes_[level]; ++device) {
                bufferMgrs_[level][device]->deleteChunksWithPrefix(keyPrefix);
            }
        }
    }

    AbstractBuffer * DataMgr::createBuffer(const MemoryLevel memoryLevel, const int deviceId, const size_t numBytes) {
        int level = static_cast <int> (memoryLevel);
        assert(deviceId < levelSizes_[level]);
        return bufferMgrs_[level][deviceId]->createBuffer(numBytes);
    }

    void DataMgr::deleteBuffer(const MemoryLevel memoryLevel, const int deviceId, AbstractBuffer *buffer) {
        int level = static_cast <int> (memoryLevel);
        assert(deviceId < levelSizes_[level]);
        bufferMgrs_[level][deviceId]->deleteBuffer(buffer);
    }


    void DataMgr::checkpoint() {

        for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
            // use reverse iterator so we start at GPU level, then CPU then DISK
            for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
                (*deviceIt)->checkpoint();
            }
        }
    }
} // Data_Namespace



















