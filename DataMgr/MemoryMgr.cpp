/**
 * @file    MemoryMgr.cpp
 * @author Todd Mostak <todd@map-d.com>
 */

#include "MemoryMgr.h"
#include "FileMgr/FileMgr.h"
#include "BufferMgr/CpuBufferMgr/CpuBufferMgr.h"
#include "BufferMgr/GpuCudaBufferMgr/GpuCudaBufferMgr.h"

#include <limits>


using namespace std;
using namespace Buffer_Namespace;
using namespace File_Namespace;

namespace Memory_Namespace {

    MemoryMgr::MemoryMgr(const int partitionKeyIndex, const string &dataDir): partitionKeyIndex_(partitionKeyIndex), dataDir_(dataDir) {
        populateMgrs();
    }

    void MemoryMgr::populateMgrs() {
        bufferMgrs_.resize(3);
        bufferMgrs_[0].push_back(new FileMgr (dataDir_)); 
        levelSizes_.push_back(1);
        bufferMgrs_[1].push_back(new CpuBufferMgr(std::numeric_limits<unsigned int>::max(), CUDA_HOST, 1 << 30,512,bufferMgrs_[0][0])); 
        levelSizes_.push_back(1);
        bufferMgrs_[2].push_back(new GpuCudaBufferMgr(1 << 30, 0, 1 << 29,512,bufferMgrs_[1][0])); 
        levelSizes_.push_back(1);
    }

    AbstractBuffer * MemoryMgr::createChunk(const MemoryLevel memoryLevel, const ChunkKey &key) {
        int level = static_cast <int> (memoryLevel);
        int device = key[partitionKeyIndex_] % levelSizes_[level];
        return bufferMgrs_[level][device] -> createChunk(key);
    }


    AbstractBuffer * MemoryMgr::getChunk(const MemoryLevel memoryLevel, const ChunkKey &key, const mapd_size_t numBytes) {
        int level = static_cast <int> (memoryLevel);
        int device = key[partitionKeyIndex_] % levelSizes_[level];
        return bufferMgrs_[level][device] -> getChunk(key, numBytes);
    }

    void MemoryMgr::deleteChunk(const ChunkKey &key) {
        // We don't know whether a given manager (of
        // correct partition key) actually has a chunk at
        // a given point. So try-except block a delete to
        // all of them.  Will change if we have MemoryMgr
        // keep track of this state
        int numLevels = bufferMgrs_.size();
        for (int level = numLevels - 1; level >= 0; --level) {
            int device = key[partitionKeyIndex_] % levelSizes_[level];
            try {
                bufferMgrs_[level][device] -> deleteChunk(key);
            }
            catch (std::runtime_error &error) {
                std::cout << "Chunk did not exist at level " <<level <<  std::endl;
            }
        }
    }


    void MemoryMgr::checkpoint() {

        for (auto levelIt = bufferMgrs_.rbegin(); levelIt != bufferMgrs_.rend(); ++levelIt) {
            // use reverse iterator so we start at GPU level, then CPU then DISK
            for (auto deviceIt = levelIt->begin(); deviceIt != levelIt->end(); ++deviceIt) {
                (*deviceIt) -> checkpoint();
            }
        }
    }
} // Memory_Namespace



















