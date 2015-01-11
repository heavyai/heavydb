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

    AbstractBuffer * MemoryMgr::createChunk(const MemoryLevel memoryLevel, ChunkKey &key) {
        int level = static_cast <int> (memoryLevel);
        int device = key[partitionKeyIndex_] % levelSizes_[level];
        return bufferMgrs_[level][device] -> createChunk(key);
    }


    AbstractBuffer * MemoryMgr::getChunk(const MemoryLevel memoryLevel, ChunkKey &key, const mapd_size_t numBytes) {
        int level = static_cast <int> (memoryLevel);
        int device = key[partitionKeyIndex_] % levelSizes_[level];
        return bufferMgrs_[level][device] -> getChunk(key, numBytes);
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



















