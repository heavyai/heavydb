//
//  MemoryMgr.cpp
//  mapd2
//
//  Created by Steven Stewart on 8/27/14.
//  Copyright (c) 2014 Map-D Technologies, Inc. All rights reserved.
//

#include "MemoryMgr.h"

MemoryMgr::MemoryMgr(const int partIdIndex, const size_t cpuBufferSize): partIdIndex_(partIdIndex) {
    abstractDataMgrVec_.resize(2);
    abstractDataMgrVec_[0].push_back(new FileMgr("."));
    abstractDataMgrVec_[1].push_back(new BufferMgr(cpuBufferSize));
}

MemoryMgr::~MemoryMgr() {
    delete abstractDataMgrVec_[0][0];
    delete abstractDataMgrVec_[1][0];
}

AbstractDatum* MemoryMgr::createChunk(const ChunkKey &key, mapd_size_t pageSize, const int groupId, mapd_size_t nbytes = 0) {
    int interNodePartId = key[partIdIndex_];
    auto deviceMapIt = firstLevelPartitionToDevicesMap_.find(interNodePartId);
    if (deviceMapIt == firstLevelPartitionToDevicesMap_.end()) {
        // we don't have registered this partition yet

    }
    else {
        vector<int> deviceVec = deviceMapIt -> second;
    }

}





