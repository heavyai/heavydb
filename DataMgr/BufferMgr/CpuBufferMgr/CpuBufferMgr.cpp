/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CpuBufferMgr.h"
#include "CpuBuffer.h"
#include <glog/logging.h>
#include "../../../CudaMgr/CudaMgr.h"

namespace Buffer_Namespace {

CpuBufferMgr::CpuBufferMgr(const int deviceId,
                           const size_t maxBufferSize,
                           CudaMgr_Namespace::CudaMgr* cudaMgr,
                           const size_t bufferAllocIncrement,
                           const size_t pageSize,
                           AbstractBufferMgr* parentMgr)
    : BufferMgr(deviceId, maxBufferSize, bufferAllocIncrement, pageSize, parentMgr), cudaMgr_(cudaMgr) {}

CpuBufferMgr::~CpuBufferMgr() {
  freeAllMem();
}

void CpuBufferMgr::addSlab(const size_t slabSize) {
  slabs_.resize(slabs_.size() + 1);
  try {
    slabs_.back() = new int8_t[slabSize];
  } catch (std::bad_alloc&) {
    slabs_.resize(slabs_.size() - 1);
    throw FailedToCreateSlab();
  }
  slabSegments_.resize(slabSegments_.size() + 1);
  slabSegments_[slabSegments_.size() - 1].push_back(BufferSeg(0, slabSize / pageSize_));
}

void CpuBufferMgr::freeAllMem() {
  for (auto bufIt = slabs_.begin(); bufIt != slabs_.end(); ++bufIt) {
    delete[] * bufIt;
  }
}

void CpuBufferMgr::allocateBuffer(BufferList::iterator segIt, const size_t pageSize, const size_t initialSize) {
  new CpuBuffer(this, segIt, deviceId_, cudaMgr_, pageSize, initialSize);  // this line is admittedly a bit weird but
                                                                           // the segment iterator passed into buffer
                                                                           // takes the address of the new Buffer in its
                                                                           // buffer member
}

}  // Buffer_Namespace
