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

#include "DataMgr/BufferMgr/CpuBufferMgr/CpuBufferMgr.h"

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/BufferMgr/CpuBufferMgr/CpuBuffer.h"

namespace Buffer_Namespace {

CpuBufferMgr::CpuBufferMgr(const int device_id,
                           const size_t max_buffer_size,
                           CudaMgr_Namespace::CudaMgr* cuda_mgr,
                           const size_t buffer_alloc_increment,
                           const size_t page_size,
                           AbstractBufferMgr* parent_mgr)
    : BufferMgr(device_id, max_buffer_size, buffer_alloc_increment, page_size, parent_mgr)
    , cuda_mgr_(cuda_mgr) {}

CpuBufferMgr::~CpuBufferMgr() {
  freeAllMem();
}

void CpuBufferMgr::addSlab(const size_t slab_size) {
  slabs_.resize(slabs_.size() + 1);
  try {
    slabs_.back() = new int8_t[slab_size];
  } catch (std::bad_alloc&) {
    slabs_.resize(slabs_.size() - 1);
    throw FailedToCreateSlab(slab_size);
  }
  slab_segments_.resize(slab_segments_.size() + 1);
  slab_segments_[slab_segments_.size() - 1].push_back(
      BufferSeg(0, slab_size / page_size_));
}

void CpuBufferMgr::freeAllMem() {
  for (auto buf_it = slabs_.begin(); buf_it != slabs_.end(); ++buf_it) {
    delete[] * buf_it;
  }
}

void CpuBufferMgr::allocateBuffer(BufferList::iterator seg_it,
                                  const size_t page_size,
                                  const size_t initial_size) {
  new CpuBuffer(this,
                seg_it,
                device_id_,
                cuda_mgr_,
                page_size,
                initial_size);  // this line is admittedly a bit weird but
                                // the segment iterator passed into buffer
                                // takes the address of the new Buffer in its
                                // buffer member
}

}  // namespace Buffer_Namespace
