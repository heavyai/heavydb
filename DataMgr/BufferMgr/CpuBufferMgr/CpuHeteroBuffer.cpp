/*
 * Copyright 2021 OmniSci, Inc.
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

#include "DataMgr/BufferMgr/CpuBufferMgr/CpuHeteroBuffer.h"
#include "CudaMgr/CudaMgr.h"

#include "Logger/Logger.h"

namespace Buffer_Namespace {

CpuHeteroBuffer::CpuHeteroBuffer(const int device_id,
                                 std::pmr::memory_resource* mem_resource,
                                 CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                 const size_t page_size,
                                 const size_t num_bytes)
    : AbstractBuffer(device_id)
    , page_size_(page_size)
    , num_pages_(0)
    , pin_count_(0)
    , buffer_(mem_resource)
    , cuda_mgr_(cuda_mgr) {
  pin();
  if (num_bytes > 0) {
    reserve(num_bytes);
  }
}

CpuHeteroBuffer::~CpuHeteroBuffer() {}
// TODO: should we accept AbstractBuffer instead of int8_t*
void CpuHeteroBuffer::read(int8_t* const dst,
                           const size_t num_bytes,
                           const size_t offset,
                           const MemoryLevel dst_buffer_type,
                           const int dst_device_id) {
  if (num_bytes == 0) {
    return;
  }
  CHECK(dst && getMemoryPtr());

  if (num_bytes + offset > this->size()) {
    LOG(FATAL) << "Buffer: Out of bounds read error";
  }
  readData(dst, num_bytes, offset, dst_buffer_type, dst_device_id);
}

void CpuHeteroBuffer::write(int8_t* src,
                            const size_t num_bytes,
                            const size_t offset,
                            const MemoryLevel src_buffer_type,
                            const int src_device_id) {
  CHECK_GT(num_bytes, size_t(0));

  if (num_bytes + offset > reservedSize()) {
    reserve(num_bytes + offset);
  }

  // write source contents to buffer
  writeData(src, num_bytes, offset, src_buffer_type, src_device_id);

  // update dirty flags for buffer and each affected page
  is_dirty_ = true;
  if (offset < size_) {
    is_updated_ = true;
  }
  if (offset + num_bytes > size_) {
    is_appended_ = true;
    size_ = offset + num_bytes;
  }

  size_t first_dirty_page = offset / page_size_;
  size_t last_dirty_page = (offset + num_bytes - 1) / page_size_;
  for (size_t i = first_dirty_page; i <= last_dirty_page; ++i) {
    page_dirty_flags_[i] = true;
  }
}

void CpuHeteroBuffer::reserve(const size_t num_bytes) {
  size_t num_pages = (num_bytes + page_size_ - 1) / page_size_;
  if (num_pages > num_pages_) {
    buffer_.resize(page_size_ * num_pages);
    page_dirty_flags_.resize(num_pages);
    num_pages_ = num_pages;
  }
}

void CpuHeteroBuffer::append(int8_t* src,
                             const size_t num_bytes,
                             const MemoryLevel src_buffer_type,
                             const int src_device_id) {
  is_dirty_ = true;
  is_appended_ = true;

  if (num_bytes + size_ > reservedSize()) {
    reserve(num_bytes + size_);
  }

  writeData(src, num_bytes, size_, src_buffer_type, src_device_id);
  size_ += num_bytes;
  // Do we worry about dirty flags here or does append avoid them
}

void CpuHeteroBuffer::readData(int8_t* const dst,
                               const size_t num_bytes,
                               const size_t offset,
                               const MemoryLevel dst_buffer_type,
                               const int dst_device_id) {
  int8_t* src = getMemoryPtr() + offset;
  if (dst_buffer_type == CPU_LEVEL) {
    memcpy(dst, src, num_bytes);
  } else if (dst_buffer_type == GPU_LEVEL) {
    CHECK_GE(dst_device_id, 0);
    cuda_mgr_->copyHostToDevice(dst, src, num_bytes, dst_device_id);
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

void CpuHeteroBuffer::writeData(int8_t* const src,
                                const size_t num_bytes,
                                const size_t offset,
                                const MemoryLevel src_buffer_type,
                                const int src_device_id) {
  CHECK(num_bytes + offset > reservedSize());
  int8_t* dst = getMemoryPtr() + offset;
  if (src_buffer_type == CPU_LEVEL) {
    memcpy(dst, src, num_bytes);
  } else if (src_buffer_type == GPU_LEVEL) {
    // std::cout << "Writing to CPU from source GPU" << std::endl;
    CHECK_GE(src_device_id, 0);
    cuda_mgr_->copyDeviceToHost(dst, src, num_bytes, src_device_id);
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

}  // namespace Buffer_Namespace