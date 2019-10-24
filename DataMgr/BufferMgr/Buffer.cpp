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

//
//  Buffer.cpp
//  mapd2
//
//  @author Steven Stewart <steve@map-d.com>
//  @author Todd Mostak <todd@map-d.com>
//
//  Copyright (c) 2014 MapD Technologies, Inc. All rights reserved.
//
#include "DataMgr/BufferMgr/Buffer.h"

#include <cassert>
#include <stdexcept>

#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Shared/Logger.h"

namespace Buffer_Namespace {

Buffer::Buffer(BufferMgr* bm,
               BufferList::iterator seg_it,
               const int device_id,
               const size_t page_size,
               const size_t num_bytes)
    : AbstractBuffer(device_id)
    , mem_(0)
    , bm_(bm)
    , seg_it_(seg_it)
    , page_size_(page_size)
    , num_pages_(0)
    , pin_count_(0) {
  pin();
  // so that the pointer value of this Buffer is stored
  seg_it_->buffer = this;
  if (num_bytes > 0) {
    reserve(num_bytes);
  }
}

Buffer::~Buffer() {}

void Buffer::reserve(const size_t num_bytes) {
#ifdef BUFFER_MUTEX
  boost::unique_lock<boost::shared_mutex> write_lock(read_write_mutex_);
#endif
  size_t num_pages = (num_bytes + page_size_ - 1) / page_size_;
  // std::cout << "NumPages reserved: " << numPages << std::endl;
  if (num_pages > num_pages_) {
    // When running out of cpu buffers, reserveBuffer() will fail and
    // trigger a SlabTooBig exception, so pageDirtyFlags_ and numPages_
    // MUST NOT be set until reserveBuffer() returns; otherwise, this
    // buffer is not properly resized, so any call to FileMgr::fetchBuffer()
    // will proceed to read(), corrupt heap memory and cause core dump later.
    seg_it_ = bm_->reserveBuffer(seg_it_, page_size_ * num_pages);
    page_dirty_flags_.resize(num_pages);
    num_pages_ = num_pages;
  }
}

void Buffer::read(int8_t* const dst,
                  const size_t num_bytes,
                  const size_t offset,
                  const MemoryLevel dst_buffer_type,
                  const int dst_device_id) {
  if (num_bytes == 0) {
    return;
  }
  CHECK(dst && mem_);
#ifdef BUFFER_MUTEX
  boost::shared_lock<boost::shared_mutex> read_lock(read_write_mutex_);
#endif

  if (num_bytes + offset > size_) {
    LOG(FATAL) << "Buffer: Out of bounds read error";
  }
  readData(dst, num_bytes, offset, dst_buffer_type, dst_device_id);
}

void Buffer::write(int8_t* src,
                   const size_t num_bytes,
                   const size_t offset,
                   const MemoryLevel src_buffer_type,
                   const int src_device_id) {
  CHECK_GT(num_bytes, size_t(0));  // cannot write 0 bytes
#ifdef BUFFER_MUTEX
  boost::unique_lock<boost::shared_mutex> write_lock(read_write_mutex_);
#endif
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
  // std::cout << "Size after write: " << size_ << std::endl;

  size_t first_dirty_page = offset / page_size_;
  size_t last_dirty_page = (offset + num_bytes - 1) / page_size_;
  for (size_t i = first_dirty_page; i <= last_dirty_page; ++i) {
    page_dirty_flags_[i] = true;
  }
}

void Buffer::append(int8_t* src,
                    const size_t num_bytes,
                    const MemoryLevel src_buffer_type,
                    const int src_device_id) {
#ifdef BUFFER_MUTEX
  boost::shared_lock<boost::shared_mutex> read_lock(
      read_write_mutex_);  // keep another thread from getting a write lock
  boost::unique_lock<boost::shared_mutex> append_lock(
      append_mutex_);  // keep another thread from getting an append lock
#endif

  is_dirty_ = true;
  is_appended_ = true;

  if (num_bytes + size_ > reservedSize()) {
    reserve(num_bytes + size_);
  }

  writeData(src, num_bytes, size_, src_buffer_type, src_device_id);
  size_ += num_bytes;
  // Do we worry about dirty flags here or does append avoid them
}

int8_t* Buffer::getMemoryPtr() {
  return mem_;
}
}  // namespace Buffer_Namespace
