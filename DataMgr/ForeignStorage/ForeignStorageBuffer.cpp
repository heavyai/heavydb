/*
 * Copyright 2020 OmniSci, Inc.
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

#include "ForeignStorageBuffer.h"

namespace foreign_storage {
ForeignStorageBuffer::ForeignStorageBuffer()
    : AbstractBuffer(0), buffer_(nullptr), reserved_byte_count_(0) {}

void ForeignStorageBuffer::read(int8_t* const destination,
                                const size_t num_bytes,
                                const size_t offset,
                                const MemoryLevel destination_buffer_type,
                                const int destination_device_id) {
  memcpy(destination, buffer_.get() + offset, num_bytes);
}

void ForeignStorageBuffer::reserve(size_t additional_num_bytes) {
  if (size_ + additional_num_bytes > reserved_byte_count_) {
    auto old_buffer = std::move(buffer_);
    reserved_byte_count_ += additional_num_bytes;
    buffer_ = std::make_unique<int8_t[]>(reserved_byte_count_);
    if (old_buffer) {
      memcpy(buffer_.get(), old_buffer.get(), size_);
    }
  }
}

void ForeignStorageBuffer::append(int8_t* source,
                                  const size_t num_bytes,
                                  const MemoryLevel source_buffer_type,
                                  const int device_id) {
  if (size_ + num_bytes > reserved_byte_count_) {
    reserve(num_bytes);
  }
  memcpy(buffer_.get() + size_, source, num_bytes);
  size_ += num_bytes;
}

int8_t* ForeignStorageBuffer::getMemoryPtr() {
  return buffer_.get();
}

size_t ForeignStorageBuffer::reservedSize() const {
  return reserved_byte_count_;
}

MemoryLevel ForeignStorageBuffer::getType() const {
  return CPU_LEVEL;
}

void ForeignStorageBuffer::write(int8_t* source,
                                 const size_t num_bytes,
                                 const size_t offset,
                                 const MemoryLevel source_buffer_type,
                                 const int source_device_id) {
  UNREACHABLE();
}

size_t ForeignStorageBuffer::pageCount() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t ForeignStorageBuffer::pageSize() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}
}  // namespace foreign_storage
