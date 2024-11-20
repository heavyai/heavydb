/*
 * Copyright 2022 HEAVY.AI, Inc.
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
ForeignStorageBuffer::ForeignStorageBuffer() : AbstractBuffer(0) {}

void ForeignStorageBuffer::read(int8_t* const destination,
                                const size_t num_bytes,
                                const size_t offset,
                                const MemoryLevel destination_buffer_type,
                                const int destination_device_id) {
  memcpy(destination, buffer_.data() + offset, num_bytes);
}

void ForeignStorageBuffer::reserve(size_t total_num_bytes) {
  buffer_.reserve(total_num_bytes);
}

void ForeignStorageBuffer::append(int8_t* source,
                                  const size_t num_bytes,
                                  const MemoryLevel source_buffer_type,
                                  const int device_id) {
  buffer_.resize(size_ + num_bytes);
  std::copy(source, source + num_bytes, buffer_.begin() + size_);
  size_ += num_bytes;
}

int8_t* ForeignStorageBuffer::getMemoryPtr() {
  return buffer_.data();
}

size_t ForeignStorageBuffer::reservedSize() const {
  return buffer_.capacity();
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
