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

#include <iostream>

namespace foreign_storage {
ForeignStorageBuffer::ForeignStorageBuffer()
    : AbstractBuffer(0), buffer(nullptr), byte_count(0), reserved_byte_count(0) {}

void ForeignStorageBuffer::read(int8_t* const destination,
                                const size_t num_bytes,
                                const size_t offset,
                                const MemoryLevel destination_buffer_type,
                                const int destination_device_id) {
  memcpy(destination, buffer.get() + offset, num_bytes);
}

void ForeignStorageBuffer::reserve(size_t additional_num_bytes) {
  auto old_buffer = std::move(buffer);
  reserved_byte_count += additional_num_bytes;
  buffer = std::make_unique<int8_t[]>(reserved_byte_count);
  if (old_buffer) {
    memcpy(buffer.get(), old_buffer.get(), byte_count);
  }
}

void ForeignStorageBuffer::append(int8_t* source,
                                  const size_t num_bytes,
                                  const MemoryLevel source_buffer_type,
                                  const int device_id) {
  if (byte_count + num_bytes > reserved_byte_count) {
    reserve(num_bytes);
  }
  memcpy(buffer.get() + byte_count, source, num_bytes);
  byte_count += num_bytes;
}

int8_t* ForeignStorageBuffer::getMemoryPtr() {
  return buffer.get();
}

size_t ForeignStorageBuffer::size() const {
  return byte_count;
}

size_t ForeignStorageBuffer::reservedSize() const {
  return reserved_byte_count;
}

MemoryLevel ForeignStorageBuffer::getType() const {
  return DISK_LEVEL;
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
