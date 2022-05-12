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

#include "PassThroughBuffer.h"

namespace foreign_storage {

PassThroughBuffer::PassThroughBuffer(const int8_t* data, const size_t data_byte_size)
    : AbstractBuffer(0)
    , data_ptr_(const_cast<int8_t*>(data))
    , data_byte_size_(data_byte_size) {
  setSize(data_byte_size_);
}

void PassThroughBuffer::read(int8_t* const destination,
                             const size_t num_bytes,
                             const size_t offset,
                             const MemoryLevel destination_buffer_type,
                             const int destination_device_id) {
  memcpy(destination, data_ptr_ + offset, num_bytes);
}

void PassThroughBuffer::reserve(size_t additional_num_bytes) {
  UNREACHABLE() << " unexpected call to purposefully unimplemented function in "
                   "PassThroughBuffer";
}

void PassThroughBuffer::append(int8_t* source,
                               const size_t num_bytes,
                               const MemoryLevel source_buffer_type,
                               const int device_id) {
  UNREACHABLE() << " unexpected call to purposefully unimplemented function in "
                   "PassThroughBuffer";
}

int8_t* PassThroughBuffer::getMemoryPtr() {
  return data_ptr_;
}

size_t PassThroughBuffer::reservedSize() const {
  UNREACHABLE() << " unexpected call to purposefully unimplemented function in "
                   "PassThroughBuffer";
  return 0;
}

MemoryLevel PassThroughBuffer::getType() const {
  return CPU_LEVEL;
}

void PassThroughBuffer::write(int8_t* source,
                              const size_t num_bytes,
                              const size_t offset,
                              const MemoryLevel source_buffer_type,
                              const int source_device_id) {
  UNREACHABLE() << " unexpected call to purposefully unimplemented function in "
                   "PassThroughBuffer";
}

size_t PassThroughBuffer::pageCount() const {
  UNREACHABLE() << " unexpected call to purposefully unimplemented function in "
                   "PassThroughBuffer";
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t PassThroughBuffer::pageSize() const {
  UNREACHABLE() << " unexpected call to purposefully unimplemented function in "
                   "PassThroughBuffer";
  return 0;  // Added to avoid "no return statement" compiler warning
}
}  // namespace foreign_storage
