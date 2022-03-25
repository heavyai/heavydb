/*
 * Copyright 2021 HEAVY.AI, Inc.
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

#include "DataMgr/ForeignStorage/TypedParquetDetectBuffer.h"

namespace foreign_storage {

TypedParquetDetectBuffer::TypedParquetDetectBuffer() : AbstractBuffer(0) {}

void TypedParquetDetectBuffer::read(
    int8_t* const destination,
    const size_t num_bytes,
    const size_t offset,
    const Data_Namespace::MemoryLevel destination_buffer_type,
    const int destination_device_id) {
  UNREACHABLE();
}

void TypedParquetDetectBuffer::reserve(size_t additional_num_bytes) {
  UNREACHABLE();
}

void TypedParquetDetectBuffer::append(
    int8_t* source,
    const size_t num_bytes,
    const Data_Namespace::MemoryLevel source_buffer_type,
    const int device_id) {
  CHECK(data_to_string_converter_);
  auto strings = data_to_string_converter_->convert(source, num_bytes);
  buffer_.insert(buffer_.end(), strings.begin(), strings.end());
}

int8_t* TypedParquetDetectBuffer::getMemoryPtr() {
  UNREACHABLE();
  return {};
}

size_t TypedParquetDetectBuffer::reservedSize() const {
  UNREACHABLE();
  return 0;
}

Data_Namespace::MemoryLevel TypedParquetDetectBuffer::getType() const {
  UNREACHABLE();
  return Data_Namespace::CPU_LEVEL;
}

void TypedParquetDetectBuffer::write(int8_t* source,
                                     const size_t num_bytes,
                                     const size_t offset,
                                     const Data_Namespace::MemoryLevel source_buffer_type,
                                     const int source_device_id) {
  UNREACHABLE();
}

size_t TypedParquetDetectBuffer::pageCount() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t TypedParquetDetectBuffer::pageSize() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

void TypedParquetDetectBuffer::reserveNumElements(size_t additional_num_elements) {
  buffer_.reserve(buffer_.size() + additional_num_elements);
}

}  // namespace foreign_storage
