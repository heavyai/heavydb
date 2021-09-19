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

#include "DataMgr/ForeignStorage/TypedParquetStorageBuffer.h"

namespace foreign_storage {
template <typename Type>
TypedParquetStorageBuffer<Type>::TypedParquetStorageBuffer() : AbstractBuffer(0) {}

template <typename Type>
void TypedParquetStorageBuffer<Type>::read(
    int8_t* const destination,
    const size_t num_bytes,
    const size_t offset,
    const Data_Namespace::MemoryLevel destination_buffer_type,
    const int destination_device_id) {
  UNREACHABLE();
}

template <typename Type>
void TypedParquetStorageBuffer<Type>::reserve(size_t additional_num_bytes) {
  UNREACHABLE();
}

template <typename Type>
void TypedParquetStorageBuffer<Type>::append(
    int8_t* source,
    const size_t num_bytes,
    const Data_Namespace::MemoryLevel source_buffer_type,
    const int device_id) {
  UNREACHABLE();
}

template <typename Type>
int8_t* TypedParquetStorageBuffer<Type>::getMemoryPtr() {
  UNREACHABLE();
  return {};
}

template <typename Type>
size_t TypedParquetStorageBuffer<Type>::reservedSize() const {
  UNREACHABLE();
  return 0;
}

template <typename Type>
Data_Namespace::MemoryLevel TypedParquetStorageBuffer<Type>::getType() const {
  UNREACHABLE();
  return Data_Namespace::CPU_LEVEL;
}

template <typename Type>
void TypedParquetStorageBuffer<Type>::write(
    int8_t* source,
    const size_t num_bytes,
    const size_t offset,
    const Data_Namespace::MemoryLevel source_buffer_type,
    const int source_device_id) {
  UNREACHABLE();
}

template <typename Type>
size_t TypedParquetStorageBuffer<Type>::pageCount() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

template <typename Type>
size_t TypedParquetStorageBuffer<Type>::pageSize() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

template <typename Type>
void TypedParquetStorageBuffer<Type>::reserveNumElements(size_t additional_num_elements) {
  buffer_.reserve(buffer_.size() + additional_num_elements);
}

template <typename Type>
void TypedParquetStorageBuffer<Type>::appendElement(const Type& element) {
  buffer_.push_back(element);
}

template <typename Type>
std::vector<Type>* TypedParquetStorageBuffer<Type>::getBufferPtr() {
  return &buffer_;
}

// Instantiate the two templates that are expected to be used
template class TypedParquetStorageBuffer<std::string>;
template class TypedParquetStorageBuffer<ArrayDatum>;

}  // namespace foreign_storage
