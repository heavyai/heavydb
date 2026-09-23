/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/AbstractBuffer.h"

namespace foreign_storage {

// TODO: `TypedParquetStorageBuffer` should not extend `AbstractBuffer`, it
// does so here for convenience. If this class lifetime becomes somewhat
// permanent it should be refactored to not extend `AbstractBuffer`.
template <typename Type>
class TypedParquetStorageBuffer : public Data_Namespace::AbstractBuffer {
 public:
  TypedParquetStorageBuffer();

  void read(int8_t* const destination,
            const size_t num_bytes,
            const size_t offset = 0,
            const Data_Namespace::MemoryLevel destination_buffer_type =
                Data_Namespace::CPU_LEVEL,
            const int destination_device_id = -1) override;

  void write(
      int8_t* source,
      const size_t num_bytes,
      const size_t offset = 0,
      const Data_Namespace::MemoryLevel source_buffer_type = Data_Namespace::CPU_LEVEL,
      const int source_device_id = -1) override;

  void reserve(size_t additional_num_bytes) override;

  void append(
      int8_t* source,
      const size_t num_bytes,
      const Data_Namespace::MemoryLevel source_buffer_type = Data_Namespace::CPU_LEVEL,
      const int device_id = -1) override;

  void reserveNumElements(size_t additional_num_elements);
  void appendElement(const Type& element);

  std::vector<Type>* getBufferPtr();

  int8_t* getMemoryPtr() override;
  size_t pageCount() const override;
  size_t pageSize() const override;
  size_t reservedSize() const override;
  Data_Namespace::MemoryLevel getType() const override;

  template <typename FindContainer>
  void eraseInvalidData(const FindContainer& invalid_indices) {
    if (invalid_indices.empty()) {
      return;
    }
    auto end_it = std::remove_if(buffer_.begin(), buffer_.end(), [&](const Type& value) {
      const Type* start = buffer_.data();
      auto index = std::distance(start, &value);
      return invalid_indices.find(index) != invalid_indices.end();
    });
    buffer_.erase(end_it, buffer_.end());
  }

 private:
  std::vector<Type> buffer_;
};

}  // namespace foreign_storage
