/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/AbstractBuffer.h"

using namespace Data_Namespace;

namespace foreign_storage {
class ForeignStorageBuffer : public AbstractBuffer {
 public:
  ForeignStorageBuffer();
  virtual ~ForeignStorageBuffer() = default;

  void read(int8_t* const destination,
            const size_t num_bytes,
            const size_t offset = 0,
            const MemoryLevel destination_buffer_type = CPU_LEVEL,
            const int destination_device_id = -1) override;

  void write(int8_t* source,
             const size_t num_bytes,
             const size_t offset = 0,
             const MemoryLevel source_buffer_type = CPU_LEVEL,
             const int source_device_id = -1) override;

  void reserve(size_t additional_num_bytes) override;

  void append(int8_t* source,
              const size_t num_bytes,
              const MemoryLevel source_buffer_type = CPU_LEVEL,
              const int device_id = -1) override;

  int8_t* getMemoryPtr() override;
  size_t pageCount() const override;
  size_t pageSize() const override;
  size_t reservedSize() const override;
  MemoryLevel getType() const override;

 private:
  std::vector<int8_t> buffer_;
};
}  // namespace foreign_storage
