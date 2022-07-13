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

#pragma once

#include "DataMgr/AbstractBuffer.h"

using namespace Data_Namespace;

namespace foreign_storage {

/**
 *  `PassThroughBuffer` wraps a pointer with specifed data size. This class
 *  does not take ownership of the pointer. The underlying buffer is never
 *  intended to be modified only read, hence the constructor takes a `const`
 *  pointer.
 */
class PassThroughBuffer : public AbstractBuffer {
 public:
  PassThroughBuffer(const int8_t* data, const size_t data_byte_size);
  PassThroughBuffer() = delete;

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
  int8_t* data_ptr_;
  size_t data_byte_size_;
};
}  // namespace foreign_storage
