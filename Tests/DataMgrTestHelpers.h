/*
 * Copyright 2020 MapD Technologies, Inc.
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

#include "DataMgr/DataMgr.h"

namespace TestHelpers {

class TestBuffer : public AbstractBuffer {
 public:
  TestBuffer(const SQLTypeInfo sql_type) : AbstractBuffer(0, sql_type) {}
  TestBuffer(const std::vector<int8_t> bytes) : AbstractBuffer(0, kTINYINT) {
    write((int8_t*)bytes.data(), bytes.size());
  }
  ~TestBuffer() override {
    if (mem_ != nullptr) {
      free(mem_);
    }
  }

  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset = 0,
            const MemoryLevel dst_buffer_type = CPU_LEVEL,
            const int dst_device_id = -1) override {
    memcpy(dst, mem_ + offset, num_bytes);
  }

  void write(int8_t* src,
             const size_t num_bytes,
             const size_t offset = 0,
             const MemoryLevel src_buffer_type = CPU_LEVEL,
             const int src_device_id = -1) override {
    reserve(num_bytes + offset);
    memcpy(mem_ + offset, src, num_bytes);
    is_dirty_ = true;
    if (offset < size_) {
      is_updated_ = true;
    }
    if (offset + num_bytes > size_) {
      is_appended_ = true;
      size_ = offset + num_bytes;
    }
  }

  void reserve(size_t num_bytes) override {
    if (mem_ == nullptr) {
      mem_ = (int8_t*)malloc(num_bytes);
    } else {
      mem_ = (int8_t*)realloc(mem_, num_bytes);
    }
    size_ = num_bytes;
  }

  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type,
              const int device_id) override {
    UNREACHABLE();
  }

  int8_t* getMemoryPtr() override { return mem_; }

  size_t pageCount() const override {
    UNREACHABLE();
    return 0;
  }

  size_t pageSize() const override {
    UNREACHABLE();
    return 0;
  }

  size_t size() const override { return size_; }

  size_t reservedSize() const override { return size_; }

  MemoryLevel getType() const override { return Data_Namespace::CPU_LEVEL; }

  bool compare(AbstractBuffer* buffer, size_t num_bytes) {
    int8_t left_array[num_bytes];
    int8_t right_array[num_bytes];
    read(left_array, num_bytes);
    buffer->read(right_array, num_bytes);
    if ((std::memcmp(left_array, right_array, num_bytes) == 0) &&
        (has_encoder == buffer->has_encoder)) {
      return true;
    }
    std::cerr << "buffers do not match:\n";
    for (size_t i = 0; i < num_bytes; ++i) {
      std::cerr << "a[" << i << "]: " << (int32_t)left_array[i] << " b[" << i
                << "]: " << (int32_t)right_array[i] << "\n";
    }
    return false;
  }

 protected:
  int8_t* mem_ = nullptr;
  size_t size_ = 0;
};

}  // namespace TestHelpers
