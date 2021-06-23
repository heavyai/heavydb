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

#include "DataMgr/AbstractBuffer.h"

namespace TestHelpers {

class TestBuffer : public Data_Namespace::AbstractBuffer {
 public:
  TestBuffer(const SQLTypeInfo sql_type) : AbstractBuffer(0, sql_type) {}
  TestBuffer(const std::vector<int8_t> bytes) : AbstractBuffer(0, kTINYINT) {
    append((int8_t*)bytes.data(), bytes.size());
  }
  TestBuffer(const std::vector<int32_t> bytes) : AbstractBuffer(0, kINT) {
    append((int8_t*)bytes.data(), bytes.size() * 4);
  }

  ~TestBuffer() override {
    if (mem_ != nullptr) {
      free(mem_);
    }
  }

  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset = 0,
            const Data_Namespace::MemoryLevel dst_buffer_type = Data_Namespace::CPU_LEVEL,
            const int dst_device_id = -1) override {
    memcpy(dst, mem_ + offset, num_bytes);
  }

  void write(
      int8_t* src,
      const size_t num_bytes,
      const size_t offset = 0,
      const Data_Namespace::MemoryLevel src_buffer_type = Data_Namespace::CPU_LEVEL,
      const int src_device_id = -1) override {
    CHECK_GE(num_bytes + offset, size_);
    reserve(num_bytes + offset);
    memcpy(mem_ + offset, src, num_bytes);
    size_ = num_bytes + offset;
    setUpdated();
  }

  void reserve(size_t num_bytes) override {
    if (mem_ == nullptr) {
      mem_ = (int8_t*)malloc(num_bytes);
    } else {
      mem_ = (int8_t*)realloc(mem_, num_bytes);
    }
    reserved_size_ = num_bytes;
  }

  void append(
      int8_t* src,
      const size_t num_bytes,
      const Data_Namespace::MemoryLevel src_buffer_type = Data_Namespace::CPU_LEVEL,
      const int device_id = -1) override {
    size_t offset = size_;
    reserve(size_ + num_bytes);
    memcpy(mem_ + offset, src, num_bytes);
    size_ += num_bytes;
    setAppended();
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

  size_t reservedSize() const override { return reserved_size_; }

  Data_Namespace::MemoryLevel getType() const override {
    return Data_Namespace::CPU_LEVEL;
  }

  bool compare(AbstractBuffer* buffer, size_t num_bytes) {
    std::vector<int8_t> left_array(num_bytes);
    std::vector<int8_t> right_array(num_bytes);
    read(left_array.data(), num_bytes);
    buffer->read(right_array.data(), num_bytes);
    if ((std::memcmp(left_array.data(), right_array.data(), num_bytes) == 0) &&
        (hasEncoder() == buffer->hasEncoder())) {
      return true;
    }
    std::cerr << "buffers do not match:\n";
    for (size_t i = 0; i < num_bytes; ++i) {
      std::cerr << "a[" << i << "]: " << (int32_t)left_array[i] << " b[" << i
                << "]: " << (int32_t)right_array[i] << "\n";
    }
    return false;
  }

  void reset() {
    reserved_size_ = 0;
    setSize(0);
    clearDirtyBits();
  }

 protected:
  int8_t* mem_ = nullptr;
  size_t reserved_size_{0};
};

}  // namespace TestHelpers
