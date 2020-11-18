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
#include <mutex>
#include <vector>
#include <memory_resource>

namespace CudaMgr_Namespace {
class CudaMgr;
}

using namespace Data_Namespace;

namespace Buffer_Namespace {
class CpuHeteroBuffer : public AbstractBuffer {
public:
  CpuHeteroBuffer(const int device_id,
                  std::pmr::memory_resource* mem_resource,
                  CudaMgr_Namespace::CudaMgr* cuda_mgr,
                  const size_t page_size = 512,
                  const size_t num_bytes = 0);

  /// Destructor
  ~CpuHeteroBuffer() override;

  CpuHeteroBuffer(const CpuHeteroBuffer&) = delete;
  CpuHeteroBuffer& operator=(const CpuHeteroBuffer&) = delete;

  /**
   * @brief Reads (copies) data from the buffer to the destination (dst) memory location.
   * Reads (copies) nbytes of data from the buffer, beginning at the specified byte
   * offset, into the destination (dst) memory location.
   *
   * @param dst       The destination address to where the buffer's data is being copied.
   * @param offset    The byte offset into the buffer from where reading (copying) begins.
   * @param nbytes    The number of bytes being read (copied) into the destination (dst).
   */
  void read(int8_t* const dst,
            const size_t num_bytes,
            const size_t offset = 0,
            const MemoryLevel dst_buffer_type = CPU_LEVEL,
            const int device_id = -1) override;

  /**
   * @brief Writes (copies) data from src into the buffer.
   * Writes (copies) nbytes of data into the buffer at the specified byte offset, from
   * the source (src) memory location.
   *
   * @param src        The source address from where data is being copied to the buffer.
   * @param num_bytes  The number of bytes being written (copied) into the buffer.
   * @param offset     The byte offset into the buffer to where writing begins.
   */
  void write(int8_t* src,
             const size_t num_bytes,
             const size_t offset = 0,
             const MemoryLevel src_buffer_type = CPU_LEVEL,
             const int device_id = -1) override;

  void reserve(const size_t num_bytes) override;

  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type = CPU_LEVEL,
              const int deviceId = -1) override;

  /**
   * @brief Returns a raw, constant (read-only) pointer to the underlying buffer.
   * @return A constant memory pointer for read-only access.
   */
  inline int8_t* getMemoryPtr() override { return buffer_.data(); };

  /// Returns the number of pages in the buffer.
  inline size_t pageCount() const override { return num_pages_; }

  /// Returns the size in bytes of each page in the buffer.
  inline size_t pageSize() const override { return page_size_; }

  inline size_t size() const override { return size_; }

  /// Returns the total number of bytes allocated for the buffer.
  inline size_t reservedSize() const override { return buffer_.size(); }

  inline MemoryLevel getType() const override { return CPU_LEVEL; }

  inline int pin() override {
    std::lock_guard<pin_mutex_type> pin_lock(pin_mutex_);
    return (++pin_count_);
  }

  inline int unPin() override {
    std::lock_guard<pin_mutex_type> pin_lock(pin_mutex_);
    return (--pin_count_);
  }

  inline int getPinCount() override {
    std::lock_guard<pin_mutex_type> pin_lock(pin_mutex_);
    return (pin_count_);
  }
private:
  void readData(int8_t* const dst,
                const size_t num_bytes,
                const size_t offset = 0,
                const MemoryLevel dst_buffer_type = CPU_LEVEL,
                const int dst_device_id = -1);

  void writeData(int8_t* const src,
                 const size_t num_bytes,
                 const size_t offset = 0,
                 const MemoryLevel src_buffer_type = CPU_LEVEL,
                 const int src_device_id = -1);


  using pin_mutex_type = std::mutex;
  using vector_type = std::pmr::vector<int8_t>;

  size_t page_size_;  /// the size of each page in the buffer
  size_t num_pages_;
  int epoch_;  /// indicates when the buffer was last flushed
  
  // TODO: Should we use the same allocator as for the buffer_?
  std::vector<bool> page_dirty_flags_;
  int pin_count_;
  pin_mutex_type pin_mutex_;

  vector_type buffer_;

  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
};
}  // namespace Buffer_Namespace