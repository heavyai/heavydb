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

/**
 * @file		Buffer.h
 * @brief
 */

#pragma once

#include <iostream>
#include <mutex>

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/BufferMgr/BufferSeg.h"

using namespace Data_Namespace;

namespace Buffer_Namespace {

class BufferMgr;

/**
 * @class   Buffer
 * @brief
 *
 * Note(s): Forbid Copying Idiom 4.1
 */
class Buffer : public AbstractBuffer {
  friend class BufferMgr;
  friend class FileMgr;

 public:
  /**
   * @brief Constructs a Buffer object.
   * The constructor requires a memory address (provided by BufferMgr), number of pages,
   * and the size in bytes of each page. Additionally, the Buffer can be initialized with
   * an epoch.
   *
   * @param mem       The beginning memory address of the buffer.
   * @param numPages  The number of pages into which the buffer's memory space is divided.
   * @param pageSize  The size in bytes of each page that composes the buffer.
   * @param epoch     A temporal reference implying the buffer is up-to-date up to the
   * epoch.
   */

  /*
  Buffer(const int8_t * mem, const size_t numPages, const size_t pageSize, const int
  epoch);
  */

  Buffer(BufferMgr* bm,
         BufferList::iterator seg_it,
         const int device_id,
         const size_t page_size = 512,
         const size_t num_bytes = 0);

  /// Destructor
  ~Buffer() override;

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

  void reserve(const size_t num_bytes) override;
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

  void append(int8_t* src,
              const size_t num_bytes,
              const MemoryLevel src_buffer_type = CPU_LEVEL,
              const int deviceId = -1) override;

  /**
   * @brief Returns a raw, constant (read-only) pointer to the underlying buffer.
   * @return A constant memory pointer for read-only access.
   */
  int8_t* getMemoryPtr() override;

  void setMemoryPtr(int8_t* new_ptr) override;
  /// Returns the total number of bytes allocated for the buffer.
  inline size_t reservedSize() const override { return page_size_ * num_pages_; }

  /// Returns the number of pages in the buffer.
  inline size_t pageCount() const override { return num_pages_; }

  /// Returns the size in bytes of each page in the buffer.
  inline size_t pageSize() const override { return page_size_; }

  inline int pin() override {
    std::lock_guard<std::mutex> pin_lock(pin_mutex_);
    return (++pin_count_);
  }

  inline int unPin() override {
    std::lock_guard<std::mutex> pin_lock(pin_mutex_);
    CHECK(pin_count_ > 0);
    return (--pin_count_);
  }

  inline int getPinCount() override {
    std::lock_guard<std::mutex> pin_lock(pin_mutex_);
    return (pin_count_);
  }

  // Added for testing.
  int32_t getSlabNum() const { return seg_it_->slab_num; }

 protected:
  int8_t* mem_;  /// pointer to beginning of buffer's memory

 private:
  Buffer(const Buffer&);             // private copy constructor
  Buffer& operator=(const Buffer&);  // private overloaded assignment operator
  virtual void readData(int8_t* const dst,
                        const size_t num_bytes,
                        const size_t offset = 0,
                        const MemoryLevel dst_buffer_type = CPU_LEVEL,
                        const int dst_device_id = -1) = 0;
  virtual void writeData(int8_t* const src,
                         const size_t num_bytes,
                         const size_t offset = 0,
                         const MemoryLevel src_buffer_type = CPU_LEVEL,
                         const int src_device_id = -1) = 0;

  BufferMgr* bm_;
  BufferList::iterator seg_it_;
  size_t page_size_;  /// the size of each page in the buffer
  size_t num_pages_;
  int epoch_;  /// indicates when the buffer was last flushed
  std::vector<bool> page_dirty_flags_;
  int pin_count_;
  std::mutex pin_mutex_;
};

}  // namespace Buffer_Namespace
