/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @author		Steven Stewart <steve@map-d.com>
 * @author		Todd Mostak <todd@map-d.com>
 */
#ifndef DATAMGR_MEMORY_BUFFER_BUFFER_H
#define DATAMGR_MEMORY_BUFFER_BUFFER_H

#include "../AbstractBuffer.h"
#include "BufferSeg.h"

#include <iostream>
#include <mutex>
//#include <boost/thread/locks.hpp>
//#include <boost/thread/mutex.hpp>

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
   * The constructor requires a memory address (provided by BufferMgr), number of pages, and
   * the size in bytes of each page. Additionally, the Buffer can be initialized with an epoch.
   *
   * @param mem       The beginning memory address of the buffer.
   * @param numPages  The number of pages into which the buffer's memory space is divided.
   * @param pageSize  The size in bytes of each page that composes the buffer.
   * @param epoch     A temporal reference implying the buffer is up-to-date up to the epoch.
   */

  /*
  Buffer(const int8_t * mem, const size_t numPages, const size_t pageSize, const int epoch);
  */

  Buffer(BufferMgr* bm,
         BufferList::iterator segIt,
         const int deviceId,
         const size_t pageSize = 512,
         const size_t numBytes = 0);

  /// Destructor
  virtual ~Buffer();

  /**
   * @brief Reads (copies) data from the buffer to the destination (dst) memory location.
   * Reads (copies) nbytes of data from the buffer, beginning at the specified byte offset,
   * into the destination (dst) memory location.
   *
   * @param dst       The destination address to where the buffer's data is being copied.
   * @param offset    The byte offset into the buffer from where reading (copying) begins.
   * @param nbytes    The number of bytes being read (copied) into the destination (dst).
   */
  virtual void read(int8_t* const dst,
                    const size_t numBytes,
                    const size_t offset = 0,
                    const MemoryLevel dstBufferType = CPU_LEVEL,
                    const int deviceId = -1);

  virtual void reserve(const size_t numBytes);
  /**
   * @brief Writes (copies) data from src into the buffer.
   * Writes (copies) nbytes of data into the buffer at the specified byte offset, from
   * the source (src) memory location.
   *
   * @param src       The source address from where data is being copied to the buffer.
   * @param offset    The byte offset into the buffer to where writing begins.
   * @param nbytes    The number of bytes being written (copied) into the buffer.
   */
  virtual void write(int8_t* src,
                     const size_t numBytes,
                     const size_t offset = 0,
                     const MemoryLevel srcBufferType = CPU_LEVEL,
                     const int deviceId = -1);

  virtual void append(int8_t* src,
                      const size_t numBytes,
                      const MemoryLevel srcBufferType = CPU_LEVEL,
                      const int deviceId = -1);

  /**
   * @brief Returns a raw, constant (read-only) pointer to the underlying buffer.
   * @return A constant memory pointer for read-only access.
   */
  virtual int8_t* getMemoryPtr();

  inline virtual size_t size() const { return size_; }

  /// Returns the total number of bytes allocated for the buffer.
  inline virtual size_t reservedSize() const { return pageSize_ * numPages_; }
  /// Returns the number of pages in the buffer.

  inline size_t pageCount() const { return numPages_; }

  /// Returns the size in bytes of each page in the buffer.

  inline size_t pageSize() const { return pageSize_; }

  /// Returns whether or not the buffer has been modified since the last flush/checkpoint.
  inline bool isDirty() const { return isDirty_; }

  inline int pin() {
    std::lock_guard<std::mutex> pinLock(pinMutex_);
    return (++pinCount_);
  }

  inline int unPin() {
    std::lock_guard<std::mutex> pinLock(pinMutex_);
    return (--pinCount_);
  }
  inline int getPinCount() {
    std::lock_guard<std::mutex> pinLock(pinMutex_);
    return (pinCount_);
  }

 protected:
  int8_t* mem_;  /// pointer to beginning of buffer's memory

 private:
  Buffer(const Buffer&);             // private copy constructor
  Buffer& operator=(const Buffer&);  // private overloaded assignment operator
  virtual void readData(int8_t* const dst,
                        const size_t numBytes,
                        const size_t offset = 0,
                        const MemoryLevel dstBufferType = CPU_LEVEL,
                        const int dstDeviceId = -1) = 0;
  virtual void writeData(int8_t* const src,
                         const size_t numBytes,
                         const size_t offset = 0,
                         const MemoryLevel srcBufferType = CPU_LEVEL,
                         const int srcDeviceId = -1) = 0;

  BufferMgr* bm_;
  BufferList::iterator segIt_;
  // size_t numBytes_;
  size_t pageSize_;  /// the size of each page in the buffer
  size_t numPages_;
  int epoch_;  /// indicates when the buffer was last flushed
  // std::vector<Page> pages_;   /// a vector of pages (page metadata) that compose the buffer
  std::vector<bool> pageDirtyFlags_;
  int pinCount_;
  std::mutex pinMutex_;
};

}  // Buffer_Namespace

#endif  // DATAMGR_MEMORY_BUFFER_BUFFER_H
