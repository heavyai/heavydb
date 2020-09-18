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
 * @file    AbstractBuffer.h
 * @author  Steven Stewart <steve@map-d.com>
 * @author  Todd Mostak <todd@map-d.com>
 */
#pragma once

#include <memory>

#ifdef BUFFER_MUTEX
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#endif

#include "Encoder.h"
#include "MemoryLevel.h"

#include "Logger/Logger.h"
#include "Shared/sqltypes.h"
#include "Shared/types.h"

namespace Data_Namespace {

/**
 * @class   AbstractBuffer
 * @brief   An AbstractBuffer is a unit of data management for a data manager.
 */

// enum BufferType {FILE_BUFFER, CPU_BUFFER, GPU_BUFFER};

class AbstractBuffer {
 public:
  AbstractBuffer(const int device_id)
      : encoder_(nullptr)
      , size_(0)
      , device_id_(device_id)
      , is_dirty_(false)
      , is_appended_(false)
      , is_updated_(false) {}

  AbstractBuffer(const int device_id, const SQLTypeInfo sql_type)
      : size_(0)
      , device_id_(device_id)
      , is_dirty_(false)
      , is_appended_(false)
      , is_updated_(false) {
    initEncoder(sql_type);
  }
  virtual ~AbstractBuffer() {}

  virtual void read(int8_t* const dst,
                    const size_t num_bytes,
                    const size_t offset = 0,
                    const MemoryLevel dst_buffer_type = CPU_LEVEL,
                    const int dst_device_id = -1) = 0;
  virtual void write(int8_t* src,
                     const size_t num_bytes,
                     const size_t offset = 0,
                     const MemoryLevel src_buffer_type = CPU_LEVEL,
                     const int src_device_id = -1) = 0;
  virtual void reserve(size_t num_bytes) = 0;
  virtual void append(int8_t* src,
                      const size_t num_bytes,
                      const MemoryLevel src_buffer_type = CPU_LEVEL,
                      const int device_id = -1) = 0;
  virtual int8_t* getMemoryPtr() = 0;
  virtual size_t pageCount() const = 0;
  virtual size_t pageSize() const = 0;
  virtual size_t reservedSize() const = 0;
  virtual MemoryLevel getType() const = 0;

  // Next three methods are dummy methods so FileBuffer does not implement these
  virtual inline int pin() { return 0; }
  virtual inline int unPin() { return 0; }
  virtual inline int getPinCount() { return 0; }

  // These getters should not vary when inherited and therefore don't need to be virtual.
  inline size_t size() const { return size_; }
  inline int getDeviceId() const { return device_id_; }
  inline bool isDirty() const { return is_dirty_; }
  inline bool isAppended() const { return is_appended_; }
  inline bool isUpdated() const { return is_updated_; }
  inline bool hasEncoder() const { return (encoder_ != nullptr); }
  inline SQLTypeInfo getSqlType() const { return sql_type_; }
  inline void setSqlType(const SQLTypeInfo& sql_type) { sql_type_ = sql_type; }
  inline Encoder* getEncoder() const {
    CHECK(hasEncoder());
    return encoder_.get();
  }

  inline void setDirty() { is_dirty_ = true; }

  inline void setUpdated() {
    is_updated_ = true;
    is_dirty_ = true;
  }

  inline void setAppended() {
    is_appended_ = true;
    is_dirty_ = true;
  }

  inline void setSize(const size_t size) { size_ = size; }
  inline void clearDirtyBits() {
    is_appended_ = false;
    is_updated_ = false;
    is_dirty_ = false;
  }

  void initEncoder(const SQLTypeInfo& tmp_sql_type);
  void syncEncoder(const AbstractBuffer* src_buffer);
  void copyTo(AbstractBuffer* destination_buffer, const size_t num_bytes = 0);
  void resetToEmpty();

 protected:
  std::unique_ptr<Encoder> encoder_;
  SQLTypeInfo sql_type_;
  size_t size_;
  int device_id_;

 private:
  bool is_dirty_;
  bool is_appended_;
  bool is_updated_;

#ifdef BUFFER_MUTEX
  boost::shared_mutex read_write_mutex_;
  boost::shared_mutex append_mutex_;
#endif
};

}  // namespace Data_Namespace
