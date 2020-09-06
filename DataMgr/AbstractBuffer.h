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

#include "Shared/Logger.h"
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
      : has_encoder(false)
      , size_(0)
      , is_dirty_(false)
      , is_appended_(false)
      , is_updated_(false)
      , device_id_(device_id) {}
  AbstractBuffer(const int device_id, const SQLTypeInfo sql_type)
      : size_(0)
      , is_dirty_(false)
      , is_appended_(false)
      , is_updated_(false)
      , device_id_(device_id) {
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
  virtual size_t size() const = 0;
  virtual size_t reservedSize() const = 0;
  virtual int getDeviceId() const { return device_id_; }
  virtual MemoryLevel getType() const = 0;

  // Next three methods are dummy methods so FileBuffer does not implement these
  virtual inline int pin() { return 0; }
  virtual inline int unPin() { return 0; }
  virtual inline int getPinCount() { return 0; }

  virtual inline bool isDirty() const { return is_dirty_; }
  virtual inline bool isAppended() const { return is_appended_; }
  virtual inline bool isUpdated() const { return is_updated_; }

  virtual inline void setDirty() { is_dirty_ = true; }

  virtual inline void setUpdated() {
    is_updated_ = true;
    is_dirty_ = true;
  }

  virtual inline void setAppended() {
    is_appended_ = true;
    is_dirty_ = true;
  }

  virtual void setSize(const size_t size) { size_ = size; }
  void clearDirtyBits() {
    is_appended_ = false;
    is_updated_ = false;
    is_dirty_ = false;
  }
  void initEncoder(const SQLTypeInfo tmp_sql_type) {
    has_encoder = true;
    sql_type = tmp_sql_type;
    encoder.reset(Encoder::Create(this, sql_type));
    LOG_IF(FATAL, encoder == nullptr)
        << "Failed to create encoder for SQL Type " << sql_type.get_type_name();
  }

  void syncEncoder(const AbstractBuffer* src_buffer) {
    has_encoder = src_buffer->has_encoder;
    if (has_encoder) {
      if (!encoder) {  // Encoder not initialized
        initEncoder(src_buffer->sql_type);
      }
      encoder->copyMetadata(src_buffer->encoder.get());
    }
  }

  std::unique_ptr<Encoder> encoder;
  bool has_encoder;
  SQLTypeInfo sql_type;

 protected:
  size_t size_;
  bool is_dirty_;
  bool is_appended_;
  bool is_updated_;
  int device_id_;

#ifdef BUFFER_MUTEX
  boost::shared_mutex read_write_mutex_;
  boost::shared_mutex append_mutex_;
#endif
};

}  // namespace Data_Namespace
