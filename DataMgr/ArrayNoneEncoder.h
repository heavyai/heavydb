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
 * @file		ArrayNoneEncoder.h
 * @author	Wei Hong <wei@mapd.com>
 * @brief		unencoded array encoder
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef ARRAY_NONE_ENCODER_H
#define ARRAY_NONE_ENCODER_H

#include <glog/logging.h>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "AbstractBuffer.h"
#include "ChunkMetadata.h"
#include "Encoder.h"

using Data_Namespace::AbstractBuffer;

class ArrayNoneEncoder : public Encoder {
 public:
  ArrayNoneEncoder(AbstractBuffer* buffer)
      : Encoder(buffer)
      , has_nulls(false)
      , initialized(false)
      , index_buf(nullptr)
      , last_offset(-1) {}

  size_t getNumElemsForBytesInsertData(const std::vector<ArrayDatum>* srcData,
                                       const int start_idx,
                                       const size_t numAppendElems,
                                       const size_t byteLimit,
                                       const bool replicating = false) {
    size_t dataSize = 0;

    size_t n = start_idx;
    for (; n < start_idx + numAppendElems; n++) {
      size_t len = (*srcData)[replicating ? 0 : n].length;
      if (dataSize + len > byteLimit)
        break;
      dataSize += len;
    }
    return n - start_idx;
  }

  ChunkMetadata appendData(int8_t*& srcData,
                           const size_t numAppendElems,
                           const SQLTypeInfo&,
                           const bool replicating = false) {
    assert(false);  // should never be called for arrays
    return ChunkMetadata{};
  }

  ChunkMetadata appendData(const std::vector<ArrayDatum>* srcData,
                           const int start_idx,
                           const size_t numAppendElems,
                           const bool replicating) {
    assert(index_buf != nullptr);  // index_buf must be set before this.
    size_t index_size = numAppendElems * sizeof(StringOffsetT);
    if (num_elems_ == 0)
      index_size += sizeof(StringOffsetT);  // plus one for the initial offset of 0.
    index_buf->reserve(index_size);
    StringOffsetT offset = 0;
    if (num_elems_ == 0) {
      index_buf->append((int8_t*)&offset,
                        sizeof(StringOffsetT));  // write the inital 0 offset
      last_offset = 0;
    } else {
      if (last_offset < 0) {
        // need to read the last offset from buffer/disk
        index_buf->read((int8_t*)&last_offset,
                        sizeof(StringOffsetT),
                        index_buf->size() - sizeof(StringOffsetT),
                        Data_Namespace::CPU_LEVEL);
        assert(last_offset >= 0);
      }
    }
    size_t data_size = 0;
    for (size_t n = start_idx; n < start_idx + numAppendElems; n++) {
      size_t len = (*srcData)[replicating ? 0 : n].length;
      data_size += len;
    }
    buffer_->reserve(data_size);

    size_t inbuf_size =
        std::min(std::max(index_size, data_size), (size_t)MAX_INPUT_BUF_SIZE);
    auto inbuf = new int8_t[inbuf_size];
    std::unique_ptr<int8_t[]> gc_inbuf(inbuf);
    for (size_t num_appended = 0; num_appended < numAppendElems;) {
      StringOffsetT* p = (StringOffsetT*)inbuf;
      size_t i;
      for (i = 0; num_appended < numAppendElems && i < inbuf_size / sizeof(StringOffsetT);
           i++, num_appended++) {
        p[i] =
            last_offset + (*srcData)[replicating ? 0 : num_appended + start_idx].length;
        last_offset = p[i];
      }
      index_buf->append(inbuf, i * sizeof(StringOffsetT));
    }

    for (size_t num_appended = 0; num_appended < numAppendElems;) {
      size_t size = 0;
      for (int i = start_idx + num_appended;
           num_appended < numAppendElems && size < inbuf_size;
           i++, num_appended++) {
        size_t len = (*srcData)[replicating ? 0 : i].length;
        if (len > inbuf_size) {
          // for large strings, append on its own
          if (size > 0)
            buffer_->append(inbuf, size);
          size = 0;
          buffer_->append((*srcData)[replicating ? 0 : i].pointer, len);
          num_appended++;
          break;
        } else if (size + len > inbuf_size)
          break;
        char* dest = (char*)inbuf + size;
        if (len > 0)
          std::memcpy((void*)dest, (void*)(*srcData)[replicating ? 0 : i].pointer, len);
        size += len;
      }
      if (size > 0)
        buffer_->append(inbuf, size);
    }
    // make sure buffer_ is flushed even if no new data is appended to it
    // (e.g. empty strings) because the metadata needs to be flushed.
    if (!buffer_->isDirty())
      buffer_->setDirty();

    // keep Chunk statistics with array elements
    for (size_t n = start_idx; n < start_idx + numAppendElems; n++) {
      update_elem_stats((*srcData)[replicating ? 0 : n]);
    }
    num_elems_ += numAppendElems;
    ChunkMetadata chunkMetadata;
    getMetadata(chunkMetadata);
    return chunkMetadata;
  }

  void getMetadata(ChunkMetadata& chunkMetadata) {
    Encoder::getMetadata(chunkMetadata);  // call on parent class
    chunkMetadata.fillChunkStats(elem_min, elem_max, has_nulls);
  }

  // Only called from the executor for synthesized meta-information.
  ChunkMetadata getMetadata(const SQLTypeInfo& ti) {
    ChunkMetadata chunk_metadata{ti, 0, 0, ChunkStats{elem_min, elem_max, has_nulls}};
    return chunk_metadata;
  }

  void updateStats(const int64_t, const bool) { CHECK(false); }

  void updateStats(const double, const bool) { CHECK(false); }

  void reduceStats(const Encoder&) { CHECK(false); }

  void writeMetadata(FILE* f) {
    // assumes pointer is already in right place
    fwrite((int8_t*)&num_elems_, sizeof(size_t), 1, f);
    fwrite((int8_t*)&elem_min, sizeof(Datum), 1, f);
    fwrite((int8_t*)&elem_max, sizeof(Datum), 1, f);
    fwrite((int8_t*)&has_nulls, sizeof(bool), 1, f);
    fwrite((int8_t*)&initialized, sizeof(bool), 1, f);
  }

  void readMetadata(FILE* f) {
    // assumes pointer is already in right place
    fread((int8_t*)&num_elems_, sizeof(size_t), 1, f);
    fread((int8_t*)&elem_min, sizeof(Datum), 1, f);
    fread((int8_t*)&elem_max, sizeof(Datum), 1, f);
    fread((int8_t*)&has_nulls, sizeof(bool), 1, f);
    fread((int8_t*)&initialized, sizeof(bool), 1, f);
  }

  void copyMetadata(const Encoder* copyFromEncoder) {
    num_elems_ = copyFromEncoder->getNumElems();
    auto array_encoder = dynamic_cast<const ArrayNoneEncoder*>(copyFromEncoder);
    elem_min = array_encoder->elem_min;
    elem_max = array_encoder->elem_max;
    has_nulls = array_encoder->has_nulls;
    initialized = array_encoder->initialized;
  }

  AbstractBuffer* get_index_buf() const { return index_buf; }

  Datum elem_min;
  Datum elem_max;
  bool has_nulls;
  bool initialized;
  void set_index_buf(AbstractBuffer* buf) {
    std::unique_lock<std::mutex> lock(EncoderMutex_);
    index_buf = buf;
  }

 private:
  std::mutex EncoderMutex_;
  AbstractBuffer* index_buf;
  StringOffsetT last_offset;

  void update_elem_stats(const ArrayDatum& array) {
    if (array.is_null || array.length == 0) {
      has_nulls = true;
      return;
    }
    switch (buffer_->sqlType.get_subtype()) {
      case kBOOLEAN: {
        const bool* bool_array = (bool*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(bool); i++) {
          if ((int8_t)bool_array[i] == NULL_BOOLEAN)
            has_nulls = true;
          else if (initialized) {
            elem_min.boolval = std::min(elem_min.boolval, bool_array[i]);
            elem_max.boolval = std::max(elem_max.boolval, bool_array[i]);
          } else {
            elem_min.boolval = bool_array[i];
            elem_max.boolval = bool_array[i];
            initialized = true;
          }
        }
      } break;
      case kINT: {
        const int32_t* int_array = (int32_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int32_t); i++) {
          if (int_array[i] == NULL_INT)
            has_nulls = true;
          else if (initialized) {
            elem_min.intval = std::min(elem_min.intval, int_array[i]);
            elem_max.intval = std::max(elem_max.intval, int_array[i]);
          } else {
            elem_min.intval = int_array[i];
            elem_max.intval = int_array[i];
            initialized = true;
          }
        }
      } break;
      case kSMALLINT: {
        const int16_t* int_array = (int16_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int16_t); i++) {
          if (int_array[i] == NULL_SMALLINT)
            has_nulls = true;
          else if (initialized) {
            elem_min.smallintval = std::min(elem_min.smallintval, int_array[i]);
            elem_max.smallintval = std::max(elem_max.smallintval, int_array[i]);
          } else {
            elem_min.smallintval = int_array[i];
            elem_max.smallintval = int_array[i];
            initialized = true;
          }
        }
      } break;
      case kTINYINT: {
        const int8_t* int_array = (int8_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int8_t); i++) {
          if (int_array[i] == NULL_TINYINT)
            has_nulls = true;
          else if (initialized) {
            elem_min.tinyintval = std::min(elem_min.tinyintval, int_array[i]);
            elem_max.tinyintval = std::max(elem_max.tinyintval, int_array[i]);
          } else {
            elem_min.tinyintval = int_array[i];
            elem_max.tinyintval = int_array[i];
            initialized = true;
          }
        }
      } break;
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL: {
        const int64_t* int_array = (int64_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int64_t); i++) {
          if (int_array[i] == NULL_BIGINT)
            has_nulls = true;
          else if (initialized) {
            elem_min.bigintval = std::min(elem_min.bigintval, int_array[i]);
            elem_max.bigintval = std::max(elem_max.bigintval, int_array[i]);
          } else {
            elem_min.bigintval = int_array[i];
            elem_max.bigintval = int_array[i];
            initialized = true;
          }
        }
      } break;
      case kFLOAT: {
        const float* flt_array = (float*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(float); i++) {
          if (flt_array[i] == NULL_FLOAT)
            has_nulls = true;
          else if (initialized) {
            elem_min.floatval = std::min(elem_min.floatval, flt_array[i]);
            elem_max.floatval = std::max(elem_max.floatval, flt_array[i]);
          } else {
            elem_min.floatval = flt_array[i];
            elem_max.floatval = flt_array[i];
            initialized = true;
          }
        }
      } break;
      case kDOUBLE: {
        const double* dbl_array = (double*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(double); i++) {
          if (dbl_array[i] == NULL_DOUBLE)
            has_nulls = true;
          else if (initialized) {
            elem_min.doubleval = std::min(elem_min.doubleval, dbl_array[i]);
            elem_max.doubleval = std::max(elem_max.doubleval, dbl_array[i]);
          } else {
            elem_min.doubleval = dbl_array[i];
            elem_max.doubleval = dbl_array[i];
            initialized = true;
          }
        }
      } break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        const time_t* tm_array = (time_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(time_t); i++) {
          if (tm_array[i] == NULL_BIGINT)
            has_nulls = true;
          else if (initialized) {
            elem_min.timeval = std::min(elem_min.timeval, tm_array[i]);
            elem_max.timeval = std::max(elem_max.timeval, tm_array[i]);
          } else {
            elem_min.timeval = tm_array[i];
            elem_max.timeval = tm_array[i];
            initialized = true;
          }
        }
      } break;
      case kCHAR:
      case kVARCHAR:
      case kTEXT: {
        assert(buffer_->sqlType.get_compression() == kENCODING_DICT);
        const int32_t* int_array = (int32_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int32_t); i++) {
          if (int_array[i] == NULL_INT)
            has_nulls = true;
          else if (initialized) {
            elem_min.intval = std::min(elem_min.intval, int_array[i]);
            elem_max.intval = std::max(elem_max.intval, int_array[i]);
          } else {
            elem_min.intval = int_array[i];
            elem_max.intval = int_array[i];
            initialized = true;
          }
        }
      } break;
      default:
        assert(false);
    }
  };

};  // class ArrayNoneEncoder

#endif  // ARRAY_NONE_ENCODER_H
