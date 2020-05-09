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

#include "Shared/Logger.h"

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
      if (dataSize + len > byteLimit) {
        break;
      }
      dataSize += len;
    }
    return n - start_idx;
  }

  std::shared_ptr<ChunkMetadata> appendData(int8_t*& src_data,
                                            const size_t num_elems_to_append,
                                            const SQLTypeInfo& ti,
                                            const bool replicating = false,
                                            const int64_t offset = -1) override {
    UNREACHABLE();  // should never be called for arrays
    return nullptr;
  }

  std::shared_ptr<ChunkMetadata> appendData(const std::vector<ArrayDatum>* srcData,
                                            const int start_idx,
                                            const size_t numAppendElems,
                                            const bool replicating) {
    CHECK(index_buf != nullptr);  // index_buf must be set before this.
    size_t index_size = numAppendElems * sizeof(ArrayOffsetT);
    if (num_elems_ == 0) {
      index_size += sizeof(ArrayOffsetT);  // plus one for the initial offset
    }
    index_buf->reserve(index_size);

    bool first_elem_is_null = false;
    ArrayOffsetT initial_offset = 0;
    if (num_elems_ == 0) {
      // If the very first ArrayDatum is NULL, initial offset will be set to 8
      // so we could negate it and write it out to index buffer to convey NULLness
      if ((*srcData)[0].is_null) {
        initial_offset = 8;
        first_elem_is_null = true;
      }
      index_buf->append((int8_t*)&initial_offset,
                        sizeof(ArrayOffsetT));  // write the inital offset
      last_offset = initial_offset;
    } else {
      // Valid last_offset is never negative
      // always need to read a valid last offset from buffer/disk
      // b/c now due to vacuum "last offset" may go backward and if
      // index chunk was not reloaded last_offset would go way off!
      index_buf->read((int8_t*)&last_offset,
                      sizeof(ArrayOffsetT),
                      index_buf->size() - sizeof(ArrayOffsetT),
                      Data_Namespace::CPU_LEVEL);
      CHECK(last_offset != -1);
      // If the loaded offset is negative it means the last value was a NULL array,
      // convert to a valid last offset
      if (last_offset < 0) {
        last_offset = -last_offset;
      }
    }
    // Need to start data from 8 byte offset if first array encoded is a NULL array
    size_t data_size = (first_elem_is_null) ? 8 : 0;
    for (size_t n = start_idx; n < start_idx + numAppendElems; n++) {
      // NULL arrays don't take any space so don't add to the data size
      if ((*srcData)[replicating ? 0 : n].is_null) {
        continue;
      }
      data_size += (*srcData)[replicating ? 0 : n].length;
    }
    buffer_->reserve(data_size);

    size_t inbuf_size =
        std::min(std::max(index_size, data_size), (size_t)MAX_INPUT_BUF_SIZE);
    auto gc_inbuf = std::make_unique<int8_t[]>(inbuf_size);
    auto inbuf = gc_inbuf.get();
    for (size_t num_appended = 0; num_appended < numAppendElems;) {
      ArrayOffsetT* p = (ArrayOffsetT*)inbuf;
      size_t i;
      for (i = 0; num_appended < numAppendElems && i < inbuf_size / sizeof(ArrayOffsetT);
           i++, num_appended++) {
        p[i] =
            last_offset + (*srcData)[replicating ? 0 : num_appended + start_idx].length;
        last_offset = p[i];
        if ((*srcData)[replicating ? 0 : num_appended + start_idx].is_null) {
          // Record array NULLness in the index buffer
          p[i] = -p[i];
        }
      }
      index_buf->append(inbuf, i * sizeof(ArrayOffsetT));
    }

    // Pad buffer_ with 8 bytes if first encoded array is a NULL array
    if (first_elem_is_null) {
      buffer_->append(inbuf, 8);
    }
    for (size_t num_appended = 0; num_appended < numAppendElems;) {
      size_t size = 0;
      for (int i = start_idx + num_appended;
           num_appended < numAppendElems && size < inbuf_size;
           i++, num_appended++) {
        if ((*srcData)[replicating ? 0 : i].is_null) {
          continue;  // NULL arrays don't take up any space in the data buffer
        }
        size_t len = (*srcData)[replicating ? 0 : i].length;
        if (len > inbuf_size) {
          // for large strings, append on its own
          if (size > 0) {
            buffer_->append(inbuf, size);
          }
          size = 0;
          buffer_->append((*srcData)[replicating ? 0 : i].pointer, len);
          num_appended++;
          break;
        } else if (size + len > inbuf_size) {
          break;
        }
        char* dest = (char*)inbuf + size;
        if (len > 0) {
          std::memcpy((void*)dest, (void*)(*srcData)[replicating ? 0 : i].pointer, len);
          size += len;
        }
      }
      if (size > 0) {
        buffer_->append(inbuf, size);
      }
    }
    // make sure buffer_ is flushed even if no new data is appended to it
    // (e.g. empty strings) because the metadata needs to be flushed.
    if (!buffer_->isDirty()) {
      buffer_->setDirty();
    }

    // keep Chunk statistics with array elements
    for (size_t n = start_idx; n < start_idx + numAppendElems; n++) {
      update_elem_stats((*srcData)[replicating ? 0 : n]);
    }
    num_elems_ += numAppendElems;
    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    getMetadata(chunk_metadata);
    return chunk_metadata;
  }

  void getMetadata(const std::shared_ptr<ChunkMetadata>& chunkMetadata) override {
    Encoder::getMetadata(chunkMetadata);  // call on parent class
    chunkMetadata->fillChunkStats(elem_min, elem_max, has_nulls);
  }

  // Only called from the executor for synthesized meta-information.
  std::shared_ptr<ChunkMetadata> getMetadata(const SQLTypeInfo& ti) override {
    auto chunk_metadata = std::make_shared<ChunkMetadata>(
        ti, 0, 0, ChunkStats{elem_min, elem_max, has_nulls});
    return chunk_metadata;
  }

  void updateStats(const int64_t, const bool) override { CHECK(false); }

  void updateStats(const double, const bool) override { CHECK(false); }

  void updateStats(const int8_t* const dst, const size_t numBytes) override {
    CHECK(false);
  }

  void reduceStats(const Encoder&) override { CHECK(false); }

  void writeMetadata(FILE* f) override {
    // assumes pointer is already in right place
    fwrite((int8_t*)&num_elems_, sizeof(size_t), 1, f);
    fwrite((int8_t*)&elem_min, sizeof(Datum), 1, f);
    fwrite((int8_t*)&elem_max, sizeof(Datum), 1, f);
    fwrite((int8_t*)&has_nulls, sizeof(bool), 1, f);
    fwrite((int8_t*)&initialized, sizeof(bool), 1, f);
  }

  void readMetadata(FILE* f) override {
    // assumes pointer is already in right place
    fread((int8_t*)&num_elems_, sizeof(size_t), 1, f);
    fread((int8_t*)&elem_min, sizeof(Datum), 1, f);
    fread((int8_t*)&elem_max, sizeof(Datum), 1, f);
    fread((int8_t*)&has_nulls, sizeof(bool), 1, f);
    fread((int8_t*)&initialized, sizeof(bool), 1, f);
  }

  void copyMetadata(const Encoder* copyFromEncoder) override {
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
  ArrayOffsetT last_offset;

  void update_elem_stats(const ArrayDatum& array) {
    if (array.is_null) {
      has_nulls = true;
    }
    switch (buffer_->sql_type.get_subtype()) {
      case kBOOLEAN: {
        if (!initialized) {
          elem_min.boolval = true;
          elem_max.boolval = false;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const bool* bool_array = (bool*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(bool); i++) {
          if ((int8_t)bool_array[i] == NULL_BOOLEAN) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.boolval = std::min(elem_min.boolval, bool_array[i]);
            elem_max.boolval = std::max(elem_max.boolval, bool_array[i]);
          } else {
            elem_min.boolval = bool_array[i];
            elem_max.boolval = bool_array[i];
            initialized = true;
          }
        }
        break;
      }
      case kINT: {
        if (!initialized) {
          elem_min.intval = 1;
          elem_max.intval = 0;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const int32_t* int_array = (int32_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int32_t); i++) {
          if (int_array[i] == NULL_INT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.intval = std::min(elem_min.intval, int_array[i]);
            elem_max.intval = std::max(elem_max.intval, int_array[i]);
          } else {
            elem_min.intval = int_array[i];
            elem_max.intval = int_array[i];
            initialized = true;
          }
        }
        break;
      }
      case kSMALLINT: {
        if (!initialized) {
          elem_min.smallintval = 1;
          elem_max.smallintval = 0;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const int16_t* int_array = (int16_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int16_t); i++) {
          if (int_array[i] == NULL_SMALLINT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.smallintval = std::min(elem_min.smallintval, int_array[i]);
            elem_max.smallintval = std::max(elem_max.smallintval, int_array[i]);
          } else {
            elem_min.smallintval = int_array[i];
            elem_max.smallintval = int_array[i];
            initialized = true;
          }
        }
        break;
      }
      case kTINYINT: {
        if (!initialized) {
          elem_min.tinyintval = 1;
          elem_max.tinyintval = 0;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const int8_t* int_array = (int8_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int8_t); i++) {
          if (int_array[i] == NULL_TINYINT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.tinyintval = std::min(elem_min.tinyintval, int_array[i]);
            elem_max.tinyintval = std::max(elem_max.tinyintval, int_array[i]);
          } else {
            elem_min.tinyintval = int_array[i];
            elem_max.tinyintval = int_array[i];
            initialized = true;
          }
        }
        break;
      }
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL: {
        if (!initialized) {
          elem_min.bigintval = 1;
          elem_max.bigintval = 0;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const int64_t* int_array = (int64_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int64_t); i++) {
          if (int_array[i] == NULL_BIGINT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.bigintval = std::min(elem_min.bigintval, int_array[i]);
            elem_max.bigintval = std::max(elem_max.bigintval, int_array[i]);
          } else {
            elem_min.bigintval = int_array[i];
            elem_max.bigintval = int_array[i];
            initialized = true;
          }
        }
        break;
      }
      case kFLOAT: {
        if (!initialized) {
          elem_min.floatval = 1.0;
          elem_max.floatval = 0.0;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const float* flt_array = (float*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(float); i++) {
          if (flt_array[i] == NULL_FLOAT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.floatval = std::min(elem_min.floatval, flt_array[i]);
            elem_max.floatval = std::max(elem_max.floatval, flt_array[i]);
          } else {
            elem_min.floatval = flt_array[i];
            elem_max.floatval = flt_array[i];
            initialized = true;
          }
        }
        break;
      }
      case kDOUBLE: {
        if (!initialized) {
          elem_min.doubleval = 1.0;
          elem_max.doubleval = 0.0;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const double* dbl_array = (double*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(double); i++) {
          if (dbl_array[i] == NULL_DOUBLE) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.doubleval = std::min(elem_min.doubleval, dbl_array[i]);
            elem_max.doubleval = std::max(elem_max.doubleval, dbl_array[i]);
          } else {
            elem_min.doubleval = dbl_array[i];
            elem_max.doubleval = dbl_array[i];
            initialized = true;
          }
        }
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        if (!initialized) {
          elem_min.bigintval = 1;
          elem_max.bigintval = 0;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const auto tm_array = reinterpret_cast<int64_t*>(array.pointer);
        for (size_t i = 0; i < array.length / sizeof(int64_t); i++) {
          if (tm_array[i] == NULL_BIGINT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.bigintval = std::min(elem_min.bigintval, tm_array[i]);
            elem_max.bigintval = std::max(elem_max.bigintval, tm_array[i]);
          } else {
            elem_min.bigintval = tm_array[i];
            elem_max.bigintval = tm_array[i];
            initialized = true;
          }
        }
        break;
      }
      case kCHAR:
      case kVARCHAR:
      case kTEXT: {
        CHECK_EQ(buffer_->sql_type.get_compression(), kENCODING_DICT);
        if (!initialized) {
          elem_min.intval = 1;
          elem_max.intval = 0;
        }
        if (array.is_null || array.length == 0) {
          break;
        }
        const int32_t* int_array = (int32_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int32_t); i++) {
          if (int_array[i] == NULL_INT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.intval = std::min(elem_min.intval, int_array[i]);
            elem_max.intval = std::max(elem_max.intval, int_array[i]);
          } else {
            elem_min.intval = int_array[i];
            elem_max.intval = int_array[i];
            initialized = true;
          }
        }
        break;
      }
      default:
        UNREACHABLE();
    }
  };

};  // class ArrayNoneEncoder

#endif  // ARRAY_NONE_ENCODER_H
