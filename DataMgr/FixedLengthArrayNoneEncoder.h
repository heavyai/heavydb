/*
 * Copyright 2018 MapD Technologies, Inc.
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
 * @file		FixedLengthArrayNoneEncoder.h
 * @author		Dmitri Shtilman <d@mapd.com>
 * @brief		unencoded fixed length array encoder
 *
 * Copyright (c) 2018 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef FIXED_LENGTH_ARRAY_NONE_ENCODER_H
#define FIXED_LENGTH_ARRAY_NONE_ENCODER_H

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

class FixedLengthArrayNoneEncoder : public Encoder {
 public:
  FixedLengthArrayNoneEncoder(AbstractBuffer* buffer, size_t as)
      : Encoder(buffer), has_nulls(false), initialized(false), array_size(as) {}

  size_t getNumElemsForBytesInsertData(const std::vector<ArrayDatum>* srcData,
                                       const int start_idx,
                                       const size_t numAppendElems,
                                       const size_t byteLimit,
                                       const bool replicating = false) {
    size_t dataSize = numAppendElems * array_size;
    if (dataSize > byteLimit)
      dataSize = byteLimit;
    return dataSize / array_size;
  }

  ChunkMetadata appendData(int8_t*& srcData,
                           const size_t numAppendElems,
                           const SQLTypeInfo&,
                           const bool replicating = false) {
    CHECK(false);  // should never be called for arrays
    return ChunkMetadata{};
  }

  ChunkMetadata appendData(const std::vector<ArrayDatum>* srcData,
                           const int start_idx,
                           const size_t numAppendElems,
                           const bool replicating = false) {
    size_t data_size = array_size * numAppendElems;
    buffer_->reserve(data_size);

    for (size_t i = start_idx; i < start_idx + numAppendElems; i++) {
      size_t len = (*srcData)[replicating ? 0 : i].length;
      if (len != array_size) {
        throw std::runtime_error("Input length doesn't match fixed-length array length");
      }
      buffer_->append((*srcData)[replicating ? 0 : i].pointer, len);

      // keep Chunk statistics with array elements
      update_elem_stats((*srcData)[replicating ? 0 : i]);
    }
    // make sure buffer_ is flushed even if no new data is appended to it
    // (e.g. empty strings) because the metadata needs to be flushed.
    if (!buffer_->isDirty())
      buffer_->setDirty();

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
    auto array_encoder =
        dynamic_cast<const FixedLengthArrayNoneEncoder*>(copyFromEncoder);
    elem_min = array_encoder->elem_min;
    elem_max = array_encoder->elem_max;
    has_nulls = array_encoder->has_nulls;
    initialized = array_encoder->initialized;
  }

  void updateMetadata(int8_t* array) {
    update_elem_stats(ArrayDatum(array_size, array, DoNothingDeleter()));
  }

  Datum elem_min;
  Datum elem_max;
  bool has_nulls;
  bool initialized;

 private:
  std::mutex EncoderMutex_;
  size_t array_size;

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
          if (int_array[i] == NULL_BIGINT) {
            has_nulls = true;
          } else if (initialized) {
            decimal_overflow_validator_.validate(int_array[i]);
            elem_min.bigintval = std::min(elem_min.bigintval, int_array[i]);
            elem_max.bigintval = std::max(elem_max.bigintval, int_array[i]);
          } else {
            decimal_overflow_validator_.validate(int_array[i]);
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

};  // class FixedLengthArrayNoneEncoder

#endif  // FIXED_LENGTH_ARRAY_NONE_ENCODER_H
