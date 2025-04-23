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
 * @file		FixedLengthArrayNoneEncoder.h
 * @brief		unencoded fixed length array encoder
 *
 */

#ifndef FIXED_LENGTH_ARRAY_NONE_ENCODER_H
#define FIXED_LENGTH_ARRAY_NONE_ENCODER_H

#include "Logger/Logger.h"

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

  size_t getNumElemsForBytesEncodedDataAtIndices(const int8_t* index_data,
                                                 const std::vector<size_t>& selected_idx,
                                                 const size_t byte_limit) override {
    size_t data_size = selected_idx.size() * array_size;
    if (data_size > byte_limit) {
      data_size = byte_limit;
    }
    return data_size / array_size;
  }

  size_t getNumElemsForBytesInsertData(const ArrayDatum* srcData,
                                       const int start_idx,
                                       const size_t numAppendElems,
                                       const size_t byteLimit,
                                       const bool replicating = false) {
    size_t dataSize = numAppendElems * array_size;
    if (dataSize > byteLimit) {
      dataSize = byteLimit;
    }
    return dataSize / array_size;
  }

  std::shared_ptr<ChunkMetadata> appendEncodedDataAtIndices(
      const int8_t* index_data,
      int8_t* data,
      const std::vector<size_t>& selected_idx) override {
    std::vector<ArrayDatum> data_subset;
    data_subset.reserve(selected_idx.size());
    for (const auto& index : selected_idx) {
      auto current_data = data + array_size * (index);
      data_subset.emplace_back(ArrayDatum(array_size,
                                          current_data,
                                          is_null_ignore_not_null(current_data),
                                          DoNothingDeleter{}));
    }
    return appendData(data_subset.data(), 0, selected_idx.size(), false);
  }

  std::shared_ptr<ChunkMetadata> appendEncodedData(const int8_t* index_data,
                                                   int8_t* data,
                                                   const size_t start_idx,
                                                   const size_t num_elements) override {
    std::vector<ArrayDatum> data_subset;
    data_subset.reserve(num_elements);
    for (size_t count = 0; count < num_elements; ++count) {
      auto current_data = data + array_size * (start_idx + count);
      data_subset.emplace_back(
          ArrayDatum(array_size, current_data, false, DoNothingDeleter{}));
    }
    return appendData(data_subset.data(), 0, num_elements, false);
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
                                            const bool replicating = false) {
    return appendData(srcData->data(), start_idx, numAppendElems, replicating);
  }

  std::shared_ptr<ChunkMetadata> appendData(const ArrayDatum* srcData,
                                            const int start_idx,
                                            const size_t numAppendElems,
                                            const bool replicating) {
    const size_t existing_data_size = num_elems_ * array_size;
    const size_t append_data_size = array_size * numAppendElems;
    buffer_->reserve(existing_data_size + append_data_size);
    std::vector<int8_t> append_buffer(append_data_size);
    int8_t* append_ptr = append_buffer.data();

    // There was some worry about the change implemented to write the append data to an
    // intermediate buffer, but testing on import and ctas of 20M points, we never append
    // more than 1.6MB and 1MB of data at a time, respectively, so at least for fixed
    // length types this should not be an issue (varlen types, which can be massive even
    // for a single field/row, are a different story however)

    if (replicating) {
      const size_t len = srcData[0].length;
      CHECK_EQ(len, array_size);
      const int8_t* replicated_ptr = srcData[0].pointer;
      for (size_t i = 0; i < numAppendElems; ++i) {
        std::memcpy(append_ptr + i * array_size, replicated_ptr, array_size);
      }
    } else {
      for (size_t i = 0; i < numAppendElems; ++i) {
        // Length of the appended array should be equal to the fixed length,
        // all others should have been discarded, assert if something slips through
        const size_t source_idx = start_idx + i;
        const size_t len = srcData[source_idx].length;
        CHECK_EQ(len, array_size);
        // NULL arrays have been filled with subtype's NULL sentinels,
        // should be appended as regular data, same size
        std::memcpy(append_ptr + i * array_size, srcData[source_idx].pointer, array_size);
      }
    }

    buffer_->append(append_ptr, append_data_size);

    if (replicating) {
      updateStats(srcData, 0, 1);
    } else {
      updateStats(srcData, start_idx, numAppendElems);
    }

    // make sure buffer_ is flushed even if no new data is appended to it
    // (e.g. empty strings) because the metadata needs to be flushed.
    if (!buffer_->isDirty()) {
      buffer_->setDirty();
    }

    num_elems_ += numAppendElems;
    return std::make_shared<ChunkMetadata>(getMetadata());
  }

  ChunkStats getChunkStats() const override {
    return ChunkStats(elem_min, elem_max, has_nulls);
  }

  ChunkStats synthesizeChunkStats(const SQLTypeInfo&) const override {
    return ChunkStats(elem_min, elem_max, has_nulls);
  }

  void updateStats(const int64_t, const bool) override { CHECK(false); }

  void updateStats(const double, const bool) override { CHECK(false); }

  void reduceStats(const Encoder&) override { CHECK(false); }

  void updateStats(const int8_t* const src_data, const size_t num_elements) override {
    UNREACHABLE();
  }

  void updateStats(const std::vector<std::string>* const src_data,
                   const size_t start_idx,
                   const size_t num_elements) override {
    UNREACHABLE();
  }

  void updateStats(const std::string* src_data,
                   const size_t start_idx,
                   const size_t num_elements) override {
    UNREACHABLE();
  }

  void updateStats(const std::vector<ArrayDatum>* const src_data,
                   const size_t start_idx,
                   const size_t num_elements) override {
    updateStats(src_data->data(), start_idx, num_elements);
  }

  void updateStats(const ArrayDatum* src_data,
                   const size_t start_idx,
                   const size_t num_elements) override {
    for (size_t n = start_idx; n < start_idx + num_elements; n++) {
      update_elem_stats(src_data[n]);
    }
  }

  void writeChunkStats(FILE* f) override {
    fwrite((int8_t*)&elem_min, sizeof(Datum), 1, f);
    fwrite((int8_t*)&elem_max, sizeof(Datum), 1, f);
    fwrite((int8_t*)&has_nulls, sizeof(bool), 1, f);
    fwrite((int8_t*)&initialized, sizeof(bool), 1, f);
  }

  void readChunkStats(FILE* f) override {
    fread((int8_t*)&elem_min, sizeof(Datum), 1, f);
    fread((int8_t*)&elem_max, sizeof(Datum), 1, f);
    fread((int8_t*)&has_nulls, sizeof(bool), 1, f);
    fread((int8_t*)&initialized, sizeof(bool), 1, f);
  }

  void copyChunkStats(const Encoder* copyFromEncoder) override {
    auto array_encoder =
        dynamic_cast<const FixedLengthArrayNoneEncoder*>(copyFromEncoder);
    elem_min = array_encoder->elem_min;
    elem_max = array_encoder->elem_max;
    has_nulls = array_encoder->has_nulls;
    initialized = array_encoder->initialized;
  }

  void updateMetadata(int8_t* array) {
    update_elem_stats(ArrayDatum(array_size, array, is_null(array), DoNothingDeleter()));
  }

  static bool is_null_ignore_not_null(const SQLTypeInfo& type, int8_t* array) {
    switch (type.get_subtype()) {
      case kBOOLEAN: {
        return (array[0] == NULL_ARRAY_BOOLEAN);
      }
      case kINT: {
        const int32_t* int_array = (int32_t*)array;
        return (int_array[0] == NULL_ARRAY_INT);
      }
      case kSMALLINT: {
        const int16_t* smallint_array = (int16_t*)array;
        return (smallint_array[0] == NULL_ARRAY_SMALLINT);
      }
      case kTINYINT: {
        const int8_t* tinyint_array = (int8_t*)array;
        return (tinyint_array[0] == NULL_ARRAY_TINYINT);
      }
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL: {
        const int64_t* bigint_array = (int64_t*)array;
        return (bigint_array[0] == NULL_ARRAY_BIGINT);
      }
      case kFLOAT: {
        const float* flt_array = (float*)array;
        return (flt_array[0] == NULL_ARRAY_FLOAT);
      }
      case kDOUBLE: {
        const double* dbl_array = (double*)array;
        return (dbl_array[0] == NULL_ARRAY_DOUBLE);
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        const int64_t* tm_array = reinterpret_cast<int64_t*>(array);
        return (tm_array[0] == NULL_ARRAY_BIGINT);
      }
      case kCHAR:
      case kVARCHAR:
      case kTEXT: {
        CHECK_EQ(type.get_compression(), kENCODING_DICT);
        const int32_t* int_array = (int32_t*)array;
        return (int_array[0] == NULL_ARRAY_INT);
      }
      default:
        UNREACHABLE();
    }
    return false;
  }

  static bool is_null(const SQLTypeInfo& type, int8_t* array) {
    if (type.get_notnull()) {
      return false;
    }
    return is_null_ignore_not_null(type, array);
  }

  bool setChunkStats(const ChunkStats& stats) override {
    auto elem_type = buffer_->getSqlType().get_elem_type();
    if (initialized && DatumEqual(elem_min, stats.min, elem_type) &&
        DatumEqual(elem_max, stats.max, elem_type) && has_nulls == stats.has_nulls) {
      return false;
    }
    elem_min = stats.min;
    elem_max = stats.max;
    has_nulls = stats.has_nulls;
    return true;
  }

  void resetChunkStats() override {
    has_nulls = false;
    initialized = false;
  }

  Datum elem_min;
  Datum elem_max;
  bool has_nulls;
  bool initialized;

 private:
  std::mutex EncoderMutex_;
  std::mutex print_mutex_;
  size_t array_size;

  bool is_null(int8_t* array) { return is_null(buffer_->getSqlType(), array); }

  bool is_null_ignore_not_null(int8_t* array) {
    return is_null_ignore_not_null(buffer_->getSqlType(), array);
  }

  void update_elem_stats(const ArrayDatum& array) {
    if (array.is_null) {
      has_nulls = true;
    }
    switch (buffer_->getSqlType().get_subtype()) {
      case kBOOLEAN: {
        if (!initialized) {
          elem_min.boolval = true;
          elem_max.boolval = false;
        }
        if (array.is_null) {
          break;
        }
        const int8_t* bool_array = array.pointer;
        for (size_t i = 0; i < array.length / sizeof(bool); i++) {
          if (bool_array[i] == NULL_BOOLEAN) {
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
        if (array.is_null) {
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
        if (array.is_null) {
          break;
        }
        const int16_t* smallint_array = (int16_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int16_t); i++) {
          if (smallint_array[i] == NULL_SMALLINT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.smallintval = std::min(elem_min.smallintval, smallint_array[i]);
            elem_max.smallintval = std::max(elem_max.smallintval, smallint_array[i]);
          } else {
            elem_min.smallintval = smallint_array[i];
            elem_max.smallintval = smallint_array[i];
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
        if (array.is_null) {
          break;
        }
        const int8_t* tinyint_array = (int8_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int8_t); i++) {
          if (tinyint_array[i] == NULL_TINYINT) {
            has_nulls = true;
          } else if (initialized) {
            elem_min.tinyintval = std::min(elem_min.tinyintval, tinyint_array[i]);
            elem_max.tinyintval = std::max(elem_max.tinyintval, tinyint_array[i]);
          } else {
            elem_min.tinyintval = tinyint_array[i];
            elem_max.tinyintval = tinyint_array[i];
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
        if (array.is_null) {
          break;
        }
        const int64_t* bigint_array = (int64_t*)array.pointer;
        for (size_t i = 0; i < array.length / sizeof(int64_t); i++) {
          if (bigint_array[i] == NULL_BIGINT) {
            has_nulls = true;
          } else if (initialized) {
            decimal_overflow_validator_.validate(bigint_array[i]);
            elem_min.bigintval = std::min(elem_min.bigintval, bigint_array[i]);
            elem_max.bigintval = std::max(elem_max.bigintval, bigint_array[i]);
          } else {
            decimal_overflow_validator_.validate(bigint_array[i]);
            elem_min.bigintval = bigint_array[i];
            elem_max.bigintval = bigint_array[i];
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
        if (array.is_null) {
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
        if (array.is_null) {
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
        if (array.is_null) {
          break;
        }
        const int64_t* tm_array = reinterpret_cast<int64_t*>(array.pointer);
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
        CHECK_EQ(buffer_->getSqlType().get_compression(), kENCODING_DICT);
        if (!initialized) {
          elem_min.intval = 1;
          elem_max.intval = 0;
        }
        if (array.is_null) {
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

};  // class FixedLengthArrayNoneEncoder

#endif  // FIXED_LENGTH_ARRAY_NONE_ENCODER_H
