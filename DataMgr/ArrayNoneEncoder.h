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
 * @file		ArrayNoneEncoder.h
 * @brief		unencoded array encoder
 *
 */

#ifndef ARRAY_NONE_ENCODER_H
#define ARRAY_NONE_ENCODER_H

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

// TODO(Misiu): All of these functions should be moved to a .cpp file.
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

  size_t getNumElemsForBytesEncodedDataAtIndices(const int8_t* index_data,
                                                 const std::vector<size_t>& selected_idx,
                                                 const size_t byte_limit) override {
    size_t num_elements = 0;
    size_t data_size = 0;
    for (const auto& offset_index : selected_idx) {
      auto element_size = getArrayDatumSizeAtIndex(index_data, offset_index);
      if (data_size + element_size > byte_limit) {
        break;
      }
      data_size += element_size;
      num_elements++;
    }
    return num_elements;
  }

  std::shared_ptr<ChunkMetadata> appendData(int8_t*& src_data,
                                            const size_t num_elems_to_append,
                                            const SQLTypeInfo& ti,
                                            const bool replicating = false,
                                            const int64_t offset = -1) override {
    UNREACHABLE();  // should never be called for arrays
    return nullptr;
  }

  std::shared_ptr<ChunkMetadata> appendEncodedDataAtIndices(
      const int8_t* index_data,
      int8_t* data,
      const std::vector<size_t>& selected_idx) override {
    std::vector<ArrayDatum> data_subset;
    data_subset.reserve(selected_idx.size());
    for (const auto& offset_index : selected_idx) {
      data_subset.emplace_back(getArrayDatumAtIndex(index_data, data, offset_index));
    }
    return appendData(&data_subset, 0, selected_idx.size(), false);
  }

  std::shared_ptr<ChunkMetadata> appendEncodedData(const int8_t* index_data,
                                                   int8_t* data,
                                                   const size_t start_idx,
                                                   const size_t num_elements) override {
    std::vector<ArrayDatum> data_subset;
    data_subset.reserve(num_elements);
    for (size_t count = 0; count < num_elements; ++count) {
      auto current_index = start_idx + count;
      data_subset.emplace_back(getArrayDatumAtIndex(index_data, data, current_index));
    }
    return appendData(&data_subset, 0, num_elements, false);
  }

  std::shared_ptr<ChunkMetadata> appendData(const std::vector<ArrayDatum>* srcData,
                                            const int start_idx,
                                            const size_t numAppendElems,
                                            const bool replicating) {
    CHECK(index_buf != nullptr);  // index_buf must be set before this.
    size_t append_index_size = numAppendElems * sizeof(ArrayOffsetT);
    if (num_elems_ == 0) {
      append_index_size += sizeof(ArrayOffsetT);  // plus one for the initial offset
    }
    index_buf->reserve(index_buf->size() + append_index_size);

    bool first_elem_padded = false;
    ArrayOffsetT initial_offset = 0;
    if (num_elems_ == 0) {
      if ((*srcData)[0].is_null || (*srcData)[0].length <= 1) {
        // Covers following potentially problematic first arrays:
        // (1) NULL array, issue - can't encode a NULL with 0 initial offset
        // otherwise, if first array is not NULL:
        // (2) length=1 array - could be followed by a {}*/NULL, covers tinyint,bool
        // (3) empty array - could be followed by {}*/NULL, or {}*|{x}|{}*|NULL, etc.
        initial_offset = DEFAULT_NULL_PADDING_SIZE;
        first_elem_padded = true;
      }
      index_buf->append((int8_t*)&initial_offset,
                        sizeof(ArrayOffsetT));  // write the initial offset
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
    size_t append_data_size = (first_elem_padded) ? DEFAULT_NULL_PADDING_SIZE : 0;
    for (size_t n = start_idx; n < start_idx + numAppendElems; n++) {
      // NULL arrays don't take any space so don't add to the data size
      if ((*srcData)[replicating ? 0 : n].is_null) {
        continue;
      }
      append_data_size += (*srcData)[replicating ? 0 : n].length;
    }
    buffer_->reserve(buffer_->size() + append_data_size);

    size_t inbuf_size = std::min(std::max(append_index_size, append_data_size),
                                 (size_t)MAX_INPUT_BUF_SIZE);
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
    if (first_elem_padded) {
      auto padding_size = DEFAULT_NULL_PADDING_SIZE;
      buffer_->append(inbuf, padding_size);
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

  void updateStats(const int8_t* const src_data, const size_t num_elements) override {
    CHECK(false);
  }

  void updateStats(const std::vector<std::string>* const src_data,
                   const size_t start_idx,
                   const size_t num_elements) override {
    UNREACHABLE();
  }

  void updateStats(const std::vector<ArrayDatum>* const src_data,
                   const size_t start_idx,
                   const size_t num_elements) override {
    for (size_t n = start_idx; n < start_idx + num_elements; n++) {
      update_elem_stats((*src_data)[n]);
    }
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

  AbstractBuffer* getIndexBuf() const { return index_buf; }

  bool resetChunkStats(const ChunkStats& stats) override {
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
  void setIndexBuffer(AbstractBuffer* buf) {
    std::unique_lock<std::mutex> lock(EncoderMutex_);
    index_buf = buf;
  }

  static constexpr size_t DEFAULT_NULL_PADDING_SIZE{8};

 private:
  std::mutex EncoderMutex_;
  AbstractBuffer* index_buf;
  ArrayOffsetT last_offset;

  void update_elem_stats(const ArrayDatum& array) {
    if (array.is_null) {
      has_nulls = true;
    }
    switch (buffer_->getSqlType().get_subtype()) {
      case kBOOLEAN: {
        if (!initialized) {
          elem_min.boolval = 1;
          elem_max.boolval = 0;
        }
        if (array.is_null || array.length == 0) {
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
        CHECK_EQ(buffer_->getSqlType().get_compression(), kENCODING_DICT);
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

 private:
  std::pair<ArrayOffsetT, ArrayOffsetT> getArrayOffsetsAtIndex(const int8_t* index_data,
                                                               size_t index) {
    auto array_offsets = reinterpret_cast<const ArrayOffsetT*>(index_data);
    auto current_index = index + 1;
    auto offset = array_offsets[current_index];
    int64_t last_offset = array_offsets[current_index - 1];
    return {offset, last_offset};
  }

  size_t getArrayDatumSizeAtIndex(const int8_t* index_data, size_t index) {
    auto [offset, last_offset] = getArrayOffsetsAtIndex(index_data, index);
    size_t array_byte_size = std::abs(offset) - std::abs(last_offset);
    return array_byte_size;
  }

  ArrayDatum getArrayDatumAtIndex(const int8_t* index_data, int8_t* data, size_t index) {
    auto [offset, last_offset] = getArrayOffsetsAtIndex(index_data, index);
    size_t array_byte_size = std::abs(offset) - std::abs(last_offset);
    bool is_null = offset < 0;
    auto current_data = data + std::abs(last_offset);
    return is_null ? ArrayDatum(0, nullptr, true, DoNothingDeleter{})
                   : ArrayDatum(array_byte_size, current_data, false, DoNothingDeleter{});
  }

};  // class ArrayNoneEncoder

#endif  // ARRAY_NONE_ENCODER_H
