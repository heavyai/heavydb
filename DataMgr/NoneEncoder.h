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

#ifndef NONE_ENCODER_H
#define NONE_ENCODER_H

#include "AbstractBuffer.h"
#include "Encoder.h"

#include <Shared/DatumFetchers.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tuple>

template <typename T>
T none_encoded_null_value() {
  return std::is_integral<T>::value ? inline_int_null_value<T>()
                                    : inline_fp_null_value<T>();
}

template <typename T>
class NoneEncoder : public Encoder {
 public:
  NoneEncoder(Data_Namespace::AbstractBuffer* buffer)
      : Encoder(buffer)
      , dataMin(std::numeric_limits<T>::max())
      , dataMax(std::numeric_limits<T>::lowest())
      , has_nulls(false) {}

  std::shared_ptr<ChunkMetadata> appendData(int8_t*& src_data,
                                            const size_t num_elems_to_append,
                                            const SQLTypeInfo&,
                                            const bool replicating = false,
                                            const int64_t offset = -1) override {
    T* unencodedData = reinterpret_cast<T*>(src_data);
    std::vector<T> encoded_data;
    if (replicating) {
      encoded_data.resize(num_elems_to_append);
    }
    for (size_t i = 0; i < num_elems_to_append; ++i) {
      size_t ri = replicating ? 0 : i;
      T data = validateDataAndUpdateStats(unencodedData[ri]);
      if (replicating) {
        encoded_data[i] = data;
      }
    }
    if (offset == -1) {
      num_elems_ += num_elems_to_append;
      buffer_->append(
          replicating ? reinterpret_cast<int8_t*>(encoded_data.data()) : src_data,
          num_elems_to_append * sizeof(T));
      if (!replicating) {
        src_data += num_elems_to_append * sizeof(T);
      }
    } else {
      num_elems_ = offset + num_elems_to_append;
      CHECK(!replicating);
      CHECK_GE(offset, 0);
      buffer_->write(
          src_data, num_elems_to_append * sizeof(T), static_cast<size_t>(offset));
    }
    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    getMetadata(chunk_metadata);
    return chunk_metadata;
  }

  void getMetadata(const std::shared_ptr<ChunkMetadata>& chunkMetadata) override {
    Encoder::getMetadata(chunkMetadata);  // call on parent class
    chunkMetadata->fillChunkStats(dataMin, dataMax, has_nulls);
  }

  // Only called from the executor for synthesized meta-information.
  std::shared_ptr<ChunkMetadata> getMetadata(const SQLTypeInfo& ti) override {
    auto chunk_metadata = std::make_shared<ChunkMetadata>(ti, 0, 0, ChunkStats{});
    chunk_metadata->fillChunkStats(dataMin, dataMax, has_nulls);
    return chunk_metadata;
  }

  // Only called from the executor for synthesized meta-information.
  void updateStats(const int64_t val, const bool is_null) override {
    if (is_null) {
      has_nulls = true;
    } else {
      const auto data = static_cast<T>(val);
      dataMin = std::min(dataMin, data);
      dataMax = std::max(dataMax, data);
    }
  }

  // Only called from the executor for synthesized meta-information.
  void updateStats(const double val, const bool is_null) override {
    if (is_null) {
      has_nulls = true;
    } else {
      const auto data = static_cast<T>(val);
      dataMin = std::min(dataMin, data);
      dataMax = std::max(dataMax, data);
    }
  }

  void updateStats(const int8_t* const src_data, const size_t num_elements) override {
    const T* unencoded_data = reinterpret_cast<const T*>(src_data);
    for (size_t i = 0; i < num_elements; ++i) {
      validateDataAndUpdateStats(unencoded_data[i]);
    }
  }

  void updateStats(const std::vector<std::string>* const src_data,
                   const size_t start_idx,
                   const size_t num_elements) override {
    UNREACHABLE();
  }

  void updateStats(const std::vector<ArrayDatum>* const src_data,
                   const size_t start_idx,
                   const size_t num_elements) override {
    UNREACHABLE();
  }

  // Only called from the executor for synthesized meta-information.
  void reduceStats(const Encoder& that) override {
    const auto that_typed = static_cast<const NoneEncoder&>(that);
    if (that_typed.has_nulls) {
      has_nulls = true;
    }
    dataMin = std::min(dataMin, that_typed.dataMin);
    dataMax = std::max(dataMax, that_typed.dataMax);
  }

  void writeMetadata(FILE* f) override {
    // assumes pointer is already in right place
    fwrite((int8_t*)&num_elems_, sizeof(size_t), 1, f);
    fwrite((int8_t*)&dataMin, sizeof(T), 1, f);
    fwrite((int8_t*)&dataMax, sizeof(T), 1, f);
    fwrite((int8_t*)&has_nulls, sizeof(bool), 1, f);
  }

  void readMetadata(FILE* f) override {
    // assumes pointer is already in right place
    fread((int8_t*)&num_elems_, sizeof(size_t), 1, f);
    fread((int8_t*)&dataMin, sizeof(T), 1, f);
    fread((int8_t*)&dataMax, sizeof(T), 1, f);
    fread((int8_t*)&has_nulls, sizeof(bool), 1, f);
  }

  bool resetChunkStats(const ChunkStats& stats) override {
    const auto new_min = DatumFetcher::getDatumVal<T>(stats.min);
    const auto new_max = DatumFetcher::getDatumVal<T>(stats.max);

    if (dataMin == new_min && dataMax == new_max && has_nulls == stats.has_nulls) {
      return false;
    }

    dataMin = new_min;
    dataMax = new_max;
    has_nulls = stats.has_nulls;
    return true;
  }

  void copyMetadata(const Encoder* copyFromEncoder) override {
    num_elems_ = copyFromEncoder->getNumElems();
    auto castedEncoder = reinterpret_cast<const NoneEncoder<T>*>(copyFromEncoder);
    dataMin = castedEncoder->dataMin;
    dataMax = castedEncoder->dataMax;
    has_nulls = castedEncoder->has_nulls;
  }

  T dataMin;
  T dataMax;
  bool has_nulls;

 private:
  T validateDataAndUpdateStats(const T& unencoded_data) {
    if (unencoded_data == none_encoded_null_value<T>()) {
      has_nulls = true;
    } else {
      decimal_overflow_validator_.validate(unencoded_data);
      dataMin = std::min(dataMin, unencoded_data);
      dataMax = std::max(dataMax, unencoded_data);
    }
    return unencoded_data;
  }
};  // class NoneEncoder

#endif  // NONE_ENCODER_H
