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

#ifndef FIXED_LENGTH_ENCODER_H
#define FIXED_LENGTH_ENCODER_H
#include "Logger/Logger.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include "AbstractBuffer.h"
#include "Encoder.h"

#include <Shared/DatumFetchers.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tuple>
#include "Shared/Iteration.h"

template <typename T, typename V>
class FixedLengthEncoder : public Encoder {
 public:
  FixedLengthEncoder(Data_Namespace::AbstractBuffer* buffer) : Encoder(buffer) {
    resetChunkStats();
  }

  size_t getNumElemsForBytesEncodedDataAtIndices(const int8_t* index_data,
                                                 const std::vector<size_t>& selected_idx,
                                                 const size_t byte_limit) override {
    UNREACHABLE()
        << "getNumElemsForBytesEncodedDataAtIndices unexpectedly called for non varlen"
           " encoder";
    return {};
  }

  std::shared_ptr<ChunkMetadata> appendEncodedDataAtIndices(
      const int8_t*,
      int8_t* data,
      const std::vector<size_t>& selected_idx) override {
    std::shared_ptr<ChunkMetadata> chunk_metadata;
    // NOTE: the use of `execute_over_contiguous_indices` is an optimization;
    // it prevents having to copy or move the indexed data and instead performs
    // an append over contiguous sections of indices.
    shared::execute_over_contiguous_indices(
        selected_idx, [&](const size_t start_pos, const size_t end_pos) {
          size_t elem_count = end_pos - start_pos;
          chunk_metadata =
              appendEncodedData(nullptr, data, selected_idx[start_pos], elem_count);
        });
    return chunk_metadata;
  }

  std::shared_ptr<ChunkMetadata> appendEncodedData(const int8_t*,
                                                   int8_t* data,
                                                   const size_t start_idx,
                                                   const size_t num_elements) override {
    auto current_data = data + start_idx * sizeof(V);
    return appendEncodedOrUnencodedData(
        current_data, num_elements, SQLTypeInfo{}, false, -1, true);
  }

  std::shared_ptr<ChunkMetadata> appendData(int8_t*& src_data,
                                            const size_t num_elems_to_append,
                                            const SQLTypeInfo& ti,
                                            const bool replicating = false,
                                            const int64_t offset = -1) override {
    return appendEncodedOrUnencodedData(
        src_data, num_elems_to_append, ti, replicating, offset, false);
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
      encodeDataAndUpdateStats(unencoded_data[i]);
    }
  }

  void updateStatsEncoded(const int8_t* const dst_data,
                          const size_t num_elements) override {
    const V* data = reinterpret_cast<const V*>(dst_data);

    std::tie(dataMin, dataMax, has_nulls) = tbb::parallel_reduce(
        tbb::blocked_range(size_t(0), num_elements),
        std::tuple(static_cast<V>(dataMin), static_cast<V>(dataMax), has_nulls),
        [&](const auto& range, auto init) {
          auto [min, max, nulls] = init;
          for (size_t i = range.begin(); i < range.end(); i++) {
            if (data[i] != std::numeric_limits<V>::min()) {
              decimal_overflow_validator_.validate(data[i]);
              min = std::min(min, data[i]);
              max = std::max(max, data[i]);
            } else {
              nulls = true;
            }
          }
          return std::tuple(min, max, nulls);
        },
        [&](auto lhs, auto rhs) {
          const auto [lhs_min, lhs_max, lhs_nulls] = lhs;
          const auto [rhs_min, rhs_max, rhs_nulls] = rhs;
          return std::tuple(std::min(lhs_min, rhs_min),
                            std::max(lhs_max, rhs_max),
                            lhs_nulls || rhs_nulls);
        });
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
    const auto that_typed = static_cast<const FixedLengthEncoder<T, V>&>(that);
    if (that_typed.has_nulls) {
      has_nulls = true;
    }
    dataMin = std::min(dataMin, that_typed.dataMin);
    dataMax = std::max(dataMax, that_typed.dataMax);
  }

  void copyMetadata(const Encoder* copyFromEncoder) override {
    num_elems_ = copyFromEncoder->getNumElems();
    auto castedEncoder =
        reinterpret_cast<const FixedLengthEncoder<T, V>*>(copyFromEncoder);
    dataMin = castedEncoder->dataMin;
    dataMax = castedEncoder->dataMax;
    has_nulls = castedEncoder->has_nulls;
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
    fread((int8_t*)&dataMin, 1, sizeof(T), f);
    fread((int8_t*)&dataMax, 1, sizeof(T), f);
    fread((int8_t*)&has_nulls, 1, sizeof(bool), f);
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

  void resetChunkStats() override {
    dataMin = std::numeric_limits<T>::max();
    dataMax = std::numeric_limits<T>::lowest();
    has_nulls = false;
  }

  T dataMin;
  T dataMax;
  bool has_nulls;

 private:
  std::shared_ptr<ChunkMetadata> appendEncodedOrUnencodedData(
      int8_t*& src_data,
      const size_t num_elems_to_append,
      const SQLTypeInfo& ti,
      const bool replicating,
      const int64_t offset,
      const bool is_encoded) {
    if (offset == 0 &&
        num_elems_to_append >=
            num_elems_) {  // we're rewriting entire buffer so fully recompute metadata
      resetChunkStats();
    }

    CHECK(!is_encoded || !replicating);  // do not support replicating of encoded data

    T* unencoded_data = reinterpret_cast<T*>(src_data);
    std::vector<V> encoded_data;
    V* data_to_write = nullptr;
    if (!is_encoded) {
      encoded_data.resize(num_elems_to_append);
      data_to_write = encoded_data.data();
      for (size_t i = 0; i < num_elems_to_append; ++i) {
        size_t ri = replicating ? 0 : i;
        encoded_data[i] = encodeDataAndUpdateStats(unencoded_data[ri]);
      }
    } else {
      data_to_write = reinterpret_cast<V*>(src_data);
      for (size_t i = 0; i < num_elems_to_append; ++i) {
        updateStatsWithAlreadyEncoded(data_to_write[i]);
      }
    }

    // assume always CPU_BUFFER?
    if (offset == -1) {
      auto append_data_size = num_elems_to_append * sizeof(V);
      buffer_->reserve(buffer_->size() + append_data_size);
      num_elems_ += num_elems_to_append;
      buffer_->append(reinterpret_cast<int8_t*>(data_to_write), append_data_size);
      if (!replicating) {
        src_data += num_elems_to_append * sizeof(T);
      }
    } else {
      num_elems_ = offset + num_elems_to_append;
      CHECK(!replicating);
      CHECK_GE(offset, 0);
      buffer_->write(reinterpret_cast<int8_t*>(data_to_write),
                     num_elems_to_append * sizeof(V),
                     static_cast<size_t>(offset));
    }
    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    getMetadata(chunk_metadata);
    return chunk_metadata;
  }

  void updateStatsWithAlreadyEncoded(const V& encoded_data) {
    if (encoded_data == std::numeric_limits<V>::min()) {
      has_nulls = true;
    } else {
      dataMin = std::min<T>(dataMin, encoded_data);
      dataMax = std::max<T>(dataMax, encoded_data);
    }
  }

  V encodeDataAndUpdateStats(const T& unencoded_data) {
    V encoded_data = static_cast<V>(unencoded_data);
    if (unencoded_data != encoded_data) {
      decimal_overflow_validator_.validate(unencoded_data);
      LOG(ERROR) << "Fixed encoding failed, Unencoded: " +
                        std::to_string(unencoded_data) +
                        " encoded: " + std::to_string(encoded_data);
    } else {
      T data = unencoded_data;
      if (data == std::numeric_limits<V>::min()) {
        has_nulls = true;
      } else {
        decimal_overflow_validator_.validate(data);
        dataMin = std::min(dataMin, data);
        dataMax = std::max(dataMax, data);
      }
    }
    return encoded_data;
  }
};  // FixedLengthEncoder

#endif  // FIXED_LENGTH_ENCODER_H
