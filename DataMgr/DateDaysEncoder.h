/*
 * Copyright 2019 OmniSci, Inc.
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

#ifndef DATE_DAYS_ENCODER_H
#define DATE_DAYS_ENCODER_H

#include "Shared/Logger.h"

#include <iostream>
#include <memory>
#include "AbstractBuffer.h"
#include "Encoder.h"

#include <Shared/DatumFetchers.h>

template <typename T, typename V>
class DateDaysEncoder : public Encoder {
 public:
  DateDaysEncoder(Data_Namespace::AbstractBuffer* buffer)
      : Encoder(buffer)
      , dataMin(std::numeric_limits<T>::max())
      , dataMax(std::numeric_limits<T>::min())
      , has_nulls(false) {}

  ChunkMetadata appendData(int8_t*& srcData,
                           const size_t numAppendElems,
                           const SQLTypeInfo& ti,
                           const bool replicating = false) override {
    CHECK(ti.is_date_in_days());
    T* unencodedData = reinterpret_cast<T*>(srcData);
    auto encodedData = std::make_unique<V[]>(numAppendElems);
    for (size_t i = 0; i < numAppendElems; ++i) {
      size_t ri = replicating ? 0 : i;
      if (unencodedData[ri] == std::numeric_limits<V>::min()) {
        has_nulls = true;
        encodedData.get()[i] = static_cast<V>(unencodedData[ri]);
      } else {
        date_days_overflow_validator_.validate(unencodedData[ri]);
        encodedData.get()[i] =
            DateConverters::get_epoch_days_from_seconds(unencodedData[ri]);
        const T data = DateConverters::get_epoch_seconds_from_days(encodedData.get()[i]);
        dataMax = std::max(dataMax, data);
        dataMin = std::min(dataMin, data);
      }
    }
    num_elems_ += numAppendElems;

    buffer_->append((int8_t*)(encodedData.get()), numAppendElems * sizeof(V));
    ChunkMetadata chunkMetadata;
    getMetadata(chunkMetadata);
    if (!replicating) {
      srcData += numAppendElems * sizeof(T);
    }
    return chunkMetadata;
  }

  void getMetadata(ChunkMetadata& chunkMetadata) override {
    Encoder::getMetadata(chunkMetadata);
    chunkMetadata.fillChunkStats(dataMin, dataMax, has_nulls);
  }

  // Only called from the executor for synthesized meta-information.
  ChunkMetadata getMetadata(const SQLTypeInfo& ti) override {
    ChunkMetadata chunk_metadata{ti, 0, 0, ChunkStats{}};
    chunk_metadata.fillChunkStats(dataMin, dataMax, has_nulls);
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

  void updateStats(const int8_t* const dst, const size_t numBytes) override {
    CHECK(false);
  }

  // Only called from the executor for synthesized meta-information.
  void reduceStats(const Encoder& that) override {
    const auto that_typed = static_cast<const DateDaysEncoder<T, V>&>(that);
    if (that_typed.has_nulls) {
      has_nulls = true;
    }
    dataMin = std::min(dataMin, that_typed.dataMin);
    dataMax = std::max(dataMax, that_typed.dataMax);
  }

  void copyMetadata(const Encoder* copyFromEncoder) override {
    num_elems_ = copyFromEncoder->getNumElems();
    auto castedEncoder = reinterpret_cast<const DateDaysEncoder<T, V>*>(copyFromEncoder);
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

  T dataMin;
  T dataMax;
  bool has_nulls;

};  // DateDaysEncoder

#endif  // DATE_DAYS_ENCODER_H
