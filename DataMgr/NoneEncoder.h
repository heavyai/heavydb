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

template <typename T>
T none_encoded_null_value() {
  return std::is_integral<T>::value ? inline_int_null_value<T>() : inline_fp_null_value<T>();
}

template <typename T>
class NoneEncoder : public Encoder {
 public:
  NoneEncoder(Data_Namespace::AbstractBuffer* buffer)
      : Encoder(buffer),
        dataMin(std::numeric_limits<T>::max()),
        dataMax(std::numeric_limits<T>::lowest()),
        has_nulls(false) {}

  ChunkMetadata appendData(int8_t*& srcData, const size_t numAppendElems) {
    T* unencodedData = reinterpret_cast<T*>(srcData);
    for (size_t i = 0; i < numAppendElems; ++i) {
      T data = unencodedData[i];
      if (data == none_encoded_null_value<T>())
        has_nulls = true;
      else {
        dataMin = std::min(dataMin, data);
        dataMax = std::max(dataMax, data);
      }
    }
    numElems += numAppendElems;
    buffer_->append(srcData, numAppendElems * sizeof(T));
    ChunkMetadata chunkMetadata;
    getMetadata(chunkMetadata);
    srcData += numAppendElems * sizeof(T);
    return chunkMetadata;
  }

  void getMetadata(ChunkMetadata& chunkMetadata) {
    Encoder::getMetadata(chunkMetadata);  // call on parent class
    chunkMetadata.fillChunkStats(dataMin, dataMax, has_nulls);
  }

  // Only called from the executor for synthesized meta-information.
  ChunkMetadata getMetadata(const SQLTypeInfo& ti) {
    ChunkMetadata chunk_metadata{ti, 0, 0, ChunkStats{}};
    chunk_metadata.fillChunkStats(dataMin, dataMax, has_nulls);
    return chunk_metadata;
  }

  // Only called from the executor for synthesized meta-information.
  void updateStats(const int64_t val, const bool is_null) {
    if (is_null) {
      has_nulls = true;
    } else {
      const auto data = static_cast<T>(val);
      dataMin = std::min(dataMin, data);
      dataMax = std::max(dataMax, data);
    }
  }

  // Only called from the executor for synthesized meta-information.
  void updateStats(const double val, const bool is_null) {
    if (is_null) {
      has_nulls = true;
    } else {
      const auto data = static_cast<T>(val);
      dataMin = std::min(dataMin, data);
      dataMax = std::max(dataMax, data);
    }
  }

  // Only called from the executor for synthesized meta-information.
  void reduceStats(const Encoder& that) {
    const auto that_typed = static_cast<const NoneEncoder&>(that);
    if (that_typed.has_nulls) {
      has_nulls = true;
    }
    dataMin = std::min(dataMin, that_typed.dataMin);
    dataMax = std::max(dataMax, that_typed.dataMax);
  }

  void writeMetadata(FILE* f) {
    // assumes pointer is already in right place
    fwrite((int8_t*)&numElems, sizeof(size_t), 1, f);
    fwrite((int8_t*)&dataMin, sizeof(T), 1, f);
    fwrite((int8_t*)&dataMax, sizeof(T), 1, f);
    fwrite((int8_t*)&has_nulls, sizeof(bool), 1, f);
  }

  void readMetadata(FILE* f) {
    // assumes pointer is already in right place
    fread((int8_t*)&numElems, sizeof(size_t), 1, f);
    fread((int8_t*)&dataMin, sizeof(T), 1, f);
    fread((int8_t*)&dataMax, sizeof(T), 1, f);
    fread((int8_t*)&has_nulls, sizeof(bool), 1, f);
  }

  void copyMetadata(const Encoder* copyFromEncoder) {
    numElems = copyFromEncoder->numElems;
    auto castedEncoder = reinterpret_cast<const NoneEncoder<T>*>(copyFromEncoder);
    dataMin = castedEncoder->dataMin;
    dataMax = castedEncoder->dataMax;
    has_nulls = castedEncoder->has_nulls;
  }

  T dataMin;
  T dataMax;
  bool has_nulls;

};  // class NoneEncoder

#endif  // NONE_ENCODER_H
