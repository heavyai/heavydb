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

#ifndef FIXED_LENGTH_ENCODER_H
#define FIXED_LENGTH_ENCODER_H
#include "Encoder.h"
#include "AbstractBuffer.h"
#include <stdexcept>
#include <iostream>
#include <memory>
#include <glog/logging.h>

template <typename T, typename V>
class FixedLengthEncoder : public Encoder {
 public:
  FixedLengthEncoder(Data_Namespace::AbstractBuffer* buffer)
      : Encoder(buffer),
        dataMin(std::numeric_limits<T>::max()),
        dataMax(std::numeric_limits<T>::min()),
        has_nulls(false) {}

  ChunkMetadata appendData(int8_t*& srcData, const size_t numAppendElems) {
    T* unencodedData = reinterpret_cast<T*>(srcData);
    auto encodedData = std::unique_ptr<V[]>(new V[numAppendElems]);
    for (size_t i = 0; i < numAppendElems; ++i) {
      encodedData.get()[i] = static_cast<V>(unencodedData[i]);
      if (unencodedData[i] != encodedData.get()[i]) {
        LOG(ERROR) << "Fixed encoding failed, Unencoded: " + std::to_string(unencodedData[i]) + " encoded: " +
                          std::to_string(encodedData.get()[i]);
      } else {
        T data = unencodedData[i];
        if (data == std::numeric_limits<V>::min())
          has_nulls = true;
        else {
          dataMin = std::min(dataMin, data);
          dataMax = std::max(dataMax, data);
        }
      }
    }
    numElems += numAppendElems;

    // assume always CPU_BUFFER?
    buffer_->append((int8_t*)(encodedData.get()), numAppendElems * sizeof(V));
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
    const auto that_typed = static_cast<const FixedLengthEncoder<T, V>&>(that);
    if (that_typed.has_nulls) {
      has_nulls = true;
    }
    dataMin = std::min(dataMin, that_typed.dataMin);
    dataMax = std::max(dataMax, that_typed.dataMax);
  }

  void copyMetadata(const Encoder* copyFromEncoder) {
    numElems = copyFromEncoder->numElems;
    auto castedEncoder = reinterpret_cast<const FixedLengthEncoder<T, V>*>(copyFromEncoder);
    dataMin = castedEncoder->dataMin;
    dataMax = castedEncoder->dataMax;
    has_nulls = castedEncoder->has_nulls;
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
    fread((int8_t*)&dataMin, 1, sizeof(T), f);
    fread((int8_t*)&dataMax, 1, sizeof(T), f);
    fread((int8_t*)&has_nulls, 1, sizeof(bool), f);
  }
  T dataMin;
  T dataMax;
  bool has_nulls;

};  // FixedLengthEncoder

#endif  // FIXED_LENGTH_ENCODER_H
