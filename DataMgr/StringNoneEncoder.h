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
 * @file		StringNoneEncoder.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		For unencoded strings
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef STRING_NONE_ENCODER_H
#define STRING_NONE_ENCODER_H

#include <glog/logging.h>
#include <cassert>
#include <string>
#include <vector>
#include "AbstractBuffer.h"
#include "ChunkMetadata.h"
#include "Encoder.h"

using Data_Namespace::AbstractBuffer;

class StringNoneEncoder : public Encoder {
 public:
  StringNoneEncoder(AbstractBuffer* buffer)
      : Encoder(buffer), index_buf(nullptr), last_offset(-1), has_nulls(false) {}

  size_t getNumElemsForBytesInsertData(const std::vector<std::string>* srcData,
                                       const int start_idx,
                                       const size_t numAppendElems,
                                       const size_t byteLimit,
                                       const bool replicating = false);

  ChunkMetadata appendData(int8_t*& srcData,
                           const size_t numAppendElems,
                           const SQLTypeInfo&,
                           const bool replicating = false) {
    CHECK(false);  // should never be called for strings
    return ChunkMetadata{};
  }

  ChunkMetadata appendData(const std::vector<std::string>* srcData,
                           const int start_idx,
                           const size_t numAppendElems,
                           const bool replicating = false);

  void getMetadata(ChunkMetadata& chunkMetadata) {
    Encoder::getMetadata(chunkMetadata);  // call on parent class
    chunkMetadata.chunkStats.min.stringval = nullptr;
    chunkMetadata.chunkStats.max.stringval = nullptr;
    chunkMetadata.chunkStats.has_nulls = has_nulls;
  }

  // Only called from the executor for synthesized meta-information.
  ChunkMetadata getMetadata(const SQLTypeInfo& ti) {
    auto chunk_stats = ChunkStats{};
    chunk_stats.min.stringval = nullptr;
    chunk_stats.max.stringval = nullptr;
    chunk_stats.has_nulls = has_nulls;
    ChunkMetadata chunk_metadata{ti, 0, 0, chunk_stats};
    return chunk_metadata;
  }

  void updateStats(const int64_t, const bool) { CHECK(false); }

  void updateStats(const double, const bool) { CHECK(false); }

  void reduceStats(const Encoder&) { CHECK(false); }

  void writeMetadata(FILE* f) {
    // assumes pointer is already in right place
    fwrite((int8_t*)&num_elems_, sizeof(size_t), 1, f);
    fwrite((int8_t*)&has_nulls, sizeof(bool), 1, f);
  }

  void readMetadata(FILE* f) {
    // assumes pointer is already in right place
    CHECK_NE(fread((int8_t*)&num_elems_, sizeof(size_t), size_t(1), f), size_t(0));
    CHECK_NE(fread((int8_t*)&has_nulls, sizeof(bool), size_t(1), f), size_t(0));
  }

  void copyMetadata(const Encoder* copyFromEncoder) {
    num_elems_ = copyFromEncoder->getNumElems();
    has_nulls = static_cast<const StringNoneEncoder*>(copyFromEncoder)->has_nulls;
  }

  AbstractBuffer* get_index_buf() const { return index_buf; }
  void set_index_buf(AbstractBuffer* buf) { index_buf = buf; }

 private:
  AbstractBuffer* index_buf;
  StringOffsetT last_offset;
  bool has_nulls;

};  // class StringNoneEncoder

#endif  // STRING_NONE_ENCODER_H
