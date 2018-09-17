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

#ifndef ENCODER_H
#define ENCODER_H

#include "../Shared/sqltypes.h"
#include "../Shared/types.h"
#include "ChunkMetadata.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Data_Namespace {
class AbstractBuffer;
}

// default max input buffer size to 1MB
#define MAX_INPUT_BUF_SIZE 1048576

class Encoder {
 public:
  static Encoder* Create(Data_Namespace::AbstractBuffer* buffer,
                         const SQLTypeInfo sqlType);
  Encoder(Data_Namespace::AbstractBuffer* buffer) : num_elems_(0), buffer_(buffer) {}
  virtual ~Encoder() {}

  virtual ChunkMetadata appendData(int8_t*& srcData,
                                   const size_t numAppendElems,
                                   const bool replicating = false) = 0;
  virtual void getMetadata(ChunkMetadata& chunkMetadata);
  // Only called from the executor for synthesized meta-information.
  virtual ChunkMetadata getMetadata(const SQLTypeInfo& ti) = 0;
  virtual void updateStats(const int64_t val, const bool is_null) = 0;
  virtual void updateStats(const double val, const bool is_null) = 0;
  virtual void reduceStats(const Encoder&) = 0;
  virtual void copyMetadata(const Encoder* copyFromEncoder) = 0;
  virtual void writeMetadata(FILE* f /*, const size_t offset*/) = 0;
  virtual void readMetadata(FILE* f /*, const size_t offset*/) = 0;

  size_t getNumElems() const { return num_elems_; }

 protected:
  size_t num_elems_;

  Data_Namespace::AbstractBuffer* buffer_;
  // ChunkMetadata metadataTemplate_;
};

#endif  // Encoder_h
