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

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Data_Namespace {
class AbstractBuffer;
}

// default max input buffer size to 1MB
#define MAX_INPUT_BUF_SIZE 1048576

class DecimalOverflowValidator {
 public:
  DecimalOverflowValidator(SQLTypeInfo type) {
    if (type.is_array()) {
      type = type.get_elem_type();
    }

    do_check_ = type.is_decimal();
    int precision = type.get_precision();
    int scale = type.get_scale();
    max_ = (int64_t)std::pow((double)10.0, precision);
    min_ = -max_;
    pow10_ = precision - scale;
  }

  template <typename T>
  void validate(T value) {
    if (std::is_integral<T>::value) {
      do_validate(static_cast<int64_t>(value));
    }
  }

  void do_validate(int64_t value) {
    if (!do_check_)
      return;

    if (value >= max_) {
      throw std::runtime_error("Decimal overflow: value is greater than 10^" +
                               std::to_string(pow10_));
    }

    if (value <= min_) {
      throw std::runtime_error("Decimal overflow: value is less than -10^" +
                               std::to_string(pow10_));
    }
  }

 private:
  bool do_check_;
  int64_t max_;
  int64_t min_;
  int pow10_;
};

template <typename INNER_VALIDATOR>
class NullAwareValidator {
 public:
  NullAwareValidator(SQLTypeInfo type, INNER_VALIDATOR* inner_validator) {
    if (type.is_array()) {
      type = type.get_elem_type();
    }

    skip_null_check_ = type.get_notnull();
    inner_validator_ = inner_validator;
  }

  template <typename T>
  void validate(T value) {
    if (skip_null_check_ || value != inline_int_null_value<T>()) {
      inner_validator_->template validate<T>(value);
    }
  }

 private:
  bool skip_null_check_;
  INNER_VALIDATOR* inner_validator_;
};

class Encoder {
 public:
  static Encoder* Create(Data_Namespace::AbstractBuffer* buffer,
                         const SQLTypeInfo sqlType);
  Encoder(Data_Namespace::AbstractBuffer* buffer);
  virtual ~Encoder() {}

  virtual ChunkMetadata appendData(int8_t*& srcData,
                                   const size_t numAppendElems,
                                   const SQLTypeInfo&,
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

  /**
   * @brief: Reset chunk level stats (min, max, nulls) using new values from the argument.
   * @return: True if an update occurred and the chunk needs to be flushed. False
   * otherwise. Default false if metadata update is unsupported. Only reset chunk stats if
   * the incoming stats differ from the current stats.
   */
  virtual bool resetChunkStats(const ChunkStats&) { return false; }

  size_t getNumElems() const { return num_elems_; }
  void setNumElems(const size_t num_elems) { num_elems_ = num_elems; }

 protected:
  size_t num_elems_;

  Data_Namespace::AbstractBuffer* buffer_;
  // ChunkMetadata metadataTemplate_;

  DecimalOverflowValidator decimal_overflow_validator_;
};

#endif  // Encoder_h
