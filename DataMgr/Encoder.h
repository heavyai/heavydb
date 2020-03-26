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

#include "../Shared/DateConverters.h"
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
    if (!do_check_) {
      return;
    }

    if (value >= max_) {
      throw std::runtime_error("Decimal overflow: value is greater than 10^" +
                               std::to_string(pow10_) + " max " + std::to_string(max_) +
                               " value " + std::to_string(value));
    }

    if (value <= min_) {
      throw std::runtime_error("Decimal overflow: value is less than -10^" +
                               std::to_string(pow10_) + " min " + std::to_string(min_) +
                               " value " + std::to_string(value));
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

class DateDaysOverflowValidator {
 public:
  DateDaysOverflowValidator(SQLTypeInfo type) {
    is_date_in_days_ =
        type.is_array() ? type.get_elem_type().is_date_in_days() : type.is_date_in_days();
    const bool is_date_16_ = is_date_in_days_ ? type.get_comp_param() == 16 : false;
    max_ = is_date_16_ ? static_cast<int64_t>(std::numeric_limits<int16_t>::max())
                       : static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    min_ = is_date_16_ ? static_cast<int64_t>(std::numeric_limits<int16_t>::min())
                       : static_cast<int64_t>(std::numeric_limits<int32_t>::min());
  }

  template <typename T>
  void validate(T value) {
    if (!is_date_in_days_ || !std::is_integral<T>::value) {
      return;
    }
    const int64_t days =
        DateConverters::get_epoch_days_from_seconds(static_cast<int64_t>(value));
    if (days > max_) {
      throw std::runtime_error("Date encoding overflow: Epoch days " +
                               std::to_string(days) + " greater than maximum capacity " +
                               std::to_string(max_));
    }
    if (days < min_) {
      throw std::runtime_error("Date encoding underflow: Epoch days " +
                               std::to_string(days) + " less than minumum capacity " +
                               std::to_string(min_));
    }
  }

 private:
  bool is_date_in_days_;
  int64_t max_;
  int64_t min_;
};

class Encoder {
 public:
  static Encoder* Create(Data_Namespace::AbstractBuffer* buffer,
                         const SQLTypeInfo sqlType);
  Encoder(Data_Namespace::AbstractBuffer* buffer);
  virtual ~Encoder() {}

  //! Append data to the chunk buffer backing this encoder.
  //! @param src_data Source data for the append
  //! @param num_elems_to_append Number of elements to append
  //! @param ti SQL Type Info for the column TODO(adb): used?
  //! @param replicating Pass one value and fill the chunk with it
  //! @param offset Write data starting at a given offset. Default is -1 which indicates
  //! an append, an offset of 0 rewrites the chunk up to `num_elems_to_append`.
  virtual ChunkMetadata appendData(int8_t*& src_data,
                                   const size_t num_elems_to_append,
                                   const SQLTypeInfo& ti,
                                   const bool replicating = false,
                                   const int64_t offset = -1) = 0;
  virtual void getMetadata(ChunkMetadata& chunkMetadata);
  // Only called from the executor for synthesized meta-information.
  virtual ChunkMetadata getMetadata(const SQLTypeInfo& ti) = 0;
  virtual void updateStats(const int64_t val, const bool is_null) = 0;
  virtual void updateStats(const double val, const bool is_null) = 0;

  // Only called from ArrowStorageInterface to update stats on chunk of data
  virtual void updateStats(const int8_t* const dst, const size_t numBytes) = 0;

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
  DateDaysOverflowValidator date_days_overflow_validator_;
};

#endif  // Encoder_h
