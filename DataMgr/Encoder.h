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

#ifndef ENCODER_H
#define ENCODER_H

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ChunkMetadata.h"
#include "Shared/DateConverters.h"
#include "Shared/sqltypes.h"
#include "Shared/types.h"

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
  void validate(T value) const {
    if (std::is_integral<T>::value) {
      do_validate(static_cast<int64_t>(value));
    }
  }

  void do_validate(int64_t value) const {
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
                               std::to_string(days) + " less than minimum capacity " +
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

  /**
   * Compute the maximum number of variable length encoded elements given a
   * byte limit
   *
   * @param index_data - (optional) index data for the encoded type
   * @param selected_idx - which indices in the encoded data to consider
   * @param byte_limit - byte limit that must be respected
   *
   * @return the number of elements
   *
   * NOTE: optional parameters above may be ignored by the implementation, but
   * may or may not be required depending on the encoder type backing the
   * implementation.
   */
  virtual size_t getNumElemsForBytesEncodedDataAtIndices(
      const int8_t* index_data,
      const std::vector<size_t>& selected_idx,
      const size_t byte_limit) = 0;

  /**
   * Append selected encoded data to the chunk buffer backing this encoder.
   *
   * @param index_data - (optional) the index data of data to append
   * @param data - the data to append
   * @param selected_idx - which indices in the encoded data to append
   *
   * @return updated chunk metadata for the chunk buffer backing this encoder
   *
   * NOTE: `index_data` must be non-null for varlen encoder types.
   */
  virtual std::shared_ptr<ChunkMetadata> appendEncodedDataAtIndices(
      const int8_t* index_data,
      int8_t* data,
      const std::vector<size_t>& selected_idx) = 0;

  /**
   * Append encoded data to the chunk buffer backing this encoder.
   *
   * @param index_data - (optional) the index data of data to append
   * @param data - the data to append
   * @param start_idx - the position to start encoding from in the `data` array
   * @param num_elements - the number of elements to encode from the `data` array
   * @return updated chunk metadata for the chunk buffer backing this encoder
   *
   * NOTE: `index_data` must be non-null for varlen encoder types.
   */
  virtual std::shared_ptr<ChunkMetadata> appendEncodedData(const int8_t* index_data,
                                                           int8_t* data,
                                                           const size_t start_idx,
                                                           const size_t num_elements) = 0;

  //! Append data to the chunk buffer backing this encoder.
  //! @param src_data Source data for the append
  //! @param num_elems_to_append Number of elements to append
  //! @param ti SQL Type Info for the column TODO(adb): used?
  //! @param replicating Pass one value and fill the chunk with it
  //! @param offset Write data starting at a given offset. Default is -1 which indicates
  //! an append, an offset of 0 rewrites the chunk up to `num_elems_to_append`.
  virtual std::shared_ptr<ChunkMetadata> appendData(int8_t*& src_data,
                                                    const size_t num_elems_to_append,
                                                    const SQLTypeInfo& ti,
                                                    const bool replicating = false,
                                                    const int64_t offset = -1) = 0;
  virtual void getMetadata(const std::shared_ptr<ChunkMetadata>& chunkMetadata);
  // Only called from the executor for synthesized meta-information.
  virtual std::shared_ptr<ChunkMetadata> getMetadata(const SQLTypeInfo& ti) = 0;
  virtual void updateStats(const int64_t val, const bool is_null) = 0;
  virtual void updateStats(const double val, const bool is_null) = 0;

  /**
   * Update statistics for data without appending.
   *
   * @param src_data - the data with which to update statistics
   * @param num_elements - the number of elements to scan in the data
   */
  virtual void updateStats(const int8_t* const src_data, const size_t num_elements) = 0;

  /**
   * Update statistics for encoded data without appending.
   *
   * @param dst_data - the data with which to update statistics
   * @param num_elements - the number of elements to scan in the data
   */
  virtual void updateStatsEncoded(const int8_t* const dst_data,
                                  const size_t num_elements) {
    UNREACHABLE();
  }

  /**
   * Update statistics for string data without appending.
   *
   * @param src_data - the string data with which to update statistics
   * @param start_idx - the offset into `src_data` to start the update
   * @param num_elements - the number of elements to scan in the string data
   */
  virtual void updateStats(const std::vector<std::string>* const src_data,
                           const size_t start_idx,
                           const size_t num_elements) = 0;

  /**
   * Update statistics for array data without appending.
   *
   * @param src_data - the array data with which to update statistics
   * @param start_idx - the offset into `src_data` to start the update
   * @param num_elements - the number of elements to scan in the array data
   */
  virtual void updateStats(const std::vector<ArrayDatum>* const src_data,
                           const size_t start_idx,
                           const size_t num_elements) = 0;

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
  virtual bool resetChunkStats(const ChunkStats&) {
    UNREACHABLE() << "Attempting to reset stats for unsupported type.";
    return false;
  }

  /**
   * Resets chunk metadata stats to their default values.
   */
  virtual void resetChunkStats() = 0;

  size_t getNumElems() const { return num_elems_; }
  void setNumElems(const size_t num_elems) { num_elems_ = num_elems; }

 protected:
  size_t num_elems_;

  Data_Namespace::AbstractBuffer* buffer_;

  DecimalOverflowValidator decimal_overflow_validator_;
  DateDaysOverflowValidator date_days_overflow_validator_;
};

#endif  // Encoder_h
