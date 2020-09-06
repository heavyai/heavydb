/*
 * Copyright 2020 OmniSci, Inc.
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

#pragma once

#include <parquet/schema.h>
#include <parquet/types.h>

#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/AbstractBuffer.h"

namespace foreign_storage {

inline int64_t get_time_conversion_denominator(
    const parquet::LogicalType::TimeUnit::unit time_unit) {
  int64_t conversion_denominator = 0;
  switch (time_unit) {
    case parquet::LogicalType::TimeUnit::MILLIS:
      conversion_denominator = 1000L;
      break;
    case parquet::LogicalType::TimeUnit::MICROS:
      conversion_denominator = 1000L * 1000L;
      break;
    case parquet::LogicalType::TimeUnit::NANOS:
      conversion_denominator = 1000L * 1000L * 1000L;
      break;
    default:
      UNREACHABLE();
  }
  return conversion_denominator;
}

template <typename V, std::enable_if_t<std::is_integral<V>::value, int> = 0>
inline V get_null_value() {
  return inline_int_null_value<V>();
}

template <typename V, std::enable_if_t<std::is_floating_point<V>::value, int> = 0>
inline V get_null_value() {
  return inline_fp_null_value<V>();
}

class ParquetInPlaceEncoder {
 public:
  ParquetInPlaceEncoder(Data_Namespace::AbstractBuffer* buffer,
                        const ColumnDescriptor* column_desciptor,
                        const parquet::ColumnDescriptor* parquet_column_descriptor)
      : buffer_(buffer)
      , omnisci_data_type_byte_size_(column_desciptor->columnType.get_size())
      , parquet_data_type_byte_size_(
            parquet::GetTypeByteSize(parquet_column_descriptor->physical_type())) {}

  /**
   * Appends Parquet data to the buffer using an in-place algorithm.  Any
   * necessary transformation or validation of the data and decoding of nulls
   * is part of appending the data. Each class inheriting from this abstract class
   * must implement the functionality to copy, nullify and encode the data.
   *
   * @param def_levels - an array containing the Dremel encoding definition levels
   * @param values_read - the number of non-null values read
   * @param levels_read - the total number of values (non-null & null) that are read
   * @param values - values that are read
   *
   * Note that the Parquet format encodes nulls using Dremel encoding.
   */
  virtual void appendData(int16_t* def_levels,
                          int64_t values_read,
                          int64_t levels_read,
                          int8_t* values) {
    if (omnisci_data_type_byte_size_ < parquet_data_type_byte_size_) {
      for (int64_t i = 0; i < values_read; ++i) {
        encodeAndCopy(values + i * parquet_data_type_byte_size_,
                      values + i * omnisci_data_type_byte_size_);
      }
    }

    if (values_read < levels_read) {  // nulls exist
      decodeNullsAndEncodeData(
          values,
          def_levels,
          values_read,
          levels_read,
          omnisci_data_type_byte_size_ >= parquet_data_type_byte_size_);
    } else if (omnisci_data_type_byte_size_ >= parquet_data_type_byte_size_) {
      for (int64_t i = levels_read - 1; i >= 0; --i) {
        encodeAndCopy(values + i * parquet_data_type_byte_size_,
                      values + i * omnisci_data_type_byte_size_);
      }
    }

    buffer_->append(values, levels_read * omnisci_data_type_byte_size_);
  }

 protected:
  virtual void setNull(int8_t* omnisci_data_bytes) = 0;
  virtual void copy(const int8_t* omnisci_data_bytes_source,
                    int8_t* omnisci_data_bytes_destination) = 0;
  virtual void encodeAndCopy(const int8_t* parquet_data_bytes,
                             int8_t* omnisci_data_bytes) = 0;

  Data_Namespace::AbstractBuffer* buffer_;
  const size_t omnisci_data_type_byte_size_;

 private:
  void decodeNullsAndEncodeData(int8_t* data_ptr,
                                int16_t* def_levels,
                                int64_t values_read,
                                int64_t levels_read,
                                const bool do_encoding) {
    for (int64_t i = levels_read - 1, j = values_read - 1; i >= 0; --i) {
      if (def_levels[i]) {  // not null
        CHECK(j >= 0);
        if (do_encoding) {
          encodeAndCopy(data_ptr + (j--) * parquet_data_type_byte_size_,
                        data_ptr + i * omnisci_data_type_byte_size_);
        } else {
          copy(data_ptr + (j--) * omnisci_data_type_byte_size_,
               data_ptr + i * omnisci_data_type_byte_size_);
        }
      } else {  // null
        setNull(data_ptr + i * omnisci_data_type_byte_size_);
      }
    }
  }

  const size_t parquet_data_type_byte_size_;
};

template <typename V, typename T>
class TypedParquetInPlaceEncoder : public ParquetInPlaceEncoder {
 public:
  TypedParquetInPlaceEncoder(Data_Namespace::AbstractBuffer* buffer,
                             const ColumnDescriptor* column_desciptor,
                             const parquet::ColumnDescriptor* parquet_column_descriptor)
      : ParquetInPlaceEncoder(buffer, column_desciptor, parquet_column_descriptor) {}

  /**
   * This is a specialization of `ParquetInPlaceEncoder::appendData` for known
   * types that allows for optimization.
   *
   * See comment for `ParquetInPlaceEncoder::appendData` for details.
   */
  void appendData(int16_t* def_levels,
                  int64_t values_read,
                  int64_t levels_read,
                  int8_t* values) override {
    if (std::is_same<V, T>::value && values_read == levels_read) {
      if (!encodingIsIdentityForSameTypes()) {
        for (int64_t i = 0; i < levels_read; ++i) {
          encodeAndCopy(values + i * omnisci_data_type_byte_size_,
                        values + i * omnisci_data_type_byte_size_);
        }
      }
      buffer_->append(values, levels_read * omnisci_data_type_byte_size_);
    } else {
      ParquetInPlaceEncoder::appendData(def_levels, values_read, levels_read, values);
    }
  }

 protected:
  void setNull(int8_t* omnisci_data_bytes) override {
    auto& omnisci_data_value = reinterpret_cast<V*>(omnisci_data_bytes)[0];
    omnisci_data_value = get_null_value<V>();
  }

  void copy(const int8_t* omnisci_data_bytes_source,
            int8_t* omnisci_data_bytes_destination) override {
    const auto& omnisci_data_value_source =
        reinterpret_cast<const V*>(omnisci_data_bytes_source)[0];
    auto& omnisci_data_value_destination =
        reinterpret_cast<V*>(omnisci_data_bytes_destination)[0];
    omnisci_data_value_destination = omnisci_data_value_source;
  }

  virtual bool encodingIsIdentityForSameTypes() const { return false; }
};

}  // namespace foreign_storage
