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

#include "ParquetEncoder.h"
#include "ParquetShared.h"

#include <parquet/schema.h>
#include <parquet/types.h>

#include "Catalog/ColumnDescriptor.h"
#include "ForeignStorageBuffer.h"

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

class ParquetInPlaceEncoder : public ParquetScalarEncoder {
 public:
  ParquetInPlaceEncoder(Data_Namespace::AbstractBuffer* buffer,
                        const size_t omnisci_data_type_byte_size,
                        const size_t parquet_data_type_byte_size)
      : ParquetScalarEncoder(buffer)
      , omnisci_data_type_byte_size_(omnisci_data_type_byte_size)
      , parquet_data_type_byte_size_(parquet_data_type_byte_size) {}

  /**
   * Appends Parquet data to the buffer using an in-place algorithm.  Any
   * necessary transformation or validation of the data and decoding of nulls
   * is part of appending the data. Each class inheriting from this abstract class
   * must implement the functionality to copy, nullify and encode the data.
   *
   * @param def_levels - an array containing the Dremel encoding definition levels
   * @param rep_levels - an array containing the Dremel encoding repetition levels
   * @param values_read - the number of non-null values read
   * @param levels_read - the total number of values (non-null & null) that are read
   * @param is_last_batch - flag indicating if this is the last read for the row group
   * @param values - values that are read
   *
   * Note that the Parquet format encodes nulls using Dremel encoding.
   */
  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  const bool is_last_batch,
                  int8_t* values) override {
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
  const size_t omnisci_data_type_byte_size_;

 private:
  void decodeNullsAndEncodeData(int8_t* data_ptr,
                                const int16_t* def_levels,
                                const int64_t values_read,
                                const int64_t levels_read,
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
      : ParquetInPlaceEncoder(
            buffer,
            column_desciptor->columnType.get_size(),
            parquet::GetTypeByteSize(parquet_column_descriptor->physical_type())) {}

  TypedParquetInPlaceEncoder(Data_Namespace::AbstractBuffer* buffer,
                             const size_t omnisci_data_type_byte_size,
                             const size_t parquet_data_type_byte_size)
      : ParquetInPlaceEncoder(buffer,
                              omnisci_data_type_byte_size,
                              parquet_data_type_byte_size) {}

  /**
   * This is a specialization of `ParquetInPlaceEncoder::appendData` for known
   * types that allows for optimization.
   *
   * See comment for `ParquetInPlaceEncoder::appendData` for details.
   */
  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  const bool is_last_batch,
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
      ParquetInPlaceEncoder::appendData(
          def_levels, rep_levels, values_read, levels_read, is_last_batch, values);
    }
  }

  void encodeAndCopyContiguous(const int8_t* parquet_data_bytes,
                               int8_t* omnisci_data_bytes,
                               const size_t num_elements) override {
    auto parquet_data_ptr = reinterpret_cast<const T*>(parquet_data_bytes);
    auto omnisci_data_ptr = reinterpret_cast<V*>(omnisci_data_bytes);
    for (size_t i = 0; i < num_elements; ++i) {
      encodeAndCopy(reinterpret_cast<const int8_t*>(&parquet_data_ptr[i]),
                    reinterpret_cast<int8_t*>(&omnisci_data_ptr[i]));
    }
  }

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

  std::shared_ptr<ChunkMetadata> getRowGroupMetadata(
      const parquet::RowGroupMetaData* group_metadata,
      const int parquet_column_index,
      const SQLTypeInfo& column_type) override {
    auto metadata = ParquetEncoder::createMetadata(column_type);
    auto column_metadata = group_metadata->ColumnChunk(parquet_column_index);

    // update statistics
    auto parquet_column_descriptor =
        group_metadata->schema()->Column(parquet_column_index);
    auto stats = validate_and_get_column_metadata_statistics(column_metadata.get());
    if (stats->HasMinMax()) {
      // validate statistics if validation applicable as part of encoding
      if (auto parquet_scalar_validator = dynamic_cast<ParquetMetadataValidator*>(this)) {
        try {
          parquet_scalar_validator->validate(stats, column_type);
        } catch (const std::exception& e) {
          std::stringstream error_message;
          error_message << e.what() << " Error validating statistics of Parquet column '"
                        << group_metadata->schema()->Column(parquet_column_index)->name()
                        << "'";
          throw std::runtime_error(error_message.str());
        }
      }

      auto [stats_min, stats_max] = getEncodedStats(parquet_column_descriptor, stats);
      auto updated_chunk_stats = getUpdatedStats(stats_min, stats_max, column_type);
      metadata->fillChunkStats(updated_chunk_stats.min,
                               updated_chunk_stats.max,
                               metadata->chunkStats.has_nulls);
    }
    metadata->chunkStats.has_nulls = stats->null_count() > 0;

    // update sizing
    metadata->numBytes = omnisci_data_type_byte_size_ * column_metadata->num_values();
    metadata->numElements = group_metadata->num_rows();

    return metadata;
  }

 protected:
  virtual bool encodingIsIdentityForSameTypes() const { return false; }

  std::pair<T, T> getUnencodedStats(std::shared_ptr<parquet::Statistics> stats) const {
    T stats_min = reinterpret_cast<T*>(stats->EncodeMin().data())[0];
    T stats_max = reinterpret_cast<T*>(stats->EncodeMax().data())[0];
    return {stats_min, stats_max};
  }

 private:
  static ChunkStats getUpdatedStats(V& stats_min,
                                    V& stats_max,
                                    const SQLTypeInfo& column_type) {
    ForeignStorageBuffer buffer;
    buffer.initEncoder(column_type);
    auto encoder = buffer.getEncoder();

    if (column_type.is_array()) {
      ArrayDatum min_datum(
          sizeof(V), reinterpret_cast<int8_t*>(&stats_min), false, DoNothingDeleter());
      ArrayDatum max_datum(
          sizeof(V), reinterpret_cast<int8_t*>(&stats_max), false, DoNothingDeleter());
      std::vector<ArrayDatum> min_max_datums{min_datum, max_datum};
      encoder->updateStats(&min_max_datums, 0, 1);
    } else {
      encoder->updateStats(reinterpret_cast<int8_t*>(&stats_min), 1);
      encoder->updateStats(reinterpret_cast<int8_t*>(&stats_max), 1);
    }
    auto updated_chunk_stats_metadata = std::make_shared<ChunkMetadata>();
    encoder->getMetadata(updated_chunk_stats_metadata);
    return updated_chunk_stats_metadata->chunkStats;
  }

  std::pair<V, V> getEncodedStats(
      const parquet::ColumnDescriptor* parquet_column_descriptor,
      std::shared_ptr<parquet::Statistics> stats) {
    V stats_min, stats_max;
    auto min_string = stats->EncodeMin();
    auto max_string = stats->EncodeMax();
    if (parquet_column_descriptor->physical_type() ==
        parquet::Type::FIXED_LEN_BYTE_ARRAY) {
      parquet::FixedLenByteArray min_byte_array, max_byte_array;
      min_byte_array.ptr = reinterpret_cast<const uint8_t*>(min_string.data());
      max_byte_array.ptr = reinterpret_cast<const uint8_t*>(max_string.data());
      encodeAndCopy(reinterpret_cast<int8_t*>(&min_byte_array),
                    reinterpret_cast<int8_t*>(&stats_min));
      encodeAndCopy(reinterpret_cast<int8_t*>(&max_byte_array),
                    reinterpret_cast<int8_t*>(&stats_max));
    } else if (parquet_column_descriptor->physical_type() == parquet::Type::BYTE_ARRAY) {
      parquet::ByteArray min_byte_array, max_byte_array;
      min_byte_array.ptr = reinterpret_cast<const uint8_t*>(min_string.data());
      min_byte_array.len = min_string.length();
      max_byte_array.ptr = reinterpret_cast<const uint8_t*>(max_string.data());
      max_byte_array.len = max_string.length();
      encodeAndCopy(reinterpret_cast<int8_t*>(&min_byte_array),
                    reinterpret_cast<int8_t*>(&stats_min));
      encodeAndCopy(reinterpret_cast<int8_t*>(&max_byte_array),
                    reinterpret_cast<int8_t*>(&stats_max));
    } else {
      encodeAndCopy(reinterpret_cast<int8_t*>(min_string.data()),
                    reinterpret_cast<int8_t*>(&stats_min));
      encodeAndCopy(reinterpret_cast<int8_t*>(max_string.data()),
                    reinterpret_cast<int8_t*>(&stats_max));
    }
    return {stats_min, stats_max};
  }
};

}  // namespace foreign_storage
