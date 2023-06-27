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

#pragma once

#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/AbstractBuffer.h"
#include "ForeignStorageBuffer.h"
#include "ParquetShared.h"

#include <parquet/metadata.h>

namespace foreign_storage {

using RejectedRowIndices = std::set<int64_t>;

class ParquetEncoder {
 public:
  ParquetEncoder(Data_Namespace::AbstractBuffer* buffer)
      : buffer_(buffer)
      , is_error_tracking_enabled_(false)
      , current_chunk_offset_(0)
      , validate_metadata_stats_(true) {}
  virtual ~ParquetEncoder() = default;

  virtual void appendDataTrackErrors(const int16_t* def_levels,
                                     const int16_t* rep_levels,
                                     const int64_t values_read,
                                     const int64_t levels_read,
                                     int8_t* values) = 0;

  virtual void appendData(const int16_t* def_levels,
                          const int16_t* rep_levels,
                          const int64_t values_read,
                          const int64_t levels_read,
                          int8_t* values) = 0;

  virtual std::shared_ptr<ChunkMetadata> getRowGroupMetadata(
      const parquet::RowGroupMetaData* group_metadata,
      const int parquet_column_index,
      const SQLTypeInfo& column_type) {
    int64_t null_count{0};
    auto metadata = createMetadata(column_type);

    if (validate_metadata_stats_ && group_metadata->num_rows() > 0) {
      // update statistics
      auto column_metadata = group_metadata->ColumnChunk(parquet_column_index);
      auto stats = validate_and_get_column_metadata_statistics(column_metadata.get());
      null_count = stats->null_count();
      validateNullCount(group_metadata->schema()->Column(parquet_column_index)->name(),
                        null_count,
                        column_type);
    }
    metadata->chunkStats.has_nulls = null_count > 0;

    // update sizing
    metadata->numElements = group_metadata->num_rows();
    return metadata;
  }

  RejectedRowIndices getRejectedRowIndices() const { return invalid_indices_; }

  virtual void disableMetadataStatsValidation() { validate_metadata_stats_ = false; }

  virtual void initializeErrorTracking() { is_error_tracking_enabled_ = true; }

  virtual void initializeColumnType(const SQLTypeInfo& column_type) {
    column_type_ = column_type;
  }

 protected:
  Data_Namespace::AbstractBuffer* buffer_;

  // Members utililzed for error tracking
  bool is_error_tracking_enabled_;
  RejectedRowIndices invalid_indices_;
  size_t current_chunk_offset_;
  SQLTypeInfo column_type_;

  // flag to disable validation of stats
  bool validate_metadata_stats_;

  static std::shared_ptr<ChunkMetadata> createMetadata(const SQLTypeInfo& column_type) {
    auto metadata = std::make_shared<ChunkMetadata>();
    ForeignStorageBuffer buffer;
    buffer.initEncoder(column_type.is_array() ? column_type.get_elem_type()
                                              : column_type);
    auto encoder = buffer.getEncoder();
    encoder->getMetadata(metadata);
    metadata->sqlType = column_type;
    return metadata;
  }

  static void throwNotNullViolation(const std::string& parquet_column_name) {
    std::stringstream error_message;
    error_message << "A null value was detected in Parquet column '"
                  << parquet_column_name << "' but HeavyDB column is set to not null";
    throw std::runtime_error(error_message.str());
  }

  static void validateNullCount(const std::string& parquet_column_name,
                                int64_t null_count,
                                const SQLTypeInfo& column_type) {
    bool has_nulls = null_count > 0;
    if (has_nulls && column_type.get_notnull()) {
      throwNotNullViolation(parquet_column_name);
    }
  }
};

using InvalidRowGroupIndices = std::set<int64_t>;

class ParquetImportEncoder {
 public:
  virtual void eraseInvalidIndicesInBuffer(
      const InvalidRowGroupIndices& invalid_indices) = 0;

  virtual void validateAndAppendData(const int16_t* def_levels,
                                     const int16_t* rep_levels,
                                     const int64_t values_read,
                                     const int64_t levels_read,
                                     int8_t* values,
                                     const SQLTypeInfo& column_type, /* may not be used */
                                     InvalidRowGroupIndices& invalid_indices) = 0;
};

class ParquetScalarEncoder : public ParquetEncoder, public ParquetImportEncoder {
 public:
  ParquetScalarEncoder(Data_Namespace::AbstractBuffer* buffer) : ParquetEncoder(buffer) {}

  virtual void setNull(int8_t* omnisci_data_bytes) = 0;
  virtual void copy(const int8_t* omnisci_data_bytes_source,
                    int8_t* omnisci_data_bytes_destination) = 0;
  virtual void encodeAndCopy(const int8_t* parquet_data_bytes,
                             int8_t* omnisci_data_bytes) = 0;

  virtual void encodeAndCopyContiguous(const int8_t* parquet_data_bytes,
                                       int8_t* omnisci_data_bytes,
                                       const size_t num_elements) = 0;

  virtual void validate(const int8_t* parquet_data,
                        const int64_t j,
                        const SQLTypeInfo& column_type) const = 0;

  virtual void validateUsingEncodersColumnType(const int8_t* parquet_data,
                                               const int64_t j) const = 0;

  virtual std::string encodedDataToString(const int8_t* bytes) const = 0;
};

}  // namespace foreign_storage
