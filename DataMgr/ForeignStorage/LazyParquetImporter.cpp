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

#include "LazyParquetImporter.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <parquet/types.h>

#include "ImportExport/ArrowImporter.h"
#include "ParquetTypeMappings.h"
#include "Shared/ArrowUtil.h"
#include "Shared/measure.h"
#include "Shared/thread_count.h"

using TypedImportBuffer = import_export::TypedImportBuffer;

namespace foreign_storage {

namespace {

std::string type_to_string(SQLTypes type) {
  return SQLTypeInfo(type, false).get_type_name();
}

std::tuple<int, int> open_parquet_table(
    const std::string& file_path,
    std::unique_ptr<parquet::arrow::FileReader>& reader) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(file_path));
  PARQUET_THROW_NOT_OK(OpenFile(infile, arrow::default_memory_pool(), &reader));
  auto file_metadata = reader->parquet_reader()->metadata();
  const auto num_row_groups = file_metadata->num_row_groups();
  const auto num_columns = file_metadata->num_columns();
  return std::make_tuple(num_row_groups, num_columns);
}

void validate_allowed_mapping(const parquet::ColumnDescriptor* descr,
                              const ColumnDescriptor* cd) {
  const auto type = cd->columnType.get_type();
  parquet::Type::type physical_type = descr->physical_type();
  auto logical_type = descr->logical_type();
  bool allowed_type =
      AllowedParquetMetadataTypeMappings::isColumnMappingSupported(cd, descr);
  if (!allowed_type) {
    std::string parquet_type;
    if (descr->logical_type()->is_none()) {
      parquet_type = parquet::TypeToString(physical_type);
    } else {
      parquet_type = logical_type->ToString();
    }
    std::string omnisci_type = type_to_string(type);
    throw std::runtime_error{"Conversion from Parquet type \"" + parquet_type +
                             "\" to OmniSci type \"" + omnisci_type +
                             "\" is not allowed. Please use an appropriate column type."};
  }
}

void validate_parquet_metadata(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const std::string& file_path,
    const ParquetForeignTableSchema& schema) {
  if (schema.numLogicalColumns() != file_metadata->num_columns()) {
    std::stringstream error_msg;
    error_msg << "Mismatched number of logical columns in table '"
              << schema.getForeignTable()->tableName << "' ("
              << schema.numLogicalColumns() << " columns) with parquet file '"
              << file_path << "' (" << file_metadata->num_columns() << " columns.)";
    throw std::runtime_error(error_msg.str());
  }

  auto column_it = schema.getLogicalColumns().begin();
  for (int i = 0; i < file_metadata->num_columns(); ++i, ++column_it) {
    const parquet::ColumnDescriptor* descr = file_metadata->schema()->Column(i);
    validate_allowed_mapping(descr, *column_it);

    auto fragment_size = schema.getForeignTable()->maxFragRows;
    int64_t max_row_group_size = 0;
    int max_row_group_index = 0;
    for (int r = 0; r < file_metadata->num_row_groups(); ++r) {
      auto group_metadata = file_metadata->RowGroup(r);
      auto num_rows = group_metadata->num_rows();
      if (num_rows > max_row_group_size) {
        max_row_group_size = num_rows;
        max_row_group_index = r;
      }

      auto column_chunk = group_metadata->ColumnChunk(i);
      bool contains_metadata = column_chunk->is_stats_set();
      if (contains_metadata) {
        auto stats = column_chunk->statistics();
        bool is_all_nulls = (stats->null_count() == group_metadata->num_rows());
        if (!stats->HasMinMax() && !is_all_nulls) {
          contains_metadata = false;
        }
      }

      if (!contains_metadata) {
        throw std::runtime_error{
            "Statistics metadata is required for all row groups. Metadata is missing for "
            "row group index: " +
            std::to_string(r) + ", column index: " + std::to_string(i) +
            ", file path: " + file_path};
      }
    }

    if (max_row_group_size > fragment_size) {
      throw std::runtime_error{
          "Parquet file has a row group size that is larger than the fragment size. "
          "Please set the table fragment size to a number that is larger than the "
          "row group size. Row group index: " +
          std::to_string(max_row_group_index) +
          ", row group size: " + std::to_string(max_row_group_size) +
          ", fragment size: " + std::to_string(fragment_size) +
          ", file path: " + file_path};
    }
  }
}

void initialize_bad_rows_tracker(import_export::BadRowsTracker& bad_rows_tracker,
                                 const int row_group,
                                 const std::string& file_path,
                                 import_export::Importer* importer) {
  bad_rows_tracker.rows.clear();
  bad_rows_tracker.nerrors = 0;
  bad_rows_tracker.file_name = file_path;
  bad_rows_tracker.row_group = row_group;
  bad_rows_tracker.importer = importer;
}

void initialize_import_buffers_vec(
    const import_export::Loader* loader,
    const ParquetForeignTableSchema& schema,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  import_buffer_vec.clear();
  for (const auto cd : schema.getLogicalAndPhysicalColumns()) {
    import_buffer_vec.emplace_back(new TypedImportBuffer(cd, loader->getStringDict(cd)));
  }
}

void initialize_row_group_metadata_vec(
    const size_t num_logical_and_physical_columns,
    LazyParquetImporter::RowGroupMetadataVector& row_group_metadata_vec) {
  row_group_metadata_vec.clear();
  row_group_metadata_vec.resize(num_logical_and_physical_columns);
}

std::unique_ptr<import_export::TypedImportBuffer>& initialize_import_buffer(
    const size_t column_id,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  auto& import_buffer =
      import_buffer_vec[column_id - 1];  // Column id starts at 1, hence the -1 offset
  import_buffer->import_buffers = &import_buffer_vec;
  import_buffer->col_idx = column_id;
  return import_buffer;
}

constexpr int MILLIS_PER_SECOND = 1000;
constexpr int MICROS_PER_SECOND = 1000 * 1000;
constexpr int NANOS_PER_SECOND = 1000 * 1000 * 1000;

/**
 * Gets the divisor to be used when converting from the given Parquet time unit to
 * the OmniSci time unit/precision. The OmniSci time unit/precision is expected to
 * either be in seconds or be an exact match of the Parquet time unit.
 *
 * @param type - OmniSci column type information
 * @param time_unit - Time unit/precision of column stored in the Parquet file
 * @return divisor
 */
int64_t get_divisor(const SQLTypeInfo& type,
                    const parquet::LogicalType::TimeUnit::unit time_unit) {
  int64_t divisor = 1;
  if (time_unit == parquet::LogicalType::TimeUnit::MILLIS) {
    if (type.get_precision() == 0) {
      divisor = MILLIS_PER_SECOND;
    } else {
      CHECK(AllowedParquetMetadataTypeMappings::isSameTimeUnit(type, time_unit));
    }
  } else if (time_unit == parquet::LogicalType::TimeUnit::MICROS) {
    if (type.get_precision() == 0) {
      divisor = MICROS_PER_SECOND;
    } else {
      CHECK(AllowedParquetMetadataTypeMappings::isSameTimeUnit(type, time_unit));
    }
  } else if (time_unit == parquet::LogicalType::TimeUnit::NANOS) {
    if (type.get_precision() == 0) {
      divisor = NANOS_PER_SECOND;
    } else {
      CHECK(AllowedParquetMetadataTypeMappings::isSameTimeUnit(type, time_unit));
    }
  } else {
    UNREACHABLE();
  }
  return divisor;
}

/**
 * Gets the min and max Parquet chunk metadata for time, timestamp, or date column types.
 */
std::pair<int64_t, int64_t> get_datetime_min_and_max(
    const ColumnDescriptor* column_descriptor,
    const parquet::ColumnDescriptor* parquet_column_descriptor,
    std::shared_ptr<parquet::Statistics> stats) {
  int64_t min{0}, max{0};
  auto& type = column_descriptor->columnType;
  auto logical_type = parquet_column_descriptor->logical_type();
  auto physical_type = parquet_column_descriptor->physical_type();
  auto encoded_min = stats->EncodeMin();
  auto encoded_max = stats->EncodeMax();

  if (type.is_timestamp()) {
    CHECK(logical_type->is_timestamp());
    CHECK_EQ(parquet::Type::INT64, physical_type);

    auto timestamp_type =
        dynamic_cast<const parquet::TimestampLogicalType*>(logical_type.get());
    CHECK(timestamp_type);

    if (!timestamp_type->is_adjusted_to_utc()) {
      throw std::runtime_error{
          "Non-UTC timezone specified in Parquet file for column \"" +
          column_descriptor->columnName +
          "\". Only UTC timezone is currently supported."};
    }

    auto divisor = get_divisor(type, timestamp_type->time_unit());
    min = reinterpret_cast<int64_t*>(encoded_min.data())[0];
    max = reinterpret_cast<int64_t*>(encoded_max.data())[0];

    min /= divisor;
    max /= divisor;
  } else if (type.is_date()) {
    CHECK_EQ(parquet::Type::INT32, physical_type);
    min = reinterpret_cast<int32_t*>(encoded_min.data())[0];
    max = reinterpret_cast<int32_t*>(encoded_max.data())[0];
  } else if (type.get_type() == kTIME) {
    CHECK(logical_type->is_time());

    auto time_type = dynamic_cast<const parquet::TimeLogicalType*>(logical_type.get());
    CHECK(time_type);

    auto time_unit = time_type->time_unit();
    auto divisor = get_divisor(type, time_unit);
    if (time_unit == parquet::LogicalType::TimeUnit::MILLIS) {
      CHECK_EQ(parquet::Type::INT32, physical_type);
      min = reinterpret_cast<int32_t*>(encoded_min.data())[0];
      max = reinterpret_cast<int32_t*>(encoded_max.data())[0];
    } else {
      CHECK(time_unit == parquet::LogicalType::TimeUnit::MICROS ||
            time_unit == parquet::LogicalType::TimeUnit::NANOS);
      CHECK_EQ(parquet::Type::INT64, physical_type);
      min = reinterpret_cast<int64_t*>(encoded_min.data())[0];
      max = reinterpret_cast<int64_t*>(encoded_max.data())[0];
    }

    min /= divisor;
    max /= divisor;
  } else {
    UNREACHABLE();
  }
  return {min, max};
}

int64_t convert_decimal_byte_array_to_int(const std::string& encoded_value) {
  auto byte_array = reinterpret_cast<const uint8_t*>(encoded_value.data());
  auto result = arrow::Decimal128::FromBigEndian(byte_array, encoded_value.length());
  CHECK(result.ok()) << result.status().message();
  auto& decimal = result.ValueOrDie();
  return static_cast<int64_t>(decimal);
}

/**
 * Gets the min and max Parquet chunk metadata for a decimal type.
 */
std::pair<int64_t, int64_t> get_decimal_min_and_max(
    const ColumnDescriptor* column_descriptor,
    const parquet::ColumnDescriptor* parquet_column_descriptor,
    std::shared_ptr<parquet::Statistics> stats) {
  auto& type = column_descriptor->columnType;
  auto logical_type = parquet_column_descriptor->logical_type();
  CHECK(type.is_decimal() && logical_type->is_decimal());

  auto decimal_type =
      dynamic_cast<const parquet::DecimalLogicalType*>(logical_type.get());
  CHECK(decimal_type);

  auto physical_type = parquet_column_descriptor->physical_type();
  auto encoded_min = stats->EncodeMin();
  auto encoded_max = stats->EncodeMax();

  int64_t min{0}, max{0};
  CHECK_GT(decimal_type->precision(), 0);
  if (physical_type == parquet::Type::BYTE_ARRAY ||
      physical_type == parquet::Type::FIXED_LEN_BYTE_ARRAY) {
    min = convert_decimal_byte_array_to_int(encoded_min);
    max = convert_decimal_byte_array_to_int(encoded_max);
  } else if (physical_type == parquet::Type::INT32) {
    min = reinterpret_cast<int32_t*>(encoded_min.data())[0];
    max = reinterpret_cast<int32_t*>(encoded_min.data())[0];
  } else if (physical_type == parquet::Type::INT64) {
    min = reinterpret_cast<int64_t*>(encoded_min.data())[0];
    max = reinterpret_cast<int64_t*>(encoded_max.data())[0];
  } else {
    UNREACHABLE();
  }
  return {min, max};
}

void read_parquet_metadata_into_import_buffer(
    const size_t num_rows,
    const int row_group,
    const ParquetForeignTableSchema& schema,
    const std::unique_ptr<parquet::RowGroupMetaData>& group_metadata,
    const ColumnDescriptor* column_descriptor,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec,
    LazyParquetImporter::RowGroupMetadataVector& row_group_metadata_vec) {
  auto column_id = column_descriptor->columnId;
  auto& import_buffer = initialize_import_buffer(column_id, import_buffer_vec);
  auto parquet_column_index = schema.getParquetColumnIndex(column_id);
  auto column_chunk = group_metadata->ColumnChunk(parquet_column_index);
  CHECK(column_chunk->is_stats_set());
  std::shared_ptr<parquet::Statistics> stats = column_chunk->statistics();
  bool is_all_nulls = (stats->null_count() == group_metadata->num_rows());
  CHECK(is_all_nulls || stats->HasMinMax());

  const auto& type = column_descriptor->columnType;
  auto parquet_column_descriptor = group_metadata->schema()->Column(parquet_column_index);
  if (!is_all_nulls) {
    if (is_datetime(type.get_type())) {
      auto [min, max] =
          get_datetime_min_and_max(column_descriptor, parquet_column_descriptor, stats);
      import_buffer->addBigint(min);
      import_buffer->addBigint(max);
    } else if (type.is_decimal()) {
      auto [min, max] =
          get_decimal_min_and_max(column_descriptor, parquet_column_descriptor, stats);
      import_buffer->addBigint(min);
      import_buffer->addBigint(max);
    } else if (!type.is_string() && !type.is_varlen()) {
      std::shared_ptr<arrow::Scalar> min, max;
      PARQUET_THROW_NOT_OK(parquet::arrow::StatisticsAsScalars(*stats, &min, &max));
      ARROW_ASSIGN_OR_THROW(auto min_array, arrow::MakeArrayFromScalar(*min, 1));
      ARROW_ASSIGN_OR_THROW(auto max_array, arrow::MakeArrayFromScalar(*max, 1));
      import_buffer->add_arrow_values(
          column_descriptor, *min_array, false, {0, 1}, &bad_rows_tracker);
      import_buffer->add_arrow_values(
          column_descriptor, *max_array, false, {0, 1}, &bad_rows_tracker);
    }
  }

  // Set the same metadata for the logical column and all its related physical columns
  auto physical_columns_count = column_descriptor->columnType.get_physical_cols();
  for (auto i = column_id - 1; i < (column_id + physical_columns_count); i++) {
    row_group_metadata_vec[i].metadata_only = true;
    row_group_metadata_vec[i].row_group_index = row_group;
    row_group_metadata_vec[i].is_all_nulls = is_all_nulls;
    row_group_metadata_vec[i].has_nulls = stats->null_count() > 0;
    row_group_metadata_vec[i].num_elements = num_rows;
  }
}

void read_parquet_data_into_import_buffer(
    const import_export::Loader* loader,
    const int row_group,
    const ParquetForeignTableSchema& schema,
    const ColumnDescriptor* column_descriptor,
    std::unique_ptr<parquet::arrow::FileReader>& reader,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  auto column_id = column_descriptor->columnId;
  auto parquet_column_index = schema.getParquetColumnIndex(column_id);
  std::shared_ptr<arrow::ChunkedArray> array;
  PARQUET_THROW_NOT_OK(
      reader->RowGroup(row_group)->Column(parquet_column_index)->Read(&array));
  auto& import_buffer = initialize_import_buffer(column_id, import_buffer_vec);
  auto num_bad_rows_before_add_values = bad_rows_tracker.rows.size();
  // TODO: the following async call & wait suppresses a false-positive
  // lock-order-inversion warning from TSAN; remove its use once
  // TypedImportBuffer is refactored
  std::async(std::launch::async, [&] {
    for (auto chunk : array->chunks()) {
      import_buffer->add_arrow_values(
          column_descriptor, *chunk, false, {0, chunk->length()}, &bad_rows_tracker);
    }
  }).wait();
  // TODO: remove this check for failure to import geo-type once null
  // geo-types are fixed
  auto logical_column = schema.getLogicalColumn(column_id);
  if (logical_column->columnType.is_geometry() &&
      bad_rows_tracker.rows.size() > num_bad_rows_before_add_values) {
    std::stringstream error_msg;
    error_msg << "Failure to import geo column '" << column_descriptor->columnName
              << "' in table '" << loader->getTableDesc()->tableName << "' for row group "
              << row_group << " and row " << *bad_rows_tracker.rows.rbegin() << ".";
    throw std::runtime_error(error_msg.str());
  }
}

void import_row_group(
    const int row_group,
    const bool metadata_scan,
    const ParquetForeignTableSchema& schema,
    const std::string& file_path,
    const Interval<ColumnType>& column_interval,
    import_export::Importer* importer,
    std::unique_ptr<parquet::arrow::FileReader>& reader,
    LazyParquetImporter::RowGroupMetadataVector& row_group_metadata_vec,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  auto loader = importer->getLoader();

  // Initialize
  initialize_import_buffers_vec(loader, schema, import_buffer_vec);
  initialize_row_group_metadata_vec(schema.numLogicalAndPhysicalColumns(),
                                    row_group_metadata_vec);
  initialize_bad_rows_tracker(bad_rows_tracker, row_group, file_path, importer);

  // Read metadata
  auto file_metadata = reader->parquet_reader()->metadata();
  std::unique_ptr<parquet::RowGroupMetaData> group_metadata =
      file_metadata->RowGroup(row_group);
  const size_t num_rows = group_metadata->num_rows();

  // Process each column into corresponding import buffer
  for (int column_id = column_interval.start; column_id <= column_interval.end;
       column_id++) {
    const auto column_descriptor = schema.getColumnDescriptor(column_id);
    if (metadata_scan) {
      // Parquet metadata only import branch
      read_parquet_metadata_into_import_buffer(num_rows,
                                               row_group,
                                               schema,
                                               group_metadata,
                                               column_descriptor,
                                               bad_rows_tracker,
                                               import_buffer_vec,
                                               row_group_metadata_vec);
    } else {
      // Regular import branch
      read_parquet_data_into_import_buffer(loader,
                                           row_group,
                                           schema,
                                           column_descriptor,
                                           reader,
                                           bad_rows_tracker,
                                           import_buffer_vec);
    }
    column_id += column_descriptor->columnType.get_physical_cols();
  }
  importer->load(import_buffer_vec, num_rows);
}

}  // namespace

void LazyParquetImporter::partialImport(const Interval<RowGroupType>& row_group_interval,
                                        const Interval<ColumnType>& column_interval,
                                        const bool metadata_scan) {
  using namespace import_export;

  CHECK_LE(row_group_interval.start, row_group_interval.end);
  CHECK_LE(column_interval.start, column_interval.end);

  std::unique_ptr<parquet::arrow::FileReader> reader;
  int num_row_groups, num_columns;
  std::tie(num_row_groups, num_columns) = open_parquet_table(file_path, reader);
  validate_parquet_metadata(reader->parquet_reader()->metadata(), file_path, schema_);

  // Check which row groups and columns to import
  auto row_groups_to_import_interval =
      metadata_scan ? Interval<RowGroupType>{0, num_row_groups - 1} : row_group_interval;
  auto columns_to_import_interval =
      metadata_scan
          ? Interval<ColumnType>{schema_.getLogicalAndPhysicalColumns().front()->columnId,
                                 schema_.getLogicalAndPhysicalColumns().back()->columnId}
          : column_interval;

  auto& import_buffers_vec = get_import_buffers_vec();
  import_buffers_vec.resize(1);
  std::vector<BadRowsTracker> bad_rows_trackers(1);
  for (int row_group = row_groups_to_import_interval.start;
       row_group <= row_groups_to_import_interval.end && !load_failed;
       ++row_group) {
    import_row_group(row_group,
                     metadata_scan,
                     schema_,
                     file_path,
                     columns_to_import_interval,
                     this,
                     reader,
                     row_group_metadata_vec_,
                     bad_rows_trackers[0],
                     import_buffers_vec[0]);
  }
}

LazyParquetImporter::LazyParquetImporter(import_export::Loader* provided_loader,
                                         const std::string& file_name,
                                         const import_export::CopyParams& copy_params,
                                         RowGroupMetadataVector& metadata_vector,
                                         ParquetForeignTableSchema& schema)
    : import_export::Importer(provided_loader, file_name, copy_params)
    , row_group_metadata_vec_(metadata_vector)
    , schema_(schema) {}
}  // namespace foreign_storage
