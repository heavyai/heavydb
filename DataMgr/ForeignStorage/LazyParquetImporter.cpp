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
    const ForeignTableSchema& schema) {
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
    const ForeignTableSchema& schema,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  import_buffer_vec.clear();
  for (const auto cd : schema.getLogicalAndPhysicalColumns()) {
    import_buffer_vec.emplace_back(new TypedImportBuffer(cd, loader->getStringDict(cd)));
  }
}

void initialize_row_group_metadata_vec(
    const size_t num_logical_and_physical_columns,
    std::vector<RowGroupMetadata>& row_group_metadata_vec) {
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

std::set<std::string> get_file_paths(std::shared_ptr<arrow::fs::FileSystem> file_system,
                                     const std::string& base_path) {
  auto timer = DEBUG_TIMER(__func__);
  std::set<std::string> file_paths;
  arrow::fs::FileSelector file_selector{};
  file_selector.base_dir = base_path;
  file_selector.recursive = true;

  auto file_info_result = file_system->GetFileInfo(file_selector);
  if (!file_info_result.ok()) {
    // This is expected when `base_path` points to a single file.
    file_paths.emplace(base_path);
  } else {
    auto& file_info_vector = file_info_result.ValueOrDie();
    for (const auto& file_info : file_info_vector) {
      if (file_info.type() == arrow::fs::FileType::File) {
        file_paths.emplace(file_info.path());
      }
    }
    if (file_paths.empty()) {
      throw std::runtime_error{"No file found at given path \"" + base_path + "\"."};
    }
  }
  return file_paths;
}

void update_array_metadata_stats(
    ArrayMetadataStats& array_stats,
    const ColumnDescriptor* column_descriptor,
    const parquet::ColumnDescriptor* parquet_column_descriptor,
    std::shared_ptr<parquet::Statistics> stats) {
  auto encoded_min = stats->EncodeMin();
  auto encoded_max = stats->EncodeMax();
  switch (parquet_column_descriptor->physical_type()) {
    case parquet::Type::BOOLEAN: {
      auto min_value = reinterpret_cast<const bool*>(encoded_min.data())[0];
      auto max_value = reinterpret_cast<const bool*>(encoded_max.data())[0];
      array_stats.updateStats(column_descriptor->columnType, min_value, max_value);
      break;
    }
    case parquet::Type::INT32: {
      auto min_value = reinterpret_cast<const int32_t*>(encoded_min.data())[0];
      auto max_value = reinterpret_cast<const int32_t*>(encoded_max.data())[0];
      array_stats.updateStats(column_descriptor->columnType, min_value, max_value);
      break;
    }
    case parquet::Type::INT64: {
      auto min_value = reinterpret_cast<const int64_t*>(encoded_min.data())[0];
      auto max_value = reinterpret_cast<const int64_t*>(encoded_max.data())[0];
      array_stats.updateStats(column_descriptor->columnType, min_value, max_value);
      break;
    }
    case parquet::Type::DOUBLE: {
      auto min_value = reinterpret_cast<const double*>(encoded_min.data())[0];
      auto max_value = reinterpret_cast<const double*>(encoded_max.data())[0];
      array_stats.updateStats(column_descriptor->columnType, min_value, max_value);
      break;
    }
    case parquet::Type::FLOAT: {
      auto min_value = reinterpret_cast<const float*>(encoded_min.data())[0];
      auto max_value = reinterpret_cast<const float*>(encoded_max.data())[0];
      array_stats.updateStats(column_descriptor->columnType, min_value, max_value);
      break;
    }
    default:
      throw std::runtime_error(
          "Unsupported physical type detected while"
          " scanning metadata of parquet column '" +
          parquet_column_descriptor->name() + "'.");
  }
}

void read_parquet_metadata_into_import_buffer(
    const size_t num_rows,
    const int row_group,
    const ForeignTableSchema& schema,
    const std::unique_ptr<parquet::RowGroupMetaData>& group_metadata,
    const ColumnDescriptor* column_descriptor,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec,
    ParquetLoaderMetadata& parquet_loader_metadata) {
  auto column_id = column_descriptor->columnId;
  auto& import_buffer = initialize_import_buffer(column_id, import_buffer_vec);
  auto parquet_column_index = schema.getParquetColumnIndex(column_id);
  auto column_chunk = group_metadata->ColumnChunk(parquet_column_index);
  CHECK(column_chunk->is_stats_set());
  std::shared_ptr<parquet::Statistics> stats = column_chunk->statistics();
  bool is_all_nulls = stats->null_count() == column_chunk->num_values();
  CHECK(is_all_nulls || stats->HasMinMax());

  const auto& type = column_descriptor->columnType;
  auto parquet_column_descriptor = group_metadata->schema()->Column(parquet_column_index);
  auto& row_group_metadata_vec = parquet_loader_metadata.row_group_metadata_vector;
  if (!is_all_nulls) {
    if (is_datetime(type.get_type()) ||
        (type.is_array() && is_datetime(type.get_elem_type().get_type()))) {
      if (type.is_array()) {
        auto sub_type_column_descriptor =
            get_sub_type_column_descriptor(column_descriptor);
        auto [min, max] = get_datetime_min_and_max(
            sub_type_column_descriptor.get(), parquet_column_descriptor, stats);
        auto& metadata = row_group_metadata_vec[column_id - 1];
        metadata.array_stats.updateStats(
            sub_type_column_descriptor->columnType, min, max);
      } else {
        auto [min, max] =
            get_datetime_min_and_max(column_descriptor, parquet_column_descriptor, stats);
        import_buffer->addBigint(min);
        import_buffer->addBigint(max);
      }
    } else if (type.is_decimal() ||
               (type.is_array() && type.get_elem_type().is_decimal())) {
      if (type.is_array()) {
        auto sub_type_column_descriptor =
            get_sub_type_column_descriptor(column_descriptor);
        auto [min, max] = get_decimal_min_and_max(
            sub_type_column_descriptor.get(), parquet_column_descriptor, stats);
        auto& metadata = row_group_metadata_vec[column_id - 1];
        metadata.array_stats.updateStats(
            sub_type_column_descriptor->columnType, min, max);
      } else {
        auto [min, max] =
            get_decimal_min_and_max(column_descriptor, parquet_column_descriptor, stats);
        import_buffer->addBigint(min);
        import_buffer->addBigint(max);
      }
    } else if (!type.is_string() && !type.is_varlen()) {
      std::shared_ptr<arrow::Scalar> min, max;
      PARQUET_THROW_NOT_OK(parquet::arrow::StatisticsAsScalars(*stats, &min, &max));
      ARROW_ASSIGN_OR_THROW(auto min_array, arrow::MakeArrayFromScalar(*min, 1));
      ARROW_ASSIGN_OR_THROW(auto max_array, arrow::MakeArrayFromScalar(*max, 1));
      import_buffer->add_arrow_values(
          column_descriptor, *min_array, false, {0, 1}, &bad_rows_tracker);
      import_buffer->add_arrow_values(
          column_descriptor, *max_array, false, {0, 1}, &bad_rows_tracker);
    } else if (type.is_array() && !type.get_elem_type().is_string()) {
      auto sub_type_column_descriptor = get_sub_type_column_descriptor(column_descriptor);
      update_array_metadata_stats(row_group_metadata_vec[column_id - 1].array_stats,
                                  sub_type_column_descriptor.get(),
                                  parquet_column_descriptor,
                                  stats);
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
    const ForeignTableSchema& schema,
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
    const ForeignTableSchema& schema,
    const Interval<ColumnType>& column_interval,
    import_export::Importer* importer,
    std::unique_ptr<parquet::arrow::FileReader>& reader,
    ParquetLoaderMetadata& parquet_loader_metadata,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  auto loader = importer->getLoader();

  // Initialize
  initialize_import_buffers_vec(loader, schema, import_buffer_vec);
  initialize_row_group_metadata_vec(schema.numLogicalAndPhysicalColumns(),
                                    parquet_loader_metadata.row_group_metadata_vector);
  initialize_bad_rows_tracker(
      bad_rows_tracker, row_group, parquet_loader_metadata.file_path, importer);

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
                                               parquet_loader_metadata);
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

void import_row_groups(
    const RowGroupInterval& row_group_interval,
    bool& load_failed,
    const bool metadata_scan,
    const ForeignTableSchema& schema,
    const Interval<ColumnType>& column_interval,
    import_export::Importer* importer,
    std::unique_ptr<parquet::arrow::FileReader>& reader,
    ParquetLoaderMetadata& parquet_loader_metadata,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  CHECK_LE(row_group_interval.start_index, row_group_interval.end_index);
  for (int row_group = row_group_interval.start_index;
       row_group <= row_group_interval.end_index && !load_failed;
       ++row_group) {
    import_row_group(row_group,
                     metadata_scan,
                     schema,
                     column_interval,
                     importer,
                     reader,
                     parquet_loader_metadata,
                     bad_rows_tracker,
                     import_buffer_vec);
  }
}

void validate_equal_schema(const parquet::arrow::FileReader* reference_file_reader,
                           const parquet::arrow::FileReader* new_file_reader,
                           const std::string& reference_file_path,
                           const std::string& new_file_path) {
  const auto reference_num_columns =
      reference_file_reader->parquet_reader()->metadata()->num_columns();
  const auto new_num_columns =
      new_file_reader->parquet_reader()->metadata()->num_columns();
  if (reference_num_columns != new_num_columns) {
    throw std::runtime_error{"Parquet file \"" + new_file_path +
                             "\" has a different schema. Please ensure that all Parquet "
                             "files use the same schema. Reference Parquet file: \"" +
                             reference_file_path + "\" has " +
                             std::to_string(reference_num_columns) +
                             " columns. New Parquet file \"" + new_file_path + "\" has " +
                             std::to_string(new_num_columns) + " columns."};
  }

  for (int i = 0; i < reference_num_columns; i++) {
    validate_equal_column_descriptor(get_column_descriptor(reference_file_reader, i),
                                     get_column_descriptor(new_file_reader, i),
                                     reference_file_path,
                                     new_file_path);
  }
}
}  // namespace

void LazyParquetImporter::metadataScan() {
  auto columns_interval =
      Interval<ColumnType>{schema_.getLogicalAndPhysicalColumns().front()->columnId,
                           schema_.getLogicalAndPhysicalColumns().back()->columnId};
  auto file_paths = get_file_paths(file_system_, base_path_);
  CHECK(!file_paths.empty());
  std::unique_ptr<parquet::arrow::FileReader> first_file_reader;
  const auto& first_file_path = *file_paths.begin();
  open_parquet_table(first_file_path, first_file_reader, file_system_);
  for (const auto& file_path : file_paths) {
    std::unique_ptr<parquet::arrow::FileReader> reader;
    open_parquet_table(file_path, reader, file_system_);
    validate_equal_schema(
        first_file_reader.get(), reader.get(), first_file_path, file_path);
    int num_row_groups = get_parquet_table_size(reader).first;
    auto row_group_interval = RowGroupInterval{file_path, 0, num_row_groups - 1};
    partialImport({row_group_interval}, columns_interval, true);
  }
}

void LazyParquetImporter::partialImport(
    const std::vector<RowGroupInterval>& row_group_intervals,
    const Interval<ColumnType>& column_interval,
    const bool is_metadata_scan) {
  CHECK_LE(column_interval.start, column_interval.end);
  for (const auto& row_group_interval : row_group_intervals) {
    std::unique_ptr<parquet::arrow::FileReader> reader;
    open_parquet_table(row_group_interval.file_path, reader, file_system_);
    validate_parquet_metadata(reader->parquet_reader()->metadata(), file_path, schema_);
    auto& import_buffers_vec = get_import_buffers_vec();
    import_buffers_vec.resize(1);
    std::vector<import_export::BadRowsTracker> bad_rows_trackers(1);
    parquet_loader_metadata_.file_path = row_group_interval.file_path;
    import_row_groups(row_group_interval,
                      load_failed,
                      is_metadata_scan,
                      schema_,
                      column_interval,
                      this,
                      reader,
                      parquet_loader_metadata_,
                      bad_rows_trackers[0],
                      import_buffers_vec[0]);
  }
}

LazyParquetImporter::LazyParquetImporter(
    import_export::Loader* provided_loader,
    const std::string& base_path,
    std::shared_ptr<arrow::fs::FileSystem> file_system,
    const import_export::CopyParams& copy_params,
    ParquetLoaderMetadata& parquet_loader_metadata,
    ForeignTableSchema& schema)
    : import_export::Importer(provided_loader, base_path, copy_params)
    , parquet_loader_metadata_(parquet_loader_metadata)
    , schema_(schema)
    , base_path_(base_path)
    , file_system_(file_system) {}
}  // namespace foreign_storage
