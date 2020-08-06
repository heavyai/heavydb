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

bool fragments_and_row_groups_align(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const int32_t fragment_size) {
  // Check to see that row groups and fragments are aligned
  using namespace parquet;
  long long running_total = 0;
  for (int r = 0; r < file_metadata->num_row_groups(); ++r) {
    std::unique_ptr<RowGroupMetaData> group_metadata = file_metadata->RowGroup(r);
    running_total += group_metadata->num_rows();
    if (running_total > fragment_size) {
      /*
       * This implies that a fragment boundary was skipped
       * NOTE: this may be salvagable for some row-groups, but for simplicity
       * set all columns as unsupported
       */
      return false;
    } else if (running_total == fragment_size) {
      /*
       * Row group aligns with fragment boundary
       */
      running_total = 0;
    }
  }
  return true;
}

std::unordered_set<int> get_columns_with_stats_available(
    const std::shared_ptr<parquet::FileMetaData>& file_metadata,
    const std::list<const ColumnDescriptor*>& columns) {
  // Check to see all row groups have stats set
  std::unordered_set<int> supported_columns;
  auto column_it = columns.begin();
  for (int i = 0; i < file_metadata->num_columns(); ++i, ++column_it) {
    std::string reason_for_not_supported = "";
    bool column_supported = false;
    const parquet::ColumnDescriptor* descr = file_metadata->schema()->Column(i);
    auto cd = *column_it;
    const auto type = cd->columnType.get_type();
    parquet::Type::type physical_type = descr->physical_type();
    auto logical_type = descr->logical_type();
    bool allowed_type =
        AllowedParquetMetadataTypeMappings::isColumnMappingSupported(cd, descr);
    if (allowed_type) {
      column_supported = true;
    } else {
      column_supported = false;
      reason_for_not_supported =
          "omnisci type, physical type and logical type combination unsupported.";
    }
    if (column_supported) {
      for (int r = 0; r < file_metadata->num_row_groups(); ++r) {
        std::unique_ptr<parquet::RowGroupMetaData> group_metadata =
            file_metadata->RowGroup(r);
        auto column_chunk = group_metadata->ColumnChunk(i);
        if (!column_chunk->is_stats_set()) {
          column_supported = false;
          reason_for_not_supported = "there exists a column chunk with unset statistics.";
          break;
        }
        auto stats = column_chunk->statistics();
        bool is_all_nulls = stats->null_count() == group_metadata->num_rows();
        if (!stats->HasMinMax() && !is_all_nulls) {
          column_supported = false;
          reason_for_not_supported =
              "there exists a column chunk with statistics without a min max set.";
          break;
        }
      }
    }
    if (column_supported) {
      supported_columns.insert(i);
      LOG(INFO) << "Column  " << i << " with name '" << cd->columnName
                << "' has omnisci type " << type_to_string(type)
                << " and foreign parquet physical type "
                << parquet::TypeToString(physical_type) << " and logical type "
                << logical_type->ToString() << " is SUPPORTED for a metadata only scan.";
    } else {
      LOG(INFO) << "Column  " << i << " with name '" << cd->columnName
                << "' has omnisci type " << type_to_string(type)
                << " and foreign parquet physical type "
                << parquet::TypeToString(physical_type) << " and logical type "
                << logical_type->ToString()
                << " is NOT SUPPORTED for a metadata only scan. The reason why is, "
                << reason_for_not_supported;
    }
  }
  return supported_columns;
}

std::unordered_set<int> get_fsi_supported_metadata_scan_columns(
    const TableDescriptor* table_desc,
    parquet::ParquetFileReader* file_reader,
    const std::list<const ColumnDescriptor*>& columns) {
  using namespace parquet;
  auto fragment_size = table_desc->maxFragRows;
  const auto& table_name = table_desc->tableName;
  auto file_metadata = file_reader->metadata();
  if (!fragments_and_row_groups_align(file_metadata, fragment_size)) {
    LOG(INFO) << "Metadata only scan for foreign table with name '" << table_name
              << "' is NOT SUPPORTED for any"
              << " column because parquet row group and omnisci fragment boundaries do "
                 "not coincide. "
              << "Consider setting fragment size to a multiple of row group size to "
                 "enable metadata only scan.";
    return std::unordered_set<int>{};
  }
  LOG(INFO) << "Metadata only scan for foreign table with name '" << table_name
            << "' is SUPPORTED.";
  return get_columns_with_stats_available(file_metadata, columns);
}

std::tuple<int, int> open_parquet_table(
    const std::string& file_path,
    std::unique_ptr<parquet::arrow::FileReader>& reader) {
  using namespace parquet::arrow;
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(file_path));
  PARQUET_THROW_NOT_OK(OpenFile(infile, arrow::default_memory_pool(), &reader));
  auto file_metadata = reader->parquet_reader()->metadata();
  const auto num_row_groups = file_metadata->num_row_groups();
  const auto num_columns = file_metadata->num_columns();
  return std::make_tuple(num_row_groups, num_columns);
}

struct ForeignTableSchema {
  std::list<const ColumnDescriptor*> logical_and_physical_columns;
  std::list<const ColumnDescriptor*> logical_columns;
  std::unordered_map<int, std::list<const ColumnDescriptor*>::iterator>
      logical_index_to_column_descriptor_it_map_;

  // TODO: remove tracking of geo columns once null geo-types are fixed
  std::vector<bool> logical_column_is_geo;

  const ColumnDescriptor* getColumnDescriptor(int logical_index) const {
    auto it = logical_index_to_column_descriptor_it_map_.find(logical_index);
    CHECK(it != logical_index_to_column_descriptor_it_map_.end());
    return *it->second;
  }

  void init() {
    auto logical_columns_it = logical_columns.begin();
    logical_column_is_geo.assign(logical_columns.size(), false);
    for (size_t logic_idx = 0; logic_idx < logical_columns.size();
         ++logic_idx, ++logical_columns_it) {
      logical_index_to_column_descriptor_it_map_[logic_idx] = logical_columns_it;
      if ((*logical_columns_it)->columnType.is_geometry()) {
        logical_column_is_geo[logic_idx] = true;
      }
    }
  }

  int numLogicalAndPhysicalColumns() const { return logical_and_physical_columns.size(); }

  int numLogicalColumns() const { return logical_columns.size(); }
};

ForeignTableSchema validate_schema(import_export::Loader* loader,
                                   const int parquet_num_columns,
                                   const std::string& file_path) {
  ForeignTableSchema schema;
  auto& catalog = loader->getCatalog();
  schema.logical_and_physical_columns = catalog.getAllColumnMetadataForTableUnlocked(
      loader->getTableDesc()->tableId, false, false, true);
  schema.logical_columns = catalog.getAllColumnMetadataForTableUnlocked(
      loader->getTableDesc()->tableId, false, false, false);
  if (schema.logical_columns.size() != static_cast<size_t>(parquet_num_columns)) {
    std::stringstream error_msg;
    error_msg << "Mismatched number of logical columns in table '"
              << loader->getTableDesc()->tableName << "' ("
              << schema.logical_columns.size() << " columns) with parquet file '"
              << file_path << "' (" << parquet_num_columns << " columns.)";
    throw std::runtime_error(error_msg.str());
  }
  schema.init();
  return schema;
}

void inititialize_bad_rows_tracker(import_export::BadRowsTracker& bad_rows_tracker,
                                   const int row_group,
                                   const std::string& file_path,
                                   import_export::Importer* importer) {
  bad_rows_tracker.rows.clear();
  bad_rows_tracker.nerrors = 0;
  bad_rows_tracker.file_name = file_path;
  bad_rows_tracker.row_group = row_group;
  bad_rows_tracker.importer = importer;
}

void inititialize_import_buffers_vec(
    const import_export::Loader* loader,
    const ForeignTableSchema& schema,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  import_buffer_vec.clear();
  for (const auto cd : schema.logical_and_physical_columns) {
    import_buffer_vec.emplace_back(new TypedImportBuffer(cd, loader->getStringDict(cd)));
  }
}

void inititialize_row_group_metadata_vec(
    const size_t num_logical_and_physical_columns,
    LazyParquetImporter::RowGroupMetadataVector& row_group_metadata_vec) {
  row_group_metadata_vec.clear();
  row_group_metadata_vec.resize(num_logical_and_physical_columns);
}

std::unique_ptr<import_export::TypedImportBuffer>& initialize_import_buffer(
    const size_t physical_idx,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  auto& import_buffer = import_buffer_vec[physical_idx];
  import_buffer->import_buffers = &import_buffer_vec;
  import_buffer->col_idx = physical_idx + 1;
  return import_buffer;
}

void read_parquet_metadata_into_import_buffer(
    const size_t array_size,
    const int logical_idx,
    const int physical_idx,
    const std::unique_ptr<parquet::RowGroupMetaData>& group_metadata,
    const ColumnDescriptor* column_desciptor,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec,
    RowGroupMetadata& metadata) {
  metadata.metadata_only = true;
  auto& import_buffer = initialize_import_buffer(physical_idx, import_buffer_vec);
  auto column_chunk = group_metadata->ColumnChunk(logical_idx);
  CHECK(column_chunk->is_stats_set());
  std::shared_ptr<parquet::Statistics> stats = column_chunk->statistics();
  bool is_all_nulls = stats->null_count() == group_metadata->num_rows();
  CHECK(is_all_nulls || stats->HasMinMax());
  if (!is_all_nulls) {
    std::shared_ptr<arrow::Scalar> min, max;
    PARQUET_THROW_NOT_OK(parquet::arrow::StatisticsAsScalars(*stats, &min, &max));
    ARROW_ASSIGN_OR_THROW(auto min_array, arrow::MakeArrayFromScalar(*min, 1));
    ARROW_ASSIGN_OR_THROW(auto max_array, arrow::MakeArrayFromScalar(*max, 1));
    import_buffer->add_arrow_values(
        column_desciptor, *min_array, false, {0, 1}, &bad_rows_tracker);
    import_buffer->add_arrow_values(
        column_desciptor, *max_array, false, {0, 1}, &bad_rows_tracker);
  }
  metadata.is_all_nulls = is_all_nulls;
  metadata.has_nulls = stats->null_count() > 0;
  metadata.num_elements = array_size;
}

void read_parquet_data_into_import_buffer(
    const import_export::Loader* loader,
    const int row_group,
    const int logical_idx,
    const int physical_idx,
    const ForeignTableSchema& schema,
    const ColumnDescriptor* column_desciptor,
    std::unique_ptr<parquet::arrow::FileReader>& reader,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec,
    RowGroupMetadata& metadata) {
  metadata.metadata_only = false;
  std::shared_ptr<arrow::ChunkedArray> array;
  PARQUET_THROW_NOT_OK(reader->RowGroup(row_group)->Column(logical_idx)->Read(&array));
  auto& import_buffer = initialize_import_buffer(physical_idx, import_buffer_vec);
  auto num_bad_rows_before_add_values = bad_rows_tracker.rows.size();
  // TODO: the following async call & wait suppresses a false-positive
  // lock-order-inversion warning from TSAN; remove its use once
  // TypedImportBuffer is refactored
  std::async(std::launch::async, [&] {
    for (auto chunk : array->chunks()) {
      import_buffer->add_arrow_values(
          column_desciptor, *chunk, false, {0, chunk->length()}, &bad_rows_tracker);
    }
  }).wait();
  // TODO: remove this check for failure to import geo-type once null
  // geo-types are fixed
  if (schema.logical_column_is_geo[logical_idx] &&
      bad_rows_tracker.rows.size() > num_bad_rows_before_add_values) {
    std::stringstream error_msg;
    error_msg << "Failure to import geo column '" << column_desciptor->columnName
              << "' in table '" << loader->getTableDesc()->tableName << "' for row group "
              << row_group << " and row " << *bad_rows_tracker.rows.rbegin() << ".";
    throw std::runtime_error(error_msg.str());
  }
}

void import_row_group(
    const int row_group,
    const bool metadata_scan,
    const ForeignTableSchema& schema,
    const std::string& file_path,
    const Interval<ColumnType>& validated_logical_column_interval,
    const std::unordered_set<int>& metadata_scan_supported_columns,
    ForeignTableColumnMap& foreign_table_column_map,
    import_export::Importer* importer,
    std::unique_ptr<parquet::arrow::FileReader>& reader,
    LazyParquetImporter::RowGroupMetadataVector& row_group_metadata_vec,
    import_export::BadRowsTracker& bad_rows_tracker,
    std::vector<std::unique_ptr<TypedImportBuffer>>& import_buffer_vec) {
  auto loader = importer->getLoader();

  // Initialize
  inititialize_import_buffers_vec(loader, schema, import_buffer_vec);
  inititialize_row_group_metadata_vec(schema.numLogicalAndPhysicalColumns(),
                                      row_group_metadata_vec);
  inititialize_bad_rows_tracker(bad_rows_tracker, row_group, file_path, importer);

  // Read metadata
  auto file_metadata = reader->parquet_reader()->metadata();
  std::unique_ptr<parquet::RowGroupMetaData> group_metadata =
      file_metadata->RowGroup(row_group);
  const size_t array_size = group_metadata->num_rows();

  // Process each logical column into corresponding import buffer
  for (int logical_idx = validated_logical_column_interval.start;
       logical_idx <= validated_logical_column_interval.end;
       ++logical_idx) {
    const auto physical_idx = foreign_table_column_map.getPhysicalIndex(logical_idx);
    const auto column_desciptor = schema.getColumnDescriptor(logical_idx);

    // Set basic metadata for import
    auto& metadata = row_group_metadata_vec[physical_idx];
    metadata.row_group_index = row_group;

    // Branch based on whether this is a metadata scan & if the logical column
    // is supported for a metadata only scan
    if (metadata_scan && metadata_scan_supported_columns.find(logical_idx) !=
                             metadata_scan_supported_columns.end()) {
      // Parquet metadata only import branch
      read_parquet_metadata_into_import_buffer(array_size,
                                               logical_idx,
                                               physical_idx,
                                               group_metadata,
                                               column_desciptor,
                                               bad_rows_tracker,
                                               import_buffer_vec,
                                               metadata);
    } else {
      // Regular import branch, may also be used to compute missing or
      // unsupported parquet metadata
      read_parquet_data_into_import_buffer(loader,
                                           row_group,
                                           logical_idx,
                                           physical_idx,
                                           schema,
                                           column_desciptor,
                                           reader,
                                           bad_rows_tracker,
                                           import_buffer_vec,
                                           metadata);
    }
  }
  importer->load(import_buffer_vec, array_size);
}

}  // namespace

void LazyParquetImporter::partialImport(
    const Interval<RowGroupType>& row_group_interval,
    const Interval<ColumnType>& logical_column_interval,
    const bool metadata_scan) {
  using namespace import_export;

  std::unique_ptr<parquet::arrow::FileReader> reader;
  int num_row_groups, num_columns;
  std::tie(num_row_groups, num_columns) = open_parquet_table(file_path, reader);

  const auto& loader = getLoader();

  auto schema = validate_schema(loader, num_columns, file_path);

  // Get metadata supported columns
  auto metadata_scan_supported_columns =
      metadata_scan
          ? get_fsi_supported_metadata_scan_columns(
                loader->getTableDesc(), reader->parquet_reader(), schema.logical_columns)
          : std::unordered_set<int>{};

  // Check which row groups and columns to import
  auto row_groups_to_import_interval =
      metadata_scan ? Interval<RowGroupType>{0, num_row_groups - 1} : row_group_interval;
  auto logical_columns_to_import_interval =
      metadata_scan ? Interval<ColumnType>{0, num_columns - 1} : logical_column_interval;

  auto& import_buffers_vec = get_import_buffers_vec();
  import_buffers_vec.resize(1);
  std::vector<BadRowsTracker> bad_rows_trackers(1);
  for (int row_group = row_groups_to_import_interval.start;
       row_group <= row_groups_to_import_interval.end && !load_failed;
       ++row_group) {
    import_row_group(row_group,
                     metadata_scan,
                     schema,
                     file_path,
                     logical_columns_to_import_interval,
                     metadata_scan_supported_columns,
                     foreign_table_column_map_,
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
                                         RowGroupMetadataVector& metadata_vector)
    : import_export::Importer(provided_loader, file_name, copy_params)
    , row_group_metadata_vec_(metadata_vector)
    , foreign_table_column_map_(
          &provided_loader->getCatalog(),
          dynamic_cast<const ForeignTable*>(provided_loader->getTableDesc())) {}

}  // namespace foreign_storage
