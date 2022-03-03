/*
 * Copyright 2019 OmniSci, Inc.
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

/*
 * @file CopyParams.h
 * @author Mehmet Sariyuce <mehmet.sariyuce@omnisci.com>
 * @brief CopyParams struct
 */

#pragma once

#include <string>

#include "ImportExport/SourceType.h"
#include "Shared/sqltypes.h"

namespace import_export {

// not too big (need much memory) but not too small (many thread forks)
constexpr static size_t kImportFileBufferSize = (1 << 23);

// import buffers may grow to this size if necessary
constexpr static size_t max_import_buffer_resize_byte_size = 1024 * 1024 * 1024;

enum class ImportHeaderRow { kAutoDetect, kNoHeader, kHasHeader };
enum class RasterPointType { kNone, kAuto, kSmallInt, kInt, kFloat, kDouble, kPoint };
enum class RasterPointTransform { kNone, kAuto, kFile, kWorld };

struct CopyParams {
  char delimiter;
  std::string null_str;
  ImportHeaderRow has_header;
  bool quoted;  // does the input have any quoted fields, default to false
  char quote;
  char escape;
  char line_delim;
  char array_delim;
  char array_begin;
  char array_end;
  int threads;
  size_t
      max_reject;  // maximum number of records that can be rejected before copy is failed
  import_export::SourceType source_type;
  bool plain_text = false;
  // s3/parquet related params
  std::string s3_access_key;  // per-query credentials to override the
  std::string s3_secret_key;  // settings in ~/.aws/credentials or environment
  std::string s3_session_token = "";
  std::string s3_region;
  std::string s3_endpoint;
  int32_t s3_max_concurrent_downloads =
      8;  // maximum number of concurrent file downloads from S3
  // kafka related params
  size_t retry_count;
  size_t retry_wait;
  size_t batch_size;
  size_t buffer_size;
  // geospatial params
  bool lonlat;
  EncodingType geo_coords_encoding;
  int32_t geo_coords_comp_param;
  SQLTypes geo_coords_type;
  int32_t geo_coords_srid;
  bool sanitize_column_names;
  std::string geo_layer_name;
  bool geo_assign_render_groups;
  bool geo_explode_collections;
  int32_t source_srid;
  std::optional<std::string> regex_path_filter;
  std::optional<std::string> file_sort_order_by;
  std::optional<std::string> file_sort_regex;
  RasterPointType raster_point_type;
  std::string raster_import_bands;
  int32_t raster_scanlines_per_thread;
  RasterPointTransform raster_point_transform;
  bool raster_point_compute_angle;
  std::string raster_import_dimensions;
  std::string add_metadata_columns;
  // odbc parameters
  std::string odbc_sql_select;
  // odbc user mapping parameters
  std::string odbc_username;
  std::string odbc_password;
  std::string odbc_credential_string;
  // odbc server parameters
  std::string odbc_dsn;
  std::string odbc_connection_string;
  // regex parameters
  std::string line_start_regex;
  std::string line_regex;

  CopyParams()
      : delimiter(',')
      , null_str("\\N")
      , has_header(ImportHeaderRow::kAutoDetect)
      , quoted(true)
      , quote('"')
      , escape('"')
      , line_delim('\n')
      , array_delim(',')
      , array_begin('{')
      , array_end('}')
      , threads(0)
      , max_reject(100000)
      , source_type(import_export::SourceType::kDelimitedFile)
      , retry_count(100)
      , retry_wait(5)
      , batch_size(1000)
      , buffer_size(kImportFileBufferSize)
      , lonlat(true)
      , geo_coords_encoding(kENCODING_GEOINT)
      , geo_coords_comp_param(32)
      , geo_coords_type(kGEOMETRY)
      , geo_coords_srid(4326)
      , sanitize_column_names(true)
      , geo_assign_render_groups(true)
      , geo_explode_collections(false)
      , source_srid(0)
      , raster_point_type(RasterPointType::kAuto)
      , raster_scanlines_per_thread(32)
      , raster_point_transform(RasterPointTransform::kAuto)
      , raster_point_compute_angle{false} {}

  CopyParams(char d, const std::string& n, char l, size_t b, size_t retries, size_t wait)
      : delimiter(d)
      , null_str(n)
      , has_header(ImportHeaderRow::kAutoDetect)
      , quoted(true)
      , quote('"')
      , escape('"')
      , line_delim(l)
      , array_delim(',')
      , array_begin('{')
      , array_end('}')
      , threads(0)
      , max_reject(100000)
      , source_type(import_export::SourceType::kDelimitedFile)
      , retry_count(retries)
      , retry_wait(wait)
      , batch_size(b)
      , buffer_size(kImportFileBufferSize)
      , lonlat(true)
      , geo_coords_encoding(kENCODING_GEOINT)
      , geo_coords_comp_param(32)
      , geo_coords_type(kGEOMETRY)
      , geo_coords_srid(4326)
      , sanitize_column_names(true)
      , geo_assign_render_groups(true)
      , geo_explode_collections(false)
      , source_srid(0)
      , raster_point_type(RasterPointType::kAuto)
      , raster_scanlines_per_thread(32)
      , raster_point_transform(RasterPointTransform::kAuto)
      , raster_point_compute_angle{false} {}
};

}  // namespace import_export
