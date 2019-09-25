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

#include "Shared/sqltypes.h"

namespace Importer_NS {

// not too big (need much memory) but not too small (many thread forks)
constexpr static size_t kImportFileBufferSize = (1 << 23);

enum class FileType {
  DELIMITED,
  POLYGON
#ifdef ENABLE_IMPORT_PARQUET
  ,
  PARQUET
#endif
};

enum class ImportHeaderRow { AUTODETECT, NO_HEADER, HAS_HEADER };

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
  FileType file_type;
  bool plain_text = false;
  // s3/parquet related params
  std::string s3_access_key;  // per-query credentials to override the
  std::string s3_secret_key;  // settings in ~/.aws/credentials or environment
  std::string s3_region;
  std::string s3_endpoint;
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

  CopyParams()
      : delimiter(',')
      , null_str("\\N")
      , has_header(ImportHeaderRow::AUTODETECT)
      , quoted(true)
      , quote('"')
      , escape('"')
      , line_delim('\n')
      , array_delim(',')
      , array_begin('{')
      , array_end('}')
      , threads(0)
      , max_reject(100000)
      , file_type(FileType::DELIMITED)
      , retry_count(100)
      , retry_wait(5)
      , batch_size(1000)
      , buffer_size(kImportFileBufferSize)
      , lonlat(true)
      , geo_coords_encoding(kENCODING_GEOINT)
      , geo_coords_comp_param(32)
      , geo_coords_type(kGEOMETRY)
      , geo_coords_srid(4326)
      , sanitize_column_names(true) {}

  CopyParams(char d, const std::string& n, char l, size_t b, size_t retries, size_t wait)
      : delimiter(d)
      , null_str(n)
      , has_header(ImportHeaderRow::AUTODETECT)
      , quoted(true)
      , quote('"')
      , escape('"')
      , line_delim(l)
      , array_delim(',')
      , array_begin('{')
      , array_end('}')
      , threads(0)
      , max_reject(100000)
      , file_type(FileType::DELIMITED)
      , retry_count(retries)
      , retry_wait(wait)
      , batch_size(b)
      , buffer_size(kImportFileBufferSize)
      , lonlat(true)
      , geo_coords_encoding(kENCODING_GEOINT)
      , geo_coords_comp_param(32)
      , geo_coords_type(kGEOMETRY)
      , geo_coords_srid(4326)
      , sanitize_column_names(true) {}
};
}  // namespace Importer_NS
