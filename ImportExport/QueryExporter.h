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

#include <Distributed/AggregatedResult.h>
#include <ImportExport/CopyParams.h>

#include <string>
#include <unordered_set>

namespace import_export {

class QueryExporter {
 public:
  enum class FileType { kCSV, kGeoJSON, kGeoJSONL, kShapefile, kFlatGeobuf };
  enum class FileCompression { kNone, kGZip, kZip };
  enum class ArrayNullHandling {
    kAbortWithWarning,
    kExportSentinels,
    kExportZeros,
    kNullEntireField
  };

  explicit QueryExporter(const FileType file_type);
  QueryExporter() = delete;
  virtual ~QueryExporter() {}

  static std::unique_ptr<QueryExporter> create(const FileType file_type);

  virtual void beginExport(const std::string& file_path,
                           const std::string& layer_name,
                           const CopyParams& copy_params,
                           const std::vector<TargetMetaInfo>& column_info,
                           const FileCompression file_compression,
                           const ArrayNullHandling array_null_handling) = 0;
  virtual void exportResults(const std::vector<AggregatedResult>& query_results) = 0;
  virtual void endExport() = 0;

 protected:
  const FileType file_type_;

  void validateFileExtensions(
      const std::string& file_path,
      const std::string& file_type,
      const std::unordered_set<std::string>& valid_extensions) const;
  std::string safeColumnName(const std::string& resname, const int column_index);
};

}  // namespace import_export
