/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ImportExport/CopyParams.h>
#include "QueryEngine/AggregatedResult.h"

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

  static std::string formatTemporal(const SQLTypeInfo& type_info, int64_t unix_time);

 protected:
  const FileType file_type_;

  void validateFileExtensions(
      const std::string& file_path,
      const std::string& file_type,
      const std::unordered_set<std::string>& valid_extensions) const;
  std::string safeColumnName(const std::string& resname, const int column_index);
};

}  // namespace import_export
