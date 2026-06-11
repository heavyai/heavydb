/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ImportExport/QueryExporter.h>

#include <string>
#include <vector>

// forward declare instead of #include GDAL header
// to avoid double def of PACKAGE_NAME etc.
class GDALDataset;
class OGRLayer;

namespace import_export {

class QueryExporterGDAL : public QueryExporter {
 public:
  explicit QueryExporterGDAL(const FileType file_type);
  QueryExporterGDAL() = delete;
  ~QueryExporterGDAL() override;

  void beginExport(const std::string& file_path,
                   const std::string& layer_name,
                   const CopyParams& copy_params,
                   const std::vector<TargetMetaInfo>& column_infos,
                   const FileCompression file_compression,
                   const ArrayNullHandling array_null_handling) final;
  void exportResults(const std::vector<AggregatedResult>& query_results) final;
  void endExport() final;

 private:
  CopyParams copy_params_;
  GDALDataset* gdal_dataset_;
  OGRLayer* ogr_layer_;
  std::vector<int> field_indices_;
  ArrayNullHandling array_null_handling_;

  void cleanUp();
};

}  // namespace import_export
