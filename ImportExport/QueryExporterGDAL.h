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
  ~QueryExporterGDAL();

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
