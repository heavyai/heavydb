/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <fstream>

#include <ImportExport/QueryExporter.h>

namespace import_export {

class QueryExporterCSV : public QueryExporter {
 public:
  QueryExporterCSV();
  ~QueryExporterCSV() override;

  void beginExport(const std::string& file_path,
                   const std::string& layer_name,
                   const CopyParams& copy_params,
                   const std::vector<TargetMetaInfo>& column_infos,
                   const FileCompression file_compression,
                   const ArrayNullHandling array_null_handling) final;
  void exportResults(const std::vector<AggregatedResult>& query_results) final;
  void endExport() final;

 private:
  std::ofstream outfile_;
  CopyParams copy_params_;
  std::string file_path_;
  FileCompression file_compression_;
};

}  // namespace import_export
