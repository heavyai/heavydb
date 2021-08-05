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

#include <ImportExport/QueryExporter.h>

#include <boost/algorithm/string.hpp>

#include <ImportExport/QueryExporterCSV.h>
#include <ImportExport/QueryExporterGDAL.h>

namespace import_export {

QueryExporter::QueryExporter(const FileType file_type) : file_type_{file_type} {}

std::unique_ptr<QueryExporter> QueryExporter::create(FileType file_type) {
  switch (file_type) {
    case FileType::kCSV:
      return std::make_unique<QueryExporterCSV>();
    case FileType::kGeoJSON:
    case FileType::kGeoJSONL:
    case FileType::kShapefile:
    case FileType::kFlatGeobuf:
      return std::make_unique<QueryExporterGDAL>(file_type);
  }
  CHECK(false);
  return nullptr;
}

void QueryExporter::validateFileExtensions(
    const std::string& file_path,
    const std::string& file_type,
    const std::unordered_set<std::string>& valid_extensions) const {
  auto extension = boost::algorithm::to_lower_copy(
      boost::filesystem::path(file_path).extension().string());
  if (valid_extensions.find(extension) == valid_extensions.end()) {
    throw std::runtime_error("Invalid file extension '" + extension +
                             "' for file type '" + file_type + "'");
  }
}

std::string QueryExporter::safeColumnName(const std::string& resname,
                                          const int column_index) {
  if (resname.size() == 0) {
    return "result_" + std::to_string(column_index);
  }
  return resname;
}

}  // namespace import_export
