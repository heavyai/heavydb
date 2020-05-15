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
#include "Import/Importer.h"
namespace foreign_storage {
struct FileRegion {
  std::string filename;
  // Offset into file
  size_t first_row_file_offset;
  // Size of region in bytes
  size_t region_size;
};

using FileRegions = std::vector<FileRegion>;

// Reads
class CsvLazyLoader : public Importer_NS::Importer {
 public:
  CsvLazyLoader(Importer_NS::Loader* providedLoader,
                const std::string& file,
                const Importer_NS::CopyParams& p);

  void scanMetadata();

  void fetchRegions(const FileRegions& file_regions);

 private:
  static void loadFileRegions(const FileRegions& file_regions,
                              const size_t start_index,
                              const size_t end_index,
                              const size_t thread_id,
                              FILE* file,
                              std::mutex& file_access_mutex,
                              Importer_NS::Importer* importer);
  std::mutex file_access_mutex_;
};
}  // namespace foreign_storage
