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

#include "LazyLoader.h"
#include "Shared/mapd_glob.h"

namespace foreign_storage {

// Performs metadata scan and lazy loading operations on CSV files
LazyLoader::LazyLoader(Loader* providedLoader,
                       const std::string& file,
                       const CopyParams& p)
    : importer(providedLoader, file, p), file_path(file) {
  auto file_paths = mapd_glob(file_path);
  // These limitations will be removed before productization
  if (file_paths.size() > 1) {
    throw std::runtime_error(
        "Only single file CSV archives supported for Foreign Storage Interface.");
  }
  if (boost::filesystem::extension(file_path) != ".csv") {
    throw std::runtime_error(
        "Only uncompressed CSV archives supported for Foreign Storage");
  }
};

// Iterates over entire file and returns file offsets and import buffers
void LazyLoader::scanMetadata() {
  importer.loadDelimited(file_path, true, true, false, {});
}

// Loads given file regions
void LazyLoader::fetchRegions(const FileRegions& file_regions) {
  importer.loadDelimited(file_path, true, true, true, file_regions);
}
}  // namespace foreign_storage
