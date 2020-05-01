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

using Importer_NS::CopyParams;
using Importer_NS::FileRegions;
using Importer_NS::Loader;

// Reads
class LazyLoader {
 public:
  LazyLoader(Loader* providedLoader, const std::string& file, const CopyParams& p);

  void scanMetadata();

  void fetchRegions(const FileRegions& file_regions);

 private:
  Importer_NS::Importer importer;
  std::string file_path;
};
}  // namespace foreign_storage
