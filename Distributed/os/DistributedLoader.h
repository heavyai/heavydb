/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef DISTRIBUTEDLOADER_H
#define DISTRIBUTEDLOADER_H

#include "Import/Importer.h"
#include "LeafAggregator.h"

class DistributedLoader : public Importer_NS::Loader {
 public:
  DistributedLoader(const Catalog_Namespace::SessionInfo& parent_session_info,
                    const TableDescriptor* t,
                    LeafAggregator* aggregator)
      : Loader(parent_session_info.getCatalog(), t) {
    CHECK(false);
  }

  bool load(
      const std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>& import_buffers,
      size_t row_count) override {
    CHECK(false);
    return false;
  }
};

#endif  // DISTRIBUTEDLOADER_H
