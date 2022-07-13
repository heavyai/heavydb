/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "ImportExport/Importer.h"
#include "LeafAggregator.h"

class DistributedLoader : public import_export::Loader {
 public:
  DistributedLoader(const Catalog_Namespace::SessionInfo& parent_session_info,
                    const TableDescriptor* t,
                    LeafAggregator* aggregator)
      : Loader(parent_session_info.getCatalog(), t) {
    CHECK(false);
  }

  bool load(const std::vector<std::unique_ptr<import_export::TypedImportBuffer>>&
                import_buffers,
            const size_t row_count,
            const Catalog_Namespace::SessionInfo* session_info) override {
    CHECK(false);
    return false;
  }
};

#endif  // DISTRIBUTEDLOADER_H
