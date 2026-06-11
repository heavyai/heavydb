/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
