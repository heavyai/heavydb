/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <vector>

#include "QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.h"

#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ForeignDataWrapper.h"
#include "InternalSystemDataWrapper.h"

namespace foreign_storage {

class InternalExecutorStatsDataWrapper : public InternalSystemDataWrapper {
 public:
  InternalExecutorStatsDataWrapper();

  InternalExecutorStatsDataWrapper(const int db_id, const ForeignTable* foreign_table);

 private:
  void initializeObjectsForTable(const std::string& table_name) override;

  void populateChunkBuffersForTable(
      const std::string& table_name,
      std::map<std::string, import_export::UnmanagedTypedImportBuffer*>& import_buffers)
      override;

  ExecutorResourceMgr_Namespace::ResourcePoolInfo executor_resource_pool_info_;
};
}  // namespace foreign_storage
