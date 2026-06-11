/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <vector>

#include "Catalog/ForeignTable.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ForeignDataWrapper.h"
#include "InternalSystemDataWrapper.h"

namespace foreign_storage {

class InternalMemoryStatsDataWrapper : public InternalSystemDataWrapper {
 public:
  InternalMemoryStatsDataWrapper();

  InternalMemoryStatsDataWrapper(const int db_id, const ForeignTable* foreign_table);

 private:
  void initializeObjectsForTable(const std::string& table_name) override;

  void populateChunkBuffersForTable(
      const std::string& table_name,
      std::map<std::string, import_export::UnmanagedTypedImportBuffer*>& import_buffers)
      override;

  std::map<std::string, std::vector<Buffer_Namespace::MemoryInfo>>
      memory_info_by_device_type_;
};
}  // namespace foreign_storage
