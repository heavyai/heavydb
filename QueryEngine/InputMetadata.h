/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_map>

#include "QueryEngine/Descriptors/InputDescriptors.h"
#include "QueryEngine/RelAlgExecutionUnit.h"
#include "Shared/DbObjectKeys.h"

namespace Catalog_Namespace {
class Catalog;
}  // namespace Catalog_Namespace

class Executor;

using TemporaryTables = std::unordered_map<int, const ResultSetPtr&>;

struct InputTableInfo {
  shared::TableKey table_key;
  Fragmenter_Namespace::TableInfo info;
};

class InputTableInfoCache {
 public:
  InputTableInfoCache(Executor* executor);
  Fragmenter_Namespace::TableInfo getTableInfo(const shared::TableKey& table_key);
  void updateTableInfo(const shared::TableKey& table_key,
                       const Fragmenter_Namespace::TableInfo& table_info,
                       const std::set<int>& device_ids_to_use);
  void clear();

 private:
  std::unordered_map<shared::TableKey, Fragmenter_Namespace::TableInfo> cache_;
  Executor* executor_;
};

ChunkMetadataMap synthesize_metadata(const ResultSet* rows);

size_t get_frag_count_of_table(const shared::TableKey& table_key, Executor* executor);

std::vector<InputTableInfo> get_table_infos(
    const std::vector<InputDescriptor>& input_descs,
    Executor* executor);

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit,
                                            Executor* executor);

Fragmenter_Namespace::TableInfo build_table_info(
    const std::vector<const TableDescriptor*>& shard_tables);
