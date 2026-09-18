/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/multi_index/mem_fun.hpp>

#include "QueryRenderer/Interface/RenderQueryInterfaceDeclarations.h"
#include "QueryRenderer/Interface/SqlSelectedTableInfo.h"
#include "Shared/DbObjectKeys.h"
#include "Shared/FullyQualifiedTableName.h"

namespace QueryRenderer {

struct HitTestTableInfo {
  const SelectedTableInfo& table_info;
  std::unordered_map<std::string, int64_t> col_rowid_map;

  inline const shared::TableKey& getTableKeyRef() const { return table_info.table_key; }
  inline const shared::FullyQualifiedTableName& getTableNameRef() const {
    return table_info.table_name;
  }
  inline bool hasColumn(const std::string& col) const {
    return table_info.hasColumn(col);
  }

  struct KeyTag;
  struct NameTag;
};

using HitTestTableContainer = ::boost::multi_index_container<
    HitTestTableInfo,
    ::boost::multi_index::indexed_by<
        ::boost::multi_index::random_access<>,

        ::boost::multi_index::hashed_unique<
            ::boost::multi_index::tag<HitTestTableInfo::KeyTag>,
            ::boost::multi_index::const_mem_fun<HitTestTableInfo,
                                                const shared::TableKey&,
                                                &HitTestTableInfo::getTableKeyRef>>,

        ::boost::multi_index::hashed_unique<
            ::boost::multi_index::tag<HitTestTableInfo::NameTag>,
            ::boost::multi_index::const_mem_fun<HitTestTableInfo,
                                                const shared::FullyQualifiedTableName&,
                                                &HitTestTableInfo::getTableNameRef>>>>;

struct HitTestCacheResults {
  std::string query_str;
  const ResultSet* result_set_ptr;
  const TargetEntries& result_targets;

  HitTestTableContainer hittest_table_info;
  bool is_projection_query;
};

struct OffsetAllRowIds {
  const int64_t rowid_offset = 0;

  inline void operator()(HitTestTableInfo& table_info) {
    std::for_each(table_info.col_rowid_map.begin(),
                  table_info.col_rowid_map.end(),
                  [&](auto& rowid_item) { rowid_item.second += rowid_offset; });
  }
};

}  // namespace QueryRenderer
