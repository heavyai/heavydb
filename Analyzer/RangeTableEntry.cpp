/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RangeTableEntry.h"

#include <Catalog/Catalog.h>

namespace Analyzer {

RangeTableEntry::~RangeTableEntry() {
  if (view_query != nullptr) {
    delete view_query;
  }
}

const std::list<const ColumnDescriptor*>& RangeTableEntry::get_column_descs() const {
  return column_descs;
}

int32_t RangeTableEntry::get_table_id() const {
  return table_desc->tableId;
}

const std::string& RangeTableEntry::get_table_name() const {
  return table_desc->tableName;
}

const TableDescriptor* RangeTableEntry::get_table_desc() const {
  return table_desc;
}

void RangeTableEntry::add_all_column_descs(const Catalog_Namespace::Catalog& catalog) {
  column_descs =
      catalog.getAllColumnMetadataForTable(table_desc->tableId, true, true, true);
}

void RangeTableEntry::expand_star_in_targetlist(
    const Catalog_Namespace::Catalog& catalog,
    std::vector<std::shared_ptr<TargetEntry>>& tlist,
    int rte_idx) {
  column_descs =
      catalog.getAllColumnMetadataForTable(table_desc->tableId, false, true, true);
  for (auto col_desc : column_descs) {
    auto cv = makeExpr<ColumnVar>(
        col_desc->columnType,
        shared::ColumnKey{
            catalog.getDatabaseId(), table_desc->tableId, col_desc->columnId},
        rte_idx);
    auto tle = std::make_shared<TargetEntry>(col_desc->columnName, cv, false);
    tlist.push_back(tle);
  }
}

const ColumnDescriptor* RangeTableEntry::get_column_desc(
    const Catalog_Namespace::Catalog& catalog,
    const std::string& name) {
  for (auto cd : column_descs) {
    if (cd->columnName == name) {
      return cd;
    }
  }
  const ColumnDescriptor* cd = catalog.getMetadataForColumn(table_desc->tableId, name);
  if (cd != nullptr) {
    column_descs.push_back(cd);
  }
  return cd;
}

}  // namespace Analyzer
