/*
 * Copyright 2019 OmniSci, Inc.
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
    auto cv = makeExpr<ColumnVar>(col_desc->columnType,
                                  table_desc->tableId,
                                  col_desc->columnId,
                                  rte_idx,
                                  col_desc->isVirtualCol);
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
