/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Interface/SqlSelectedTableInfo.h"

#include <numeric>

#include "Analyzer/Analyzer.h"
#include "Catalog/Catalog.h"

namespace QueryRenderer {

namespace {

int64_t get_total_num_rows_for_table(const Catalog_Namespace::Catalog& cat,
                                     const TableDescriptor& td) {
  const auto physical_tds = cat.getPhysicalTablesDescriptors(&td);
  return std::accumulate(
      physical_tds.begin(),
      physical_tds.end(),
      0ll,
      [](int64_t curr_val, auto const* physical_td) {
        if (physical_td && physical_td->fragmenter) {
          return curr_val + static_cast<int64_t>(physical_td->fragmenter->getNumRows());
        }
        return curr_val;
      });
}

}  // namespace

bool SelectedTableInfo::operator==(const SelectedTableInfo& other) const {
  return table_key == other.table_key && table_name == other.table_name;
}

bool SelectedTableInfo::operator==(const TableDescriptor& other_td) const {
  return table_key.table_id == other_td.tableId &&
         table_name.table_name == other_td.tableName;
}

PhysicalTableInfo::PhysicalTableInfo(const TableDescriptor& td,
                                     const Catalog_Namespace::Catalog& cat)
    : SelectedTableInfo({cat.getDatabaseId(), td.tableId}, {cat.name(), td.tableName})
    , num_rows(get_total_num_rows_for_table(cat, td)) {}

PhysicalTableInfo::PhysicalTableInfo(const shared::TableKey& table_key,
                                     const shared::FullyQualifiedTableName& table_name,
                                     const int64_t num_rows)
    : SelectedTableInfo(table_key, table_name), num_rows(num_rows) {}

bool PhysicalTableInfo::hasColumn(const std::string& col) const {
  const auto cat = Catalog_Namespace::SysCatalog::instance().getCatalog(table_key.db_id);
  CHECK(cat);
  return cat->getMetadataForColumn(table_key.table_id, col) != nullptr;
}

boost::optional<std::string> PhysicalTableInfo::getOwningColName(
    const Analyzer::TargetEntry& result_target) const {
  boost::optional<std::string> rtn;
  auto expr = result_target.get_expr();
  CHECK(expr) << "No expr for result_target: " << result_target.toString();
  std::set<const Analyzer::ColumnVar*,
           bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)>
      colvar_set(Analyzer::ColumnVar::colvar_comp);

  expr->collect_column_var(colvar_set, false);
  if (colvar_set.size() == 1) {
    auto colvar = (*colvar_set.begin());
    const auto& column_key = colvar->getColumnKey();
    if (column_key.db_id == table_key.db_id &&
        column_key.table_id == table_key.table_id) {
      const auto cd = Catalog_Namespace::get_metadata_for_column(column_key);
      CHECK(cd) << column_key.db_id << ":" << column_key.table_id << ":"
                << column_key.column_id;
      rtn = cd->columnName;
    }
  }
  return rtn;
}

ViewTableInfo::ViewTableInfo(
    const TableDescriptor& td,
    std::unordered_set<shared::FullyQualifiedTableName>&& target_tables,
    TargetEntries&& target_columns,
    const Catalog_Namespace::Catalog& cat)
    : SelectedTableInfo({cat.getDatabaseId(), td.tableId}, {cat.name(), td.tableName})
    , target_table_names(std::move(target_tables))
    , target_entries(std::move(target_columns)) {}

bool ViewTableInfo::hasColumn(const std::string& col) const {
  // TODO(croot): use a better datastructure for the lookup? This vector size is likely
  // very small, so this will suffice.
  return std::find_if(
             target_entries.begin(), target_entries.end(), [&col](const auto& target) {
               return target->get_resname() == col;
             }) != target_entries.end();
}

boost::optional<std::string> ViewTableInfo::getOwningColName(
    const Analyzer::TargetEntry& result_target) const {
  boost::optional<std::string> rtn;
  if (hasColumn(result_target.get_resname())) {
    rtn = result_target.get_resname();
  }
  return rtn;
}

boost::optional<std::string> ViewTableInfo::getAliasedColumn(
    const shared::TableKey& table_key,
    const ColumnId col_id) const {
  boost::optional<std::string> rtn;
  auto itr = std::find_if(target_entries.begin(),
                          target_entries.end(),
                          [&table_key, col_id](const auto& target) {
                            const auto expr = target->get_expr();
                            CHECK(expr) << "No expr for target: " << target->toString();
                            auto col_expr =
                                dynamic_cast<const Analyzer::ColumnVar*>(expr);
                            return col_expr && col_expr->getColumnKey() ==
                                                   shared::ColumnKey{table_key, col_id};
                          });

  if (itr != target_entries.end()) {
    rtn = (*itr)->get_resname();
  }
  return rtn;
}

void SQLSelectedTableContainer::clear() {
  phys_tables.clear();
  views.clear();
}

}  // namespace QueryRenderer
