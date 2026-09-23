/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/optional.hpp>

#include "QueryRenderer/Interface/RenderQueryInterfaceDeclarations.h"
#include "QueryRenderer/Interface/TableTypes.h"
#include "Shared/DbObjectKeys.h"
#include "Shared/FullyQualifiedTableName.h"

namespace QueryRenderer {

/**
 * SelectedTableInfo: interface class for table info maintained by the renderer
 */
struct SelectedTableInfo {
  shared::TableKey table_key;
  shared::FullyQualifiedTableName table_name;

  explicit SelectedTableInfo(const shared::TableKey& table_key,
                             const shared::FullyQualifiedTableName& table_name)
      : table_key(table_key), table_name(table_name) {}
  virtual ~SelectedTableInfo() {}

  bool operator==(const SelectedTableInfo& other) const;
  bool operator==(const TableDescriptor& other_td) const;

  /**
   * Returns true if the selected from table has a column
   */
  virtual bool hasColumn(const std::string& col) const = 0;

  /**
   * Given the result_target (which should be a TargetEntry for a resulting column from
   a
   * query), if that result_target originates only from one column, and that column
   exists
   * in this table, then the original column name from the table is returned. Otherwise
   it
   * returns an undefined string.
   *
   * For example, using the query "SELECT orig_col as my_col FROM table", if the
   * TargetEntry for the "my_col" column of the result set is used as the result_target
   * argument, then returns "orig_col".
   *
   * If the query is "SELECT orig_col / 2 as my_col ..." or "SELECT CAST(orig_col AS
   * FLOAT) as my_col ...", then "orig_col" should be returned.
   *
   * If the query is "SELECT CASE WHEN other_col == 1 THEN orig_col ELSE orig_col / 2
   END
   * as my_col FROM table", this would return an undefined string as the expression for
   * the "my_col" column references multiple columns.
   */
  virtual boost::optional<std::string> getOwningColName(
      const Analyzer::TargetEntry& result_target) const = 0;

  struct KeyTag;
  struct NameTag;
};

template <typename T>
using SelectedTableContainer = ::boost::multi_index_container<
    T,
    ::boost::multi_index::indexed_by<
        ::boost::multi_index::random_access<>,

        ::boost::multi_index::hashed_unique<
            ::boost::multi_index::tag<typename T::KeyTag>,
            ::boost::multi_index::member<SelectedTableInfo,
                                         shared::TableKey,
                                         &SelectedTableInfo::table_key>>>>;

struct PhysicalTableInfo : public SelectedTableInfo {
  PhysicalTableInfo(const TableDescriptor& td, const Catalog_Namespace::Catalog& cat);
  explicit PhysicalTableInfo(const shared::TableKey& table_key,
                             const shared::FullyQualifiedTableName& table_name,
                             const int64_t num_rows);

  bool hasColumn(const std::string& col) const override;

  boost::optional<std::string> getOwningColName(
      const Analyzer::TargetEntry& result_target) const override;

  int64_t num_rows;
};

using PhysicalTableInfoContainer = SelectedTableContainer<PhysicalTableInfo>;

struct ViewTableInfo : public SelectedTableInfo {
  ViewTableInfo(const TableDescriptor& td,
                std::unordered_set<shared::FullyQualifiedTableName>&& target_tables,
                TargetEntries&& target_columns,
                const Catalog_Namespace::Catalog& cat);

  bool hasColumn(const std::string& col) const override;

  boost::optional<std::string> getOwningColName(
      const Analyzer::TargetEntry& result_target) const override;

  /**
   * Returns the column name in a view if it aliases a specific physical column.
   * Otherwise it returns an undefined string.
   */
  boost::optional<std::string> getAliasedColumn(const shared::TableKey& table_key,
                                                const ColumnId col_id) const;

  std::unordered_set<shared::FullyQualifiedTableName> target_table_names;
  TargetEntries target_entries;
};

using ViewTableInfoContainer = SelectedTableContainer<ViewTableInfo>;

struct SQLSelectedTableContainer {
 public:
  PhysicalTableInfoContainer phys_tables;
  ViewTableInfoContainer views;

  void clear();
};

}  // namespace QueryRenderer
