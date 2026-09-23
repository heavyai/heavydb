/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Analyzer.h"

#include <Catalog/ColumnDescriptor.h>
#include <Catalog/TableDescriptor.h>

namespace Analyzer {

/*
 * @type RangeTableEntry
 * @brief Range table contains all the information about the tables/views
 * and columns referenced in a query.  It is a list of RangeTableEntry's.
 */
class RangeTableEntry {
 public:
  RangeTableEntry(const std::string& r, const TableDescriptor* t, Query* v)
      : rangevar(r), table_desc(t), view_query(v) {}
  virtual ~RangeTableEntry();
  /* @brief get_column_desc tries to find the column in column_descs and returns the
   * column descriptor if found. otherwise, look up the column from Catalog, add the
   * descriptor to column_descs and return the descriptor.  return nullptr if not found
   * @param catalog the catalog for the current database
   * @param name name of column to look up
   */
  const ColumnDescriptor* get_column_desc(const Catalog_Namespace::Catalog& catalog,
                                          const std::string& name);
  const std::list<const ColumnDescriptor*>& get_column_descs() const;
  const std::string& get_rangevar() const { return rangevar; }
  int32_t get_table_id() const;
  const std::string& get_table_name() const;
  const TableDescriptor* get_table_desc() const;
  const Query* get_view_query() const { return view_query; }
  void expand_star_in_targetlist(const Catalog_Namespace::Catalog& catalog,
                                 std::vector<std::shared_ptr<TargetEntry>>& tlist,
                                 int rte_idx);
  void add_all_column_descs(const Catalog_Namespace::Catalog& catalog);

 private:
  std::string rangevar;  // range variable name, e.g., FROM emp e, dept d
  const TableDescriptor* table_desc;
  std::list<const ColumnDescriptor*>
      column_descs;   // column descriptors for all columns referenced in this query
  Query* view_query;  // parse tree for the view query
};

}  // namespace Analyzer