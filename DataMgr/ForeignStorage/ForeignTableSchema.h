/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <vector>
#include "Catalog/CatalogFwd.h"
#include "Interval.h"

namespace foreign_storage {
class ForeignTableSchema {
 public:
  ForeignTableSchema(const int32_t db_id, const ForeignTable* foreign_table);

  /**
   * Gets a pointer to the column descriptor object for the given column id.
   */
  const ColumnDescriptor* getColumnDescriptor(const int column_id) const;

  /**
   * Get a list of pointers of valid column descriptors (returned by
   * `getLogicalAndPhysicalColumns`) that are also within the given interval.
   */
  std::list<const ColumnDescriptor*> getColumnsInInterval(
      const Interval<ColumnType>& column_interval) const;

  /**
   * Gets the logical column that is associated with the given column id.
   * Given column id can be for a physical column or logical column (in
   * this case, the column descriptor for the same column is returned)
   */
  const ColumnDescriptor* getLogicalColumn(const int column_id) const;

  /**
   * Gets the Parquet column index that corresponds to the given
   * column id.
   */
  int getParquetColumnIndex(const int column_id) const;

  /**
   * Gets all the logical and physical columns for the foreign table.
   */
  const std::list<const ColumnDescriptor*>& getLogicalAndPhysicalColumns() const;

  /**
   * Gets the total number of logical and physical columns for the foreign table.
   */
  int numLogicalAndPhysicalColumns() const;

  /**
   * Gets all the logical columns for the foreign table.
   */
  const std::list<const ColumnDescriptor*>& getLogicalColumns() const;

  /**
   * Gets the total number of logical columns for the foreign table.
   */
  int numLogicalColumns() const;

  const ForeignTable* getForeignTable() const;

 private:
  std::vector<int>::const_iterator getLogicalColumnIdIterator(const int column_id) const;

  std::list<const ColumnDescriptor*> logical_and_physical_columns_;
  std::list<const ColumnDescriptor*> logical_columns_;
  std::vector<int> logical_column_ids_;
  const ForeignTable* foreign_table_;
  std::shared_ptr<Catalog_Namespace::Catalog> catalog_;
};
}  // namespace foreign_storage
