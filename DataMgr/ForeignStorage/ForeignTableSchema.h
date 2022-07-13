/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <vector>
#include "Catalog/CatalogFwd.h"

namespace foreign_storage {
class ForeignTableSchema {
 public:
  ForeignTableSchema(const int32_t db_id, const ForeignTable* foreign_table);

  /**
   * Gets a pointer to the column descriptor object for the given column id.
   */
  const ColumnDescriptor* getColumnDescriptor(const int column_id) const;

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
