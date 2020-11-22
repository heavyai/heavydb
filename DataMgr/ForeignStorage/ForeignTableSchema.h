/*
 * Copyright 2020 OmniSci, Inc.
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

#include "Catalog/Catalog.h"

namespace foreign_storage {
class ForeignTableSchema {
 public:
  ForeignTableSchema(const int32_t db_id, const ForeignTable* foreign_table) {
    catalog_ = Catalog_Namespace::SysCatalog::instance().checkedGetCatalog(db_id);
    foreign_table_ = foreign_table;
    logical_and_physical_columns_ = catalog_->getAllColumnMetadataForTableUnlocked(
        foreign_table->tableId, false, false, true);
    logical_columns_ = catalog_->getAllColumnMetadataForTableUnlocked(
        foreign_table->tableId, false, false, false);

    for (const auto column : logical_columns_) {
      logical_column_ids_.emplace_back(column->columnId);
    }
  }

  /**
   * Gets a pointer to the column descriptor object for the given column id.
   */
  const ColumnDescriptor* getColumnDescriptor(const int column_id) const {
    auto column =
        catalog_->getMetadataForColumnUnlocked(foreign_table_->tableId, column_id);
    CHECK(column);
    return column;
  }

  /**
   * Gets the logical column that is associated with the given column id.
   * Given column id can be for a physical column or logical column (in
   * this case, the column descriptor for the same column is returned)
   */
  const ColumnDescriptor* getLogicalColumn(const int column_id) const {
    auto logical_column_id = *getLogicalColumnIdIterator(column_id);
    CHECK_LE(logical_column_id, column_id);
    return getColumnDescriptor(logical_column_id);
  }

  /**
   * Gets the Parquet column index that corresponds to the given
   * column id.
   */
  int getParquetColumnIndex(const int column_id) const {
    auto column_index =
        std::distance(logical_column_ids_.begin(), getLogicalColumnIdIterator(column_id));
    CHECK_GE(column_index, 0);
    return column_index;
  }

  /**
   * Gets all the logical and physical columns for the foreign table.
   */
  const std::list<const ColumnDescriptor*>& getLogicalAndPhysicalColumns() const {
    return logical_and_physical_columns_;
  }

  /**
   * Gets the total number of logical and physical columns for the foreign table.
   */
  int numLogicalAndPhysicalColumns() const {
    return logical_and_physical_columns_.size();
  }

  /**
   * Gets all the logical columns for the foreign table.
   */
  const std::list<const ColumnDescriptor*>& getLogicalColumns() const {
    return logical_columns_;
  }

  /**
   * Gets the total number of logical columns for the foreign table.
   */
  int numLogicalColumns() const { return logical_columns_.size(); }

  /**
   * Gets a pointer to the foreign table object.
   */
  const ForeignTable* getForeignTable() const { return foreign_table_; }

 private:
  std::vector<int>::const_iterator getLogicalColumnIdIterator(const int column_id) const {
    auto it = std::upper_bound(
        logical_column_ids_.begin(), logical_column_ids_.end(), column_id);
    CHECK(it != logical_column_ids_.begin());
    it--;
    return it;
  }

  std::list<const ColumnDescriptor*> logical_and_physical_columns_;
  std::list<const ColumnDescriptor*> logical_columns_;
  std::vector<int> logical_column_ids_;
  const ForeignTable* foreign_table_;
  std::shared_ptr<Catalog_Namespace::Catalog> catalog_;
};
}  // namespace foreign_storage
