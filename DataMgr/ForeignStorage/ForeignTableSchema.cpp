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

#include "ForeignTableSchema.h"
#include "Catalog/Catalog.h"

namespace foreign_storage {
ForeignTableSchema::ForeignTableSchema(const int32_t db_id,
                                       const ForeignTable* foreign_table) {
  catalog_ = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog_);
  foreign_table_ = foreign_table;
  logical_and_physical_columns_ =
      catalog_->getAllColumnMetadataForTable(foreign_table->tableId, false, false, true);
  logical_columns_ =
      catalog_->getAllColumnMetadataForTable(foreign_table->tableId, false, false, false);

  for (const auto column : logical_columns_) {
    logical_column_ids_.emplace_back(column->columnId);
  }
}

const ColumnDescriptor* ForeignTableSchema::getColumnDescriptor(
    const int column_id) const {
  auto column = catalog_->getMetadataForColumn(foreign_table_->tableId, column_id);
  CHECK(column);
  return column;
}

const ColumnDescriptor* ForeignTableSchema::getLogicalColumn(const int column_id) const {
  auto logical_column_id = *getLogicalColumnIdIterator(column_id);
  CHECK_LE(logical_column_id, column_id);
  return getColumnDescriptor(logical_column_id);
}

std::list<const ColumnDescriptor*> ForeignTableSchema::getColumnsInInterval(
    const Interval<ColumnType>& column_interval) const {
  auto column_start = column_interval.start;
  auto column_end = column_interval.end;
  std::list<const ColumnDescriptor*> columns;
  for (const auto column : getLogicalAndPhysicalColumns()) {
    auto column_id = column->columnId;
    if (column_id >= column_start && column_id <= column_end) {
      columns.push_back(column);
    }
  }
  return columns;
}

int ForeignTableSchema::getParquetColumnIndex(const int column_id) const {
  auto column_index =
      std::distance(logical_column_ids_.begin(), getLogicalColumnIdIterator(column_id));
  CHECK_GE(column_index, 0);
  return column_index;
}

const std::list<const ColumnDescriptor*>&
ForeignTableSchema::getLogicalAndPhysicalColumns() const {
  return logical_and_physical_columns_;
}

int ForeignTableSchema::numLogicalAndPhysicalColumns() const {
  return logical_and_physical_columns_.size();
}

const std::list<const ColumnDescriptor*>& ForeignTableSchema::getLogicalColumns() const {
  return logical_columns_;
}

int ForeignTableSchema::numLogicalColumns() const {
  return logical_columns_.size();
}

const ForeignTable* ForeignTableSchema::getForeignTable() const {
  return foreign_table_;
}

std::vector<int>::const_iterator ForeignTableSchema::getLogicalColumnIdIterator(
    const int column_id) const {
  auto it =
      std::upper_bound(logical_column_ids_.begin(), logical_column_ids_.end(), column_id);
  CHECK(it != logical_column_ids_.begin());
  it--;
  return it;
}
}  // namespace foreign_storage
