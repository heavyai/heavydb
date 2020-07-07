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

#include <set>

#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "Interval.h"

namespace foreign_storage {

class ForeignTableColumnMap {
 public:
  ForeignTableColumnMap(const int db_id, const ForeignTable* foreign_table)
      : catalog_(Catalog_Namespace::Catalog::get(db_id).get())
      , foreign_table_(foreign_table)
      , is_mapped_(false) {}

  ForeignTableColumnMap(const Catalog_Namespace::Catalog* catalog,
                        const ForeignTable* foreign_table)
      : catalog_(catalog), foreign_table_(foreign_table), is_mapped_(false) {}

  /**
   * Find the logical column index that corresponds to a physical index.
   *
   * @param physical_index - physical column index to find corresponding logical index
   *
   * @return Logical column index of given physical index
   */
  int getLogicalIndex(int physical_index) {
    ensureMapped();
    auto upper_bound = logical_columns_physical_index_.upper_bound(physical_index);
    CHECK(upper_bound != logical_columns_physical_index_.begin());
    CHECK(upper_bound != logical_columns_physical_index_.end());
    --upper_bound;
    return std::distance(logical_columns_physical_index_.begin(), upper_bound);
  }

  /**
   * Find (inclusive) interval of physical columns given a physical index.
   *
   * @param physical_index - the physical index to search for
   *
   * @return An interval of (start,end) inclusive bounds specifying an interval
   * of columns that correspond to this physical column's span
   */
  Interval<ColumnType> getPhysicalColumnSpan(int physical_index) {
    ensureMapped();
    auto upper_bound = logical_columns_physical_index_.upper_bound(physical_index);
    CHECK(upper_bound != logical_columns_physical_index_.begin());
    CHECK(upper_bound != logical_columns_physical_index_.end());
    auto lower_bound = upper_bound;
    --lower_bound;
    return {*lower_bound, (*upper_bound) - 1};
  }

  /**
   * Find the physical index that corresponds to a logical index.
   *
   * @param logical_index - logical index to find corresponding physical index
   *
   * @return Physical index of the logical index given
   */
  int getPhysicalIndex(int logical_index) {
    ensureMapped();
    auto it = logical_to_physical_index.find(logical_index);
    CHECK(it != logical_to_physical_index.end());
    return it->second;
  }

 private:
  std::set<int> logical_columns_physical_index_;
  std::unordered_map<int, int> logical_to_physical_index;

  const Catalog_Namespace::Catalog* catalog_;
  const ForeignTable* foreign_table_;
  bool is_mapped_;

  void mapColumns() {
    CHECK(catalog_);
    auto columns = catalog_->getAllColumnMetadataForTableUnlocked(
        foreign_table_->tableId, false, false, false);
    int col_idx = 0;
    int logical_idx = 0;
    for (const auto& cd : columns) {
      logical_columns_physical_index_.insert(col_idx);
      logical_to_physical_index[logical_idx++] = col_idx;
      col_idx += 1 + cd->columnType.get_physical_cols();
    }
    logical_columns_physical_index_.insert(col_idx);
  }

  void ensureMapped() {
    if (!is_mapped_) {
      mapColumns();
      is_mapped_ = true;
    }
  }
};

}  // namespace foreign_storage
