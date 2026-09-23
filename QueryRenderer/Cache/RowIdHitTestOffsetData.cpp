/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RowIdHitTestOffsetData.h"

#include "QueryRenderer/Data/EmbeddedDataUtils.h"

#include <QueryEngine/ResultSet.h>
#include <Shared/Rendering/HitTestTypes.h>

namespace QueryRenderer {

namespace {
inline uint32_t numBits(const uint64_t number) {
  return static_cast<uint64_t>(std::log2(number)) + 1;
}

constexpr bool g_traverse_agg_functions = false;
constexpr uint32_t max_bits_for_hittesting = 64;
}  // namespace

RowIdHitTestOffsetData::RowIdHitTestOffsetData(
    const std::string& in_sql_str,
    const RenderQueryOutput& render_query_output)
    : sql_str{in_sql_str} {
  if (render_query_output.numResultRows() > 0) {
    // Builds up the rowid bit-shift offsets from all the rowid columns in the sql.
    auto const& selected_phys_tables = render_query_output.getSqlSelectedPhysicalTables();
    auto const& key_lookup = selected_phys_tables.get<SelectedTableInfo::KeyTag>();

    // rowid_regex captures "rowid", "rowid0", "rowid1" right now, as our shaders only
    // handle 3 separate rowids.
    auto rowid_regex = HitTestTypes::get_rowid_regex();
    std::smatch rowid_match;
    uint32_t total_bits = 0;

    // rowid_cols, rowid_tables, & rowid_offsets are const, so grab an un-const version of
    // those members and clear them out. These will be initialized/modified only in the
    // constructor
    auto& columns = const_cast<std::vector<const Analyzer::TargetEntry*>&>(rowid_cols);
    auto& tables = const_cast<std::vector<const SelectedTableInfo*>&>(rowid_tables);
    auto& offsets = const_cast<std::vector<uint32_t>&>(rowid_offsets);
    columns.clear();
    tables.clear();
    offsets.clear();

    for (auto& target : render_query_output.getOutputTargetEntries()) {
      if (std::regex_match(target->get_resname(), rowid_match, rowid_regex)) {
        // found a column from the sql that matches our rowid regex, so grab the expr and
        // collect the ColumnVar to build out the table and bit-shift data
        auto expr = target->get_expr();
        CHECK(expr) << "No expr for target: " << target->toString();

        // Collect all the ColumnVar nodes from this expr, but don't collect from agg
        // functions
        std::set<const Analyzer::ColumnVar*,
                 bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)>
            colvar_set(Analyzer::ColumnVar::colvar_comp);
        expr->collect_column_var(colvar_set, g_traverse_agg_functions);

        if (colvar_set.size()) {
          // rowid columns should only back-reference a single table
          RUNTIME_EX_ASSERT(colvar_set.size() == 1,
                            "Invalid rowid column \"" + target->get_resname() +
                                "\" in query \"" + sql_str +
                                "\". It should reference exactly 1 rowid column from a "
                                "table, but it references " +
                                std::to_string(colvar_set.size()) + " columns.");

          auto col_target = *colvar_set.begin();
          const auto table_key = col_target->getTableKey();
          if (table_key.table_id < 0) {
            // apparently ColumnVar Expr objects can have negative table ids. I'm not sure
            // exactly the use case, but it can happen in at least 1 place -
            // i.e. RelAlgTranslator::translateInput(). The only noticeable time I've hit
            // this case is with an aggregate query such as this: "SELECT
            // zipcodes_2017.rowid, CASE WHEN rowid IN (25101) THEN LAST_SAMPLE(ZCTA5CE10)
            // END as color FROM zipcodes_2017 GROUP BY zipcodes_2017.rowid" which is a
            // legacy poly render query. If this is the only time we will see negative
            // table ids, then this should be safe as the aggregate is ultimately a
            // non-in-situ query which means we'll extract the rowid directly from the
            // results anyway, but just noting here.
            // TODO(croot): if we ever get around to gpu-reductions, meaning aggregate
            // queries in theory could be run in-situ, then what?
            continue;
          }

          // Double check that we have the table info for this rowid in the query.
          auto itr = key_lookup.find(table_key);
          if (itr == key_lookup.end()) {
            CHECK(false) << "Cannot find " << table_key << " in used tables of size "
                         << key_lookup.size() << ". Query: " << sql_str;
          }

          columns.push_back(target.get());
          tables.push_back(&(*itr));

          // caculate the number of bits required to handle rowid for this table.
          uint32_t num_bits_for_rows{0};
          if (itr->num_rows == 0) {
            // if we reach here that means we've either got a query referencing a table
            // that actually does not have any rows (in which case the result size
            // probably is 0), or there is an edge case or bug when trying to calculate
            // the total number of rows for a table. I (croot) am not confident that the
            // 'get_total_num_rows_for_table' function in SqlSelectedTableInfo.cpp
            // captures all cases. Nor am I confident that the table descriptor state from
            // which we grab the total number of rows for a table is properly updated in
            // all cases. So the following is a fail safe to at least carry forward with
            // some kind of hit-testing if we hit such an odd state.
            //
            // In this case if the table with 0 rows is the only table referenced in the
            // query, then we can let it consume all of the rowid bits. This assumes that
            // any later hit-test queries (i.e. SELECT ... FROM <table> WHERE
            // rowid=<rowid>) will work if the num rows for the table is incorrect.
            //
            // If there are multiple tables referenced in the query, then we will throw an
            // error as we won't have all the info needed to unpack the rowids correctly.
            // If this error is hit and we know the table is supposed to have rows then
            // we'll probably want to get all table create and modify statements to
            // reproduce. NOTE: if a table is rendered that actually has 0 rows, then this
            // error would be thrown incorrectly. The problem is that we don't have the
            // number of query result rows at our disposal without more plumbing. I feel
            // this is an ok hit for now until we change evaluation order so that this
            // function would only be called in the event that there's actually data to
            // render.
            RUNTIME_EX_ASSERT(
                selected_phys_tables.size() == 1,
                "Could not calculate the total number of rows for table \"" +
                    itr->table_name.table_name + "\" in catalog \"" +
                    itr->table_name.db_name + "\" referenced in query \"" + sql_str +
                    "\". A total number of rows is required for hit-testing queries "
                    "with " +
                    std::to_string(selected_phys_tables.size()) + " tables referenced.");
            num_bits_for_rows = max_bits_for_hittesting;
          } else {
            num_bits_for_rows = numBits(itr->num_rows);
          }
          offsets.push_back(num_bits_for_rows);  // build out the bit-shift offsets
          total_bits += num_bits_for_rows;
        }
      }
    }

    // Validate that the number of bits required for all rowids fit within 64 bits.

    // TODO(croot): should this just be a warning instead and carry on with hit-testing
    // ignored?
    RUNTIME_EX_ASSERT(
        total_bits <= max_bits_for_hittesting,
        "Cannot build a hit-testing packing policy in less than 64 bits for query \"" +
            sql_str + "\". Total number of bits required: " + std::to_string(total_bits) +
            ".");
  }
}

/**
 * Unpacks a rowid from the id buffers from the render and returns the rowids and their
 * respective table data.
 */
HitTestTableContainer RowIdHitTestOffsetData::unpackRowId(
    const int64_t row_id_to_unpack,
    const PhysicalTableInfoContainer& used_tables) const {
  auto rowid = row_id_to_unpack;
  HitTestTableContainer hittest_table_info;

  if (rowid_offsets.size()) {
    // unpack rowids from their bit-shift offsets.
    for (size_t i = 0; i < rowid_offsets.size(); ++i) {
      auto curr_rowid = rowid & (static_cast<int64_t>(std::exp2(rowid_offsets[i])) - 1);
      if (curr_rowid > 0) {
        hittest_table_info.push_back(
            {*rowid_tables[i], {{kDefaultIdColumnName, curr_rowid}}});
      }
      rowid = rowid >> rowid_offsets[i];
    }
  } else {
    // this means we could not properly backtrace the TargetValue expr to a valid table
    // (see comments in RowIdHitTestOffsetData::initialize() for more info). In this case
    // we assume the primary table from the query is the table the rowid is being from.
    auto& first_table = *used_tables.begin();
    hittest_table_info.push_back(
        {first_table, {{kDefaultIdColumnName, row_id_to_unpack}}});
  }
  return hittest_table_info;
}

}  // namespace QueryRenderer
