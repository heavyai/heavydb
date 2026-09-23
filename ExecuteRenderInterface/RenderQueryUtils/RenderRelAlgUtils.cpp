/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/RenderRelAlgUtils.h"

#include "Shared/Rendering/HitTestTypes.h"

namespace QueryRenderer {

namespace {

/**
 * Injects rowid columns into the top level projection node of an RA tree representation
 * of a render projection query.
 * @param rel_scan_tree The RelScanTree representation of the RelAlgDag being modified.
 */
void inject_rowid_into_RA(RelScanTree& rel_scan_tree) {
  // stores all projected tables without a projected rowid
  std::unordered_set<const RelScan*> no_rowid_tables;
  constexpr int max_rowids = HitTestTypes::max_num_rowids();

  if (rel_scan_tree.size() == 0) {
    // early out, no valid scanned tables
    return;
  } else if (rel_scan_tree.size() > max_rowids) {
    // can only support a max number of rowids, determined by the renderer, so
    // deep, multi-level joins may not be fully hittest-able
    LOG(WARNING) << "Hit-testing supports a max " << max_rowids
                 << " tables per-query. This render query references "
                 << rel_scan_tree.size()
                 << " tables. Not all tables will be hittest-able.";
  }

  // create a container that maintains the state of the scanned tables with projected
  // rowid. The container will b truncated for any deep, multi-level join
  for (size_t i = 0; i < std::min(size_t(max_rowids), rel_scan_tree.size()); i++) {
    no_rowid_tables.insert(&rel_scan_tree[i]);
  }

  // Now gather all the projected rowid columns in the query. We obviously don't need
  // to re-project rowid if it's already projected.
  // NOTE: rowid_regex captures "rowid", "rowid0", "rowid1" right now, as renderer
  // handles up-to 3 separate rowids currently.
  auto rowid_regex = HitTestTypes::get_rowid_regex();
  std::smatch rowid_match;

  // array for storing used rowid output slots - "rowid"/"rowid0"/"rowid1"
  // when true, the slot is used.
  std::array<bool, HitTestTypes::max_num_rowids()> used_rowids;
  std::fill(used_rowids.begin(), used_rowids.end(), false);

  auto const& root_project_node = rel_scan_tree.getRootProjectNode();

  // look for rowids in the projected columns
  for (size_t i = 0; i < root_project_node.size(); ++i) {
    auto col_name = root_project_node.getFieldName(i);
    if ((std::regex_match(col_name, rowid_match, rowid_regex))) {
      // gather the prospective RelScan and column index for the rowid column and validate

      // NOTE: need to use std::tie here since rowid_col_table is captured in a lambda
      // below. This can be moved to a structured binding in c++20
      const RelScan* rowid_col_table{nullptr};
      uint32_t rowid_idx{0};
      std::tie(rowid_col_table, rowid_idx) = rel_scan_tree.getScanNodeForOutputIndex(i);

      if (!rowid_col_table) {
        // Found a rowid output that is built via an aggregate/compound
        // This is not supported.
        // TODO(croot): should we properly support this case? If so, how should this rowid
        // be handled?
        throw std::runtime_error("Projected rowid output \"" + col_name +
                                 "\" in query RA does not directly reference a rowid "
                                 "column in a table. Projected rowid output must have a "
                                 "1-to-1 relationship to a single table.");
      }

      auto referenced_col = rowid_col_table->getFieldName(rowid_idx);
      if (referenced_col != "rowid") {
        // Found a rowid output that doesn't reference the real rowid column in a
        // table
        throw std::runtime_error("Projected rowid output \"" + col_name +
                                 "\" in query RA references the \"" + referenced_col +
                                 "\" column in table \"" +
                                 rowid_col_table->getTableDescriptor()->tableName +
                                 "\". It must reference the rowid column.");
      }

      // found a valid rowid, mark the rowid output slot as used
      // NOTE: can use std::erase_if in C++20
      auto itr = std::find_if(no_rowid_tables.begin(),
                              no_rowid_tables.end(),
                              [&](auto const* table_scan_node) {
                                return table_scan_node == rowid_col_table;
                              });
      CHECK(itr != no_rowid_tables.end());
      no_rowid_tables.erase(itr);

      // make sure to use the right output slot - again, rowid names will be rowid,
      // rowid0, rowid1, so increment the slot index if output column is suffixed with
      // a numeric value
      int rowid_slot_idx =
          (rowid_match[1].length() > 0 ? std::stoi(rowid_match[1]) + 1 : 0);
      used_rowids[rowid_slot_idx] = true;
    }
  }

  if (no_rowid_tables.size() > 0) {
    // now inject rowid columns into the RA for all unreferenced tables, using rowid
    // names from unused output slots
    int rowid_slot_idx = 0;
    for (auto const* table_scan : no_rowid_tables) {
      // search this table's columns for 'rowid'
      // search from the end, because 99% of the time it will be the
      // last column, and if it's not, it's only because other columns
      // have been appended
      int rowid_column = -1;
      for (int i = table_scan->size() - 1; i >= 0; i--) {
        if (table_scan->getFieldName(i) == "rowid") {
          rowid_column = i;
          break;
        }
      }

      // if there isn't one, just throw
      if (rowid_column < 0) {
        throw std::runtime_error("Failed to find a 'rowid' column in table '" +
                                 table_scan->getTableDescriptor()->tableName + "'");
      }

      // report what we find, to assist repro
      VLOG(1) << "inject_rowid_into_RA: Found 'rowid' at column " << rowid_column
              << " of table '" << table_scan->getTableDescriptor()->tableName << "' with "
              << table_scan->size() << " columns";

      // find an unused output slot
      CHECK_LE(rowid_slot_idx, max_rowids - 1);
      while (used_rowids[rowid_slot_idx]) {
        rowid_slot_idx++;
      }

      // re-adjust the index for the output name
      std::string injected_rowid =
          "rowid" + (rowid_slot_idx == 0 ? "" : std::to_string(rowid_slot_idx - 1));

      rel_scan_tree.injectInputColumn(*table_scan, rowid_column, injected_rowid);

      VLOG(1) << "inject_rowid_into_RA: Injected rowid column \"" << injected_rowid
              << "\" into render query for table "
              << table_scan->getTableDescriptor()->tableName;
      rowid_slot_idx++;
    }
  }
}

}  // namespace

/**
 * Alters an RA tree for a render query (i.e. auto-inject rowid columns to hit-test
 * projections)
 * @param nodes RA tree nodes in vector form. The last node of the vector is the root of
 *              the tree.
 * @param render_info The render query info and options.
 */
void RenderRelAlgUtils::alterRAForRender(RelScanTree* rel_scan_tree,
                                         const RenderInfo& render_info) {
  auto& render_opts = render_info.getRenderQueryOptions();
  if (!rel_scan_tree || !render_opts.shouldAlterRA()) {
    return;
  }

  if (render_info.couldRunInSitu() &&
      ((render_opts.injectRowIdForHitTesting() && !render_opts.useLegacyHitTestLogic()) ||
       render_opts.injectRowIdForPPLL())) {
    inject_rowid_into_RA(*rel_scan_tree);
  }
}

}  // namespace QueryRenderer
