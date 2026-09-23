/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/SQLSelectedTablesBuilder.h"

namespace QueryRenderer {

namespace {
template <typename T, typename... Targs>
static void handle_multi_index_push_front(T& container,
                                          const TableDescriptor& td,
                                          Targs&&... Fargs) {
  auto insert_pair = container.emplace_front(td, std::forward<Targs>(Fargs)...);
  if (!insert_pair.second) {
    // if insertion failed, that should mean the table already exists in the list
    CHECK(insert_pair.first != container.end());
    CHECK(insert_pair.first->table_id == td.tableId);
    CHECK(*insert_pair.first == td)
        << insert_pair.first->table_id << ":" << insert_pair.first->table_name
        << " != " << td.tableId << ":" << td.tableName;

    // Just move the existing table to the front of the list
    container.relocate(container.begin(), insert_pair.first);
  }
}
}  // namespace

void SQLSelectedTablesBuilder::push_back(const Catalog_Namespace::Catalog& cat,
                                         const TableDescriptor& td) {
  if (td.isView) {
    auto [view_tables, view_cols] = resolveView(cat, td);
    sql_selected_tables.views.emplace_back(
        td, std::move(view_tables), std::move(view_cols), cat);
  } else {
    sql_selected_tables.phys_tables.emplace_back(td, cat);
  }
}

std::pair<SQLSelectedTablesBuilder::ViewUsedTables,
          SQLSelectedTablesBuilder::ViewColumnTargets>
SQLSelectedTablesBuilder::resolveView(const Catalog_Namespace::Catalog& cat,
                                      const TableDescriptor& td) {
  auto& render_info = parent_render_query_runner.getRenderInfo();
  RenderInfo my_render_info(
      render_info.render_session_key,
      render_info.getRenderQueryOptions(),
      heavyai::InSituFlags::kForcedNonInSitu);  // forces non insitu, not entirely
                                                // necessary, but just-in-case

  // Resolves the view via a sql_validate
  // NOTE: due to https://jira.omnisci.com/browse/BE-4338, we can't just use td.viewSql
  // because view permissions are lost the moment we use the view's sql since the view is
  // then bypassed in calcite. We need to maintain the view table so permissions are
  // properly checked during the calcite parse. The workaround is to do a SELECT * LIMIT 0
  // on the view so all columns are retrieved yet permissions retained.

  // NOTE: table read locks should have been acquired somewhere up the callstack
  auto query_ra = RenderQueryRunner::parseToRAAndAcquireTableLocks(
      parent_render_query_runner.getRenderHandler(),
      "SELECT * FROM " + cat.name() + "." + td.tableName + " LIMIT 0",
      my_render_info,
      nullptr);

  // TODO(croot): should we use a new Executor here?
  // NOTE: do not need to use the return ExecutionOptions from the validate.
  // All we care about here is the tables/targets populated in the RenderInfo object
  // afterwards
  RenderQueryRunner::validateQuery(
      parent_render_query_runner.getRenderHandler(),
      my_render_info,
      RenderQueryRunner::buildRelAlgDagAndClassifyRender(my_render_info, query_ra),
      parent_render_query_runner.getRenderQueryExecutionOptions());

  return {my_render_info.table_names, my_render_info.targets};
}

}  // namespace QueryRenderer
