/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExecuteRenderInterface/RenderQueryUtils/RenderQueryRunner.h"
#include "QueryRenderer/Interface/SqlSelectedTableInfo.h"

namespace QueryRenderer {

/**
 * This is a utility builder struct that is used to populate a list of the
 * selected-from tables from a query, differentiated between physical tables and
 * views. This builder handles the differentiation logic and resolves views. Resolving
 * views involves evaluating the view table's sql to obtain it's selected-from tables and
 * a list of the TargetEntries of its columns for later lookups by the renderer.
 *
 * This struct lives in the thrift-handler level of the code to avoid putting query
 * execution logic inside the QueryRenderer lib.
 */
struct SQLSelectedTablesBuilder {
  using ViewUsedTables = decltype(::QueryRenderer::ViewTableInfo::target_table_names);
  using ViewColumnTargets = decltype(::QueryRenderer::ViewTableInfo::target_entries);

  // render query runner that creates an instance of this class
  // this instance should not exceed the lifetime of its parent
  RenderQueryRunner& parent_render_query_runner;

  // sql_selected_tables is the resulting list of selected-from tables
  ::QueryRenderer::SQLSelectedTableContainer& sql_selected_tables;

  explicit SQLSelectedTablesBuilder(
      RenderQueryRunner& parent_render_query_runner,
      QueryRenderer::SQLSelectedTableContainer& sql_selected_tables)
      : parent_render_query_runner{parent_render_query_runner}
      , sql_selected_tables{sql_selected_tables} {}
  SQLSelectedTablesBuilder& operator=(const SQLSelectedTablesBuilder&) = delete;

  // pushes a new selected-from table onto the back of the list
  void push_back(const Catalog_Namespace::Catalog& cat, const TableDescriptor& td);

 private:
  /**
   * Evaluates the view sql via a validate sql execution and returns the selected-from
   * table names and result-set targets from the query.
   *
   * NOTE: read locks for the view's tables must be acquired before calling.
   */
  std::pair<ViewUsedTables, ViewColumnTargets> resolveView(
      const Catalog_Namespace::Catalog& cat,
      const TableDescriptor& td);
};

}  // namespace QueryRenderer
