/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderHandlerImpl.h"

#include "ExecuteRenderInterface/RenderQueryUtils/HitTesting/ProjHitTestColSqlBuilder.h"
#include "QueryEngine/ThriftSerializers.h"
#include "QueryRenderer/Cache/HitTestCacheResults.h"
#include "QueryRenderer/QueryRenderManager.h"
#include "Shared/Rendering/HitTestTypes.h"
#include "Shared/sqltypes.h"
#include "ThriftHandler/DBHandler.h"

namespace {

struct HitTestProjectionQueryInfo {
  std::set<const QueryRenderer::HitTestTableInfo*> table_infos;
  std::vector<std::string> projections;

  // helper function if you know there is just one table info
  const QueryRenderer::HitTestTableInfo& getSingleTableInfo() const {
    CHECK_EQ(table_infos.size(), 1u);
    return *(*table_infos.begin());
  }
};

std::string build_hittest_projection_query(const HitTestProjectionQueryInfo& info) {
  // gather these
  std::vector<std::string> table_names;
  std::vector<std::string> rowid_filters;

  // iterate the data
  for (auto const* ti : info.table_infos) {
    // get the table info and key
    CHECK(ti);
    auto const& table_key = ti->getTableKeyRef();

    // get and validate the table descriptor
    auto const catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(table_key.db_id);
    CHECK(catalog);
    auto const* td = catalog->getMetadataForTable(table_key.table_id);
    if (!td) {
      std::stringstream ss;
      ss << "Table " << table_key << " not found for Hit-Test Request.";
      throw std::runtime_error(ss.str());
    }

    // append table name
    table_names.push_back(catalog->name() + "." + td->tableName);

    // append rowid filter
    auto const& rowid_map = ti->col_rowid_map;
    CHECK_EQ(rowid_map.size(), 1u);
    auto const value = rowid_map.begin()->second;
    rowid_filters.push_back(catalog->name() + "." + td->tableName +
                            ".rowid = " + std::to_string(value));
  }

  // concat the lists
  auto const all_tables = boost::algorithm::join(table_names, ", ");
  auto const all_rowid_filters = boost::algorithm::join(rowid_filters, " AND ");
  auto const all_projections = boost::algorithm::join(info.projections, ", ");

  // assemble query string
  // TODO(croot): what about non-projection queries?
  // TODO(croot): what about poly tables?
  return "SELECT " + all_projections + " FROM " + all_tables + " WHERE " +
         all_rowid_filters;
}

void build_query_infos(const QueryRenderer::TargetEntries& result_targets,
                       const QueryRenderer::HitTestTableContainer& all_table_info,
                       const std::string& original_projection,
                       const std::string& query_str,
                       const bool is_projection_query,
                       const bool has_rows,
                       std::function<void()> populate_return_column,
                       std::function<void(const std::string&)> invalidate_return_column,
                       size_t& result_target_idx,
                       HitTestProjectionQueryInfo& query_info) {
  // check for col_name being a projection "something AS x"
  // and replace with just the "x"
  static std::regex project_as_regex(
      R"(^\s*[\S\s]+\s+as\s+(\w+)\s*$)",
      std::regex_constants::ECMAScript | std::regex_constants::icase);
  std::smatch project_as_match;
  auto col_name{original_projection};
  if (std::regex_match(col_name, project_as_match, project_as_regex)) {
    col_name = project_as_match[1];
  }

  // iterate through results to see if we've already generated results
  // for this particular column.
  const Analyzer::ColumnVar* col_var = nullptr;
  Analyzer::Expr* target_expr = nullptr;
  bool colvar_references_multiple_tables = false;
  bool have_results_for_column = false;
  for (result_target_idx = 0; result_target_idx < result_targets.size();
       ++result_target_idx) {
    auto target = result_targets[result_target_idx];
    CHECK(target) << "Undefined target at result_target_idx " << result_target_idx;
    if (target->get_resname() == col_name) {
      // found the column
      if (!has_rows) {
        // if the hit-test query doesn't have rows, we'll need to generate a
        // simple projection query (i.e. "SELECT col1, col2 FROM <table> where
        // rowid=<rowid>") to extract the data from a table. So let's get at the
        // original column data for this target. However, we can only support
        // columns that reference a single table for now. For example, if we
        // have a column from a join query that references columns from each
        // table in the join, then we'll throw an error.
        target_expr = target->get_expr();
        CHECK(target_expr) << "No expr for target: " << target->toString();
        std::set<const Analyzer::ColumnVar*,
                 bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)>
            colvar_set(Analyzer::ColumnVar::colvar_comp);
        target_expr->collect_column_var(colvar_set, false);
        if (!colvar_set.empty()) {
          col_var = *(colvar_set.begin());
          const auto table_key = col_var->getTableKey();
          if (static_cast<size_t>(std::count_if(colvar_set.begin(),
                                                colvar_set.end(),
                                                [&table_key](const auto col_var_ptr) {
                                                  return col_var_ptr->getTableKey() ==
                                                         table_key;
                                                })) != colvar_set.size()) {
            colvar_references_multiple_tables = true;
          }
        }
      }
      have_results_for_column = true;
      break;
    }
  }

  //
  // limited handling of colvar-references-multiple-tables case
  //

  if (colvar_references_multiple_tables) {
    // must be a projection query
    // we already know it doesn't have rows
    // doesn't matter if a project-as or not, as we're gonna rebuild the SQL
    // anyway
    if (is_projection_query) {
      // we still have the target expression
      CHECK(target_expr);
      // fetch the colvar set again
      std::set<const Analyzer::ColumnVar*,
               bool (*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)>
          colvar_set(Analyzer::ColumnVar::colvar_comp);
      target_expr->collect_column_var(colvar_set, false);
      // we know it's not empty, from before
      CHECK(!colvar_set.empty());
      for (auto const* cv : colvar_set) {
        // get table info for this colvar
        auto& all_table_by_key =
            all_table_info.get<::QueryRenderer::HitTestTableInfo::KeyTag>();
        auto table_itr = all_table_by_key.find(cv->getTableKey());
        CHECK(table_itr != all_table_by_key.end());
        auto const* ti = &(*table_itr);
        CHECK(ti);

        // add it as a table to be projected from in this query
        query_info.table_infos.emplace(ti);
      }

      // build SQL for the target expression
      QueryRenderer::ProjHitTestColSqlBuilder sql_builder(*target_expr, col_name);
      auto const sql = sql_builder.build();
      query_info.projections.push_back(sql);

      // done
      return;
    } else {
      // error as before
      throw std::runtime_error(
          "Column \"" + col_name + "\" from the query \"" + query_str +
          "\" references columns from separate tables. Hit-testing columns of "
          "this type is not currently supported (not a projection query)");
    }
  }

  if (has_rows && have_results_for_column) {
    // Found the column in the generated results, so convert it
    // into the return format expected
    populate_return_column();
  } else if (is_projection_query) {
    // TODO(croot): handle custom projections of existing columns. For
    // example, if the original SQL was something like this: "SELECT lon as x,
    // lat as y
    // ... FROM ... WHERE ..." and the column expression supplied by the user
    // for the hit-test was this: "conv_4326_900913_x(x) as new_x", we should
    // be able to run that expression using the expression from the original
    // query as an argument with the rowid sql. Using the example here, that
    // would ultimately equate to "SELECT conv_4326_900913_x(lon) as new_x
    // WHERE rowid=<>", but it's difficult to decipher whether the intent is
    // custom expressions against columns from the original query, or legacy
    // expressions. Best way to determine this is to check whether the expr
    // supplied by the user is found in the original sql, but as immerse only
    // generates legacy expressions, this isn't necessary currently.
    auto table_ptr_to_use = &(*all_table_info.begin());
    if (col_var) {
      auto& all_table_by_key =
          all_table_info.get<::QueryRenderer::HitTestTableInfo::KeyTag>();
      auto table_itr = all_table_by_key.find(col_var->getTableKey());
      if (table_itr != all_table_by_key.end()) {
        table_ptr_to_use = &(*table_itr);
      }
    }
    if (project_as_match.empty() && have_results_for_column) {
      auto expr = result_targets[result_target_idx]->get_expr();
      CHECK(expr) << "Invalid target entry "
                  << result_targets[result_target_idx]->get_resname()
                  << " at result_target_idx " << result_target_idx;

      // TODO(croot): this could be overkill. A regex on the original query
      // would be "safer", and perhaps easier, as it would assure that the
      // reconstruction of the SQL for the column would be intact, whereas the
      // sql builder is prone to bugs and incompleteness. However, I was
      // scared off from doing that due to how complex the regex would need to
      // be to handle the embedded, hierarchical nature of all the sql
      // expression possibilities, but it may not be all that bad. My
      // regex-foo is just not very strong.
      QueryRenderer::ProjHitTestColSqlBuilder sql_builder(*expr, col_name);
      query_info.table_infos.emplace(table_ptr_to_use);
      query_info.projections.push_back(sql_builder.build());
    } else {
      if (project_as_match.empty()) {
        // TODO(croot): this does not handle the case where the user is
        // requesting a column that exists in multiple tables from the query. To
        // resolve this arbitrary column, the user would need to supply the
        // column like this: <table>.<column>, but this is not supported yet.
        auto table_itr = std::find_if(all_table_info.begin(),
                                      all_table_info.end(),
                                      [&col_name](const auto table_info_ptr) {
                                        return table_info_ptr.hasColumn(col_name);
                                      });

        if (table_itr == all_table_info.end()) {
          LOG(WARNING) << "Cannot find the column '" << col_name
                       << "' in any of the hit-test tables from query '" << query_str
                       << "'.";
          invalidate_return_column(col_name);
          return;
        }
        table_ptr_to_use = &(*table_itr);
      }
      query_info.table_infos.emplace(table_ptr_to_use);
      query_info.projections.push_back(original_projection);
    }
  } else {
    LOG(WARNING) << "Cannot find column '" << col_name
                 << "' in hit-test cache for query '" << query_str << "'.";
    invalidate_return_column(col_name);
  }
}

std::string build_hittest_query_string_from_rows(
    const HitTestProjectionQueryInfo& info,
    const std::vector<std::pair<const Analyzer::TargetEntry*, size_t>>& rowid_targets,
    const ResultSet* result_row_ptr,
    const std::string& query_str,
    const TPixel& pixel) {
  // get the single table info
  auto const& hittest_table_info = info.getSingleTableInfo();

  // the rowids needed to do queries against the original tables haven't
  // been found yet. Look for it in the results. This can happen if a query
  // could run in-situ but was forced non-insitu for some reason.

  // copies the rowid column-name -> rowid number map
  // this is required because we'll be modifying the rowid item in that map
  // here
  decltype(hittest_table_info.col_rowid_map) col_rowid_map_copy;
  for (auto& rowid_item : rowid_targets) {
    // checking that the table is an owner of this result target. Ownership
    // is determined by checking whether the result_target is just a simple
    // alias of a column in a physical table or if the result target is
    // actually referenced by a view.
    auto owning_col_name =
        hittest_table_info.table_info.getOwningColName(*rowid_item.first);

    if (owning_col_name) {
      // verified that a rowid column belongs to this particular table.
      // Extract the result of the rowid projection from the results to use
      // in the hit-test query
      auto col_rowid_itr = hittest_table_info.col_rowid_map.find(*owning_col_name);
      CHECK(col_rowid_itr != hittest_table_info.col_rowid_map.end())
          << hittest_table_info.table_info.table_name << ":" << *owning_col_name << " - "
          << rowid_item.first->get_resname();
      const auto crt_row = result_row_ptr->getRowAt(col_rowid_itr->second);
      if (crt_row.empty()) {
        throw std::runtime_error(
            "Invalid hit-test result index " + std::to_string(col_rowid_itr->second) +
            " for column \"" + col_rowid_itr->first + "\" in render query \"" +
            query_str + "\" at pixel [" + std::to_string(pixel.x) + ", " +
            std::to_string(pixel.y) + "].");
      }

      const auto scalar_val = boost::get<ScalarTargetValue>(&crt_row[rowid_item.second]);
      CHECK(scalar_val) << "\"" + col_rowid_itr->first + "\" is not a scalar column";
      const auto int_val = boost::get<int64_t>(scalar_val);
      CHECK(int_val) << "\"" + col_rowid_itr->first + "\" must be an integer";

      col_rowid_map_copy[col_rowid_itr->first] = *int_val;
    }
  }

  // all rowid columns must be accounted for
  if (col_rowid_map_copy.size() != hittest_table_info.col_rowid_map.size()) {
    const auto& table_name = hittest_table_info.getTableNameRef();
    throw std::runtime_error(
        "Not all rowid columns are accounted for when hit-testing query "
        "\"" +
        query_str + "\". " + std::to_string(hittest_table_info.col_rowid_map.size()) +
        " rowid columns expected, but found " +
        std::to_string(col_rowid_map_copy.size()) +
        " rowid columns in the result for table \"" + table_name.table_name +
        "\" in catalog \"" + table_name.db_name + "\".");
  }

  // make a copy of info but with the replacement col_rowid_map
  QueryRenderer::HitTestTableInfo hittest_table_info_copy{hittest_table_info};
  hittest_table_info_copy.col_rowid_map = col_rowid_map_copy;
  HitTestProjectionQueryInfo info_copy{{&hittest_table_info_copy}, info.projections};

  return build_hittest_projection_query(info_copy);
}

}  // namespace

void RenderHandler::Impl::getResultRowForPixelLocal(
    TPixelTableRowResult& _return,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
    const TPixel& pixel,
    const VegaTableColNamesMap& table_col_names,
    const bool column_format,
    const std::string& vega_table_name,
    QueryRenderer::ResultCacheId cache_id,
    int64_t row_id) {
  TQueryResult ret;
  ret.row_set.is_columnar = column_format;

  // All render queries should now be cached in some way. Non-insitu queries will
  // have their result sets stored, whereas insisu queries will not. In-situ
  // queries however will have additional bit-shift offset data supplied to handle
  // multiple-rowids (to support join hit-testing).
  auto itr = table_col_names.find(vega_table_name);
  if (itr == table_col_names.end() || itr->second.empty()) {
    return;
  }

  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     ExecutorDeviceType::CPU,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     0,
                                                     0),
                         {}};

  // Now grab all the relevant query and rowid info from the initial hit-test
  // info. We'll need the result set (if it exists), the result targets, all the
  // relevant table info from the query, all the unpacked rowids, and whether the
  // query is a projection query. The projection query bool is used to determine
  // whether we can use the unpacked rowids to back-reference into the existing
  // tables for more column information.

  // NOTE: we're grabbing the result set with the
  // appropriate geo return type (if results have geo)
  // TODO(croot): handle other geo return types here, such as geojson

  // NOTE(scb): removed GeoReturnType since we are always passed WktString and it
  // was forcing inclusion of the massive ResultSet.h in QueryRenderManager.h just
  // to get the enum declaration! In the future if we need this enum it needs to
  // be handled differently.
  auto hittest_cache_results = render_manager_->getQueryCacheResults(
      cache_id, row_id /*, ResultSet::GeoReturnType::WktString*/);

  // TODO(croot): use c++17 structured bindings to flatten this
  auto& query_str = hittest_cache_results.query_str;
  auto result_row_ptr = hittest_cache_results.result_set_ptr;
  const auto& result_targets = hittest_cache_results.result_targets;
  auto& all_table_info = hittest_cache_results.hittest_table_info;
  CHECK_GT(all_table_info.size(), 0u);
  auto is_projection_query = hittest_cache_results.is_projection_query;

  if (!result_row_ptr || !result_targets.size()) {
    // cached results don't exist for the query.
    throw std::runtime_error("Cannot retrieve results for query \"" + query_str +
                             "\". The results have not been cached for hit-testing.");
  }

  const bool has_rows =
      !result_row_ptr->isExplain() && !result_row_ptr->definitelyHasNoRows();

  // used and modified by both build_query_infos() and populate_trow()
  size_t query_idx{}, result_target_idx{};
  TRow trow;

  auto const num_queries = itr->second.size();

  // this process needs to be done in the build_query_infos loop
  // but (of course) it requires completely different data from
  // the rest of that function, and only happens in some conditions,
  // and is nothing to do with actually building the query infos,
  // so to avoid having to ALSO pass all this different stuff into
  // build_query_infos(), define it as a lambda here and pass in the
  // lambda to be called if needed
  auto populate_return_column = [&]() {
    if (ret.row_set.row_desc.empty()) {
      ret.row_set.row_desc.resize(num_queries);
    }
    const auto& target_entry = *result_targets[result_target_idx];
    const auto expr = target_entry.get_expr();
    CHECK(expr) << "Invalid target entry " << target_entry.get_resname()
                << " at result_target_idx " << result_target_idx;
    const auto& type_info = expr->get_type_info();
    ret.row_set.row_desc[query_idx] = ThriftSerializers::target_meta_info_to_thrift(
        TargetMetaInfo(target_entry.get_resname(), type_info), query_idx);

    if (column_format) {
      if (ret.row_set.columns.empty()) {
        ret.row_set.columns.resize(num_queries);
      }
      const auto crt_row = result_row_ptr->getRowAt(row_id);
      if (crt_row.empty()) {
        throw std::runtime_error("get_result_row_for_pixel(): invalid result entry id " +
                                 std::to_string(row_id) + " for query \"" + query_str +
                                 "\" at pixel [" + std::to_string(pixel.x) + ", " +
                                 std::to_string(pixel.y) + "].");
      }
      TColumn tcol;
      db_handler_->value_to_thrift_column(crt_row[result_target_idx], type_info, tcol);
      ret.row_set.columns[query_idx] = tcol;
    } else {
      if (trow.cols.empty()) {
        trow.cols.resize(num_queries);
      }
      const auto crt_row = result_row_ptr->getRowAt(row_id);
      if (crt_row.empty()) {
        throw std::runtime_error("get_result_row_for_pixel(): invalid result entry id " +
                                 std::to_string(row_id) + " for query \"" + query_str +
                                 "\" at pixel [" + std::to_string(pixel.x) + ", " +
                                 std::to_string(pixel.y) + "].");
      }
      trow.cols[query_idx] =
          db_handler_->value_to_thrift(crt_row[result_target_idx], type_info);
    }
  };

  // the equivalent of the above, but mark the value as invalid
  auto invalidate_return_column = [&](const std::string& col_name) {
    static const NullableString kRETURN_STRING{
        "$HEAVYAI_ERROR_COLUMN_NOT_FOUND_IN_HIT_TEST_CACHE$"};
    static const SQLTypeInfo kRETURN_TYPEINFO(kTEXT);

    if (ret.row_set.row_desc.empty()) {
      ret.row_set.row_desc.resize(num_queries);
    }
    ret.row_set.row_desc[query_idx] = ThriftSerializers::target_meta_info_to_thrift(
        TargetMetaInfo(col_name, kRETURN_TYPEINFO), query_idx);

    if (column_format) {
      if (ret.row_set.columns.empty()) {
        ret.row_set.columns.resize(num_queries);
      }
      TColumn tcol;
      db_handler_->value_to_thrift_column(kRETURN_STRING, kRETURN_TYPEINFO, tcol);
      ret.row_set.columns[query_idx] = tcol;
    } else {
      if (trow.cols.empty()) {
        trow.cols.resize(num_queries);
      }
      trow.cols[query_idx] =
          db_handler_->value_to_thrift(kRETURN_STRING, kRETURN_TYPEINFO);
    }
  };

  std::vector<HitTestProjectionQueryInfo> hit_test_projection_query_infos(num_queries);

  // iterate through the columns requested by the user, and look for
  // it in the generated results (results are either cached or generated
  // by the above query)
  // If the particular column is not found, then keep those as we'll
  // run a final query to get those columns for a specific rowid of
  // the original table
  for (query_idx = 0; query_idx < num_queries; ++query_idx) {
    build_query_infos(result_targets,
                      all_table_info,
                      itr->second[query_idx],
                      query_str,
                      is_projection_query,
                      has_rows,
                      populate_return_column,
                      invalidate_return_column,
                      result_target_idx,
                      hit_test_projection_query_infos[query_idx]);
  }

  // Run a final query to generate results of the columns that weren't found
  // in the original render query.
  // Requires a rowid column of a reference-able table for backreferencing.
  if (is_projection_query) {
    auto rowid_regex = HitTestTypes::get_rowid_regex();
    std::smatch rowid_match;

    // if the hit-test cache has results, build up all the rowid columns for
    // back-referencing
    std::vector<std::pair<const Analyzer::TargetEntry*, size_t>> rowid_targets;
    if (has_rows) {
      for (result_target_idx = 0; result_target_idx < result_targets.size();
           ++result_target_idx) {
        if (std::regex_match(result_targets[result_target_idx]->get_resname(),
                             rowid_match,
                             rowid_regex)) {
          rowid_targets.emplace_back(result_targets[result_target_idx].get(),
                                     result_target_idx);
        }
      }
    }

    for (size_t query_idx = 0; query_idx < num_queries; ++query_idx) {
      auto const& info = hit_test_projection_query_infos[query_idx];
      if (info.table_infos.size() == 0u) {
        // no source tables gathered for this query, leave output slot empty
        continue;
      }

      std::string hittest_query_str;
      if (has_rows) {
        hittest_query_str = build_hittest_query_string_from_rows(
            info, rowid_targets, result_row_ptr, query_str, pixel);
      } else {
        hittest_query_str = build_hittest_projection_query(info);
      }

      ExecutionResult tmp_exec_result;
      lockmgr::LockedTableDescriptors locks;
      auto query_state = db_handler_->create_query_state(session_info, hittest_query_str);
      db_handler_->sql_execute_impl(tmp_exec_result,
                                    query_state->createQueryStateProxy(),
                                    column_format,
                                    ExecutorDeviceType::CPU,
                                    -1,
                                    -1,
                                    /*uses_calcite=*/true,
                                    locks);
      TQueryResult tmp_result;
      DBHandler::convertData(tmp_result,
                             tmp_exec_result,
                             query_state->createQueryStateProxy(),
                             column_format,
                             -1,
                             -1);

      // build out the results of from the hit-test query, placed in the
      // appropriate slot index
      auto const num_projections = info.projections.size();
      CHECK_EQ(tmp_result.row_set.row_desc.size(), num_projections);
      CHECK(column_format || tmp_result.row_set.rows.size() == 1)
          << "column_format: " << column_format
          << ", row size: " << tmp_result.row_set.rows.size();

      if (ret.row_set.row_desc.empty()) {
        ret.row_set.row_desc.resize(num_queries);
      }
      for (size_t projection_index = 0; projection_index < num_projections;
           ++projection_index) {
        ret.row_set.row_desc[query_idx] = tmp_result.row_set.row_desc[projection_index];

        if (column_format) {
          if (ret.row_set.columns.empty()) {
            ret.row_set.columns.resize(num_queries);
          }
          ret.row_set.columns[query_idx] = tmp_result.row_set.columns[projection_index];
        } else {
          // NOTE: already checked that the number of rows in tmp result is 1
          CHECK(!trow.cols.empty());
          trow.cols[query_idx] = tmp_result.row_set.rows[0].cols[projection_index];
        }
      }
    }
  }

  if (!trow.cols.empty()) {
    ret.row_set.rows.push_back(trow);
  }
  _return.row_set = ret.row_set;
}
