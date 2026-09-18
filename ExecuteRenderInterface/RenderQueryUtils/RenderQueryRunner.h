/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExecuteRenderInterface/RenderHandlerImpl.h"
#include "ExecuteRenderInterface/RenderQueryUtils/Metadata/RasterMeshMetadata.h"
#include "ExecuteRenderInterface/RenderQueryUtils/RenderQueryExecutionOptions.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryRenderer/Interface/RenderQueryRunnerInterface.h"

namespace QueryRenderer {

class RenderQueryRunner : public RenderQueryRunnerInterface {
 public:
  explicit RenderQueryRunner(DBHandler& db_handler,
                             RenderHandler::Impl& render_handler,
                             QueryRenderManager& render_manager,
                             const RenderSessionKey& render_session_key);

  RenderQueryRunner() = delete;
  ~RenderQueryRunner() override = default;

  static void setUseParallelExecutors(const bool use_parallel_executors);

  void notifyQueryExecutionComplete() const final;

  RenderInfo& getRenderInfo() { return render_info_; }
  QueryRenderManager& getRenderManager() { return render_manager_; }
  RenderHandler::Impl& getRenderHandler() { return render_handler_; }
  RenderQueryExecutionOptions getRenderQueryExecutionOptions() const {
    return render_query_exec_opts_;
  }

  RenderQueryParseData executeQueryParse(
      RenderQueryExecuteTimer& render_timer,
      const std::string& query_str,
      const JSONLocation*,
      const RenderQueryOptions& query_opts,
      const RenderQuerySpecialtyType render_query_type) final;

  RenderQueryExecuteData executeQuery(
      RenderQueryExecuteTimer& render_timer,
      const std::string& query_str,
      const JSONLocation* data_loc,
      const RenderQueryOptions& query_opts,
      const RenderQuerySpecialtyType render_query_type,
      const heavyai::InSituFlags insitu_flags = heavyai::InSituFlags::kInSitu) final;

  std::vector<int32_t> getStringIds(
      const QueryDataLayout& query_data_layout,
      const std::string& column_name,
      const std::vector<std::string>& column_values_to_convert,
      const ResultSet& results,
      const bool warn = false) const final;

  std::vector<std::string> getStringsFromIds(const QueryDataLayout& query_data_layout,
                                             const std::string& column_name,
                                             const std::vector<int32_t>& column_value_ids,
                                             const ResultSet& results) const final;

  /**
   * Parses a SQL query and produces a JSON representation of the Relational Algebra (RA)
   * tree. See: https://calcite.apache.org/docs/algebra.html
   * A list of referenced tables used in the query is stored in the RenderInfo argument,
   * and optionally acquires the table locks for the referenced tables.
   * Table-level locks will only be acquired if the QueryLockState argument != nullptr
   */
  inline static std::string parseToRAAndAcquireTableLocks(
      RenderHandler::Impl& render_handler,
      const std::string& sql_str,
      RenderInfo& render_info,
      RenderHandler::Impl::QueryLockState* query_lock_state) {
    return render_handler.parseToRelAlgAndAcquireLocks(
        sql_str, render_info, query_lock_state);
  }

  static std::unique_ptr<RelAlgDag> buildRelAlgDagAndClassifyRender(
      RenderInfo& render_info,
      const std::string& query_ra);

  static ExecutionResult validateQuery(
      RenderHandler::Impl& render_handler,
      RenderInfo& render_info,
      std::unique_ptr<RelAlgDag> rel_alg_dag,
      const RenderQueryExecutionOptions& render_query_exec_opts);

  static ExecutionResult executeQuery(
      RenderHandler::Impl& render_handler,
      RenderInfo& render_info,
      std::unique_ptr<RelAlgDag> rel_alg_dag,
      const RenderQueryExecutionOptions& render_query_exec_opts);

 protected:
  static bool use_parallel_executors_;
  RenderHandler::Impl& render_handler_;
  QueryRenderManager& render_manager_;

  // NOTE: need to put the render_info in this class so that it is properly
  // destroyed before exception handling one level up. This is to ensure that the
  // render_allocator unmaps its buffers beforehand as cleanly as possible.
  RenderInfo render_info_;

  RenderQueryExecutionOptions render_query_exec_opts_;

  RasterMeshMetadata raster_mesh_metadata_;

  const DBHandler& getDBHandler() const {
    CHECK(render_handler_.db_handler_);
    return *render_handler_.db_handler_;
  }

  virtual RenderQueryExecuteData executeRenderQueryImpl(
      RenderQueryExecuteTimer& render_timer,
      const std::string& query_ra,
      std::unique_ptr<RelAlgDag> rel_alg_dag,
      RenderHandler::Impl::QueryLockState&& query_lock_state,
      const JSONLocation* data_loc,
      const RenderQuerySpecialtyType render_query_type);

  void initRenderAllocator();

  SQLSelectedTableContainer getReferencedTablesPostQueryRun(
      const RenderQuerySpecialtyType render_query_type);

  std::pair<ExecutionResult, QueryDataLayoutShPtr> getDataLayoutFromRelAlgDag(
      std::unique_ptr<RelAlgDag> rel_alg_dag);

  RenderQueryExecuteData executeRenderQuery(
      RenderQueryExecuteTimer& render_timer,
      std::unique_ptr<RelAlgDag> rel_alg_dag,
      RenderHandler::Impl::QueryLockState&& query_lock_state,
      const JSONLocation* data_loc,
      const RenderQuerySpecialtyType render_query_type);
};

}  // namespace QueryRenderer
