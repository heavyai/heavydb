/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderHandlerImpl.h"

#include "ExecuteRenderInterface/RenderQueryUtils/RenderQueryRunner.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/Enums.h"
#include "QueryRenderer/Interface/ResultCacheTypes.h"
#include "QueryRenderer/PngData.h"
#include "QueryRenderer/QueryRenderManager.h"
#include "QueryRenderer/Utils/StringUtils.h"
#include "ThriftHandler/DBHandler.h"
#include "gen-cpp/calciteserver_types.h"

RenderHandler::Impl::Impl(DBHandler* db_handler,
                          gfx::GfxContext* gfx_context,
                          const size_t render_mem_bytes,
                          const size_t max_concurrent_render_sessions,
                          const bool compositor_use_last_gpu,
                          const bool enable_auto_clear_render_mem,
                          const int render_oom_retry_threshold,
                          const bool renderer_use_parallel_executors,
                          const SystemParameters system_parameters,
                          const bool renderer_enable_slab_allocation)
    : db_handler_(db_handler)
    , enable_auto_clear_render_mem_(enable_auto_clear_render_mem)
    , render_oom_retry_threshold_(render_oom_retry_threshold)
    , system_parameters_(system_parameters) {
  CHECK(db_handler_);
  ::QueryRenderer::RenderQueryRunner::setUseParallelExecutors(
      renderer_use_parallel_executors);
  render_manager_ = std::make_unique<::QueryRenderer::QueryRenderManager>(
      gfx_context,
      db_handler_->data_mgr_.get(),
      render_mem_bytes,
      max_concurrent_render_sessions,
      compositor_use_last_gpu,
      gfx::RasterSampleCount::k4,
      renderer_enable_slab_allocation);
}

RenderHandler::Impl::~Impl() {}

void RenderHandler::Impl::disconnect(const TSessionId& session) {
  std::lock_guard<std::mutex> render_lock(render_mutex_);
  render_manager_->removeSessionId(session);
}

void RenderHandler::Impl::render_vega(
    TRenderResult& _return,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
    const int64_t widget_id,
    std::string&& vega_json,
    const int32_t compression_level,
    const std::string& nonce) {
  RENDER_LOG_SCOPE();
  _return.execution_time_ms = 0;
  _return.render_time_ms = 0;
  _return.nonce = nonce;

  QueryRenderer::PngData::validateCompressionLevel(compression_level);

  QueryRenderer::RenderRequestInfo render_request_info;

  ScopeGuard log_at_exit = [&_return, &nonce] {
    LOG(INFO) << "render_vega-COMPLETED nonce:" << nonce
              << " Total Execution: " << _return.execution_time_ms
              << " (ms), Total Render: " << _return.render_time_ms << " (ms)";
  };

  {
    // take render lock
    std::lock_guard<std::mutex> render_lock(render_mutex_);

    LOG(INFO) << "render_vega :" << *session_info << ":widget_id:" << widget_id
              << ":compression_level:" << compression_level << ":vega_json:" << vega_json
              << ":nonce:" << nonce;

    CHECK(session_info);

    auto vega = std::move(vega_json);
    const QueryRenderer::RenderSession* render_session{nullptr};

    render_request_info = check_oom_w_retry(*this, [&](const size_t retry_cnt) {
      render_session = &render_manager_->getOrCreateRenderSession(
          session_info, widget_id, std::move(vega));

      CHECK(render_session);

      try {
        return render_manager_->runRenderRequest(
            *render_session,
            std::make_unique<QueryRenderer::RenderQueryRunner>(
                *db_handler_, *this, *render_manager_, render_session->getKey()));
      } catch (...) {
        // intentional copy of the vega in the case the render_session is cleared out
        // during a retry
        vega = render_session->getVegaJSON();
        std::rethrow_exception(std::current_exception());
      }
    });

    _return.vega_metadata = render_session->getRenderContext().serializeToVega();
  }  // release render lock

  // now do the PNG encoding
  // any renderData that gets to here must be a RenderPixels
  // Thrift is now free to call render_vega again from a different thread
  // @TODO(se) do the un-multiply here too?
  auto png_timer = timer_start();
  auto const& pixels = render_request_info.renderData;
  RUNTIME_EX_ASSERT((pixels.width > 0) && (pixels.height > 0) && (!pixels.pixels.empty()),
                    "Unable to encode PNG, invalid RenderPixels");
  _return.image = QueryRenderer::QueryRenderManager::encodePNG(pixels, compression_level);
  auto png_time_ms = timer_stop(png_timer);

  _return.execution_time_ms = render_request_info.total_execution_time_ms;
  _return.render_time_ms = render_request_info.total_render_time_ms + png_time_ms;
}

std::string RenderHandler::Impl::dump_table_col_names(
    const VegaTableColNamesMap& table_col_names) {
  std::ostringstream oss;
  for (const auto& table_col : table_col_names) {
    oss << ":" << table_col.first;
    for (const auto& col : table_col.second) {
      oss << "," << col;
    }
  }
  return oss.str();
}

void RenderHandler::Impl::get_result_row_for_pixel(
    TPixelTableRowResult& _return,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
    const int64_t widget_id,
    const TPixel& pixel,
    const VegaTableColNamesMap& table_col_names,
    const bool column_format,
    const int32_t pixel_radius,
    const std::string& nonce) {
  RENDER_LOG_SCOPE();
  _return.nonce = nonce;
  std::lock_guard<std::mutex> render_lock(render_mutex_);
  CHECK(session_info);

  LOG(INFO) << "get_result_row_for_pixel :" << *session_info << ":widget_id:" << widget_id
            << ":pixel.x:" << pixel.x << ":pixel.y:" << pixel.y
            << ":column_format:" << column_format << ":pixel_radius:" << pixel_radius
            << ":table_col_names" << dump_table_col_names(table_col_names)
            << ":nonce:" << nonce;

  int64_t total_time = 0;
  ScopeGuard log_at_exit = [&total_time, &nonce] {
    LOG(INFO) << "get_result_row_for_pixel-COMPLETED nonce:" << nonce
              << ", Execute Time: " << total_time << " (ms)";
  };

  total_time = measure<>::execution([&]() {
    CHECK(render_manager_);

    _return.pixel = pixel;

    if (!render_manager_->hasRenderSession(
            QueryRenderer::RenderSessionKey(session_info, widget_id))) {
      _return.table_id = {-1};
      _return.row_id = {-1};
      _return.vega_table_name = "";
      return;
    }

    // render session known to exist
    auto const& render_session =
        render_manager_->getOrCreateRenderSession(session_info, widget_id, "");

    QueryRenderer::ResultCacheId cache_id;
    int64_t row_id;
    std::string vega_table_name;
    int16_t node_idx;
    std::tie(cache_id, row_id, vega_table_name, node_idx) =
        render_manager_->getIdAt(render_session, pixel.x, pixel.y, pixel_radius);

    _return.vega_table_name = vega_table_name;
    _return.table_id = {cache_id};
    _return.row_id = {row_id};

    // NOTE: 0 for cache_id or -1 for row_id indicates nothing was hit
    if (cache_id != 0 && row_id >= 0) {
      getResultRowForPixelLocal(_return,
                                session_info,
                                pixel,
                                table_col_names,
                                column_format,
                                vega_table_name,
                                cache_id,
                                row_id);
    }
  });
}

void RenderHandler::Impl::clear_gpu_memory() {
  std::lock_guard<std::mutex> render_lock(render_mutex_);
  clear_gpu_memory_impl();
}

void RenderHandler::Impl::clear_gpu_memory_impl() {
  CHECK(render_manager_);
  LOG(INFO) << "clearing render context/memory on gpus "
            << QueryRenderer::to_string(render_manager_->getAllGpuIds());
  render_manager_->clearGpuMemory();
}

void RenderHandler::Impl::clear_cpu_memory() {
  std::lock_guard<std::mutex> render_lock(render_mutex_);
  CHECK(render_manager_);
  render_manager_->clearCpuMemory();
}

std::string RenderHandler::Impl::get_renderer_status_json() const {
  CHECK(render_manager_);
  return render_manager_->getRendererStatusJSON();
}

bool RenderHandler::Impl::validate_renderer_status_json(
    const std::string& other_renderer_status_json) const {
  CHECK(render_manager_);
  return render_manager_->validateRendererStatusJSON(other_renderer_status_json);
}

void RenderHandler::Impl::shutdown() {
  render_manager_.reset();
}

RenderHandler::Impl::QueryLockState::QueryLockState()
    : execute_read_lock_(legacylockmgr::getExecuteReadLock()) {}

void RenderHandler::Impl::QueryLockState::takeOwnershipOfTableLocks(
    lockmgr::LockedTableDescriptors&& table_descriptor_locks) {
  RENDER_LOG_SCOPE();
  CHECK(execute_read_lock_ && execute_read_lock_.owns_lock())
      << "Execute read lock needs to be acquired first before table descriptor locks";
  CHECK_EQ(table_descriptor_locks_.size(), 0ul);
  table_descriptor_locks_ = std::move(table_descriptor_locks);
}

std::string RenderHandler::Impl::parseToRelAlgAndAcquireLocks(
    const std::string& query_str,
    RenderInfo& render_info,
    QueryLockState* query_lock_state) {
  RENDER_LOG_SCOPE();
  auto query_state =
      db_handler_->create_query_state(render_info.getSessionInfoPtr(), query_str);

  const bool acquire_locks = query_lock_state != nullptr;
  RENDER_LOG() << "*** calling parse_to_ra (calcite)";
  auto [thrift_plan_result, locks] =
      db_handler_->parse_to_ra(query_state->createQueryStateProxy(),
                               query_str,
                               {},
                               acquire_locks,
                               system_parameters_);

  // grabs all the selected-from tables, even views. This is used by the renderer to
  // resolve view hit-testing.
  // NOTE: the same table name could exist in both the primary and resolved tables.

  auto& primary_tables = thrift_plan_result.primary_accessed_objects.tables_selected_from;
  for (auto& name : primary_tables) {
    render_info.table_names.emplace(name[1], name[0]);
  }
  auto& resolved_tables =
      thrift_plan_result.resolved_accessed_objects.tables_selected_from;
  for (auto& name : resolved_tables) {
    render_info.table_names.emplace(name[1], name[0]);
  }

  if (query_lock_state) {
    query_lock_state->takeOwnershipOfTableLocks(std::move(locks));
  }
  return thrift_plan_result.plan_result;
}

ExecutionResult RenderHandler::Impl::sqlExecute(
    const std::shared_ptr<const Catalog_Namespace::SessionInfo>& session_info,
    const std::string& sql_query_str) {
  RENDER_LOG_SCOPE();
  ExecutionResult exec_result;
  lockmgr::LockedTableDescriptors locks;
  auto query_state = db_handler_->create_query_state(session_info, sql_query_str);
  db_handler_->sql_execute_impl(exec_result,
                                query_state->createQueryStateProxy(),
                                /*column_format=*/true,
                                session_info->get_executor_device_type(),
                                -1,
                                -1,
                                /*uses_calcite=*/true,
                                locks);
  return exec_result;
}

void RenderHandler::Impl::dispatch_query_task(
    std::shared_ptr<QueryDispatchQueue::Task> query_task) {
  return db_handler_->dispatch_query_task(std::move(query_task), false);
}
