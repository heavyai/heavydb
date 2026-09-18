/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ThriftHandler/RenderHandler.h"

#include <boost/core/noncopyable.hpp>

#include "GfxDriver/RenderError.h"
#include "LockMgr/LockMgr.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/Rendering/RenderInfo.h"
#include "QueryRenderer/Interface/ResultCacheTypes.h"
#include "Shared/measure.h"

namespace QueryRenderer {
class RenderQueryRunner;
class RenderSession;
}  // namespace QueryRenderer

class RenderHandler::Impl : public boost::noncopyable {
  // check_oom should only be called by check_oom_w_retry
  template <class UnaryFunction>
  static decltype(auto) check_oom(RenderHandler::Impl& render_handler, UnaryFunction f) {
    try {
      return f();
    } catch (gfx::OutOfGpuMemoryError& e) {
      if (render_handler.enable_auto_clear_render_mem_) {
        // auto clear gpu memory - assumes lock is acquired on the
        // QueryRenderManager::_renderMtx mutex
        LOG(INFO) << "Caught an out-of-gpu-memory error in the renderer. The gpu memory "
                     "consumed by the renderer will be purged.";
        render_handler.clear_gpu_memory_impl();
      }
      std::rethrow_exception(std::current_exception());
    }
  }

 public:
  Impl() = delete;
  explicit Impl(DBHandler* db_handler,
                gfx::GfxContext* gfx_context,
                const size_t render_mem_bytes,
                const size_t max_concurrent_render_sessions,
                const bool compositor_use_last_gpu,
                const bool enable_auto_clear_render_mem,
                const int render_oom_retry_threshold,
                const bool renderer_use_parallel_executors,
                const SystemParameters system_parameters,
                const bool renderer_enable_slab_allocation);

  ~Impl();

  template <class UnaryFunction>
  static decltype(auto) check_oom_w_retry(RenderHandler::Impl& render_handler,
                                          UnaryFunction f) {
    size_t retry_cnt{0};
    auto call_f_w_retry_cnt = [&]() -> decltype(auto) { return f(retry_cnt++); };

    auto timer = timer_start();
    try {
      return check_oom(render_handler, call_f_w_retry_cnt);
    } catch (gfx::OutOfGpuMemoryError& e) {
      if (render_handler.enable_auto_clear_render_mem_) {
        // do a single retry if the query took less than configured threshold (in ms)
        auto time_ms = timer_stop(timer);
        if (time_ms < render_handler.render_oom_retry_threshold_) {
          LOG(INFO) << "Caught an out-of-gpu-memory error \"" << e.what()
                    << "\" while trying to render the vega. Attempting retry #"
                    << retry_cnt << ".";
          return check_oom(render_handler, call_f_w_retry_cnt);
        }
        LOG(INFO) << "Caught an out-of-gpu-memory error while trying to render the "
                     "vega but didn't retry because it took "
                  << time_ms << "ms on the first attempt. Retry threshold is "
                  << render_handler.render_oom_retry_threshold_ << "ms.";
      }
      std::rethrow_exception(std::current_exception());
    } catch (gfx::DeviceLostError& e) {
      LOG(INFO) << "Caught Vulkan device lost error. Restarting renderer";
      render_handler.clear_gpu_memory_impl();
      std::rethrow_exception(std::current_exception());
    }
  }

  /**
   * A moveable-only class that holds onto the locks required for query execution via
   * RelAlgExecutor. These locks are as follows (and should be acquired in this order):
   *
   *   1) ExecutorOuterLock: This is the outermost lock that should be acquired for all
   * query executions
   *      - acquired upon instantiation of this class
   *   2) Table Descriptor locks: Read locks on the table referenced in the query
   *      - acquired via parse_to_ra
   *
   * The locks are released when this class is destroyed.
   *
   * The locking order here should follow that of the sql_validate/sql_execute endpoints.
   */
  class QueryLockState {
   public:
    QueryLockState();

    // set to be moveable-only, due to the locks
    QueryLockState(QueryLockState&&) = default;
    QueryLockState& operator=(QueryLockState&&) = default;

    /**
     * Should be called immediately after a MapDRenderHandler::parse_to_ra to take
     * ownership of the locks acquired via that API
     */
    void takeOwnershipOfTableLocks(
        lockmgr::LockedTableDescriptors&& table_descriptor_locks);

   private:
    legacylockmgr::ExecutorReadLock execute_read_lock_;
    lockmgr::LockedTableDescriptors table_descriptor_locks_;
  };

  /**
   * Parses a SQL query and produces a JSON representation of the Relational Algebra (RA)
   * tree. See: https://calcite.apache.org/docs/algebra.html
   * A list of referenced tables used in the query is stored in the RenderInfo argument,
   * and optionally acquires the table locks for the referenced tables.
   * Table-level locks Will only be acquired if the QueryLockState argument != nullptr
   */
  std::string parseToRelAlgAndAcquireLocks(const std::string& query_str,
                                           RenderInfo& render_info,
                                           QueryLockState* query_lock_state);

  /**
   * Parses and executes a standalone sql query
   */
  ExecutionResult sqlExecute(
      const std::shared_ptr<const Catalog_Namespace::SessionInfo>& session_info,
      const std::string& sql_query_str);

 private:
  void disconnect(const TSessionId& session);

  void render_vega(TRenderResult& _return,
                   const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
                   const int64_t widget_id,
                   std::string&& vega_json,
                   const int32_t compression_level,
                   const std::string& nonce);

  using VegaTableColNamesMap = std::map<std::string, std::vector<std::string>>;
  static std::string dump_table_col_names(const VegaTableColNamesMap& table_col_names);
  void get_result_row_for_pixel(
      TPixelTableRowResult& _return,
      const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
      const int64_t widget_id,
      const TPixel& pixel,
      const VegaTableColNamesMap& table_col_names,
      const bool column_format,
      const int32_t pixel_radius,
      const std::string& nonce);

  void clear_gpu_memory();
  void clear_gpu_memory_impl();
  void clear_cpu_memory();

  std::string get_renderer_status_json() const;
  bool validate_renderer_status_json(const std::string& other_renderer_status_json) const;

  void shutdown();

  std::mutex render_mutex_;
  DBHandler* db_handler_;
  const bool enable_auto_clear_render_mem_;
  const int render_oom_retry_threshold_;
  SystemParameters system_parameters_;

  std::unique_ptr<QueryRenderer::QueryRenderManager> render_manager_;

  void dispatch_query_task(std::shared_ptr<QueryDispatchQueue::Task> query_task);

  void getResultRowForPixelLocal(
      TPixelTableRowResult& _return,
      const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
      const TPixel& pixel,
      const VegaTableColNamesMap& table_col_names,
      const bool column_format,
      const std::string& vega_table_name,
      QueryRenderer::ResultCacheId cache_id,
      int64_t row_id);

  friend class RenderHandler;
  friend class QueryRenderer::RenderQueryRunner;
};
