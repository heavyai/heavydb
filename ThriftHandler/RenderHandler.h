/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file   RenderHandler.h
 * @brief
 *
 */

#pragma once

#include "Shared/SystemParameters.h"
#include "gen-cpp/Heavy.h"

class DBHandler;

namespace Catalog_Namespace {
class SessionInfo;
}

namespace QueryRenderer {
class QueryRenderManager;
}  // namespace QueryRenderer

namespace Parser {
class DDLStmt;
}

namespace gfx {
class GfxContext;
}

class RenderHandler {
 public:
  // forward declaration of the implementation class to be defined later.
  // This is public as there can be certain functionality at lower levels that may want to
  // work directly with the implementation layer.
  class Impl;

  explicit RenderHandler(DBHandler* db_handler,
                         gfx::GfxContext* gfx_context,
                         const size_t render_mem_bytes,
                         const size_t max_conncurrent_render_sessions,
                         const bool compositor_use_last_gpu,
                         const bool enable_auto_clear_render_mem,
                         const int render_oom_retry_threshold,
                         const bool renderer_use_parallel_executors,
                         const SystemParameters system_parameters,
                         const bool renderer_enable_slab_allocation);
  ~RenderHandler();

 private:
  void disconnect(const TSessionId& session);
  void render_vega(TRenderResult& _return,
                   const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
                   const int64_t widget_id,
                   std::string&& vega_json,
                   const int32_t compression_level,
                   const std::string& nonce);

  void get_result_row_for_pixel(
      TPixelTableRowResult& _return,
      const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
      const int64_t widget_id,
      const TPixel& pixel,
      const std::map<std::string, std::vector<std::string>>& table_col_names,
      const bool column_format,
      const int32_t pixelRadius,
      const std::string& nonce);

  void clear_gpu_memory();
  void clear_cpu_memory();

  std::string get_renderer_status_json() const;
  bool validate_renderer_status_json(const std::string& other_renderer_status_json) const;

  void shutdown();

  std::unique_ptr<Impl> impl_;

  friend class DBHandler;
};
