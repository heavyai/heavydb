/*
 * Copyright 2019 OmniSci, Inc.
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

/*
 * File:   RenderHandler.h
 * Author: Chris Root
 *
 * Created on Dec 18, 2019, 10:00 AM
 */

#pragma once

#include "Shared/SystemParameters.h"
#include "gen-cpp/OmniSci.h"

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

class RenderHandler {
 public:
  // forward declaration of the implementation class to be defined later.
  // This is public as there can be certain functionality at lower levels that may want to
  // work directly with the implementation layer.
  class Impl;

  explicit RenderHandler(DBHandler* db_handler,
                         const size_t render_mem_bytes,
                         const size_t render_poly_cache_bytes,
                         const size_t max_conncurrent_render_sessions,
                         const bool enable_auto_clear_render_mem,
                         const int render_oom_retry_threshold,
                         const SystemParameters system_parameters);
  ~RenderHandler();

 private:
  void disconnect(const TSessionId& session);
  void render_vega(TRenderResult& _return,
                   const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
                   const int64_t widget_id,
                   const std::string& vega_json,
                   const int32_t compression_level,
                   const std::string& nonce);

  void start_render_query(TPendingRenderQuery& _return,
                          const TSessionId& session,
                          const int64_t widget_id,
                          const int16_t node_idx,
                          const std::string& vega_json);

  void execute_next_render_step(TRenderStepResult& _return,
                                const TPendingRenderQuery& pending_render,
                                const TRenderAggDataMap& merged_data);

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

  QueryRenderer::QueryRenderManager* get_render_manager();

  void shutdown();

  std::unique_ptr<Impl> impl_;

  friend class DBHandler;
};