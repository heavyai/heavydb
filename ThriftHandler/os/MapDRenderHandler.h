/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * File:   MapDRenderHandler.h
 * Author: Chris Root
 *
 * Created on Nov 6, 2017, 10:00 AM
 */

#ifndef MAPDRENDERHANDLER_H_
#define MAPDRENDERHANDLER_H_

#include "../MapDHandler.h"

class MapDRenderHandler {
 public:
  ~MapDRenderHandler() {}

 private:
  MapDRenderHandler(MapDHandler* mapd_handler,
                    const size_t render_mem_bytes,
                    const int num_gpus,
                    const int start_gpu) {
    throw std::runtime_error(
        "Rendering is only supported in the Enterprise and Community Editions");
  }

  void disconnect(const TSessionId& session) {}

  void render_vega(TRenderResult& _return,
                   const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
                   const int64_t widget_id,
                   const std::string& vega_json,
                   const int32_t compression_level,
                   const std::string& nonce) {
    CHECK(false);
  }

  void start_render_query(TPendingRenderQuery& _return,
                          const TSessionId& session,
                          const int64_t widget_id,
                          const int16_t node_idx,
                          const std::string& vega_json) {
    CHECK(false);
  }

  void execute_next_render_step(TRenderStepResult& _return,
                                const TPendingRenderQuery& pending_render,
                                const TRenderAggDataMap& merged_data) {
    CHECK(false);
  }

  void get_result_row_for_pixel(
      TPixelTableRowResult& _return,
      const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
      const int64_t widget_id,
      const TPixel& pixel,
      const std::map<std::string, std::vector<std::string>>& table_col_names,
      const bool column_format,
      const int32_t pixelRadius,
      const std::string& nonce) {
    CHECK(false);
  }

  void clear_gpu_memory() { CHECK(false); }
  void clear_cpu_memory() { CHECK(false); }

  ::QueryRenderer::QueryRenderManager* get_render_manager() {
    CHECK(false);
    return nullptr;
  }

  void handle_ddl(Parser::DDLStmt*) { CHECK(false); }
  void shutdown() { CHECK(false); }

  friend class MapDHandler;
};

#endif /* MAPDRENDERHANDLER_H_ */
