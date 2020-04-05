/*
 * Copyright 2019 OmnSci, Inc.
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
 * File:   MapDRenderHandler.cpp
 * Author: Chris Root
 *
 * Created on Dec 18, 2019, 10:00 AM
 */

#include "ThriftHandler/MapDRenderHandler.h"

#include "Shared/Logger.h"

class MapDRenderHandler::Impl {};

MapDRenderHandler::MapDRenderHandler(MapDHandler* mapd_handler,
                                     const size_t render_mem_bytes,
                                     const size_t render_poly_cache_bytes,
                                     const size_t max_concurrent_render_sessions,
                                     const bool enable_auto_clear_render_mem,
                                     const int render_oom_retry_threshold,
                                     const SystemParameters mapd_parameters)
    : impl_(nullptr) {
  throw std::runtime_error(
      "Rendering is only supported in the Enterprise and Community Editions");
}

MapDRenderHandler::~MapDRenderHandler() = default;

void MapDRenderHandler::disconnect(const TSessionId& session) {
  CHECK(impl_);
}

void MapDRenderHandler::render_vega(
    TRenderResult& _return,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
    const int64_t widget_id,
    const std::string& vega_json,
    const int32_t compression_level,
    const std::string& nonce) {
  CHECK(impl_);
}

void MapDRenderHandler::start_render_query(TPendingRenderQuery& _return,
                                           const TSessionId& session,
                                           const int64_t widget_id,
                                           const int16_t node_idx,
                                           const std::string& vega_json) {
  CHECK(impl_);
}

void MapDRenderHandler::execute_next_render_step(
    TRenderStepResult& _return,
    const TPendingRenderQuery& pending_render,
    const TRenderAggDataMap& merged_data) {
  CHECK(impl_);
}

void MapDRenderHandler::get_result_row_for_pixel(
    TPixelTableRowResult& _return,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
    const int64_t widget_id,
    const TPixel& pixel,
    const std::map<std::string, std::vector<std::string>>& table_col_names,
    const bool column_format,
    const int32_t pixelRadius,
    const std::string& nonce) {
  CHECK(impl_);
}

void MapDRenderHandler::clear_gpu_memory() {
  CHECK(impl_);
}

void MapDRenderHandler::clear_cpu_memory() {
  CHECK(impl_);
}

QueryRenderer::QueryRenderManager* MapDRenderHandler::get_render_manager() {
  CHECK(impl_);
  return nullptr;
}

void MapDRenderHandler::handle_ddl(Parser::DDLStmt*) {
  CHECK(impl_);
}
void MapDRenderHandler::shutdown() {
  CHECK(impl_);
}
