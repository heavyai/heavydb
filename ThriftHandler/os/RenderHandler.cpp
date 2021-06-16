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
 * File:   RenderHandler.cpp
 * Author: Chris Root
 *
 * Created on Dec 18, 2019, 10:00 AM
 */

#include "ThriftHandler/RenderHandler.h"

#include "Logger/Logger.h"

class RenderHandler::Impl {};

RenderHandler::RenderHandler(DBHandler* db_handler,
                             const size_t render_mem_bytes,
                             const size_t max_concurrent_render_sessions,
                             const bool compositor_use_last_gpu,
                             const bool enable_auto_clear_render_mem,
                             const int render_oom_retry_threshold,
                             const bool renderer_use_vulkan_driver,
                             const SystemParameters system_parameters)
    : impl_(nullptr) {
  throw std::runtime_error(
      "Rendering is only supported in the Enterprise and Community Editions");
}

RenderHandler::~RenderHandler() = default;

void RenderHandler::disconnect(const TSessionId& session) {
  CHECK(impl_);
}

void RenderHandler::render_vega(
    TRenderResult& _return,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
    const int64_t widget_id,
    const std::string& vega_json,
    const int32_t compression_level,
    const std::string& nonce) {
  CHECK(impl_);
}

void RenderHandler::start_render_query(TPendingRenderQuery& _return,
                                       const TSessionId& session,
                                       const int64_t widget_id,
                                       const int16_t node_idx,
                                       const std::string& vega_json) {
  CHECK(impl_);
}

void RenderHandler::execute_next_render_step(TRenderStepResult& _return,
                                             const TPendingRenderQuery& pending_render,
                                             const TRenderAggDataMap& merged_data) {
  CHECK(impl_);
}

void RenderHandler::get_result_row_for_pixel(
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

void RenderHandler::clear_gpu_memory() {
  CHECK(impl_);
}

void RenderHandler::clear_cpu_memory() {
  CHECK(impl_);
}

std::string RenderHandler::get_renderer_status_json() const {
  CHECK(impl_);
  return std::string();
}

bool RenderHandler::validate_renderer_status_json(
    const std::string& other_renderer_status_json) const {
  CHECK(impl_);
  return false;
}

void RenderHandler::shutdown() {
  CHECK(impl_);
}
