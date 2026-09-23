/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ThriftHandler/RenderHandler.h"
#include <stdexcept>

#ifdef HAVE_RENDERING
#include "ExecuteRenderInterface/RenderHandlerImpl.h"
#else
class RenderHandler::Impl {};
#endif  // HAVE_RENDERING

namespace {
#ifndef HAVE_RENDERING
void throw_render_disabled() {
  throw std::runtime_error("Backend rendering is disabled.");
}
#endif  // HAVE_RENDERING
}  // namespace

RenderHandler::RenderHandler(DBHandler* db_handler,
                             gfx::GfxContext* gfx_context,
                             const size_t render_mem_bytes,
                             const size_t max_concurrent_render_sessions,
                             const bool compositor_use_last_gpu,
                             const bool enable_auto_clear_render_mem,
                             const int render_oom_retry_threshold,
                             const bool renderer_use_parallel_executors,
                             const SystemParameters system_parameters,
                             const bool renderer_enable_slab_allocation)
#ifdef HAVE_RENDERING
    : impl_(std::make_unique<Impl>(db_handler,
                                   gfx_context,
                                   render_mem_bytes,
                                   max_concurrent_render_sessions,
                                   compositor_use_last_gpu,
                                   enable_auto_clear_render_mem,
                                   render_oom_retry_threshold,
                                   renderer_use_parallel_executors,
                                   system_parameters,
                                   renderer_enable_slab_allocation)){}
#else
{
}
#endif  // HAVE_RENDERING

    RenderHandler::~RenderHandler() = default;

void RenderHandler::disconnect(const TSessionId& session) {
#ifdef HAVE_RENDERING
  impl_->disconnect(session);
#else
  throw_render_disabled();
#endif  // HAVE_RENDERING
}

void RenderHandler::render_vega(
    TRenderResult& _return,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
    const int64_t widget_id,
    std::string&& vega_json,
    const int32_t compression_level,
    const std::string& nonce) {
#ifdef HAVE_RENDERING
  impl_->render_vega(
      _return, session_info, widget_id, std::move(vega_json), compression_level, nonce);
#else
  throw_render_disabled();
#endif  // HAVE_RENDERING
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
#ifdef HAVE_RENDERING
  impl_->get_result_row_for_pixel(_return,
                                  session_info,
                                  widget_id,
                                  pixel,
                                  table_col_names,
                                  column_format,
                                  pixelRadius,
                                  nonce);
#else
  throw_render_disabled();
#endif  // HAVE_RENDERING
}

void RenderHandler::clear_gpu_memory() {
#ifdef HAVE_RENDERING
  impl_->clear_gpu_memory();
#else
  throw_render_disabled();
#endif  // HAVE_RENDERING
}

void RenderHandler::clear_cpu_memory() {
#ifdef HAVE_RENDERING
  impl_->clear_cpu_memory();
#else
  throw_render_disabled();
#endif  // HAVE_RENDERING
}

std::string RenderHandler::get_renderer_status_json() const {
#ifdef HAVE_RENDERING
  return impl_->get_renderer_status_json();
#else
  throw_render_disabled();
  return std::string();
#endif  // HAVE_RENDERING
}

bool RenderHandler::validate_renderer_status_json(
    const std::string& other_renderer_status_json) const {
#ifdef HAVE_RENDERING
  return impl_->validate_renderer_status_json(other_renderer_status_json);
#else
  throw_render_disabled();
  return false;
#endif  // HAVE_RENDERING
}

void RenderHandler::shutdown() {
#ifdef HAVE_RENDERING
  impl_->shutdown();
#else
  throw_render_disabled();
#endif  // HAVE_RENDERING
}
