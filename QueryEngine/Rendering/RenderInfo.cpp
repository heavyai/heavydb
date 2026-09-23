/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/Rendering/RenderInfo.h"
#include "Shared/Rendering/RenderQueryOptions.h"
#ifdef HAVE_RENDERING
#include "QueryRenderer/Interface/RenderSessionKey.h"
#endif  // HAVE_RENDERING

RenderInfo::RenderInfo(const QueryRenderer::RenderSessionKey& in_render_session_key,
                       const RenderQueryOptions& in_render_query_opts,
                       const heavyai::InSituFlags in_insitu_flags)
    : heavyai::InSituFlagsOwnerInterface(in_insitu_flags)
    , render_session_key{in_render_session_key}
    , cuda_using_buffers_{true}
    , render_query_opts_{in_render_query_opts} {}

const Catalog_Namespace::SessionInfo& RenderInfo::getSessionInfo() const {
#ifdef HAVE_RENDERING
  return render_session_key.getSessionInfo();
#else
  CHECK(false);
  static const Catalog_Namespace::SessionInfo tmp(
      nullptr,
      Catalog_Namespace::UserMetadata(-1, "", "", false, -1, false, false, ""),
      ExecutorDeviceType::CPU,
      "");
  return tmp;
#endif  // HAVE_RENDERING
}

std::shared_ptr<Catalog_Namespace::SessionInfo const> RenderInfo::getSessionInfoPtr()
    const {
#ifdef HAVE_RENDERING
  return render_session_key.getSessionInfoPtr();
#else
  UNREACHABLE();
  return {};
#endif  // HAVE_RENDERING
}

void RenderInfo::forceNonInSitu() {
  insitu_flags_ |= heavyai::InSituFlags::kNonInSitu;
}

void RenderInfo::setNonInSitu() {
  insitu_flags_ = heavyai::InSituFlags::kNonInSitu;
}

bool RenderInfo::useCudaBuffers() const {
  return cuda_using_buffers_;
}

void RenderInfo::disableCudaBuffers() {
  cuda_using_buffers_ = false;
}

std::shared_ptr<QueryRenderer::QueryDataLayout> RenderInfo::getQueryVboLayout() const {
  return query_vbo_layout;
}

void RenderInfo::setQueryVboLayout(
    const std::shared_ptr<QueryRenderer::QueryDataLayout>& vbo_layout) {
  CHECK(!query_vbo_layout);
  query_vbo_layout = vbo_layout;
}

std::shared_ptr<QueryRenderer::QueryDataLayout> RenderInfo::getQuerySsboLayout() const {
  return query_ssbo_layout;
}

void RenderInfo::setQuerySsboLayout(
    const std::shared_ptr<QueryRenderer::QueryDataLayout>& ssbo_layout) {
  CHECK(!query_ssbo_layout);
  query_ssbo_layout = ssbo_layout;
}

const RenderQueryOptions& RenderInfo::getRenderQueryOptions() const {
  return render_query_opts_;
}

void RenderInfo::reset(const RenderQueryOptions& in_query_opts,
                       const heavyai::InSituFlags in_insitu_flags) {
  render_query_opts_ = in_query_opts;
  insitu_flags_ = in_insitu_flags;
  cuda_using_buffers_ = true;
  query_vbo_layout = nullptr;
  query_ssbo_layout = nullptr;
  targets.clear();
  table_names.clear();
}

void RenderInfo::setRenderQueryStr(std::string const& query_str) {
  render_query_str_ = query_str;
}
std::string RenderInfo::getRenderQueryStr() const {
  return render_query_str_;
}
