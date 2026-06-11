/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/Rendering/RenderInfo.h"
#include "Shared/Rendering/RenderQueryOptions.h"

RenderInfo::RenderInfo(const ::QueryRenderer::RenderSessionKey& in_render_session_key,
                       const RenderQueryOptions& in_render_query_opts,
                       const heavyai::InSituFlags in_insitu_flags)
    : heavyai::InSituFlagsOwnerInterface(in_insitu_flags)
    , render_session_key(in_render_session_key)
    , render_query_opts_(in_render_query_opts) {
  CHECK(false);
}

const Catalog_Namespace::SessionInfo& RenderInfo::getSessionInfo() const {
  CHECK(false);
  static const Catalog_Namespace::SessionInfo tmp(
      nullptr,
      Catalog_Namespace::UserMetadata(-1, "", "", false, -1, false, false, ""),
      ExecutorDeviceType::CPU,
      "");
  return tmp;
}

std::shared_ptr<Catalog_Namespace::SessionInfo const> RenderInfo::getSessionInfoPtr()
    const {
  UNREACHABLE();
  return {};
}

void RenderInfo::forceNonInSitu() {
  CHECK(false);
}

void RenderInfo::setNonInSitu() {
  CHECK(false);
}

bool RenderInfo::useCudaBuffers() const {
  CHECK(false);
  return false;
}

void RenderInfo::disableCudaBuffers() {
  CHECK(false);
}

std::shared_ptr<QueryRenderer::QueryDataLayout> RenderInfo::getQueryVboLayout() const {
  CHECK(false);
  return nullptr;
}

void RenderInfo::setQueryVboLayout(
    const std::shared_ptr<QueryRenderer::QueryDataLayout>& vbo_layout) {
  CHECK(false);
}

std::shared_ptr<QueryRenderer::QueryDataLayout> RenderInfo::getQuerySsboLayout() const {
  CHECK(false);
  return nullptr;
}

void RenderInfo::setQuerySsboLayout(
    const std::shared_ptr<QueryRenderer::QueryDataLayout>& ssbo_layout) {
  CHECK(false);
}

const RenderQueryOptions& RenderInfo::getRenderQueryOptions() const {
  CHECK(false);
  return render_query_opts_;
}

void RenderInfo::reset(std::unique_ptr<RenderQueryOptions> in_query_opts,
                       const heavyai::InSituFlags in_insitu_flags) {
  CHECK(false);
}

void RenderInfo::setRenderQueryStr(std::string const& query_str) {
  CHECK(false);
}
std::string RenderInfo::getRenderQueryStr() const {
  CHECK(false);
  return "";
}
