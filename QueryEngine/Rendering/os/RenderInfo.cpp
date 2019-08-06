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

#include "../RenderInfo.h"
#include "Shared/Rendering/RenderQueryOptions.h"

RenderInfo::RenderInfo(
    const std::shared_ptr<const ::QueryRenderer::RenderSession> in_render_session,
    RenderQueryOptions in_render_query_opts,
    const bool force_non_in_situ_data)
    : render_session(in_render_session)
    , render_query_opts_(std::move(in_render_query_opts)) {
  CHECK(false);
}

const Catalog_Namespace::SessionInfo& RenderInfo::getSessionInfo() const {
  CHECK(false);
  static const Catalog_Namespace::SessionInfo tmp(
      nullptr, Catalog_Namespace::UserMetadata(), ExecutorDeviceType::CPU, "");
  return tmp;
}

void RenderInfo::setForceNonInSituData() {
  CHECK(false);
}

bool RenderInfo::queryRanWithInSituData() const {
  CHECK(false);
  return false;
}

bool RenderInfo::hasInSituData() const {
  CHECK(false);
  return false;
}

bool RenderInfo::isInSituDataFlagUnset() const {
  CHECK(false);
  return false;
}

bool RenderInfo::isPotentialInSituRender() const {
  CHECK(false);
  return false;
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

bool RenderInfo::setInSituDataIfUnset(const bool is_in_situ_data) {
  CHECK(false);
  return false;
}

const RenderQueryOptions* RenderInfo::getRenderQueryOptsPtr() const {
  CHECK(false);
  return nullptr;
}

const RenderQueryOptions& RenderInfo::getRenderQueryOpts() const {
  CHECK(false);
  return render_query_opts_;
}

void RenderInfo::reset(RenderQueryOptions in_query_opts,
                       const bool disallow_in_situ_only_if_final_ED_is_aggregate_in) {
  CHECK(false);
}
