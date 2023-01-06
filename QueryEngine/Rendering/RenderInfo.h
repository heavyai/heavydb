/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include "Analyzer/Analyzer.h"
#include "Catalog/Catalog.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/Rendering/RenderAllocator.h"
#include "Shared/FullyQualifiedTableName.h"
#include "Shared/Rendering/InSituFlags.h"
#include "Shared/Rendering/RenderQueryOptions.h"

namespace QueryRenderer {
struct RenderSessionKey;
}

class RenderInfo : public heavyai::InSituFlagsOwnerInterface {
 public:
  std::unique_ptr<RenderAllocatorMap> render_allocator_map_ptr;
  const ::QueryRenderer::RenderSessionKey& render_session_key;

  // Info for all the column targets retrieved in in a query. Used to extract column/table
  // info when rendering.
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> targets;

  // All the "selected from" tables in a query. Includes resolved and un-resolved views.
  std::unordered_set<shared::FullyQualifiedTableName> table_names;

  RenderInfo(const ::QueryRenderer::RenderSessionKey& in_render_session_key,
             const RenderQueryOptions& in_render_query_opts,
             const heavyai::InSituFlags in_insitu_flags = heavyai::InSituFlags::kInSitu);

  const Catalog_Namespace::SessionInfo& getSessionInfo() const;
  std::shared_ptr<Catalog_Namespace::SessionInfo const> getSessionInfoPtr() const;

  void forceNonInSitu();
  void setNonInSitu();

  bool useCudaBuffers() const;
  void disableCudaBuffers();

  std::shared_ptr<QueryRenderer::QueryDataLayout> getQueryVboLayout() const;
  void setQueryVboLayout(
      const std::shared_ptr<QueryRenderer::QueryDataLayout>& vbo_layout);
  std::shared_ptr<QueryRenderer::QueryDataLayout> getQuerySsboLayout() const;
  void setQuerySsboLayout(
      const std::shared_ptr<QueryRenderer::QueryDataLayout>& ssbo_layout);

  const RenderQueryOptions& getRenderQueryOptions() const;

  bool setInSituDataIfUnset(const bool is_in_situ_data);

  void reset(std::unique_ptr<RenderQueryOptions> in_query_opts,
             const bool in_force_non_in_situ_data);

 private:
  enum class InSituState { UNSET, IS_IN_SITU, IS_NOT_IN_SITU };
  InSituState
      in_situ_data;  // Should be set to true if query results can be written directly to
                     // CUDA-mapped buffers for rendering. Should be set to false
                     // otherwise, meaning results are written to CPU first, and buffered
                     // back to GPU for rendering. An alternative meaning is that when
                     // false, you've encountered a non-projection query. Can only be set
                     // once for the lifetime of the object.
  bool force_non_in_situ_data;
  bool cuda_using_buffers_;

  std::shared_ptr<QueryRenderer::QueryDataLayout> query_vbo_layout;
  std::shared_ptr<QueryRenderer::QueryDataLayout> query_ssbo_layout;
  RenderQueryOptions render_query_opts_;
};
