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

#ifndef QUERYENGINE_RENDERINFO_H
#define QUERYENGINE_RENDERINFO_H

#include <Analyzer/Analyzer.h>
#include <Catalog/Catalog.h>
#include <Shared/Rendering/RenderQueryOptions.h>
#include "../Descriptors/RowSetMemoryOwner.h"
#include "RenderAllocator.h"

namespace QueryRenderer {
struct RenderSession;
}  // namespace QueryRenderer

class RenderInfo {
 public:
  std::unique_ptr<RenderAllocatorMap> render_allocator_map_ptr;
  const std::shared_ptr<const ::QueryRenderer::RenderSession> render_session;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner;

  // Info for all the column targets retrieved in in a query. Used to extract column/table
  // info when rendering.
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> targets;

  // All the "selected from" tables in a query. Includes resolved and un-resolved views.
  std::unordered_set<std::string> table_names;
  bool disallow_in_situ_only_if_final_ED_is_aggregate;

  RenderInfo(
      const std::shared_ptr<const ::QueryRenderer::RenderSession> in_render_session,
      RenderQueryOptions in_render_query_opts,
      const bool force_non_in_situ_data = false);

  const Catalog_Namespace::SessionInfo& getSessionInfo() const;
  void setForceNonInSituData();
  bool queryRanWithInSituData() const;
  bool hasInSituData() const;
  bool isInSituDataFlagUnset() const;
  bool couldRunInSitu() const;
  bool isPotentialInSituRender() const;
  bool useCudaBuffers() const;
  void disableCudaBuffers();

  std::shared_ptr<QueryRenderer::QueryDataLayout> getQueryVboLayout() const;
  void setQueryVboLayout(
      const std::shared_ptr<QueryRenderer::QueryDataLayout>& vbo_layout);
  std::shared_ptr<QueryRenderer::QueryDataLayout> getQuerySsboLayout() const;
  void setQuerySsboLayout(
      const std::shared_ptr<QueryRenderer::QueryDataLayout>& ssbo_layout);

  const RenderQueryOptions* getRenderQueryOptsPtr() const;
  const RenderQueryOptions& getRenderQueryOpts() const;

  bool setInSituDataIfUnset(const bool is_in_situ_data);

  void reset(RenderQueryOptions in_query_opts,
             const bool disallow_in_situ_only_if_final_ED_is_aggregate_in);

 private:
  enum class InSituState { UNSET, IS_IN_SITU, IS_NOT_IN_SITU };
  InSituState
      in_situ_data;  // Should be set to true if query results can be written directly
                     // to CUDA-mapped opengl buffers for rendering. Should be set
                     // to false otherwise, meaning results are written to CPU first,
                     // and buffered back to GPU for rendering.
                     // An alternative meaning is that when false, you've encountered
                     // a non-projection query.
                     // Can only be set once for the lifetime of the object.
  bool force_non_in_situ_data;

  enum class RendererBufferMode { CUDA, GL };
  RendererBufferMode buffer_mode_;  // The Renderer buffer mode determines how query
                                    // results are bused to the Rendering engine.

  std::shared_ptr<QueryRenderer::QueryDataLayout> query_vbo_layout;
  std::shared_ptr<QueryRenderer::QueryDataLayout> query_ssbo_layout;
  RenderQueryOptions render_query_opts_;
};

#endif  // QUERYENGINE_RENDERINFO_H
