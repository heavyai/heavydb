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

#include "RenderAllocator.h"
#include "../ResultRows.h"

class RenderInfo {
 public:
  std::unique_ptr<RenderAllocatorMap> render_allocator_map_ptr;
  const std::string session_id;
  const int render_widget_id;
  const std::string render_vega;
  const size_t render_small_groups_buffer_entry_count{2 * 1024 * 1024};
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner;
  std::vector<std::shared_ptr<Analyzer::TargetEntry>> targets;  // Info for all the column targets retrieved in
                                                                // in a query. Used to extract column/table info
                                                                // when rendering.
  std::vector<std::string> table_names;  // the names of all the tables used in a query in hierarchical order.
                                         // For example, for join queries, the outer join table will be the
                                         // first item in this list

  RenderInfo(const std::string& session_id,
             const int render_widget_id,
             const std::string& render_vega = "",
             const bool force_non_in_situ_data = false);

  void setForceNonInSituData();
  bool queryRanWithInSituData() const;
  bool hasInSituData() const;
  bool isInSituDataFlagUnset() const;
  bool isPotentialInSituRender() const;
  bool useCudaBuffers() const;
  void disableCudaBuffers();
  bool hasVega() const;

  std::shared_ptr<QueryRenderer::QueryDataLayout> getQueryVboLayout() const;
  void setQueryVboLayout(const std::shared_ptr<QueryRenderer::QueryDataLayout>& vbo_layout);
  std::shared_ptr<QueryRenderer::QueryDataLayout> getQuerySsboLayout() const;
  void setQuerySsboLayout(const std::shared_ptr<QueryRenderer::QueryDataLayout>& ssbo_layout);

  bool setInSituDataIfUnset(const bool is_in_situ_data);

  void reset();

 private:
  enum class InSituState { UNSET, IS_IN_SITU, IS_NOT_IN_SITU };
  InSituState in_situ_data;  // Should be set to true if query results can be written directly
                             // to CUDA-mapped opengl buffers for rendering. Should be set
                             // to false otherwise, meaning results are written to CPU first,
                             // and buffered back to GPU for rendering.
                             // An alternative meaning is that when false, you've encountered
                             // a non-projection query.
                             // Can only be set once for the lifetime of the object.
  bool force_non_in_situ_data;

  enum class RendererBufferMode { CUDA, GL };
  RendererBufferMode
      buffer_mode_;  // The Renderer buffer mode determines how query results are bused to the Rendering engine.

  std::shared_ptr<QueryRenderer::QueryDataLayout> query_vbo_layout;
  std::shared_ptr<QueryRenderer::QueryDataLayout> query_ssbo_layout;
};

#endif  // QUERYENGINE_RENDERINFO_H
