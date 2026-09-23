/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

  void setRenderQueryStr(std::string const& query_str);
  std::string getRenderQueryStr() const;

  void reset(const RenderQueryOptions& in_query_opts,
             const heavyai::InSituFlags in_insitu_flags);

 private:
  bool cuda_using_buffers_;

  std::shared_ptr<QueryRenderer::QueryDataLayout> query_vbo_layout;
  std::shared_ptr<QueryRenderer::QueryDataLayout> query_ssbo_layout;
  RenderQueryOptions render_query_opts_;
  std::string render_query_str_;
};
