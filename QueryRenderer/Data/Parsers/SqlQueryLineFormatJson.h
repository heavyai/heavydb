/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

struct SqlQueryLineFormatJson {
  static void validate(const JSONLocation& parent_loc);

  struct LineDrawBufferData {
    std::shared_ptr<std::vector<double>> primary_vertices_ptr;
    std::shared_ptr<std::vector<double>> secondary_vertices_ptr;
    std::shared_ptr<std::vector<unsigned int>> indices_ptr;
  };

  struct LineVertexDataTemplate {
    const std::vector<unsigned int> target_column_indices;
    const size_t align_bytes;
  };

  struct LineVertexColumnInfo {
    std::string name;
    int target_column_index;
    bool is_reference;
  };

  static QueryDataLayout::LayoutType SetVertexLayout(const JSONLocation& data_loc);

  struct LineVertexQueryTargetInfo {
    std::unordered_map<unsigned int, LineVertexColumnInfo> vertex_target_map;
    // Separate query targets by rendering buffer
    std::vector<unsigned int> primary_vertex_query_indices;
    std::vector<unsigned int> secondary_vertex_query_indices;
  };  // TODO(adb): pass this by ref?

  using QueryColumnInfoMap =
      std::unordered_map<std::string, std::pair<unsigned int, SQLTypes>>;

  static LineVertexQueryTargetInfo SetVertexQueryTargets(
      const JSONLocation& data_loc,
      const QueryColumnInfoMap& query_column_info_map,
      const QueryDataLayout::LayoutType vertex_layout);

  static bool ShouldAttemptInSituRender(const JSONLocation& data_loc);
};

}  // namespace QueryRenderer
