/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "QueryRenderer/Interface/RenderQueryRunnerInterface.h"

class ResultSet;
class ExecutionResult;
class RenderInfo;

namespace QueryRenderer {

class QueryRenderManager;
class JSONLocation;
struct RasterMeshMetadata;

struct RenderResultProcessor {
  // returns number of rows in results
  static uint64_t processExecuteQueryResults(
      QueryRenderManager& render_manager,
      const ExecutionResult& execute_results,
      const JSONLocation* data_loc,
      const RenderQuerySpecialtyType render_query_type,
      RenderInfo& render_info,
      RasterMeshMetadata* raster_mesh_metadata = nullptr);
};

}  // namespace QueryRenderer
