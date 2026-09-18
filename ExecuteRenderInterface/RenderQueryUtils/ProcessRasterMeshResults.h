/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include "ExecuteRenderInterface/RenderQueryUtils/ProcessRowResults.h"

class ExecutionResult;
class RenderInfo;
class JSONLocation;

namespace QueryRenderer {
struct RasterMeshMetadata;
class QueryRenderManager;

uint64_t process_raster_mesh_non_in_situ(QueryRenderManager& render_manager,
                                         const ExecutionResult& results,
                                         const RasterMeshMetadata& raster_mesh_metadata,
                                         const JSONLocation& data_loc,
                                         RenderInfo& render_info);
}  // namespace QueryRenderer
