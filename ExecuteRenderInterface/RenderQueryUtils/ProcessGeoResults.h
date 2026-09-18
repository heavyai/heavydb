/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

class ExecutionResult;
class RenderInfo;

namespace QueryRenderer {
class QueryRenderManager;
class JSONLocation;

enum class InSituGeoRenderType { kPOLYGONS, kLINES };

// POLYGON, MULTIPOLYGON, LINESTRING, MULTILINESTRING
uint64_t process_geo_in_situ(QueryRenderManager& render_manager,
                             const InSituGeoRenderType in_situ_geo_render_type,
                             const ExecutionResult& results,
                             const JSONLocation* data_loc,
                             RenderInfo& render_info);

// POLYGON, MULTIPOLYGON
uint64_t process_polygons_non_in_situ(QueryRenderManager& render_manager,
                                      const ExecutionResult& results,
                                      const JSONLocation* data_loc,
                                      RenderInfo& render_info);

// LINESTRING, MULTILINESTRING
uint64_t process_lines_non_in_situ(QueryRenderManager& render_manager,
                                   const ExecutionResult& results,
                                   const JSONLocation* data_loc,
                                   RenderInfo& render_info);

}  // namespace QueryRenderer
