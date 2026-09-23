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

// POINT, MULTIPOINT, LINESTRING, MULTILINESTRING
uint64_t process_rows_in_situ(RenderInfo& render_info);

// scalar points only
uint64_t process_rows_non_in_situ(QueryRenderManager& render_manager,
                                  const ExecutionResult& results,
                                  RenderInfo& render_info);

}  // namespace QueryRenderer
