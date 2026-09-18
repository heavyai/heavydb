/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace QueryRenderer {

struct RenderQueryExecuteTimer {
  int64_t queue_time_ms = 0;
  int64_t query_parse_time_ms = 0;
  int64_t query_execution_time_ms = 0;
  int64_t vega_parse_time_ms = 0;
  int64_t render_time_ms = 0;

  inline int64_t getFullExecutionTime() const {
    return query_parse_time_ms + query_execution_time_ms;
  }
  inline int64_t getFullRenderTime() const { return vega_parse_time_ms + render_time_ms; }
};

}  // namespace QueryRenderer
