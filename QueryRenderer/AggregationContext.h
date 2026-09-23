/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Types.h"

namespace QueryRenderer {

class QueryRendererContext;

//
// class AggregationContext
//
// Component class for QueryRendererContext
//
class AggregationContext {
 public:
  explicit AggregationContext(QueryRendererContext& render_ctx);

  size_t getNumAggRenderableMarks() const;
  size_t getNumSoloAggRenderableMarks() const;

 private:
  QueryRendererContext& render_context_;
};

}  // namespace QueryRenderer
