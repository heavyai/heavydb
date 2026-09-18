/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "Events/NotifyRefEventQueue.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Scales/Types.h"
#include "Utils/RapidJSONUtils.h"

namespace QueryRenderer {

class QueryRendererContext;

class VegaParser {
 public:
  VegaParser(QueryRendererContext& ctx);
  ~VegaParser() = default;

  void parse(const std::string& json_str);

 private:
  QueryRendererContext& ctx_;

  using ProjectionEvents = NotifyRefEventQueue<ProjectionShPtr>;
  using ScaleEvents = NotifyRefEventQueue<ScaleShPtr>;

  void parseInternal(const std::string& json_str);

  void parseMetadata(const JSONLocation& root_loc);
  void parseViewRenderOptions(const JSONLocation& root_loc);
  void parseData(const JSONLocation& root_loc);
  ProjectionEvents parseProjections(const JSONLocation& root_loc);
  ScaleEvents parseScales(const JSONLocation& root_loc);
  void parseMarks(const JSONLocation& root_loc);
};

}  // namespace QueryRenderer
