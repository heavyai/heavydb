/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "QueryRenderer/Interface/RenderQueryRunnerInterface.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {
//
// class BaseEmbeddedDataTable
//
// Mix-in base class for DataTables that are driven by data embedded in the vega json
// Also provides the Json interface for VegaParser
//
class BaseEmbeddedDataTable : public JSONRefObject {
 public:
  BaseEmbeddedDataTable(QueryRendererContext& ctx,
                        const std::string& name,
                        const JSONLocation& json_loc,
                        const RenderQuerySpecialtyType render_query_type);

  ~BaseEmbeddedDataTable() override = default;

  // Json serialization
  void toJSONInternal(rapidjson::Value& obj,
                      rapidjson::Document::AllocatorType& allocator) const final {
    THROW_RUNTIME_EX(
        createJSONRefError("Evaluating data table to JSON is not currently supported."));
  }

 protected:
  RenderQuerySpecialtyType render_query_type_;
};

}  // namespace QueryRenderer
