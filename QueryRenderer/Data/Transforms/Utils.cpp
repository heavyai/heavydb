/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Utils.h"

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/QueryDataTableQueues.h"
#include "QueryRenderer/Data/Transforms/Aggregate/AggXform.h"
#include "QueryRenderer/Data/Transforms/Enums.h"
#include "QueryRenderer/Data/Transforms/Formula/FormulaXform.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

XformShPtr createTransform(QueryRendererContext& ctx,
                           const BaseDataTableShPtr& source_data_ptr,
                           const JSONLocation& obj_loc) {
  RENDER_LOG_SCOPE();
  CHECK(source_data_ptr);
  RUNTIME_EX_ASSERT(obj_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        obj_loc, "Transforms must be defined by an array"));

  auto curr_src_data = source_data_ptr;
  XformShPtr curr_xform;
  for (size_t i = 0; i < obj_loc.size(); ++i) {
    auto const obj_item_loc = obj_loc[i];
    RUNTIME_EX_ASSERT(
        obj_item_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            obj_item_loc,
            "All elements in a transform array must be objects. The element at index " +
                std::to_string(i) + " is not an object."));

    auto const type_loc = obj_item_loc.getMember(JSONSchema_v1::Xform::kTypeProp);
    RUNTIME_EX_ASSERT(
        type_loc.isValid() && type_loc.isString(),
        RapidJSONUtils::createJsonParseError(
            type_loc.isValid() ? type_loc : obj_item_loc,
            "Transform object at index " + std::to_string(i) + " has an invalid \"" +
                std::string(JSONSchema_v1::Xform::kTypeProp) + "\" property. \"" +
                std::string(JSONSchema_v1::Xform::kTypeProp) +
                "\" is required and must be one of the strings " +
                get_xform_types_as_string() + "."));

    auto xform_type = convert_string_to_xform_type_enum(type_loc.getString());
    RUNTIME_EX_ASSERT(
        xform_type >= 0,
        RapidJSONUtils::createJsonParseError(
            type_loc,
            "Transform object at index " + std::to_string(i) + " has an invalid \"" +
                std::string(JSONSchema_v1::Xform::kTypeProp) + "\" property \"" +
                std::string(type_loc.getString()) + "\". It must be one of the strings " +
                get_xform_types_as_string()));

    switch (static_cast<XformType>(xform_type)) {
      case XformType::kAggregate:
        curr_xform = std::make_shared<AggXform>(ctx, curr_src_data);
        break;
      case XformType::kFormula:
        curr_xform = std::make_shared<FormulaXform>(ctx, curr_src_data);
        break;
      case XformType::kMaxXformType:
        CHECK(false);
        break;
    }

    // Queue up call to initialize
    ctx.getDataTableQueues().addToSourceTableQueue(curr_xform, obj_item_loc, true);
    // Change curr_src_data to point to the new xform and continue loop
    curr_src_data = std::dynamic_pointer_cast<BaseDataTable>(curr_xform);
  }

  return curr_xform;
}

}  // namespace QueryRenderer
