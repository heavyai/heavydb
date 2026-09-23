/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/ValidateUtils.h"

#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/QueryDataLayout.h"

namespace QueryRenderer {

void ValidateUtils::validateNumOrDictEncodedStrInput(const BaseDataTableShPtr& in_data,
                                                     const LayoutAttrInfoSet& inputs,
                                                     const AggOp* curr_op) {
  auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(in_data);
  if (data) {
    for (auto const& input_info : inputs) {
      auto query_data_layout = getDataLayoutForAttribute(in_data, input_info.attr_name);
      CHECK(query_data_layout);
      auto const& type_info =
          query_data_layout->getAttrSQLTypeInfoRef(input_info.attr_name);
      RUNTIME_EX_ASSERT(
          type_info.is_number() ||
              (type_info.is_string() && type_info.get_compression() == kENCODING_DICT),
          "Input \"" + input_info.attr_name + "\" is of type " +
              type_info.get_type_name() +
              ". Only numeric types and dict-encoded strings are currently supported as "
              "inputs for "
              "aggregate transform operator " +
              to_string(curr_op->getOpType()) + ".");
    }
  } else {
    XformOp::validateNumericAttrType(in_data, inputs, curr_op);
  }
}

}  // namespace QueryRenderer
