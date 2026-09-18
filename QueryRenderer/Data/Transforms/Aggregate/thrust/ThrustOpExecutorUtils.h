/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/BufferWrapper.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Interface/AggDataTypes.h"
#include "QueryRenderer/Interop/InteropBufferInfo.h"
#include "QueryRenderer/Interop/LayoutAttrInfo.h"
#include "QueryRenderer/Utils/thrust/ThrustBufferColumn.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContext.h"
namespace QueryRenderer {

struct ThrustOpExecutionState {
  InteropBufferInfo interop_buffer_info;
  const LayoutAttrInfo& input_info;
  const QueryDataType column_data_type;
  std::shared_ptr<ThrustBufferColumn<int64_t>> key_col;
};

struct ThrustOpExecutorUtils {
  inline static ThrustOpExecutionState createThrustOpExecutionState(
      const InteropBufferInfo& interop_buffer_info,
      const LayoutAttrInfo& input_info) {
    // NOTE: the ThrustOpExecutionState constructor below makes a copy of the
    // InteropBufferInfo argument in order to offset its handle according to the
    // LayoutAttrInfo input.
    ThrustOpExecutionState execution_state{
        interop_buffer_info,
        input_info,
        LayoutAttrInfo::getTypeFromLayoutAttr(input_info),
        nullptr};

    auto* layout_buffer = execution_state.interop_buffer_info.layout_buffer;
    CHECK(layout_buffer->hasLayout());
    auto* layout_mgr = layout_buffer->getLayoutManager();
    auto offset = layout_mgr->getOffsetBytes(execution_state.input_info.buffer_layout);

    execution_state.interop_buffer_info.mapped_buffer_descriptor.handle += offset;
    execution_state.interop_buffer_info.mapped_buffer_descriptor.num_bytes =
        layout_mgr->getNumUsedBytes(input_info.buffer_layout);

    if (execution_state.input_info.buffer_layout->hasAttribute("key") &&
        execution_state.input_info.buffer_layout->getAttributeType("key") ==
            gfx::BufferAttrType::kInt64) {
      execution_state.key_col = createThrustBufferColumn<int64_t>(
          execution_state.interop_buffer_info.mapped_buffer_descriptor,
          "key",
          execution_state.input_info.buffer_layout);
    }

    return execution_state;
  }

  template <template <typename, int> class Functor, typename... Targs>
  static AggDataList runMultiTypeOp(DataMgrThrustContext& thrust_context,
                                    const QueryDataType column_data_type,
                                    Targs&&... Fargs) {
    switch (column_data_type) {
      case QueryDataType::INT:
        return Functor<QueryDataTypeSelector<QueryDataType::INT>::type, 1>::eval(
            thrust_context, Fargs...);
        break;
      case QueryDataType::UINT:
        return Functor<QueryDataTypeSelector<QueryDataType::UINT>::type, 1>::eval(
            thrust_context, Fargs...);
        break;
      case QueryDataType::FLOAT:
        return Functor<QueryDataTypeSelector<QueryDataType::FLOAT>::type, 1>::eval(
            thrust_context, Fargs...);
        break;
      case QueryDataType::DOUBLE:
        return Functor<QueryDataTypeSelector<QueryDataType::DOUBLE>::type, 1>::eval(
            thrust_context, Fargs...);
        break;
      case QueryDataType::UINT64:
        return Functor<QueryDataTypeSelector<QueryDataType::UINT64>::type, 1>::eval(
            thrust_context, Fargs...);
        break;
      case QueryDataType::INT64:
        return Functor<QueryDataTypeSelector<QueryDataType::INT64>::type, 1>::eval(
            thrust_context, Fargs...);
        break;
      default:
        throw std::runtime_error(
            "Cannot run a valid thrust aggregation on data of type " +
            to_string(column_data_type));
    }
  }
};

}  // namespace QueryRenderer
