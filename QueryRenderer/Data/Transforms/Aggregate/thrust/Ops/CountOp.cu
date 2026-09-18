/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/Ops/Utils.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

#include <thrust/count.h>

namespace QueryRenderer {

AggDataList ThrustOpExecutorImpl<active_device_system>::executeCountOp(
    ThrustOpExecutionState state) {
  uint32_t val{0};
  if (state.key_col) {
    val = ::thrust::count_if(thrust_context_.getDevicePolicy(),
                             state.key_col->begin(),
                             state.key_col->end(),
                             detail::is_valid(state.interop_buffer_info.invalid_key));
  } else {
    val = state.interop_buffer_info.layout_buffer->getLayoutManager()->numItems(
        state.input_info.buffer_layout);
  }
  return {std::make_shared<AnyDataType>(QueryDataType::UINT, val)};
}

}  // namespace QueryRenderer
