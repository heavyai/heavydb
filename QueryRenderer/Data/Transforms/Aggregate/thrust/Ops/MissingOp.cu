/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#include <thrust/count.h>

#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

namespace {

template <typename T, int NUM_ELEMS = 1>
struct MissingEval {
  struct is_key_null {
    const int64_t invalid_key;
    const T null_val;

    is_key_null(const int64_t invalid_key, const T null_val)
        : invalid_key{invalid_key}, null_val{null_val} {}

    __host__ __device__ bool operator()(const ::thrust::tuple<T, int64_t>& val) const {
      return ::thrust::get<1>(val) != invalid_key && ::thrust::get<0>(val) == null_val;
    }
  };

  struct is_null {
    const T null_val;

    is_null(T null_val) : null_val{null_val} {}

    __host__ __device__ bool operator()(const T& val) { return val == null_val; }
  };

  static AggDataList eval(DataMgrThrustContext& execution_context,
                          const ThrustOpExecutionState& state) {
    uint32_t val{0};
    auto col = createThrustBufferColumn<T, NUM_ELEMS>(
        state.interop_buffer_info.mapped_buffer_descriptor,
        state.input_info.attr_name,
        state.input_info.buffer_layout);
    auto const null_val = getNullValueFromTypeInfo<T>(state.input_info.type_info);
    if (state.key_col) {
      auto zip_begin = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(col->begin(), state.key_col->begin()));
      auto zip_end = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(col->end(), state.key_col->end()));
      val = ::thrust::count_if(
          execution_context.getDevicePolicy(),
          zip_begin,
          zip_end,
          is_key_null(state.interop_buffer_info.invalid_key, null_val));
    } else {
      val = ::thrust::count_if(execution_context.getDevicePolicy(),
                               col->begin(),
                               col->end(),
                               is_null(null_val));
    }
    return {std::make_shared<AnyDataType>(QueryDataType::UINT, val)};
  }
};

}  // namespace

AggDataList ThrustOpExecutorImpl<active_device_system>::executeMissingOp(
    ThrustOpExecutionState state) {
  return ThrustOpExecutorUtils::runMultiTypeOp<MissingEval>(
      thrust_context_, state.column_data_type, state);
}

}  // namespace QueryRenderer
