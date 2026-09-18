/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#include <thrust/transform_reduce.h>

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/Ops/Utils.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

namespace {
template <typename T, int NUM_ELEMS = 1>
struct SumEval {
  struct nullTransform {
    const T null_val;
    nullTransform(const T null_val) : null_val{null_val} {}
    __host__ __device__ T operator()(const T& v) const {
      if (v == null_val) {
        return T(0);
      }
      return v;
    }
  };
  struct keySum : public nullTransform {
    const int64_t invalid_key;

    keySum(const int64_t invalid_key, const T null_val)
        : nullTransform(null_val), invalid_key{invalid_key} {}

    __host__ __device__ T operator()(const ::thrust::tuple<T, int64_t>& val) const {
      if (::thrust::get<1>(val) == invalid_key) {
        return T(0);
      }
      return nullTransform::operator()(::thrust::get<0>(val));
    }
  };

  static AggDataList getReturnVal(const SQLTypeInfo&, const T& val) {
    return {std::make_shared<AnyDataType>(
        TypeToQueryDataTypeSelector<T>::getQueryDataType(), val)};
  }

  static AggDataList eval(DataMgrThrustContext& execution_context,
                          const ThrustOpExecutionState& state) {
    T val{0};
    auto col = createThrustBufferColumn<T, NUM_ELEMS>(
        state.interop_buffer_info.mapped_buffer_descriptor,
        state.input_info.attr_name,
        state.input_info.buffer_layout);
    auto const null_val = getNullValueFromTypeInfo<T>(state.input_info.type_info);

    // TODO(croot): handle overflow
    if (state.key_col) {
      auto zip_begin = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(col->begin(), state.key_col->begin()));
      auto zip_end = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(col->end(), state.key_col->end()));
      val = ::thrust::transform_reduce(
          execution_context.getDevicePolicy(),
          zip_begin,
          zip_end,
          keySum(state.interop_buffer_info.invalid_key, null_val),
          T(0),
          ::thrust::plus<T>());
    } else {
      val = ::thrust::transform_reduce(execution_context.getDevicePolicy(),
                                       col->begin(),
                                       col->end(),
                                       nullTransform(null_val),
                                       T(0),
                                       ::thrust::plus<T>());
    }
    return getReturnVal(state.input_info.type_info, val);
  }
};

template <>
AggDataList SumEval<int64_t>::getReturnVal(const SQLTypeInfo& type_info,
                                           const int64_t& val) {
  if (type_info.is_decimal()) {
    detail::DecimalConverter conv(type_info.get_scale());
    return {std::make_shared<AnyDataType>(QueryDataType::DOUBLE, conv(val))};
  }
  return {std::make_shared<AnyDataType>(
      TypeToQueryDataTypeSelector<int64_t>::getQueryDataType(), val)};
}
}  // namespace

AggDataList ThrustOpExecutorImpl<active_device_system>::executeSumOp(
    ThrustOpExecutionState state) {
  return ThrustOpExecutorUtils::runMultiTypeOp<SumEval>(
      thrust_context_, state.column_data_type, state);
}

}  // namespace QueryRenderer
