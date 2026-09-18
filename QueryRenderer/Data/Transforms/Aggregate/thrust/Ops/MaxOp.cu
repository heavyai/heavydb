/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#include <thrust/extrema.h>

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/Ops/Utils.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"
namespace QueryRenderer {

namespace {
template <typename T, int NUM_ELEMS = 1>
struct MaximumEval {
  struct nullCompare {
    const T null_val;
    nullCompare(const T null_val) : null_val{null_val} {}
    __host__ __device__ bool operator()(const T& v1, const T& v2) const {
      if (v2 == null_val) {
        return false;
      } else if (v1 == null_val) {
        return true;
      }
      return v1 < v2;
    }
  };

  struct tupleCompare : nullCompare {
    const int64_t invalid_key;

    tupleCompare(const int64_t invalid_key, const T null_val)
        : nullCompare{null_val}, invalid_key{invalid_key} {}

    __host__ __device__ bool operator()(const ::thrust::tuple<T, int64_t>& a,
                                        const ::thrust::tuple<T, int64_t>& b) const {
      if (::thrust::get<1>(b) == invalid_key) {
        return false;
      } else if (::thrust::get<1>(a) == invalid_key) {
        return true;
      }
      return nullCompare::operator()(::thrust::get<0>(a), ::thrust::get<0>(b));
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
    if (state.key_col) {
      auto zip_begin = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(col->begin(), state.key_col->begin()));
      auto zip_end = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(col->end(), state.key_col->end()));
      auto max_elem = ::thrust::max_element(
          execution_context.getDevicePolicy(),
          zip_begin,
          zip_end,
          tupleCompare(state.interop_buffer_info.invalid_key, null_val));
      if (max_elem == zip_end ||
          ::thrust::get<1>(*max_elem) == state.interop_buffer_info.invalid_key) {
        val = null_val;
      } else {
        val = ::thrust::get<0>(*max_elem);
      }
    } else {
      auto max_elem = ::thrust::max_element(execution_context.getDevicePolicy(),
                                            col->begin(),
                                            col->end(),
                                            nullCompare(null_val));
      // the < 0 check below is to handle the special case where all values are null. The
      // thrust implementation of max_element here sets an initial value of
      // std::numerical_limits<T>::min() with an iterator offset of -1. See
      // <thrust/system/detail/generic/extrema.inl>::max_element
      // If all the elements are null, the nullCompare() check will always return false,
      // which means the dummy initial item created by thrust is considered the min, so
      // the -1 is returned as the iterator offset from col->begin().
      if (max_elem == col->end() || max_elem - col->begin() < 0) {
        val = null_val;
      } else {
        val = *max_elem;
      }
    }
    if (val == null_val) {
      val = getNullValue<T>();
    }
    return getReturnVal(state.input_info.type_info, val);
  }
};

template <>
AggDataList MaximumEval<int64_t>::getReturnVal(const SQLTypeInfo& type_info,
                                               const int64_t& val) {
  if (type_info.is_decimal()) {
    detail::DecimalConverter conv(type_info.get_scale());
    return {std::make_shared<AnyDataType>(QueryDataType::DOUBLE, conv(val))};
  }
  return {std::make_shared<AnyDataType>(
      TypeToQueryDataTypeSelector<int64_t>::getQueryDataType(), val)};
}
}  // namespace

AggDataList ThrustOpExecutorImpl<active_device_system>::executeMaxOp(
    ThrustOpExecutionState state) {
  return ThrustOpExecutorUtils::runMultiTypeOp<MaximumEval>(
      thrust_context_, state.column_data_type, state);
}

}  // namespace QueryRenderer
