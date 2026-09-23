/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#include <thrust/transform_reduce.h>

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/Ops/Utils.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/NumericUtils.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

namespace {

template <typename T, int NUM_ELEMS = 1>
struct SqDiffSumEval {
  struct isNull {
    const T null_val;
    isNull(const T null_val) : null_val{null_val} {}
    __host__ __device__ bool operator()(const T& v) const {
      if (v == null_val) {
        return true;
      }
      return false;
    }
  };

  // squarediff<T> computes the squared difference from a value
  template <typename FType>
  struct squarediff : public isNull {
    FType val;
    squarediff(const FType& val, const T null_val) : isNull(null_val), val{val} {}

    __host__ __device__ FType operator()(const T& x) const {
      if (isNull::operator()(x)) {
        return FType(0);
      }
      auto diff = static_cast<FType>(x) - val;
      return diff * diff;
    }
  };

  template <typename FType>
  struct keysquareddiff : public squarediff<FType> {
    const int64_t invalid_key;

    keysquareddiff(const int64_t invalid_key, const FType& val, const T null_val)
        : squarediff<FType>(val, null_val), invalid_key{invalid_key} {}

    __host__ __device__ FType operator()(const ::thrust::tuple<T, int64_t>& val) const {
      if (::thrust::get<1>(val) == invalid_key) {
        return FType(0);
      }

      return squarediff<FType>::operator()(::thrust::get<0>(val));
    }
  };

  static AggDataList eval(DataMgrThrustContext& execution_context,
                          const ThrustOpExecutionState& state,
                          const AnyDataType& mean) {
    using OutType = typename FloatingPtTypeSelector<T>::type;
    OutType val{0};
    auto col = createThrustBufferColumn<T, NUM_ELEMS>(
        state.interop_buffer_info.mapped_buffer_descriptor,
        state.input_info.attr_name,
        state.input_info.buffer_layout);
    auto const null_val = getNullValueFromTypeInfo<T>(state.input_info.type_info);
    auto mean_val = mean.getVal<OutType>();
    if (isNullValue(mean_val)) {
      val = mean_val;
    } else {
      if (state.key_col) {
        auto zip_begin = ::thrust::make_zip_iterator(
            ::thrust::make_tuple(col->begin(), state.key_col->begin()));
        auto zip_end = ::thrust::make_zip_iterator(
            ::thrust::make_tuple(col->end(), state.key_col->end()));
        val = ::thrust::transform_reduce(
            execution_context.getDevicePolicy(),
            zip_begin,
            zip_end,
            keysquareddiff<OutType>(
                state.interop_buffer_info.invalid_key, mean_val, null_val),
            OutType(0),
            ::thrust::plus<OutType>());
      } else {
        val = ::thrust::transform_reduce(execution_context.getDevicePolicy(),
                                         col->begin(),
                                         col->end(),
                                         squarediff<OutType>(mean_val, null_val),
                                         OutType(0),
                                         ::thrust::plus<OutType>());
      }
    }
    return {std::make_shared<AnyDataType>(
        TypeToQueryDataTypeSelector<OutType>::getQueryDataType(), val)};
  }
};

template <>
struct SqDiffSumEval<int64_t> {
  struct isNull {
    const int64_t null_val;
    isNull(const int64_t null_val) : null_val{null_val} {}
    __host__ __device__ bool operator()(const int64_t& v) const {
      if (v == null_val) {
        return true;
      }
      return false;
    }
  };

  // squarediff<int64_t> computes the squared difference from a value
  struct squarediff : public isNull {
    double val;
    detail::DecimalConverter conv;
    const bool conv_decimal;
    squarediff(const double val, const SQLTypeInfo& type_info)
        : isNull(getNullValueFromTypeInfo<int64_t>(type_info))
        , val{val}
        , conv(type_info.get_scale())
        , conv_decimal{type_info.is_decimal()} {}

    __host__ __device__ double operator()(const int64_t& x) const {
      if (isNull::operator()(x)) {
        return double(0);
      }
      auto diff = (conv_decimal ? conv(x) : static_cast<double>(x)) - val;
      return diff * diff;
    }
  };

  struct keysquareddiff : public squarediff {
    const int64_t invalid_key;

    keysquareddiff(const int64_t invalid_key,
                   const double& val,
                   const SQLTypeInfo& type_info)
        : squarediff(val, type_info), invalid_key{invalid_key} {}

    __host__ __device__ double operator()(
        const ::thrust::tuple<int64_t, int64_t>& val) const {
      if (::thrust::get<1>(val) == invalid_key) {
        return double(0);
      }

      return squarediff::operator()(::thrust::get<0>(val));
    }
  };

  static AggDataList eval(DataMgrThrustContext& execution_context,
                          const ThrustOpExecutionState& state,
                          const AnyDataType& mean) {
    double val{0.0};
    auto col = createThrustBufferColumn<int64_t>(
        state.interop_buffer_info.mapped_buffer_descriptor,
        state.input_info.attr_name,
        state.input_info.buffer_layout);
    auto mean_val = mean.getVal<double>();
    if (isNullValue(mean_val)) {
      val = mean_val;
    } else {
      if (state.key_col) {
        auto zip_begin = ::thrust::make_zip_iterator(
            ::thrust::make_tuple(col->begin(), state.key_col->begin()));
        auto zip_end = ::thrust::make_zip_iterator(
            ::thrust::make_tuple(col->end(), state.key_col->end()));
        val = ::thrust::transform_reduce(
            execution_context.getDevicePolicy(),
            zip_begin,
            zip_end,
            keysquareddiff(state.interop_buffer_info.invalid_key,
                           mean.getVal<double>(),
                           state.input_info.type_info),
            double(0),
            ::thrust::plus<double>());
      } else {
        val = ::thrust::transform_reduce(
            execution_context.getDevicePolicy(),
            col->begin(),
            col->end(),
            squarediff(mean.getVal<double>(), state.input_info.type_info),
            double(0),
            ::thrust::plus<double>());
      }
    }
    return {std::make_shared<AnyDataType>(QueryDataType::DOUBLE, val)};
  }
};

}  // namespace

AggDataList ThrustOpExecutorImpl<active_device_system>::executeSqDiffSumOp(
    ThrustOpExecutionState state,
    const AnyDataType& avg_result) {
  return ThrustOpExecutorUtils::runMultiTypeOp<SqDiffSumEval>(
      thrust_context_, state.column_data_type, state, avg_result);
}

}  // namespace QueryRenderer
