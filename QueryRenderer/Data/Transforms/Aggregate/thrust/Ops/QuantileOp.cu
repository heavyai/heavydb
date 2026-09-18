/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <thrust/binary_search.h>

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/QuantileProps.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustAllocatorDeviceVector.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

namespace {

template <typename T, typename TT>
void find_quantiles(DataMgrThrustContext& execution_context,
                    std::vector<T>& quantiles,
                    const QuantileProps& props,
                    const std::vector<T>& vals,
                    const AnyDataType& distinct_cnts) {
  auto const& cnts = distinct_cnts.getVectorRef<TT>();
  auto thrust_cnts =
      make_device_vector_from_context<TT>(execution_context, distinct_cnts.size());
#ifdef HAVE_CUDA
  cuMemcpyHtoD((CUdeviceptr)::thrust::raw_pointer_cast(thrust_cnts.data()),
               &cnts[0],
               cnts.size() * sizeof(TT));
#else
  ::thrust::copy(
      execution_context.getDevicePolicy(), cnts.begin(), cnts.end(), thrust_cnts.begin());
#endif  // HAVE_CUDA
  ::thrust::inclusive_scan(execution_context.getDevicePolicy(),
                           thrust_cnts.begin(),
                           thrust_cnts.end(),
                           thrust_cnts.begin());
  const uint32_t population_size = thrust_cnts[thrust_cnts.size() - 1];
  if (population_size == 1) {
    quantiles.resize(quantiles.size() + props.num_quantiles - 1, vals[0]);
  } else {
    auto const population_size_d = static_cast<double>(population_size);
    const double quantile_diff =
        (population_size_d - 1) / std::max(static_cast<double>(props.num_quantiles), 1.0);
    double curr_quantile = quantile_diff;
    for (decltype(props.num_quantiles) i = 0; i < props.num_quantiles - 1;
         ++i, curr_quantile += quantile_diff) {
      auto const quantile_idx = static_cast<TT>(std::ceil(curr_quantile));
      auto const range = ::thrust::equal_range(execution_context.getDevicePolicy(),
                                               thrust_cnts.begin(),
                                               thrust_cnts.end(),
                                               quantile_idx);
      auto const idx1 = range.first - thrust_cnts.begin();
      auto const value0 = vals[idx1];
      if (range.first != range.second) {
        CHECK(range.second != thrust_cnts.end());
        auto const value1 = vals[idx1 + 1];
        quantiles.push_back(value0 +
                            (value1 - value0) *
                                (curr_quantile - static_cast<double>(quantile_idx) + 1));
      } else {
        quantiles.push_back(value0);
      }
    }
  }
}

template <typename T, int NUM_ELEMS = 1>
struct QuantileEval {
  static AggDataList eval(DataMgrThrustContext& execution_context,
                          const QuantileProps& props,
                          const AnyDataType& distinct_vals,
                          const AnyDataType& distinct_cnts) {
    auto const& vals = distinct_vals.getVectorRef<T>();
    if (!vals.size()) {
      return {};
    }

    std::vector<T> quantiles;
    if (props.num_quantiles == 1) {
      quantiles = {distinct_vals.getValAtIndex<T>(0)};
    } else {
      if (props.include_extrema) {
        quantiles.push_back(distinct_vals.getValAtIndex<T>(0));
      }

      auto thrust_vals =
          make_device_vector_from_context<T>(execution_context, vals.size());
#ifdef HAVE_CUDA
      cuMemcpyHtoD((CUdeviceptr)::thrust::raw_pointer_cast(thrust_vals.data()),
                   &vals[0],
                   vals.size() * sizeof(T));
#else
      ::thrust::copy(execution_context.getDevicePolicy(),
                     vals.begin(),
                     vals.end(),
                     thrust_vals.begin());
#endif  // HAVE_CUDA
      if (distinct_cnts.getType() == QueryDataType::UINT) {
        find_quantiles<T, uint32_t>(
            execution_context, quantiles, props, vals, distinct_cnts);
      } else {
        find_quantiles<T, int>(execution_context, quantiles, props, vals, distinct_cnts);
      }

      if (props.include_extrema) {
        quantiles.push_back(distinct_vals.getValAtIndex<T>(distinct_vals.size() - 1));
        CHECK_EQ(quantiles.size(), static_cast<size_t>(props.num_quantiles) + 1);
      } else {
        CHECK_EQ(quantiles.size(), static_cast<size_t>(props.num_quantiles) - 1);
      }
    }

    return {std::make_shared<AnyDataType>(quantiles)};
  }
};
}  // namespace
AggDataList ThrustOpExecutorImpl<active_device_system>::executeQuantileOp(
    const AggDataList& distinct_histogram_results,
    const QuantileProps& props) {
  return ThrustOpExecutorUtils::runMultiTypeOp<QuantileEval>(
      thrust_context_,
      distinct_histogram_results[0]->getType(),
      props,
      *distinct_histogram_results[0],
      *distinct_histogram_results[1]);
}

}  // namespace QueryRenderer
