/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <thrust/sort.h>

#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustAllocatorDeviceVector.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

namespace {
template <typename T, int NUM_ELEMS = 1>
struct TopBottomKEval {
  static AggDataList eval(DataMgrThrustContext& execution_context,
                          const bool ascending,
                          const uint16_t k,
                          const AnyDataType& distinct_vals,
                          const AnyDataType& distinct_cnts) {
    auto const& vals = distinct_vals.getVectorRef<T>();
    if (vals.size() < k) {
      return {std::make_shared<AnyDataType>(vals)};
    }

    auto thrust_vals = make_device_vector_from_context<T>(execution_context, vals.size());
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
      auto const& cnts = distinct_cnts.getVectorRef<uint32_t>();
      auto thrust_cnts = make_device_vector_from_context<uint32_t>(execution_context,
                                                                   distinct_cnts.size());
#ifdef HAVE_CUDA
      cuMemcpyHtoD((CUdeviceptr)::thrust::raw_pointer_cast(thrust_cnts.data()),
                   &cnts[0],
                   cnts.size() * sizeof(uint32_t));
#else
      ::thrust::copy(execution_context.getDevicePolicy(),
                     cnts.begin(),
                     cnts.end(),
                     thrust_cnts.begin());
#endif  // HAVE_CUDA
      if (ascending) {
        ::thrust::sort_by_key(execution_context.getDevicePolicy(),
                              thrust_cnts.begin(),
                              thrust_cnts.end(),
                              thrust_vals.begin(),
                              ::thrust::less<uint32_t>());
      } else {
        ::thrust::sort_by_key(execution_context.getDevicePolicy(),
                              thrust_cnts.begin(),
                              thrust_cnts.end(),
                              thrust_vals.begin(),
                              ::thrust::greater<uint32_t>());
      }
    } else {
      auto const& cnts = distinct_cnts.getVectorRef<int>();
      auto thrust_cnts = make_device_vector_from_context<uint32_t>(execution_context,
                                                                   distinct_cnts.size());
#ifdef HAVE_CUDA
      cuMemcpyHtoD((CUdeviceptr)::thrust::raw_pointer_cast(thrust_cnts.data()),
                   &cnts[0],
                   cnts.size() * sizeof(int));
#else
      ::thrust::copy(execution_context.getDevicePolicy(),
                     cnts.begin(),
                     cnts.end(),
                     thrust_cnts.begin());
#endif  // HAVE_CUDA
      if (ascending) {
        ::thrust::sort_by_key(execution_context.getDevicePolicy(),
                              thrust_cnts.begin(),
                              thrust_cnts.end(),
                              thrust_vals.begin(),
                              ::thrust::less<int>());
      } else {
        ::thrust::sort_by_key(execution_context.getDevicePolicy(),
                              thrust_cnts.begin(),
                              thrust_cnts.end(),
                              thrust_vals.begin(),
                              ::thrust::greater<int>());
      }
    }

    std::vector<T> h_results(k);
#ifdef HAVE_CUDA
    cuMemcpyDtoH(&h_results[0],
                 (CUdeviceptr)::thrust::raw_pointer_cast(thrust_vals.data()),
                 k * sizeof(T));
#else
    ::thrust::copy(thrust_vals.begin(), thrust_vals.begin() + k, h_results.begin());
#endif  // HAVE_CUDA
    return {std::make_shared<AnyDataType>(std::move(h_results))};
  }
};
}  // namespace

AggDataList ThrustOpExecutorImpl<active_device_system>::executeTopBottomKOp(
    const AggDataList& dependency_results,
    const bool is_ascending,
    const uint16_t k) {
  // Note: dependency_results should have already been properly validated by the time we
  // get here
  return ThrustOpExecutorUtils::runMultiTypeOp<TopBottomKEval>(
      thrust_context_,
      dependency_results[0]->getType(),
      is_ascending,
      k,
      *dependency_results[0],
      *dependency_results[1]);
}

}  // namespace QueryRenderer
