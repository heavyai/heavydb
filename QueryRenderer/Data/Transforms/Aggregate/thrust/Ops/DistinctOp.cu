/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/Ops/Utils.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustAllocatorDeviceVector.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

namespace {
template <typename T, int NUM_ELEMS = 1>
struct DistinctEval {
  static AggDataList getReturnVal(const SQLTypeInfo&,
                                  ::thrust::device_ptr<T> distinct_vec,
                                  const size_t num_results) {
#ifdef HAVE_CUDA
    std::vector<T> h_results(num_results);

    // NOTE: cuMemcpy is much faster than thrust copy routines such as
    // std::vector<T>(result_vec.begin(), new_ends.first)) This is likely because the
    // std::vector copy does an iteration through the iterators. A direct memcpy is
    // much, much faster.
    cuMemcpyDtoH(&h_results[0],
                 (CUdeviceptr)::thrust::raw_pointer_cast(distinct_vec),
                 num_results * sizeof(T));
#else
    // TODO(croot): improve
    std::vector<T> h_results(distinct_vec, distinct_vec + num_results);
#endif
    return {std::make_shared<AnyDataType>(std::move(h_results))};
  }

  static AggDataList eval(DataMgrThrustContext& execution_context,
                          const ThrustOpExecutionState& state) {
    // NOTE; not using count as a dependency here because we need the individual counts
    // per gpu. If we grabbed the count as a dependency we'd get the total of all
    // the entire distributed set.
    uint32_t num_items = 0;
    if (state.key_col) {
      num_items =
          ::thrust::count_if(execution_context.getDevicePolicy(),
                             state.key_col->begin(),
                             state.key_col->end(),
                             detail::is_valid(state.interop_buffer_info.invalid_key));
    } else {
      num_items = state.interop_buffer_info.layout_buffer->getLayoutManager()->numItems(
          state.input_info.buffer_layout);
    }

    if (!num_items) {
      return {std::make_shared<AnyDataType>(std::vector<T>())};
    }

    // TODO(croot): use data mgr to allocate these extra arrays
    auto distinct_vec = make_device_vector_from_context<T>(execution_context, num_items);
    auto col = createThrustBufferColumn<T, NUM_ELEMS>(
        state.interop_buffer_info.mapped_buffer_descriptor,
        state.input_info.attr_name,
        state.input_info.buffer_layout);
    if (state.key_col) {
      ::thrust::copy_if(execution_context.getDevicePolicy(),
                        col->begin(),
                        col->end(),
                        state.key_col->begin(),
                        distinct_vec.begin(),
                        detail::is_valid(state.interop_buffer_info.invalid_key));
    } else {
      ::thrust::copy(execution_context.getDevicePolicy(),
                     col->begin(),
                     col->end(),
                     distinct_vec.begin());
    }

    ::thrust::sort(
        execution_context.getDevicePolicy(), distinct_vec.begin(), distinct_vec.end());
    auto new_end = ::thrust::unique(
        execution_context.getDevicePolicy(), distinct_vec.begin(), distinct_vec.end());

    auto const num_results = new_end - distinct_vec.begin();
    return getReturnVal(state.input_info.type_info, distinct_vec.data(), num_results);
  }
};

template <>
AggDataList DistinctEval<int64_t>::getReturnVal(
    const SQLTypeInfo& type_info,
    ::thrust::device_ptr<int64_t> distinct_vec,
    const size_t num_results) {
#ifdef HAVE_CUDA
  std::vector<int64_t> h_results(num_results);
  // NOTE: cuMemcpy is much faster than thrust copy routines such as
  // std::vector<T>(result_vec.begin(), new_ends.first)) This is likely because the
  // std::vector copy does an iteration through the iterators. A direct memcpy is much,
  // much faster.
  cuMemcpyDtoH(&h_results[0],
               (CUdeviceptr)::thrust::raw_pointer_cast(distinct_vec),
               num_results * sizeof(int64_t));
#else
  std::vector<int64_t> h_results(distinct_vec, distinct_vec + num_results);
#endif  // HAVE_CUDA

  if (type_info.is_decimal()) {
    detail::DecimalConverter conv(type_info.get_scale());
    std::vector<double> conv_h_results(num_results);
    // TODO(croot): multi-thread this conversion or use thrust, or do one or the other
    // depending on available memory? Right now I'm making the assumption that the
    // cardinality of a decimal column will be small, in which case this conversion will
    // be quick.
    std::transform(h_results.begin(),
                   h_results.end(),
                   conv_h_results.begin(),
                   [&conv](const int64_t v) { return conv(v); });
    return {std::make_shared<AnyDataType>(std::move(conv_h_results))};
  }
  return {std::make_shared<AnyDataType>(std::move(h_results))};
}
}  // namespace

AggDataList ThrustOpExecutorImpl<active_device_system>::executeDistinctOp(
    ThrustOpExecutionState state) {
  return ThrustOpExecutorUtils::runMultiTypeOp<DistinctEval>(
      thrust_context_, state.column_data_type, state);
}

}  // namespace QueryRenderer
