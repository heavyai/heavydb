/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplSpecialized.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif  // HAVE_CUDA
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramProps.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/Ops/Utils.h"
#include "QueryRenderer/Utils/AnyDataType.h"
#include "QueryRenderer/Utils/thrust/ThrustAllocatorDeviceVector.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

#ifndef HAVE_CUDA
using std::min;
#endif

namespace {
template <typename T, int NUM_ELEMS = 1>
struct DistinctHistogramEval {
  static AggDataList getReturnVal(const SQLTypeInfo&,
                                  const T* result_vec,
                                  const uint32_t* result_cnt_vec,
                                  const size_t num_results) {
// TODO(croot): use data mgr to allocate these extra arrays
#ifdef HAVE_CUDA
    std::vector<T> h_results(num_results);
    std::vector<uint32_t> h_results_cnt(num_results);

    // NOTE: cuMemcpy is much faster than thrust copy routines such as
    // std::vector<T>(result_vec.begin(), new_ends.first)) This is likely because the
    // std::vector copy does an iteration through the iterators. A direct memcpy is
    // much, much faster.
    cuMemcpyDtoH(&h_results[0], (CUdeviceptr)result_vec, num_results * sizeof(T));
    cuMemcpyDtoH(
        &h_results_cnt[0], (CUdeviceptr)result_cnt_vec, num_results * sizeof(uint32_t));
#else
    std::vector<T> h_results(result_vec, result_vec + num_results);
    std::vector<uint32_t> h_results_cnt(result_cnt_vec, result_cnt_vec + num_results);
#endif  // HAVE_CUDA

    return {std::make_shared<AnyDataType>(std::move(h_results)),
            std::make_shared<AnyDataType>(std::move(h_results_cnt))};
  }

  struct QuantileMedianIndices {
    double qdiff_;
    double half_qdiff_;
    QuantileMedianIndices(const double quantile_diff)
        : qdiff_(quantile_diff), half_qdiff_(quantile_diff / 2.0) {}

    __host__ __device__ uint32_t operator()(const uint32_t val) {
      return uint32_t(ceil(double(val) * qdiff_ + half_qdiff_));
    }
  };

  struct QuantileIndexOffset {
    uint32_t half_qdiff_;
    uint32_t data_sz_;
    QuantileIndexOffset(const double quantile_diff, const uint32_t data_sz)
        : half_qdiff_(ceil(quantile_diff / 2.0)), data_sz_(data_sz - 1) {}
    __host__ __device__ uint32_t operator()(const uint32_t val) {
      return min(data_sz_, val + half_qdiff_);
    }
  };

  static AggDataList eval(DataMgrThrustContext& execution_context,
                          const ThrustOpExecutionState& state,
                          const DistinctHistogramProps& props) {
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
      return {state.input_info.type_info.is_decimal()
                  ? std::make_shared<AnyDataType>(std::vector<double>())
                  : std::make_shared<AnyDataType>(std::vector<T>()),
              std::make_shared<AnyDataType>(std::vector<uint32_t>())};
    }

    //
    // The following is an example of what the data looks like after each step
    // of building the histogram.
    //
    // Let's start with this example:
    // distinct_vec =    [ 1, 5, 3, 10, 8, 5, 10, 6, 7, 3, 1, 1]
    //
    // Ater sorting:
    // distinct_vec =    [ 1, 1, 1, 3, 3, 5, 5, 6, 7, 8, 10, 10]
    //
    // After reduce by key: // uses a const iterator (which means all 1 values)
    // result_vec =    [ 1, 3, 5, 6, 7, 8, 10]
    // result_cnt_vec = [ 3, 2, 2, 1, 1, 1,  2]
    //
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

    auto const num_results = ::thrust::inner_product(execution_context.getDevicePolicy(),
                                                     distinct_vec.begin(),
                                                     distinct_vec.end() - 1,
                                                     distinct_vec.begin() + 1,
                                                     size_t(1),
                                                     ::thrust::plus<size_t>(),
                                                     ::thrust::not_equal_to<T>());

    auto final_num_results = num_results;

    auto result_vec = make_device_vector_from_context<T>(execution_context, num_results);
    auto result_cnt_vec =
        make_device_vector_from_context<uint32_t>(execution_context, num_results);
    auto new_ends = ::thrust::reduce_by_key(execution_context.getDevicePolicy(),
                                            distinct_vec.begin(),
                                            distinct_vec.end(),
                                            ::thrust::make_constant_iterator(1),
                                            result_vec.begin(),
                                            result_cnt_vec.begin());

    CHECK_EQ(static_cast<size_t>(new_ends.first - result_vec.begin()), num_results);

    T* vals_to_use;
    uint32_t* cnts_to_use;

    auto reduced_vals = make_device_vector_from_context<T>(execution_context);
    auto reduced_cnts = make_device_vector_from_context<uint32_t>(execution_context);
    if (props.approximate && num_results > props.num_bins) {
      auto final_reduce_size = props.num_bins;
      // The following algorithm is the following:
      // We'll quantile the input set of data according to the number of bins we want to
      // use to make the approximation. To build out these quantiles, we'll determine
      // approximately how many values should be in each bin. We'll take the median
      // value in each bin to be the new "approximate" value for the bin, and we'll
      // collapse all the counts for each bin accordingly. Because the input data is
      // already histogram-binned, we're working around that datastructure, which is a
      // little tricky.
      //
      // TODO(croot): we may be overcomplicating this by doing the lookups against the
      // unique values and their respective counts. Why don't we lookup directly in the
      // original values? That would certainly be easier. The caveat being that every
      // bin would have the exact same # of values in the end. Maybe that would be ok
      //
      // We'll keep using the above example input data to illustrate what's being done
      // here. result_vec =    [ 1, 3, 5, 6, 7, 8, 10] result_cnt_vec = [ 3, 2, 2, 1, 1,
      // 1, 2]
      //
      // We'll maintain 2 examples here to illustrate. One where we take the above
      // inputs and reduce to 2 & 10 examples.
      //
      // after inclusive_scan:
      // result_cnt_vec = [ 3, 5, 7, 8, 9, 10, 12]
      ::thrust::inclusive_scan(execution_context.getDevicePolicy(),
                               result_cnt_vec.begin(),
                               result_cnt_vec.end(),
                               result_cnt_vec.begin());

      const double quantile_diff =
          (static_cast<double>(num_items) - 1) / static_cast<double>(final_reduce_size);

      auto bin_indices =
          make_device_vector_from_context<uint32_t>(execution_context, final_reduce_size);
      auto input_indices =
          make_device_vector_from_context<uint32_t>(execution_context, final_reduce_size);

      // We're first going to calculate the median value for each bin. The index for the
      // median value will therefore = curridx * quantile_diff + 0.5*quantile_diff
      // Example 1: bin_indices = [ 0, 1 ]
      //
      // Example 2:
      // bin_indices = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
      ::thrust::sequence(
          execution_context.getDevicePolicy(), bin_indices.begin(), bin_indices.end());

      // Example 1:
      // quantile_diff = 12 - 1 / 2 = 5.5
      // bin_indices = [ 3, 8 ]
      //
      // Example 2:
      // quantile_diff = 12 - 1 / 10 = 1.1
      // bin_indices = [ 1, 2, 3, 4, 5, 7, 8, 9, 10, 11 ]
      ::thrust::transform(execution_context.getDevicePolicy(),
                          bin_indices.begin(),
                          bin_indices.end(),
                          bin_indices.begin(),
                          QuantileMedianIndices(quantile_diff));

      // Example 1:
      // input_indices = [ 1, 4 ]
      //
      // Example 2:
      // input_indices = [ 0, 0, 1, 1, 2, 3, 4, 5, 6, 6]
      ::thrust::upper_bound(execution_context.getDevicePolicy(),
                            result_cnt_vec.begin(),
                            result_cnt_vec.end(),
                            bin_indices.begin(),
                            bin_indices.end(),
                            input_indices.begin());

      // Because of the initial histogram, we can have duplicates here, so we need to
      // clear those out
      //
      // Example 1:
      // bin_indices = [ 3, 8 ]
      // input_indices = [ 1, 4 ]
      //
      // Example 2:
      // bin_indices = [ 2, 4, 5, 7, 8, 9, 11]
      // input_indices = [ 0, 1, 2, 3, 4, 5, 6]
      auto new_ends = ::thrust::unique_by_key(execution_context.getDevicePolicy(),
                                              input_indices.rbegin(),
                                              input_indices.rend(),
                                              bin_indices.rbegin());
      auto offset = new_ends.first - input_indices.rbegin();
      auto inv_offset = final_reduce_size - offset;
      auto qstart = bin_indices.begin() + inv_offset;
      auto qend = bin_indices.end();
      auto lstart = input_indices.begin() + inv_offset;
      auto lend = input_indices.end();

      // update the final reduce size according to any duplicate purging
      final_reduce_size = offset;

      // resize the approximate return values and counts
      reduced_cnts.resize(final_reduce_size);
      reduced_vals.resize(final_reduce_size);

      // Now gather the estimated values from the initial unique values:
      // Example 1:
      // reduced_vals = [ 3, 7 ]
      //
      // Example 2:
      // reduced_vals = [ 1, 3, 5, 6, 7, 8, 10 ]
      ::thrust::gather(execution_context.getDevicePolicy(),
                       lstart,
                       lend,
                       result_vec.begin(),
                       reduced_vals.begin());

      // Now offset the initial bin indices to get to the end of the bin.
      // From the end of the bin we'll get the count. This will be used to suggest
      // how many values are in that bin
      // Example 1:
      // bin_indices = [6, 11]
      //
      // Example 2:
      // bin_indices = [3, 5, 6, 8, 9, 10, 11]
      ::thrust::transform(execution_context.getDevicePolicy(),
                          qstart,
                          qend,
                          qstart,
                          QuantileIndexOffset(quantile_diff, num_items));

      // Now gather the counts for each index
      // First, let's grab the indices from the counts to use
      // Example 1:
      // input_indices = [2, 6]
      //
      // Example 2:
      // input_indices = [ 0, 1, 2, 3, 4, 5, 6 ]
      ::thrust::lower_bound(execution_context.getDevicePolicy(),
                            result_cnt_vec.begin(),
                            result_cnt_vec.end(),
                            qstart,
                            qend,
                            lstart);

      // Now gather the count values
      // Example 1:
      // reduce_cnts = [7, 12]
      //
      // Example 2
      // reduce_cnts = [ 3, 5, 7, 8, 9, 10, 12]
      ::thrust::gather(execution_context.getDevicePolicy(),
                       lstart,
                       lend,
                       result_cnt_vec.begin(),
                       reduced_cnts.begin());

      // Because of the initial histogram, again we can have duplicates here, so we need
      // to clear those out. This should be ok for approximates
      auto reduced_new_ends = ::thrust::unique_by_key(execution_context.getDevicePolicy(),
                                                      reduced_cnts.begin(),
                                                      reduced_cnts.end(),
                                                      reduced_vals.begin());
      auto red_vals_start = reduced_vals.begin();
      auto red_vals_end = reduced_new_ends.second;
      auto red_cnts_start = reduced_cnts.begin();
      auto red_cnts_end = reduced_new_ends.first;
      final_reduce_size = red_vals_end - red_vals_start;

      // Now invert the initial inclusive_scan:
      // Example 1:
      // reduced_cnts = [7, 5]
      //
      // Example 2:
      // reduced_cnts = [3, 2, 2, 1, 1, 1, 2]
      ::thrust::adjacent_difference(execution_context.getDevicePolicy(),
                                    red_cnts_start,
                                    red_cnts_end,
                                    red_cnts_start);

      // So in the end
      // Example 1:
      // reduced_vals = [1, 6]
      // reduced_cnts = [7, 5]
      //
      // Example 2:
      // reduced_vals = [ 1, 3, 5, 6, 7, 8, 10 ]
      // reduced_cnts = [3, 2, 2, 1, 1, 1, 2]
      final_num_results = final_reduce_size;
      vals_to_use = ::thrust::raw_pointer_cast(reduced_vals.data());
      cnts_to_use = ::thrust::raw_pointer_cast(reduced_cnts.data());
    } else {
      vals_to_use = ::thrust::raw_pointer_cast(result_vec.data());
      cnts_to_use = ::thrust::raw_pointer_cast(result_cnt_vec.data());
    }

    return getReturnVal(
        state.input_info.type_info, vals_to_use, cnts_to_use, final_num_results);
  }
};

template <>
AggDataList DistinctHistogramEval<int64_t>::getReturnVal(const SQLTypeInfo& type_info,
                                                         const int64_t* result_vec,
                                                         const uint32_t* result_cnt_vec,
                                                         const size_t num_results) {
#ifdef HAVE_CUDA
  // TODO(croot): use data mgr to allocate these extra arrays
  std::vector<int64_t> h_results(num_results);
  std::vector<uint32_t> h_results_cnt(num_results);

  // NOTE: cuMemcpy is much faster than thrust copy routines such as
  // std::vector<T>(result_vec.begin(), new_ends.first)) This is likely because the
  // std::vector copy does an iteration through the iterators. A direct memcpy is much,
  // much faster.
  cuMemcpyDtoH(&h_results[0], (CUdeviceptr)result_vec, num_results * sizeof(int64_t));
  cuMemcpyDtoH(
      &h_results_cnt[0], (CUdeviceptr)result_cnt_vec, num_results * sizeof(uint32_t));
#else
  std::vector<int64_t> h_results(result_vec, result_vec + num_results);
  std::vector<uint32_t> h_results_cnt(result_cnt_vec, result_cnt_vec + num_results);
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
    return {std::make_shared<AnyDataType>(std::move(conv_h_results)),
            std::make_shared<AnyDataType>(std::move(h_results_cnt))};
  }

  return {std::make_shared<AnyDataType>(std::move(h_results)),
          std::make_shared<AnyDataType>(std::move(h_results_cnt))};
}
}  // namespace

AggDataList ThrustOpExecutorImpl<active_device_system>::executeDistinctHistogramOp(
    ThrustOpExecutionState state,
    const DistinctHistogramProps& props) {
  return ThrustOpExecutorUtils::runMultiTypeOp<DistinctHistogramEval>(
      thrust_context_, state.column_data_type, state, props);
}

}  // namespace QueryRenderer
