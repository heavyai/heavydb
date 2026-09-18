/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustPolygonsInterface.h"

#include <limits>
#include <type_traits>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/system_error.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "DataMgr/Allocators/thrust/TypedThrustAllocator.h"
#include "GfxDriver/Render/PPLLConstants.h"
#include "GfxDriver/RenderError.h"
#include "Logger/Logger.h"
#include "QueryEngine/GpuRtConstants.h"
#include "QueryRenderer/Utils/thrust/ThrustAllocatorDeviceVector.h"
#include "QueryRenderer/Utils/thrust/ThrustDeviceSystem.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

using Data_Namespace::ThrustAllocatorDeviceVector;
using Data_Namespace::TypedThrustAllocator;

//
// THRUST FUNCTORS
//

namespace detail_polys {

struct GetRowIDFunctor {
  int8_t* query_output_buffer_base_ptr_;
  uint32_t query_output_buffer_stride_;
  uint32_t rowid_buffer_col_idx_;

  GetRowIDFunctor(int8_t* query_output_buffer_base_ptr,
                  uint32_t query_output_buffer_stride,
                  uint32_t rowid_buffer_col_idx)
      : query_output_buffer_base_ptr_{query_output_buffer_base_ptr}
      , query_output_buffer_stride_{query_output_buffer_stride}
      , rowid_buffer_col_idx_{rowid_buffer_col_idx} {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t row_index = ::thrust::get<0>(t);

    // row base
    int8_t* p_row_bytes =
        query_output_buffer_base_ptr_ + (row_index * query_output_buffer_stride_);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // key (first value)
    int64_t key = *p_row_values;

    // order by valid rowids
    uint32_t order = UINT32_MAX;
    if (key != EMPTY_KEY_64) {
      // get rowid
      int64_t rowid = p_row_values[rowid_buffer_col_idx_];
      if (rowid >= 0) {
        // valid
        order = (int32_t)rowid;
      }
    }

    // outputs
    ::thrust::get<1>(t) = order;
  }
};

template <typename T, bool IsInSitu>
struct PolyBufferAccessor {
  int8_t* buffer_ptr_;
  uint32_t buffer_ptr_idx_;
};

template <typename T>
struct PolyBufferAccessor<T, true> {
  int8_t* buffer_ptr_;
  uint32_t buffer_ptr_idx_;

  explicit PolyBufferAccessor(uint32_t buffer_ptr_idx)
      : buffer_ptr_{nullptr}, buffer_ptr_idx_{buffer_ptr_idx} {}

  inline __host__ __device__ T* operator()(const int64_t* p_row_values) {
    return reinterpret_cast<T*>(p_row_values[buffer_ptr_idx_]);
  }
};

template <typename T>
struct PolyBufferAccessor<T, false> {
  int8_t* buffer_ptr_;
  uint32_t buffer_ptr_idx_;

  explicit PolyBufferAccessor(int8_t* buffer_ptr, uint32_t buffer_ptr_idx)
      : buffer_ptr_{buffer_ptr}, buffer_ptr_idx_{buffer_ptr_idx} {}

  inline __host__ __device__ T* operator()(const int64_t* p_row_values) {
    return reinterpret_cast<T*>(buffer_ptr_ + p_row_values[buffer_ptr_idx_]);
  }
};

template <bool IsInSitu>
struct GetNumVertsAndPolysFunctor {
  int8_t* query_output_buffer_base_ptr_;
  uint32_t query_output_buffer_stride_;
  PolyBufferAccessor<int32_t, IsInSitu> ring_sizes_accessor_;
  uint32_t ring_sizes_num_idx_;

  GetNumVertsAndPolysFunctor(int8_t* query_output_buffer_base_ptr,
                             uint32_t query_output_buffer_stride,
                             PolyBufferAccessor<int32_t, IsInSitu> ring_sizes_accessor)
      : query_output_buffer_base_ptr_{query_output_buffer_base_ptr}
      , query_output_buffer_stride_{query_output_buffer_stride}
      , ring_sizes_accessor_{std::move(ring_sizes_accessor)}
      , ring_sizes_num_idx_{ring_sizes_accessor_.buffer_ptr_idx_ + 1} {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t sorted_row_index = ::thrust::get<0>(t);

    // compute total number of verts for this row
    uint32_t vert_count = 0;

    // row base
    int8_t* p_row_bytes =
        query_output_buffer_base_ptr_ + (sorted_row_index * query_output_buffer_stride_);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // ring sizes ptr
    // buffer 64-bit value is the ptr
    // source values are int32_t so cast the ptr to that
    int32_t* p_ring_sizes = ring_sizes_accessor_(p_row_values);

    // ring sizes num
    // buffer 64-bit value can be cast down to uint32_t
    uint32_t num_ring_sizes = (uint32_t)p_row_values[ring_sizes_num_idx_];

    // verts per ring
    for (uint32_t i = 0; i < num_ring_sizes; i++) {
      vert_count += p_ring_sizes[i];
    }

    // repeats (3 per ring)
    // separators (between each ring, but not at the end)
    if (num_ring_sizes > 0) {
      vert_count += (num_ring_sizes * 4) + 3;
    }

    // outputs
    ::thrust::get<1>(t) = vert_count;
    ::thrust::get<2>(t) = num_ring_sizes;
  }
};

template <typename CoordT, bool IsInSitu>
struct CopyVertexDataFunctor {
  int8_t* query_output_buffer_base_ptr_;
  uint32_t query_output_buffer_stride_;
  PolyBufferAccessor<CoordT, IsInSitu> coords_accessor_;
  PolyBufferAccessor<int32_t, IsInSitu> rings_accessor_;
  uint32_t ring_sizes_num_idx_;
  CoordT* pd_vertex_data_;
  CoordT separator_value_;

  CopyVertexDataFunctor(int8_t* query_output_buffer_base_ptr,
                        uint32_t query_output_buffer_stride,
                        PolyBufferAccessor<CoordT, IsInSitu> coords_accessor,
                        PolyBufferAccessor<int32_t, IsInSitu> rings_accessor,
                        CoordT* pd_vertex_data,
                        CoordT separator_value)
      : query_output_buffer_base_ptr_{query_output_buffer_base_ptr}
      , query_output_buffer_stride_{query_output_buffer_stride}
      , coords_accessor_{std::move(coords_accessor)}
      , rings_accessor_{std::move(rings_accessor)}
      , ring_sizes_num_idx_{rings_accessor_.buffer_ptr_idx_ + 1}
      , pd_vertex_data_{pd_vertex_data}
      , separator_value_{separator_value} {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t sorted_row_index = ::thrust::get<0>(t);
    uint32_t this_first_vert_per_row = ::thrust::get<1>(t);

    // row base
    int8_t* p_row_bytes =
        query_output_buffer_base_ptr_ + (sorted_row_index * query_output_buffer_stride_);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // coords ptr
    // buffer 64-bit value is the ptr
    // cast that ptr to the coord data type
    CoordT* p_coords = coords_accessor_(p_row_values);

    // ring sizes ptr
    // buffer 64-bit value is the ptr
    // source values are int32_t so cast the ptr to that
    int32_t* p_ring_sizes = rings_accessor_(p_row_values);

    // ring sizes num
    // buffer 64-bit value can be cast down to uint32_t
    uint32_t num_ring_sizes = (uint32_t)p_row_values[ring_sizes_num_idx_];

    // empty?
    if (num_ring_sizes == 0) {
      return;
    }

    // copy coords to verts
    uint32_t vertex_data_idx = this_first_vert_per_row * 2;
    uint32_t coords_ring_start = 0;
    uint32_t coords_ring_idx = 0;

    // copy rings
    uint32_t last_ring = num_ring_sizes - 1;
    for (uint32_t i = 0; i < num_ring_sizes; i++) {
      // ring vertices
      coords_ring_idx = coords_ring_start;
      for (int32_t j = 0; j < p_ring_sizes[i] * 2; j++) {
        pd_vertex_data_[vertex_data_idx++] = p_coords[coords_ring_idx++];
      }

      // repeats
      coords_ring_idx = coords_ring_start;
      for (uint32_t j = 0; j < 6; j++) {
        pd_vertex_data_[vertex_data_idx++] = p_coords[coords_ring_idx++];
      }

      // separator (between rings only)
      if (i < last_ring) {
        pd_vertex_data_[vertex_data_idx++] = separator_value_;
        pd_vertex_data_[vertex_data_idx++] = separator_value_;
      }

      // next ring
      coords_ring_start += p_ring_sizes[i] * 2;
    }
  }
};

template <bool IsInSitu>
struct PPLLBuildDrawDataFunctor {
  int8_t* query_output_buffer_base_ptr_;
  uint32_t query_output_buffer_stride_;
  PolyBufferAccessor<int32_t, IsInSitu> rings_accessor_;
  uint32_t ring_sizes_num_idx_;
  uint32_t* pd_line_draw_data_;
  uint32_t* pd_poly_draw_data_;
  uint32_t* pd_poly_rowids_data_;

  PPLLBuildDrawDataFunctor(int8_t* query_output_buffer_base_ptr,
                           uint32_t query_output_buffer_stride,
                           PolyBufferAccessor<int32_t, IsInSitu> rings_accessor,
                           uint32_t* pd_line_draw_data,
                           uint32_t* pd_poly_draw_data,
                           uint32_t* pd_poly_rowids_data)
      : query_output_buffer_base_ptr_{query_output_buffer_base_ptr}
      , query_output_buffer_stride_{query_output_buffer_stride}
      , rings_accessor_{std::move(rings_accessor)}
      , ring_sizes_num_idx_{rings_accessor_.buffer_ptr_idx_ + 1}
      , pd_line_draw_data_{pd_line_draw_data}
      , pd_poly_draw_data_{pd_poly_draw_data}
      , pd_poly_rowids_data_{pd_poly_rowids_data} {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t sorted_row_index = ::thrust::get<0>(t);
    uint32_t output_index = ::thrust::get<1>(t);
    uint32_t this_first_poly_per_row = ::thrust::get<2>(t);
    uint32_t this_first_vert_per_row = ::thrust::get<3>(t);
    uint32_t this_num_verts_per_row = ::thrust::get<4>(t);

    // row base
    int8_t* p_row_bytes =
        query_output_buffer_base_ptr_ + (sorted_row_index * query_output_buffer_stride_);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // ring sizes ptr
    // buffer 64-bit value is the ptr
    // source values are int32_t so cast the ptr to that
    int32_t* p_ring_sizes = rings_accessor_(p_row_values);

    // ring sizes num
    // buffer 64-bit value can be cast down to uint32_t
    uint32_t num_ring_sizes = (uint32_t)p_row_values[ring_sizes_num_idx_];

    // indirect draw struct start indices
    uint32_t line_index = output_index * 4;
    uint32_t poly_index = this_first_poly_per_row * 4;

    // typedef  struct {
    //    uint  vertex_count;
    //    uint  instance_count;
    //    uint  first_vertex;
    //    uint  first_instance;
    // } IndirectDrawVertexData;

    if (num_ring_sizes > 0) {
      // line draw data (all but the last four verts, including repeats and separators)
      pd_line_draw_data_[line_index++] = this_num_verts_per_row - 4;
      pd_line_draw_data_[line_index++] = 1;
      pd_line_draw_data_[line_index++] = this_first_vert_per_row;
      pd_line_draw_data_[line_index++] = 0;
    } else {
      // empty line draw data
      pd_line_draw_data_[line_index++] = 0;
      pd_line_draw_data_[line_index++] = 0;
      pd_line_draw_data_[line_index++] = 0;
      pd_line_draw_data_[line_index++] = 0;
    }

    // poly draw data (all verts for each ring, skipping repeats and separators)
    for (uint32_t i = 0; i < num_ring_sizes; i++) {
      pd_poly_draw_data_[poly_index++] = p_ring_sizes[i];
      pd_poly_draw_data_[poly_index++] = 1;
      pd_poly_draw_data_[poly_index++] = this_first_vert_per_row;
      pd_poly_draw_data_[poly_index++] = 0;
      this_first_vert_per_row += (p_ring_sizes[i] + 4);
    }

    // poly rowids data
    uint32_t poly_rowid_index = this_first_poly_per_row;
    for (uint32_t i = 0; i < num_ring_sizes; i++) {
      pd_poly_rowids_data_[poly_rowid_index++] = output_index;
    }
  }
};

struct BuildPerRowDataFunctor {
  int8_t* query_output_buffer_base_ptr_;
  uint32_t query_output_buffer_stride_;
  int8_t* pd_per_row_data_;
  uint32_t per_row_buffer_stride_;
  uint32_t per_row_buffer_num_cols_;
  uint32_t* pd_per_row_buffer_col_idxs_;
  uint32_t* pd_per_row_buffer_col_offsets_;
  uint32_t* pd_per_row_buffer_col_types_;

  BuildPerRowDataFunctor(int8_t* query_output_buffer_base_ptr,
                         uint32_t query_output_buffer_stride,
                         int8_t* pd_per_row_data,
                         uint32_t per_row_buffer_stride,
                         uint32_t per_row_buffer_num_cols,
                         uint32_t* pd_per_row_buffer_col_idxs,
                         uint32_t* pd_per_row_buffer_col_offsets,
                         uint32_t* pd_per_row_buffer_col_types)
      : query_output_buffer_base_ptr_{query_output_buffer_base_ptr}
      , query_output_buffer_stride_{query_output_buffer_stride}
      , pd_per_row_data_{pd_per_row_data}
      , per_row_buffer_stride_{per_row_buffer_stride}
      , per_row_buffer_num_cols_{per_row_buffer_num_cols}
      , pd_per_row_buffer_col_idxs_{pd_per_row_buffer_col_idxs}
      , pd_per_row_buffer_col_offsets_{pd_per_row_buffer_col_offsets}
      , pd_per_row_buffer_col_types_{pd_per_row_buffer_col_types} {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t output_index = ::thrust::get<0>(t);
    uint32_t row_index = ::thrust::get<1>(t);

    // row base
    int8_t* p_row_bytes =
        query_output_buffer_base_ptr_ + (row_index * query_output_buffer_stride_);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // output base
    int8_t* p_output_bytes = pd_per_row_data_ + (output_index * per_row_buffer_stride_);

    // outputs
    // @TODO simon.eves
    // is there a more efficient implementation?
    // also need to implement the rowid/projection-query logic... how?
    for (uint32_t i = 0; i < per_row_buffer_num_cols_; i++) {
      switch (pd_per_row_buffer_col_types_[i]) {
        case (uint32_t)::gfx::BufferAttrType::kInt: {
          // cast to int32
          int64_t* p_row_value_int64 = p_row_values + pd_per_row_buffer_col_idxs_[i];
          int32_t* p_output_int32 = reinterpret_cast<int32_t*>(
              p_output_bytes + pd_per_row_buffer_col_offsets_[i]);
          *p_output_int32 = (int32_t)(*p_row_value_int64);
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kUint: {
          // cast to uint32
          uint64_t* p_row_value_uint64 =
              reinterpret_cast<uint64_t*>(p_row_values + pd_per_row_buffer_col_idxs_[i]);
          uint32_t* p_output_uint32 = reinterpret_cast<uint32_t*>(
              p_output_bytes + pd_per_row_buffer_col_offsets_[i]);
          *p_output_uint32 = (uint32_t)(*p_row_value_uint64);
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kInt64: {
          // copy int64
          int64_t* p_row_value_int64 = p_row_values + pd_per_row_buffer_col_idxs_[i];
          int64_t* p_output_int64 = reinterpret_cast<int64_t*>(
              p_output_bytes + pd_per_row_buffer_col_offsets_[i]);
          *p_output_int64 = *p_row_value_int64;
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kUint64: {
          // copy uint64
          uint64_t* p_row_value_uint64 =
              reinterpret_cast<uint64_t*>(p_row_values + pd_per_row_buffer_col_idxs_[i]);
          uint64_t* p_output_uint64 = reinterpret_cast<uint64_t*>(
              p_output_bytes + pd_per_row_buffer_col_offsets_[i]);
          *p_output_uint64 = *p_row_value_uint64;
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kFloat: {
          // cast double to float
          double* p_row_value_double =
              reinterpret_cast<double*>(p_row_values + pd_per_row_buffer_col_idxs_[i]);
          float* p_output_float = reinterpret_cast<float*>(
              p_output_bytes + pd_per_row_buffer_col_offsets_[i]);
          *p_output_float = (float)(*p_row_value_double);
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kDouble: {
          // copy double
          double* p_row_value_double =
              reinterpret_cast<double*>(p_row_values + pd_per_row_buffer_col_idxs_[i]);
          double* p_output_double = reinterpret_cast<double*>(
              p_output_bytes + pd_per_row_buffer_col_offsets_[i]);
          *p_output_double = *p_row_value_double;
          break;
        }
        default:
          break;
      }
    }
  }
};

template <bool IsInSitu>
struct GetBatchInfoFunctor {
  int8_t* query_output_buffer_base_ptr_;
  uint32_t query_output_buffer_stride_;
  PolyBufferAccessor<int32_t, IsInSitu> ring_sizes_accessor_;
  uint32_t ring_sizes_num_idx_;
  uint32_t row_count_;
  uint32_t max_rows_per_batch_;
  uint32_t* pd_sorted_row_indices_;

  GetBatchInfoFunctor(int8_t* query_output_buffer_base_ptr,
                      uint32_t query_output_buffer_stride,
                      PolyBufferAccessor<int32_t, IsInSitu> ring_sizes_accessor,
                      uint32_t row_count,
                      uint32_t max_rows_per_batch,
                      uint32_t* pd_sorted_row_indices)
      : query_output_buffer_base_ptr_{query_output_buffer_base_ptr}
      , query_output_buffer_stride_{query_output_buffer_stride}
      , ring_sizes_accessor_{std::move(ring_sizes_accessor)}
      , ring_sizes_num_idx_{ring_sizes_accessor_.buffer_ptr_idx_ + 1}
      , row_count_{row_count}
      , max_rows_per_batch_{max_rows_per_batch}
      , pd_sorted_row_indices_{pd_sorted_row_indices} {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t batch_index = ::thrust::get<0>(t);

    // prepare to sum for batch
    uint32_t polys_in_batch{0u};

    // first and last rows for batch
    uint32_t first_row = batch_index * max_rows_per_batch_;
    uint32_t next_first_row = first_row + max_rows_per_batch_;
    next_first_row = (next_first_row > row_count_) ? row_count_ : next_first_row;
    uint32_t rows_in_batch = next_first_row - first_row;

    // loop over those rows
    for (uint32_t row = first_row; row < next_first_row; row++) {
      // sorted row index
      uint32_t sorted_row_index = pd_sorted_row_indices_[row];

      // row base
      int8_t* p_row_bytes = query_output_buffer_base_ptr_ +
                            (sorted_row_index * query_output_buffer_stride_);
      int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

      // ring sizes num
      // buffer 64-bit value can be cast down to uint32_t
      uint32_t num_ring_sizes = (uint32_t)p_row_values[ring_sizes_num_idx_];

      // add num polys for this row
      polys_in_batch += num_ring_sizes;
    }

    // outputs
    ::thrust::get<1>(t) = rows_in_batch;
    ::thrust::get<2>(t) = polys_in_batch;
  }
};

void run_thrust_stage_checked(const std::string& stage_name,
                              std::function<void()> run_lambda,
                              std::function<void()> fail_lambda,
                              uint32_t row_count) {
  try {
    run_lambda();
  } catch (std::bad_alloc& e) {
    if (fail_lambda) {
      fail_lambda();
    }
    std::stringstream ss;
    ss << "In-Situ Polys Thrust Out-of-Memory Error in " << stage_name << ": " << e.what()
       << " (row count = " << row_count << ")";
    throw gfx::OutOfGpuMemoryError(ss.str());
  } catch (::thrust::system_error& e) {
    if (fail_lambda) {
      fail_lambda();
    }
    std::stringstream ss;
    ss << "In-Situ Polys Thrust System Error in " << stage_name << ": " << e.what()
       << " (row count = " << row_count << ")";
    throw std::runtime_error(ss.str());
  }
}

}  // namespace detail_polys

//
// MAIN CLASS
//

template <ThrustDeviceSystem device_system>
class PolygonDataConverterImpl final : public PolygonDataConverterInterface {
 public:
  PolygonDataConverterImpl() : PolygonDataConverterInterface() {}
  ~PolygonDataConverterImpl() override = default;

  void ConvertStage1(DataMgrThrustContext& thrust_context,
                     uint32_t query_row_count,
                     int8_t* query_output_buffer_base_ptr,
                     uint32_t query_output_buffer_stride,
                     int32_t ring_sizes_ptr_buffer_col_idx,
                     int32_t rowid_buffer_col_idx,
                     uint32_t& row_count,
                     uint32_t& vert_count,
                     uint32_t& poly_count,
                     int8_t* rings_sizes_buffer_base_ptr) override {
    if (!state_) {
      state_ = std::make_unique<InternalThrustState>(thrust_context);
    }

    auto clean_up_on_error = [&] {
      row_count = 0;
      vert_count = 0;
      poly_count = 0;
    };

    clean_up_on_error();

    //
    // SIZE THESE
    //

    auto resize_query_row_count = [&] {
      state_->d_sorted_row_index.resize(query_row_count);
      state_->d_rowids.resize(query_row_count);
    };

    detail_polys::run_thrust_stage_checked("resize query row count",
                                           resize_query_row_count,
                                           clean_up_on_error,
                                           query_row_count);

    //
    // GET ROWID PER ROW
    //

    auto get_sort_value_per_row = [&] {
      ::thrust::counting_iterator<uint32_t> row_index_it(0);
      ::thrust::for_each(thrust_context.getDevicePolicy(),
                         ::thrust::make_zip_iterator(::thrust::make_tuple(
                             row_index_it, state_->d_rowids.begin())),
                         ::thrust::make_zip_iterator(::thrust::make_tuple(
                             row_index_it + query_row_count, state_->d_rowids.end())),
                         detail_polys::GetRowIDFunctor(
                             ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                             query_output_buffer_stride,
                             rowid_buffer_col_idx));
    };

    detail_polys::run_thrust_stage_checked(
        "get rowid values", get_sort_value_per_row, clean_up_on_error, query_row_count);

    //
    // CREATE INDEX ARRAY AND SORT BY ROWID
    //

    // this gives us the master output order
    // all the invalid rows will be sorted to the end

    auto create_index_array_and_sort_by_rowid = [&] {
      ::thrust::sequence(thrust_context.getDevicePolicy(),
                         state_->d_sorted_row_index.begin(),
                         state_->d_sorted_row_index.end());
      ::thrust::sort_by_key(thrust_context.getDevicePolicy(),
                            state_->d_rowids.begin(),
                            state_->d_rowids.end(),
                            state_->d_sorted_row_index.begin());
    };

    detail_polys::run_thrust_stage_checked("create index array and sort by rowid",
                                           create_index_array_and_sort_by_rowid,
                                           clean_up_on_error,
                                           query_row_count);

    //
    // TOTAL NUM ROWS
    //

    // first we need to find the first invalid row by searching for the first instance of
    // the invalid value
    //
    // so the last value in the vector will be the last group
    // and hence the number of groups is this plus one

    auto get_row_count = [&] {
      auto firstInvalidRowIt = ::thrust::lower_bound(thrust_context.getDevicePolicy(),
                                                     state_->d_rowids.begin(),
                                                     state_->d_rowids.end(),
                                                     UINT32_MAX);
      row_count = firstInvalidRowIt - state_->d_rowids.begin();
    };

    detail_polys::run_thrust_stage_checked(
        "get row count", get_row_count, clean_up_on_error, query_row_count);

    //
    // ANY NON-EMPTY ROWS?
    //

    if (row_count == 0) {
      vert_count = 0;
      poly_count = 0;
      // not an error
      return;
    }

    //
    // RESIZE THE INDEX ARRAYS (discard everything beyond row_count) AND SIZE THE REST
    //

    auto resize_other_arrays = [&] { state_->resizeAll(row_count); };

    detail_polys::run_thrust_stage_checked(
        "resize other arrays", resize_other_arrays, clean_up_on_error, row_count);

    //
    // GET NUM VERTS AND POLYS PER ROW
    //

    // we use the sorted index

    auto get_num_verts_and_polys_per_row = [&] {
      auto start_itr = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(state_->d_sorted_row_index.begin(),
                               state_->d_num_verts_per_row.begin(),
                               state_->d_num_polys_per_row.begin()));
      auto end_itr = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(state_->d_sorted_row_index.end(),
                               state_->d_num_verts_per_row.end(),
                               state_->d_num_polys_per_row.end()));

      if (rings_sizes_buffer_base_ptr) {
        ::thrust::for_each(
            thrust_context.getDevicePolicy(),
            start_itr,
            end_itr,
            detail_polys::GetNumVertsAndPolysFunctor<false>(
                ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                query_output_buffer_stride,
                detail_polys::PolyBufferAccessor<int32_t, false>(
                    rings_sizes_buffer_base_ptr, ring_sizes_ptr_buffer_col_idx)));
      } else {
        ::thrust::for_each(thrust_context.getDevicePolicy(),
                           start_itr,
                           end_itr,
                           detail_polys::GetNumVertsAndPolysFunctor<true>(
                               ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                               query_output_buffer_stride,
                               detail_polys::PolyBufferAccessor<int32_t, true>(
                                   ring_sizes_ptr_buffer_col_idx)));
      }
    };

    detail_polys::run_thrust_stage_checked("get num verts and polys per row",
                                           get_num_verts_and_polys_per_row,
                                           clean_up_on_error,
                                           row_count);

    //
    // COMPUTE FIRST VERTS AND POLYS PER ROW AND TOTAL VERTS AND POLYS
    //

    // the firsts will be in sorted order, which is what we want
    // we need an exclusive-scan in each case but this does not compute
    // the final total, so we have to add back on the num of the last row

    auto get_vert_count = [&] {
      ::thrust::exclusive_scan(thrust_context.getDevicePolicy(),
                               state_->d_num_verts_per_row.begin(),
                               state_->d_num_verts_per_row.end(),
                               state_->d_first_vert_per_row.begin());

      vert_count = state_->d_first_vert_per_row[row_count - 1] +
                   state_->d_num_verts_per_row[row_count - 1];
    };

    detail_polys::run_thrust_stage_checked(
        "get vert count", get_vert_count, clean_up_on_error, row_count);

    auto get_poly_count = [&] {
      ::thrust::exclusive_scan(thrust_context.getDevicePolicy(),
                               state_->d_num_polys_per_row.begin(),
                               state_->d_num_polys_per_row.end(),
                               state_->d_first_poly_per_row.begin());

      poly_count = state_->d_first_poly_per_row[row_count - 1] +
                   state_->d_num_polys_per_row[row_count - 1];
    };

    detail_polys::run_thrust_stage_checked(
        "get poly count", get_poly_count, clean_up_on_error, row_count);
  }

  void ConvertStage2(
      DataMgrThrustContext& thrust_context,
      uint32_t row_count,
      int8_t* query_output_buffer_base_ptr,
      uint32_t query_output_buffer_stride,
      int32_t coords_ptr_buffer_col_idx,
      int32_t ring_sizes_ptr_buffer_col_idx,
      const EncodingType coords_encoding,
      const std::vector<uint32_t> per_row_data_buffer_col_idxs,
      const std::vector<uint32_t> per_row_data_buffer_col_offsets,
      const std::vector<::gfx::BufferAttrType> per_row_data_buffer_col_types,
      uint32_t per_row_data_buffer_stride,
      const ::QueryRenderer::PolyBufferMemoryDescriptors& buffer_memory_descriptors,
      std::vector<uint32_t>& num_rows_per_batch,
      std::vector<uint32_t>& num_polys_per_batch,
      int8_t* coords_buffer_base_ptr,
      int8_t* ring_sizes_buffer_base_ptr) override {
    CHECK(state_ != nullptr);
    //
    // GET RAW BUFFER POINTERS FROM CUDA HANDLES
    //

    auto pd_line_draw_data = ::thrust::device_pointer_cast(reinterpret_cast<uint32_t*>(
        buffer_memory_descriptors.line_indirect_vbo_descriptor.handle));
    auto pd_poly_draw_data = ::thrust::device_pointer_cast(reinterpret_cast<uint32_t*>(
        buffer_memory_descriptors.poly_indirect_vbo_descriptor.handle));
    auto pd_per_row_data = ::thrust::device_pointer_cast(
        reinterpret_cast<int8_t*>(buffer_memory_descriptors.ssbo_descriptor.handle));

    //
    // COPY AND TRANSFORM VERTEX DATA
    //

    // we can now allocate the final renderable vertex buffer
    // now that we know the total num verts
    // then run a functor to copy in the vertex data

    auto copy_vertex_data = [&] {
      auto start_copy_itr = ::thrust::make_zip_iterator(::thrust::make_tuple(
          state_->d_sorted_row_index.begin(), state_->d_first_vert_per_row.begin()));
      auto end_copy_itr = ::thrust::make_zip_iterator(::thrust::make_tuple(
          state_->d_sorted_row_index.end(), state_->d_first_vert_per_row.end()));
      if (coords_encoding == kENCODING_GEOINT) {
        auto pd_vertex_data = ::thrust::device_pointer_cast(
            reinterpret_cast<int32_t*>(buffer_memory_descriptors.vbo_descriptor.handle));
        auto const separator_value = -std::numeric_limits<int32_t>::max();
        if (coords_buffer_base_ptr) {
          ::thrust::for_each(
              thrust_context.getDevicePolicy(),
              start_copy_itr,
              end_copy_itr,
              detail_polys::CopyVertexDataFunctor<int32_t, false>(
                  ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                  query_output_buffer_stride,
                  detail_polys::PolyBufferAccessor<int32_t, false>(
                      coords_buffer_base_ptr, coords_ptr_buffer_col_idx),
                  detail_polys::PolyBufferAccessor<int32_t, false>(
                      ring_sizes_buffer_base_ptr, ring_sizes_ptr_buffer_col_idx),
                  ::thrust::raw_pointer_cast(pd_vertex_data),
                  separator_value));
        } else {
          ::thrust::for_each(thrust_context.getDevicePolicy(),
                             start_copy_itr,
                             end_copy_itr,
                             detail_polys::CopyVertexDataFunctor<int32_t, true>(
                                 ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                                 query_output_buffer_stride,
                                 detail_polys::PolyBufferAccessor<int32_t, true>(
                                     coords_ptr_buffer_col_idx),
                                 detail_polys::PolyBufferAccessor<int32_t, true>(
                                     ring_sizes_ptr_buffer_col_idx),
                                 ::thrust::raw_pointer_cast(pd_vertex_data),
                                 separator_value));
        }
      } else {
        auto pd_vertex_data = ::thrust::device_pointer_cast(
            reinterpret_cast<double*>(buffer_memory_descriptors.vbo_descriptor.handle));
        auto const separator_value = -std::numeric_limits<double>::max();
        if (coords_buffer_base_ptr) {
          ::thrust::for_each(
              thrust_context.getDevicePolicy(),
              start_copy_itr,
              end_copy_itr,
              detail_polys::CopyVertexDataFunctor<double, false>(
                  ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                  query_output_buffer_stride,
                  detail_polys::PolyBufferAccessor<double, false>(
                      coords_buffer_base_ptr, coords_ptr_buffer_col_idx),
                  detail_polys::PolyBufferAccessor<int32_t, false>(
                      ring_sizes_buffer_base_ptr, ring_sizes_ptr_buffer_col_idx),
                  ::thrust::raw_pointer_cast(pd_vertex_data),
                  separator_value));
        } else {
          ::thrust::for_each(thrust_context.getDevicePolicy(),
                             start_copy_itr,
                             end_copy_itr,
                             detail_polys::CopyVertexDataFunctor<double, true>(
                                 ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                                 query_output_buffer_stride,
                                 detail_polys::PolyBufferAccessor<double, true>(
                                     coords_ptr_buffer_col_idx),
                                 detail_polys::PolyBufferAccessor<int32_t, true>(
                                     ring_sizes_ptr_buffer_col_idx),
                                 ::thrust::raw_pointer_cast(pd_vertex_data),
                                 separator_value));
        }
      }
    };

    detail_polys::run_thrust_stage_checked(
        "copy and transform vertex data", copy_vertex_data, nullptr, row_count);

    //
    // BUILD INDIRECT DRAW DATA
    //

    auto pd_poly_rowids_data = ::thrust::device_pointer_cast(reinterpret_cast<uint32_t*>(
        buffer_memory_descriptors.poly_rowids_ssbo_descriptor.handle));

    auto build_indirect_draw_data = [&] {
      ::thrust::counting_iterator<uint32_t> output_index_it1(0);
      auto start_cnt_itr = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(state_->d_sorted_row_index.begin(),
                               output_index_it1,
                               state_->d_first_poly_per_row.begin(),
                               state_->d_first_vert_per_row.begin(),
                               state_->d_num_verts_per_row.begin()));
      auto end_cnt_itr = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(state_->d_sorted_row_index.end(),
                               output_index_it1 + row_count,
                               state_->d_first_poly_per_row.end(),
                               state_->d_first_vert_per_row.end(),
                               state_->d_num_verts_per_row.end()));

      if (ring_sizes_buffer_base_ptr) {
        ::thrust::for_each(
            thrust_context.getDevicePolicy(),
            start_cnt_itr,
            end_cnt_itr,
            detail_polys::PPLLBuildDrawDataFunctor<false>(
                ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                query_output_buffer_stride,
                detail_polys::PolyBufferAccessor<int32_t, false>(
                    ring_sizes_buffer_base_ptr, ring_sizes_ptr_buffer_col_idx),
                ::thrust::raw_pointer_cast(pd_line_draw_data),
                ::thrust::raw_pointer_cast(pd_poly_draw_data),
                ::thrust::raw_pointer_cast(pd_poly_rowids_data)));
      } else {
        ::thrust::for_each(thrust_context.getDevicePolicy(),
                           start_cnt_itr,
                           end_cnt_itr,
                           detail_polys::PPLLBuildDrawDataFunctor<true>(
                               ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                               query_output_buffer_stride,
                               detail_polys::PolyBufferAccessor<int32_t, true>(
                                   ring_sizes_ptr_buffer_col_idx),
                               ::thrust::raw_pointer_cast(pd_line_draw_data),
                               ::thrust::raw_pointer_cast(pd_poly_draw_data),
                               ::thrust::raw_pointer_cast(pd_poly_rowids_data)));
      }
    };

    detail_polys::run_thrust_stage_checked(
        "build indirect draw data", build_indirect_draw_data, nullptr, row_count);

    //
    // BUILD PER-ROW DATA
    //

    auto d_per_row_data_buffer_col_idxs =
        make_device_vector_from_context<uint32_t>(thrust_context);
    auto d_per_row_data_buffer_col_offsets =
        make_device_vector_from_context<uint32_t>(thrust_context);
    auto d_per_row_data_buffer_col_types =
        make_device_vector_from_context<uint32_t>(thrust_context);

    uint32_t per_row_data_buffer_num_cols = per_row_data_buffer_col_idxs.size();

    auto copy_per_row_data = [&] {
      d_per_row_data_buffer_col_idxs = per_row_data_buffer_col_idxs;
      d_per_row_data_buffer_col_offsets = per_row_data_buffer_col_offsets;
      d_per_row_data_buffer_col_types.resize(per_row_data_buffer_num_cols);
      for (uint32_t i = 0; i < per_row_data_buffer_num_cols; i++) {
        d_per_row_data_buffer_col_types[i] = (uint32_t)per_row_data_buffer_col_types[i];
      }
    };

    detail_polys::run_thrust_stage_checked(
        "copy per-row data", copy_per_row_data, nullptr, row_count);

    auto build_per_row_data = [&] {
      ::thrust::counting_iterator<uint32_t> output_index_it2(0);
      ::thrust::for_each(
          thrust_context.getDevicePolicy(),
          ::thrust::make_zip_iterator(
              ::thrust::make_tuple(output_index_it2, state_->d_sorted_row_index.begin())),
          ::thrust::make_zip_iterator(::thrust::make_tuple(
              output_index_it2 + row_count, state_->d_sorted_row_index.end())),
          detail_polys::BuildPerRowDataFunctor(
              ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
              query_output_buffer_stride,
              ::thrust::raw_pointer_cast(pd_per_row_data),
              per_row_data_buffer_stride,
              per_row_data_buffer_num_cols,
              ::thrust::raw_pointer_cast(&d_per_row_data_buffer_col_idxs[0]),
              ::thrust::raw_pointer_cast(&d_per_row_data_buffer_col_offsets[0]),
              ::thrust::raw_pointer_cast(&d_per_row_data_buffer_col_types[0])));
    };

    detail_polys::run_thrust_stage_checked(
        "build per-row data", build_per_row_data, nullptr, row_count);

    //
    // COMPUTE BATCH INFO
    //
    // Values determined by empirical testing against dataset with
    // a large number of deeply overlapping small polygons
    // Larger batch size is generally better, but can lead to deeper pixels
    // More batches increases loop overhead (including a queue wait and struct readback)
    uint32_t batch_size = 16000;  // target batch size
    uint32_t num_batches = (row_count + batch_size - 1) / batch_size;

    // Limit batch count to prevent batch loop overhead from overtaking benefits
    if (num_batches > gfx::kMaxNumPPLLPrimitiveBatches) {
      // Subtract one to account for rounding up in the next step
      batch_size = row_count / (gfx::kMaxNumPPLLPrimitiveBatches - 1);
      num_batches = (row_count + batch_size - 1) / batch_size;
    }

    auto d_num_rows_per_batch = make_device_vector_from_context<uint32_t>(thrust_context);
    auto d_num_polys_per_batch =
        make_device_vector_from_context<uint32_t>(thrust_context);

    auto size_batch_arrays = [&] {
      d_num_rows_per_batch.resize(num_batches);
      d_num_polys_per_batch.resize(num_batches);
    };

    detail_polys::run_thrust_stage_checked(
        "size batch arrays", size_batch_arrays, nullptr, row_count);

    //
    // COMPUTE INFO
    //

    auto compute_batch_info = [&] {
      ::thrust::counting_iterator<uint32_t> batch_zero(0);
      auto start_batch_itr = ::thrust::make_zip_iterator(::thrust::make_tuple(
          batch_zero, d_num_rows_per_batch.begin(), d_num_polys_per_batch.begin()));
      auto end_batch_itr =
          ::thrust::make_zip_iterator(::thrust::make_tuple(batch_zero + num_batches,
                                                           d_num_rows_per_batch.end(),
                                                           d_num_polys_per_batch.end()));

      if (ring_sizes_buffer_base_ptr) {
        ::thrust::for_each(
            thrust_context.getDevicePolicy(),
            start_batch_itr,
            end_batch_itr,
            detail_polys::GetBatchInfoFunctor<false>(
                ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                query_output_buffer_stride,
                detail_polys::PolyBufferAccessor<int32_t, false>(
                    ring_sizes_buffer_base_ptr, ring_sizes_ptr_buffer_col_idx),
                row_count,
                batch_size,
                ::thrust::raw_pointer_cast(&state_->d_sorted_row_index[0])));
      } else {
        ::thrust::for_each(
            thrust_context.getDevicePolicy(),
            start_batch_itr,
            end_batch_itr,
            detail_polys::GetBatchInfoFunctor<true>(
                ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                query_output_buffer_stride,
                detail_polys::PolyBufferAccessor<int32_t, true>(
                    ring_sizes_ptr_buffer_col_idx),
                row_count,
                batch_size,
                ::thrust::raw_pointer_cast(&state_->d_sorted_row_index[0])));
      }
    };

    detail_polys::run_thrust_stage_checked(
        "compute batch info", compute_batch_info, nullptr, row_count);

    //
    // COPY HOST RESULTS BACK
    //

    auto resize_results = [&] {
      num_rows_per_batch.resize(num_batches);
      num_polys_per_batch.resize(num_batches);
    };

    detail_polys::run_thrust_stage_checked(
        "resize results", resize_results, nullptr, row_count);

    auto copy_back_results = [&] {
      ::thrust::copy_n(
          d_num_rows_per_batch.begin(), num_batches, num_rows_per_batch.begin());
      ::thrust::copy_n(
          d_num_polys_per_batch.begin(), num_batches, num_polys_per_batch.begin());
    };

    detail_polys::run_thrust_stage_checked(
        "copy back results", copy_back_results, nullptr, row_count);

    //
    // FREE UP DEVICE MEMORY
    //

    auto free_device_memory = [&] { state_->clearAll(); };

    detail_polys::run_thrust_stage_checked(
        "free device memory", free_device_memory, nullptr, row_count);
  }

 private:
  struct InternalThrustState {
    __host__ InternalThrustState(DataMgrThrustContext& thrust_context)
        : d_sorted_row_index(make_device_vector_from_context<uint32_t>(thrust_context))
        , d_rowids(make_device_vector_from_context<uint32_t>(thrust_context))
        , d_num_verts_per_row(make_device_vector_from_context<uint32_t>(thrust_context))
        , d_num_polys_per_row(make_device_vector_from_context<uint32_t>(thrust_context))
        , d_first_vert_per_row(make_device_vector_from_context<uint32_t>(thrust_context))
        , d_first_poly_per_row(
              make_device_vector_from_context<uint32_t>(thrust_context)) {}

    void resizeAll(const size_t new_size) {
      d_sorted_row_index.resize(new_size);
      d_rowids.resize(new_size);
      d_num_verts_per_row.resize(new_size);
      d_num_polys_per_row.resize(new_size);
      d_first_vert_per_row.resize(new_size);
      d_first_poly_per_row.resize(new_size);
    }

    void clearAll() {
      d_sorted_row_index.clear();
      d_rowids.clear();
      d_num_verts_per_row.clear();
      d_num_polys_per_row.clear();
      d_first_vert_per_row.clear();
      d_first_poly_per_row.clear();
    }

    ThrustAllocatorDeviceVector<uint32_t> d_sorted_row_index;
    ThrustAllocatorDeviceVector<uint32_t> d_rowids;
    ThrustAllocatorDeviceVector<uint32_t> d_num_verts_per_row;
    ThrustAllocatorDeviceVector<uint32_t> d_num_polys_per_row;
    ThrustAllocatorDeviceVector<uint32_t> d_first_vert_per_row;
    ThrustAllocatorDeviceVector<uint32_t> d_first_poly_per_row;
  };

  std::unique_ptr<InternalThrustState> state_;
};

}  // namespace QueryRenderer
