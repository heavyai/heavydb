/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ExecuteRenderInterface/RenderQueryUtils/thrust/ThrustLinesInterface.h"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/system_error.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "DataMgr/Allocators/thrust/TypedThrustAllocator.h"
#include "GfxDriver/RenderError.h"
#include "Logger/Logger.h"
#include "QueryEngine/GpuRtConstants.h"
#include "QueryRenderer/Utils/thrust/ThrustAllocatorDeviceVector.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContextInternal.h"

namespace QueryRenderer {

using Data_Namespace::ThrustAllocatorDeviceVector;
using Data_Namespace::TypedThrustAllocator;

//
// THRUST FUNCTORS
//

namespace detail_lines {

struct GetValidRowIndexFunctor {
  int8_t* _query_output_buffer_base_ptr;
  uint32_t _query_output_buffer_stride;
  uint32_t _coords_num_idx;
  uint32_t _coords_bytes_to_verts_shift;

  GetValidRowIndexFunctor(int8_t* query_output_buffer_base_ptr,
                          uint32_t query_output_buffer_stride,
                          uint32_t coords_ptr_idx,
                          uint32_t coords_bytes_to_verts_shift)
      : _query_output_buffer_base_ptr(query_output_buffer_base_ptr)
      , _query_output_buffer_stride(query_output_buffer_stride)
      , _coords_num_idx(coords_ptr_idx + 1)
      , _coords_bytes_to_verts_shift(coords_bytes_to_verts_shift) {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t row_index = ::thrust::get<0>(t);

    // row base
    int8_t* p_row_bytes =
        _query_output_buffer_base_ptr + (row_index * _query_output_buffer_stride);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // key (first value)
    int64_t key = *p_row_values;

    // coords num bytes
    // buffer 64-bit value can be cast down to uint32_t
    uint32_t num_coords_bytes = (uint32_t)p_row_values[_coords_num_idx];
    uint32_t num_verts = num_coords_bytes >> _coords_bytes_to_verts_shift;

    // for NULL geo rows, where coords has just two
    // values which are the null sentinels, num_verts
    // will come out as 1, so the code below will already
    // tag the row as invalid due to insufficient points
    //
    // we should really check that the X and Y values are
    // the expected sentinels, but the result will be the
    // same so why bother

    // valid index
    uint32_t valid_row_index;
    if (key == EMPTY_KEY_64 || num_verts < 2) {
      // invalid key or insufficient verts, sort to the end
      valid_row_index = UINT32_MAX;
    } else {
      // valid row, use this index
      valid_row_index = row_index;
    }

    // outputs
    ::thrust::get<1>(t) = valid_row_index;
  }
};

struct GetNumVertsAndLinesFunctor {
  int8_t* query_output_buffer_base_ptr_;
  uint32_t query_output_buffer_stride_;
  uint32_t coords_num_idx_;
  uint32_t linestring_sizes_num_idx_;
  uint32_t coords_bytes_to_verts_shift_;

  GetNumVertsAndLinesFunctor(int8_t* query_output_buffer_base_ptr,
                             uint32_t query_output_buffer_stride,
                             uint32_t coords_ptr_idx,
                             uint32_t linestring_sizes_ptr_idx,
                             uint32_t coords_bytes_to_verts_shift)
      : query_output_buffer_base_ptr_{query_output_buffer_base_ptr}
      , query_output_buffer_stride_{query_output_buffer_stride}
      , coords_num_idx_{coords_ptr_idx + 1}
      , linestring_sizes_num_idx_{linestring_sizes_ptr_idx + 1}
      , coords_bytes_to_verts_shift_{coords_bytes_to_verts_shift} {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t valid_row_index = ::thrust::get<0>(t);

    // compute total number of verts for this row
    uint32_t vert_count = 0;

    // row base
    int8_t* p_row_bytes =
        query_output_buffer_base_ptr_ + (valid_row_index * query_output_buffer_stride_);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // coords num bytes
    // buffer 64-bit value can be cast down to uint32_t
    uint32_t num_coords_bytes = static_cast<uint32_t>(p_row_values[coords_num_idx_]);
    uint32_t num_verts = num_coords_bytes >> coords_bytes_to_verts_shift_;

    // verts per line
    vert_count += num_verts;

    // number of linestring sizes
    // if num_idx = 1 (ptr_idx = 0) there are no linestring_sizes, so assume size of 1
    uint32_t num_linestring_sizes;
    if (linestring_sizes_num_idx_ == 1) {
      num_linestring_sizes = 1;
    } else {
      num_linestring_sizes =
          static_cast<uint32_t>(p_row_values[linestring_sizes_num_idx_]);
    }

    // repeats (one at the end of each line)
    vert_count += num_linestring_sizes * 2;

    // separators (one between each line, if more than one)
    vert_count += num_linestring_sizes - 1;

    // outputs
    ::thrust::get<1>(t) = vert_count;
    ::thrust::get<2>(t) = num_linestring_sizes;
  }
};

template <typename CoordT>
struct CopyVertexDataFunctor {
  int8_t* query_output_buffer_base_;
  uint32_t query_output_buffer_stride_;
  uint32_t coords_ptr_idx_;
  uint32_t coords_num_idx_;
  uint32_t linestring_sizes_ptr_idx_;
  uint32_t linestring_sizes_num_idx_;
  CoordT* pd_vertex_data_;
  uint32_t coords_bytes_to_verts_shift_;
  CoordT separator_value_;

  CopyVertexDataFunctor(int8_t* query_output_buffer_base,
                        uint32_t query_output_buffer_stride,
                        uint32_t coords_ptr_idx,
                        uint32_t linestring_sizes_ptr_idx,
                        CoordT* pd_vertex_data,
                        uint32_t coords_bytes_to_verts_shift,
                        CoordT separator_value)
      : query_output_buffer_base_{query_output_buffer_base}
      , query_output_buffer_stride_{query_output_buffer_stride}
      , coords_ptr_idx_{coords_ptr_idx}
      , coords_num_idx_{coords_ptr_idx + 1}
      , linestring_sizes_ptr_idx_{linestring_sizes_ptr_idx}
      , linestring_sizes_num_idx_{linestring_sizes_ptr_idx + 1}
      , pd_vertex_data_{pd_vertex_data}
      , coords_bytes_to_verts_shift_{coords_bytes_to_verts_shift}
      , separator_value_{separator_value} {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t valid_row_index = ::thrust::get<0>(t);
    uint32_t this_first_vert_per_row = ::thrust::get<1>(t);

    // row base
    int8_t* p_row_bytes =
        query_output_buffer_base_ + (valid_row_index * query_output_buffer_stride_);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // coords ptr
    // buffer 64-bit value is the ptr
    // cast that ptr to the coord data type
    CoordT* p_coords = reinterpret_cast<CoordT*>(p_row_values[coords_ptr_idx_]);

    // coords num
    uint32_t num_coords_bytes = static_cast<uint32_t>(p_row_values[coords_num_idx_]);
    uint32_t num_verts = num_coords_bytes >> coords_bytes_to_verts_shift_;

    // number of linestring sizes
    // if num_idx = 1 (ptr_idx = 0) there are no linestring_sizes, so assume size of 1
    uint32_t num_linestring_sizes;
    uint32_t* p_linestring_sizes;
    if (linestring_sizes_num_idx_ == 1) {
      num_linestring_sizes = 1;
      p_linestring_sizes = nullptr;
    } else {
      num_linestring_sizes =
          static_cast<uint32_t>(p_row_values[linestring_sizes_num_idx_]);
      p_linestring_sizes =
          reinterpret_cast<uint32_t*>(p_row_values[linestring_sizes_ptr_idx_]);
    }

    // copy coords to verts
    uint32_t vertex_data_idx = this_first_vert_per_row * 2;
    uint32_t coords_line_idx = 0;

    // copy rings
    uint32_t last_line = num_linestring_sizes - 1;
    for (uint32_t i = 0; i < num_linestring_sizes; i++) {
      // line size?
      uint32_t linestring_size;
      if (p_linestring_sizes) {
        // this line's actual size
        linestring_size = p_linestring_sizes[i];
      } else {
        // only one line, size = num_verts
        linestring_size = num_verts;
      }

      // repeat first point
      pd_vertex_data_[vertex_data_idx++] = p_coords[coords_line_idx];
      pd_vertex_data_[vertex_data_idx++] = p_coords[coords_line_idx + 1];

      // copy points
      for (uint32_t j = 0; j < linestring_size; j++) {
        pd_vertex_data_[vertex_data_idx++] = p_coords[coords_line_idx++];
        pd_vertex_data_[vertex_data_idx++] = p_coords[coords_line_idx++];
      }

      // repeat last point
      pd_vertex_data_[vertex_data_idx++] = p_coords[coords_line_idx - 2];
      pd_vertex_data_[vertex_data_idx++] = p_coords[coords_line_idx - 1];

      // separator (between lines only)
      if (i < last_line) {
        pd_vertex_data_[vertex_data_idx++] = separator_value_;
        pd_vertex_data_[vertex_data_idx++] = separator_value_;
      }
    }
  }
};

struct BuildDrawDataFunctor {
  uint32_t* _pd_line_draw_data;

  BuildDrawDataFunctor(uint32_t* pd_line_draw_data)
      : _pd_line_draw_data(pd_line_draw_data) {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // inputs
    uint32_t output_row_index = ::thrust::get<0>(t);
    uint32_t this_first_vert_per_row = ::thrust::get<1>(t);
    uint32_t this_num_verts_per_row = ::thrust::get<2>(t);

    // indirect draw struct start indices
    uint32_t line_index = output_row_index * 4;

    // typedef  struct {
    //    uint  vertex_count;
    //    uint  instance_count;
    //    uint  first_vertex;
    //    uint  base_instance;
    // } IndirectDrawVertexData;

    if (this_num_verts_per_row > 0) {
      // line draw data
      _pd_line_draw_data[line_index++] = this_num_verts_per_row;
      _pd_line_draw_data[line_index++] = 1;
      _pd_line_draw_data[line_index++] = this_first_vert_per_row;
      _pd_line_draw_data[line_index++] = 0;
    } else {
      // empty line draw data
      _pd_line_draw_data[line_index++] = 0;
      _pd_line_draw_data[line_index++] = 0;
      _pd_line_draw_data[line_index++] = 0;
      _pd_line_draw_data[line_index++] = 0;
    }
  }
};

struct BuildPerRowDataFunctor {
  int8_t* _query_output_buffer_base_ptr;
  uint32_t _query_output_buffer_stride;
  int8_t* _pd_per_row_data;
  uint32_t _per_row_buffer_stride;
  uint32_t _per_row_buffer_num_cols;
  uint32_t* _pd_per_row_buffer_col_idxs;
  uint32_t* _pd_per_row_buffer_col_offsets;
  uint32_t* _pd_per_row_buffer_col_types;

  BuildPerRowDataFunctor(int8_t* query_output_buffer_base_ptr,
                         uint32_t query_output_buffer_stride,
                         int8_t* pd_per_row_data,
                         uint32_t per_row_buffer_stride,
                         uint32_t per_row_buffer_num_cols,
                         uint32_t* pd_per_row_buffer_col_idxs,
                         uint32_t* pd_per_row_buffer_col_offsets,
                         uint32_t* pd_per_row_buffer_col_types)
      : _query_output_buffer_base_ptr(query_output_buffer_base_ptr)
      , _query_output_buffer_stride(query_output_buffer_stride)
      , _pd_per_row_data(pd_per_row_data)
      , _per_row_buffer_stride(per_row_buffer_stride)
      , _per_row_buffer_num_cols(per_row_buffer_num_cols)
      , _pd_per_row_buffer_col_idxs(pd_per_row_buffer_col_idxs)
      , _pd_per_row_buffer_col_offsets(pd_per_row_buffer_col_offsets)
      , _pd_per_row_buffer_col_types(pd_per_row_buffer_col_types) {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    uint32_t valid_row_index = ::thrust::get<0>(t);
    uint32_t output_row_index = ::thrust::get<1>(t);

    // row base
    int8_t* p_row_bytes =
        _query_output_buffer_base_ptr + (valid_row_index * _query_output_buffer_stride);
    int64_t* p_row_values = reinterpret_cast<int64_t*>(p_row_bytes);

    // output base
    int8_t* p_output_bytes =
        _pd_per_row_data + (output_row_index * _per_row_buffer_stride);

    // outputs
    // @TODO simon.eves
    // is there a more efficient implementation?
    // also need to implement the rowid/projection-query logic... how?
    for (uint32_t i = 0; i < _per_row_buffer_num_cols; i++) {
      switch (_pd_per_row_buffer_col_types[i]) {
        case (uint32_t)::gfx::BufferAttrType::kInt: {
          // cast to int32
          int64_t* p_row_value_int64 = p_row_values + _pd_per_row_buffer_col_idxs[i];
          int32_t* p_output_int32 = reinterpret_cast<int32_t*>(
              p_output_bytes + _pd_per_row_buffer_col_offsets[i]);
          *p_output_int32 = (int32_t)(*p_row_value_int64);
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kUint: {
          // cast to uint32
          uint64_t* p_row_value_uint64 =
              reinterpret_cast<uint64_t*>(p_row_values + _pd_per_row_buffer_col_idxs[i]);
          uint32_t* p_output_uint32 = reinterpret_cast<uint32_t*>(
              p_output_bytes + _pd_per_row_buffer_col_offsets[i]);
          *p_output_uint32 = (uint32_t)(*p_row_value_uint64);
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kInt64: {
          // copy int64
          int64_t* p_row_value_int64 = p_row_values + _pd_per_row_buffer_col_idxs[i];
          int64_t* p_output_int64 = reinterpret_cast<int64_t*>(
              p_output_bytes + _pd_per_row_buffer_col_offsets[i]);
          *p_output_int64 = *p_row_value_int64;
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kUint64: {
          // copy uint64
          uint64_t* p_row_value_uint64 =
              reinterpret_cast<uint64_t*>(p_row_values + _pd_per_row_buffer_col_idxs[i]);
          uint64_t* p_output_uint64 = reinterpret_cast<uint64_t*>(
              p_output_bytes + _pd_per_row_buffer_col_offsets[i]);
          *p_output_uint64 = *p_row_value_uint64;
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kFloat: {
          // cast double to float
          double* p_row_value_double =
              reinterpret_cast<double*>(p_row_values + _pd_per_row_buffer_col_idxs[i]);
          float* p_output_float = reinterpret_cast<float*>(
              p_output_bytes + _pd_per_row_buffer_col_offsets[i]);
          *p_output_float = (float)(*p_row_value_double);
          break;
        }
        case (uint32_t)::gfx::BufferAttrType::kDouble: {
          // copy double
          double* p_row_value_double =
              reinterpret_cast<double*>(p_row_values + _pd_per_row_buffer_col_idxs[i]);
          double* p_output_double = reinterpret_cast<double*>(
              p_output_bytes + _pd_per_row_buffer_col_offsets[i]);
          *p_output_double = *p_row_value_double;
          break;
        }
        default:
          break;
      }
    }
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
    ss << "In-Situ Lines Thrust Out-of-Memory Error in " << stage_name << ": " << e.what()
       << " (row count = " << row_count << ")";
    throw gfx::OutOfGpuMemoryError(ss.str());
  } catch (::thrust::system_error& e) {
    if (fail_lambda) {
      fail_lambda();
    }
    std::stringstream ss;
    ss << "In-Situ Lines Thrust System Error in " << stage_name << ": " << e.what()
       << " (row count = " << row_count << ")";
    throw std::runtime_error(ss.str());
  }
}

}  // namespace detail_lines

//
// MAIN CLASS
//

template <ThrustDeviceSystem device_system>
class LineDataConverterImpl final : public LineDataConverterInterface {
 public:
  LineDataConverterImpl() : LineDataConverterInterface() {}
  ~LineDataConverterImpl() override = default;

  void ConvertStage1(DataMgrThrustContext& thrust_context,
                     uint32_t query_row_count,
                     int8_t* query_output_buffer_base_ptr,
                     uint32_t query_output_buffer_stride,
                     uint32_t coords_ptr_buffer_col_idx,
                     uint32_t linestring_sizes_ptr_buffer_col_idx,
                     const EncodingType coords_encoding,
                     uint32_t& row_count,
                     uint32_t& vert_count,
                     uint32_t& line_count) override {
    if (!state_) {
      state_ = std::make_unique<InternalThrustState>(thrust_context);
    }

    // how many bits to shift to convert from num bytes to num verts
    // compressed is four bytes per value, eight bytes per vert, shift 3
    // uncompressed is eight bytes per value, sixteen bytes per vert, shift 4
    uint32_t coords_bytes_to_verts_shift = (coords_encoding == kENCODING_GEOINT) ? 3 : 4;

    auto clean_up_on_error = [&] {
      row_count = 0;
      vert_count = 0;
      line_count = 0;
    };

    clean_up_on_error();

    //
    // SIZE THIS
    //

    auto resize_valid_rows = [&] { state_->d_valid_row_indices.resize(query_row_count); };

    detail_lines::run_thrust_stage_checked(
        "resize valid rows", resize_valid_rows, clean_up_on_error, query_row_count);

    //
    // GET COMPACTED VALID ROW INDICES
    //

    auto get_compacted_valid_row_indices = [&] {
      // pull array of either valid or invalid indices
      ::thrust::counting_iterator<uint32_t> row_index_it(0);
      ::thrust::for_each(
          thrust_context.getDevicePolicy(),
          ::thrust::make_zip_iterator(
              ::thrust::make_tuple(row_index_it, state_->d_valid_row_indices.begin())),
          ::thrust::make_zip_iterator(::thrust::make_tuple(
              row_index_it + query_row_count, state_->d_valid_row_indices.end())),
          detail_lines::GetValidRowIndexFunctor(
              ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
              query_output_buffer_stride,
              coords_ptr_buffer_col_idx,
              coords_bytes_to_verts_shift));
    };

    detail_lines::run_thrust_stage_checked("get compacted valid row indices",
                                           get_compacted_valid_row_indices,
                                           clean_up_on_error,
                                           query_row_count);

    auto sort_row_indices = [&] {
      // sort (invalid sorted to the end)
      ::thrust::sort(thrust_context.getDevicePolicy(),
                     state_->d_valid_row_indices.begin(),
                     state_->d_valid_row_indices.end());
    };

    detail_lines::run_thrust_stage_checked(
        "sort row indices", sort_row_indices, clean_up_on_error, query_row_count);

    auto find_row_count = [&] {
      // valid count is index of first invalid
      auto firstInvalidRowIt = ::thrust::lower_bound(thrust_context.getDevicePolicy(),
                                                     state_->d_valid_row_indices.begin(),
                                                     state_->d_valid_row_indices.end(),
                                                     UINT32_MAX);
      row_count = firstInvalidRowIt - state_->d_valid_row_indices.begin();
    };

    detail_lines::run_thrust_stage_checked(
        "find row count", find_row_count, clean_up_on_error, query_row_count);

    //
    // ANY NON-EMPTY ROWS?
    //

    if (row_count == 0) {
      vert_count = 0;
      line_count = 0;
      // not an error
      return;
    }

    //
    // RESIZE THE VALID INDICES ARRAY (discard all beyond row_count) AND SIZE THE REST
    //

    auto resize_other_arrays = [&] { state_->resizeAll(row_count); };

    detail_lines::run_thrust_stage_checked(
        "resize other arrays", resize_other_arrays, clean_up_on_error, row_count);

    //
    // GET NUM VERTS AND LINES PER ROW
    //

    auto get_num_verts_and_lines_per_row = [&] {
      // run the functor
      ::thrust::for_each(thrust_context.getDevicePolicy(),
                         ::thrust::make_zip_iterator(::thrust::make_tuple(
                             state_->d_valid_row_indices.begin(),
                             state_->d_num_verts_per_valid_row.begin(),
                             state_->d_num_lines_per_valid_row.begin())),
                         ::thrust::make_zip_iterator(::thrust::make_tuple(
                             state_->d_valid_row_indices.end(),
                             state_->d_num_verts_per_valid_row.end(),
                             state_->d_num_lines_per_valid_row.end())),
                         detail_lines::GetNumVertsAndLinesFunctor(
                             ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                             query_output_buffer_stride,
                             coords_ptr_buffer_col_idx,
                             linestring_sizes_ptr_buffer_col_idx,
                             coords_bytes_to_verts_shift));
    };

    detail_lines::run_thrust_stage_checked("get num verts and lines per row",
                                           get_num_verts_and_lines_per_row,
                                           clean_up_on_error,
                                           row_count);

    //
    // COMPUTE FIRST VERTS AND LINES PER ROW AND TOTAL VERTS AND LINES
    //

    // the firsts will be in sorted order, which is what we want
    // we need an exclusive-scan in each case but this does not compute
    // the final total, so we have to add back on the num of the last row

    auto get_vert_count = [&] {
      ::thrust::exclusive_scan(thrust_context.getDevicePolicy(),
                               state_->d_num_verts_per_valid_row.begin(),
                               state_->d_num_verts_per_valid_row.end(),
                               state_->d_first_vert_per_valid_row.begin());

      vert_count = state_->d_first_vert_per_valid_row[row_count - 1] +
                   state_->d_num_verts_per_valid_row[row_count - 1];
    };

    detail_lines::run_thrust_stage_checked(
        "get vert count", get_vert_count, clean_up_on_error, row_count);

    auto get_line_count = [&] {
      ::thrust::exclusive_scan(thrust_context.getDevicePolicy(),
                               state_->d_num_lines_per_valid_row.begin(),
                               state_->d_num_lines_per_valid_row.end(),
                               state_->d_first_line_per_valid_row.begin());

      line_count = state_->d_first_line_per_valid_row[row_count - 1] +
                   state_->d_num_lines_per_valid_row[row_count - 1];
    };

    detail_lines::run_thrust_stage_checked(
        "get line count", get_line_count, clean_up_on_error, row_count);
  }

  void ConvertStage2(
      DataMgrThrustContext& thrust_context,
      int8_t* query_output_buffer_base_ptr,
      uint32_t query_output_buffer_stride,
      uint32_t coords_ptr_buffer_col_idx,
      uint32_t linestring_sizes_ptr_buffer_col_idx,
      const EncodingType coords_encoding,
      const std::vector<uint32_t> per_row_data_buffer_col_idxs,
      const std::vector<uint32_t> per_row_data_buffer_col_offsets,
      const std::vector<::gfx::BufferAttrType> per_row_data_buffer_col_types,
      uint32_t per_row_data_buffer_stride,
      const ::QueryRenderer::LineBufferMemoryDescriptors& buffer_memory_descriptors)
      final {
    CHECK(state_ != nullptr);
    //
    // GET RAW BUFFER POINTERS FROM CUDA HANDLES
    //

    auto pd_line_draw_data = ::thrust::device_pointer_cast(reinterpret_cast<uint32_t*>(
        buffer_memory_descriptors.line_indirect_vbo_descriptor.handle));
    auto pd_per_row_data = ::thrust::device_pointer_cast(
        reinterpret_cast<int8_t*>(buffer_memory_descriptors.ssbo_descriptor.handle));

    //
    // COPY VERTEX DATA
    //

    // how many bits to shift to convert from num bytes to num verts
    // compressed is four bytes per value, eight bytes per vert, shift 3
    // uncompressed is eight bytes per value, sixteen bytes per vert, shift 4
    uint32_t coords_bytes_to_verts_shift = (coords_encoding == kENCODING_GEOINT) ? 3 : 4;

    // we can now allocate the final renderable vertex buffer
    // now that we know the total num verts
    // then run a functor to copy in the vertex and bounds data

    uint32_t row_count = state_->d_valid_row_indices.size();

    auto copy_vertex_data = [&] {
      if (coords_encoding == kENCODING_GEOINT) {
        auto pd_vertex_data = ::thrust::device_pointer_cast(
            reinterpret_cast<int32_t*>(buffer_memory_descriptors.vbo_descriptor.handle));
        auto const separator_value = -std::numeric_limits<int32_t>::max();
        ::thrust::for_each(thrust_context.getDevicePolicy(),
                           ::thrust::make_zip_iterator(::thrust::make_tuple(
                               state_->d_valid_row_indices.begin(),
                               state_->d_first_vert_per_valid_row.begin(),
                               state_->d_num_verts_per_valid_row.begin())),
                           ::thrust::make_zip_iterator(::thrust::make_tuple(
                               state_->d_valid_row_indices.end(),
                               state_->d_first_vert_per_valid_row.end(),
                               state_->d_num_verts_per_valid_row.end())),
                           detail_lines::CopyVertexDataFunctor<int32_t>(
                               ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                               query_output_buffer_stride,
                               coords_ptr_buffer_col_idx,
                               linestring_sizes_ptr_buffer_col_idx,
                               ::thrust::raw_pointer_cast(pd_vertex_data),
                               coords_bytes_to_verts_shift,
                               separator_value));
      } else {
        auto pd_vertex_data = ::thrust::device_pointer_cast(
            reinterpret_cast<double*>(buffer_memory_descriptors.vbo_descriptor.handle));
        auto const separator_value = -std::numeric_limits<double>::max();
        ::thrust::for_each(thrust_context.getDevicePolicy(),
                           ::thrust::make_zip_iterator(::thrust::make_tuple(
                               state_->d_valid_row_indices.begin(),
                               state_->d_first_vert_per_valid_row.begin(),
                               state_->d_num_verts_per_valid_row.begin())),
                           ::thrust::make_zip_iterator(::thrust::make_tuple(
                               state_->d_valid_row_indices.end(),
                               state_->d_first_vert_per_valid_row.end(),
                               state_->d_num_verts_per_valid_row.end())),
                           detail_lines::CopyVertexDataFunctor<double>(
                               ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
                               query_output_buffer_stride,
                               coords_ptr_buffer_col_idx,
                               linestring_sizes_ptr_buffer_col_idx,
                               ::thrust::raw_pointer_cast(pd_vertex_data),
                               coords_bytes_to_verts_shift,
                               separator_value));
      }
    };

    detail_lines::run_thrust_stage_checked(
        "copy and transform vertex data", copy_vertex_data, nullptr, row_count);

    //
    // BUILD INDIRECT DRAW DATA
    //

    auto build_indirect_draw_data = [&] {
      ::thrust::counting_iterator<uint32_t> output_row_index_it1(0);
      ::thrust::for_each(thrust_context.getDevicePolicy(),
                         ::thrust::make_zip_iterator(::thrust::make_tuple(
                             output_row_index_it1,
                             state_->d_first_vert_per_valid_row.begin(),
                             state_->d_num_verts_per_valid_row.begin())),
                         ::thrust::make_zip_iterator(::thrust::make_tuple(
                             output_row_index_it1 + row_count,
                             state_->d_first_vert_per_valid_row.end(),
                             state_->d_num_verts_per_valid_row.end())),
                         detail_lines::BuildDrawDataFunctor(
                             ::thrust::raw_pointer_cast(pd_line_draw_data)));
    };

    detail_lines::run_thrust_stage_checked(
        "build indirect draw data", build_indirect_draw_data, nullptr, row_count);

    //
    // BUILD PER-ROW DATA
    //

    auto d_per_row_data_buffer_col_idxs =
        QueryRenderer::make_device_vector_from_context<uint32_t>(thrust_context);
    auto d_per_row_data_buffer_col_offsets =
        QueryRenderer::make_device_vector_from_context<uint32_t>(thrust_context);
    auto d_per_row_data_buffer_col_types =
        QueryRenderer::make_device_vector_from_context<uint32_t>(thrust_context);

    uint32_t per_row_data_buffer_num_cols = per_row_data_buffer_col_idxs.size();

    auto copy_per_row_data = [&] {
      d_per_row_data_buffer_col_idxs = per_row_data_buffer_col_idxs;
      d_per_row_data_buffer_col_offsets = per_row_data_buffer_col_offsets;
      d_per_row_data_buffer_col_types.resize(per_row_data_buffer_num_cols);
      for (uint32_t i = 0; i < per_row_data_buffer_num_cols; i++) {
        d_per_row_data_buffer_col_types[i] = (uint32_t)per_row_data_buffer_col_types[i];
      }
    };

    detail_lines::run_thrust_stage_checked(
        "copy per-row data", copy_per_row_data, nullptr, row_count);

    auto build_per_row_data = [&] {
      ::thrust::counting_iterator<uint32_t> output_row_index_it2(0);
      ::thrust::for_each(
          thrust_context.getDevicePolicy(),
          ::thrust::make_zip_iterator(::thrust::make_tuple(
              state_->d_valid_row_indices.begin(), output_row_index_it2)),
          ::thrust::make_zip_iterator(::thrust::make_tuple(
              state_->d_valid_row_indices.end(), output_row_index_it2 + row_count)),
          detail_lines::BuildPerRowDataFunctor(
              ::thrust::raw_pointer_cast(query_output_buffer_base_ptr),
              query_output_buffer_stride,
              ::thrust::raw_pointer_cast(pd_per_row_data),
              per_row_data_buffer_stride,
              per_row_data_buffer_num_cols,
              ::thrust::raw_pointer_cast(&d_per_row_data_buffer_col_idxs[0]),
              ::thrust::raw_pointer_cast(&d_per_row_data_buffer_col_offsets[0]),
              ::thrust::raw_pointer_cast(&d_per_row_data_buffer_col_types[0])));
    };

    detail_lines::run_thrust_stage_checked(
        "build per-row data", build_per_row_data, nullptr, row_count);

    //
    // FREE UP DEVICE MEMORY
    //

    auto free_device_memory = [&] { state_->clearAll(); };

    detail_lines::run_thrust_stage_checked(
        "free device memory", free_device_memory, nullptr, row_count);
  }

 private:
  struct InternalThrustState {
    InternalThrustState(DataMgrThrustContext& thrust_context)
        : d_valid_row_indices(make_device_vector_from_context<uint32_t>(thrust_context))
        , d_num_verts_per_valid_row(
              make_device_vector_from_context<uint32_t>(thrust_context))
        , d_num_lines_per_valid_row(
              make_device_vector_from_context<uint32_t>(thrust_context))
        , d_first_vert_per_valid_row(
              make_device_vector_from_context<uint32_t>(thrust_context))
        , d_first_line_per_valid_row(
              make_device_vector_from_context<uint32_t>(thrust_context)) {}

    void resizeAll(const size_t new_size) {
      d_valid_row_indices.resize(new_size);
      d_num_verts_per_valid_row.resize(new_size);
      d_num_lines_per_valid_row.resize(new_size);
      d_first_vert_per_valid_row.resize(new_size);
      d_first_line_per_valid_row.resize(new_size);
    }

    void clearAll() {
      d_valid_row_indices.clear();
      d_num_verts_per_valid_row.clear();
      d_num_lines_per_valid_row.clear();
      d_first_vert_per_valid_row.clear();
      d_first_line_per_valid_row.clear();
    }

    ThrustAllocatorDeviceVector<uint32_t> d_valid_row_indices;
    ThrustAllocatorDeviceVector<uint32_t> d_num_verts_per_valid_row;
    ThrustAllocatorDeviceVector<uint32_t> d_num_lines_per_valid_row;
    ThrustAllocatorDeviceVector<uint32_t> d_first_vert_per_valid_row;
    ThrustAllocatorDeviceVector<uint32_t> d_first_line_per_valid_row;
  };

  std::unique_ptr<InternalThrustState> state_;
};

}  // namespace QueryRenderer
