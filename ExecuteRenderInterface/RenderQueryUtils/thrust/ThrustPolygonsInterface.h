/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/Enums.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContext.h"
#include "Shared/sqltypes.h"

namespace QueryRenderer {

class PolygonDataConverterInterface {
 public:
  PolygonDataConverterInterface() = default;
  virtual ~PolygonDataConverterInterface() = default;

  //
  // Stage 1
  //
  // Create an index array (monotonically increasing)
  // and sort it by row_id
  // Read the number of Verts and Polys for each row
  // from the query buffer (in sorted index order)
  // Prefix-sum those array and get the totals
  // Return the totals so that buffers can be created
  //

  virtual void ConvertStage1(DataMgrThrustContext& thrust_context,
                             uint32_t query_row_count,
                             int8_t* query_output_buffer_base_ptr,
                             uint32_t query_output_buffer_stride,
                             int32_t ring_sizes_ptr_buffer_col_idx,
                             int32_t rowid_buffer_col_idx,
                             uint32_t& row_count,
                             uint32_t& vert_count,
                             uint32_t& poly_count,
                             int8_t* rings_sizes_buffer_base_ptr = nullptr) = 0;

  //
  // Stage 2
  //
  // This completes the process by populating the given QueryBuffers
  // and returns data structures required for the draw process
  //

  virtual void ConvertStage2(
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
      const PolyBufferMemoryDescriptors& buffer_memory_descriptors,
      std::vector<uint32_t>& num_rows_per_batch,
      std::vector<uint32_t>& num_polys_per_batch,
      int8_t* coords_buffer_base_ptr = nullptr,
      int8_t* rings_buffer_base_ptr = nullptr) = 0;
};

}  // namespace QueryRenderer
