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

class LineDataConverterInterface {
 public:
  LineDataConverterInterface() = default;
  virtual ~LineDataConverterInterface() = default;

  //
  // Stage 1
  //
  // Read the number of Verts and Lines for each row
  // from the query buffer
  // Prefix-sum those array and get the totals
  // Return the totals so that buffers can be created
  //

  virtual void ConvertStage1(DataMgrThrustContext& thrust_context,
                             uint32_t query_row_count,
                             int8_t* query_output_buffer_base_ptr,
                             uint32_t query_output_buffer_stride,
                             uint32_t coords_ptr_buffer_col_idx,
                             uint32_t linestring_sizes_ptr_buffer_col_idx,
                             const EncodingType coords_encoding,
                             uint32_t& row_count,
                             uint32_t& vert_count,
                             uint32_t& line_count) = 0;

  //
  // Stage 2
  //
  // This completes the process by populating the given QueryBuffers
  // and returns data structures required for the draw process
  //

  virtual void ConvertStage2(
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
      const LineBufferMemoryDescriptors& buffer_memory_descriptors) = 0;
};

}  // namespace QueryRenderer
