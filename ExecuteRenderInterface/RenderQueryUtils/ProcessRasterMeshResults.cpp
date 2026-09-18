/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/ProcessRasterMeshResults.h"

#include "CudaMgr/CudaMgr.h"
#include "ExecuteRenderInterface/RenderQueryUtils/Metadata/RasterMeshMetadata.h"
#include "ExecuteRenderInterface/RenderQueryUtils/ProcessResultsUtils.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryRenderer/Cache/RasterMeshMgr.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/QueryRenderManager.h"

namespace QueryRenderer {

// raster mesh (expects uniform grid of points)
uint64_t process_raster_mesh_non_in_situ(QueryRenderManager& render_manager,
                                         const ExecutionResult& results,
                                         const RasterMeshMetadata& raster_mesh_metadata,
                                         const JSONLocation& data_loc,
                                         RenderInfo& render_info) {
  RENDER_LOG_SCOPE();
  const int gpu_idx = render_manager.getLeastSubscribedGpuId();
  VLOG(1) << "Selected gpu " << gpu_idx << " for non-insitu raster mesh render";

  const auto name_loc = data_loc.getMember(JSONSchema_v1::Data::kNameProp);
  CHECK(name_loc.isString());

  auto const& rows = results.getRows();
  auto const entry_count = rows->entryCount();
  auto const row_count = rows->rowCount(entry_count > kMinRowCountWorthMultiThreading);
  const bool force_singlethreaded = entry_count > row_count;

  auto const raster_width = raster_mesh_metadata.width;
  auto const raster_height = raster_mesh_metadata.height;

  if (row_count > 0u) {
    CHECK_EQ(raster_width * raster_height, row_count)
        << raster_width << "x" << raster_height << " != " << row_count;
  }

  const auto& result_targets = results.getTargetsMeta();
  const auto rowid_status = get_rowid_status(result_targets, render_info);

  std::vector<unsigned int> target_column_indices(result_targets.size());
  std::iota(target_column_indices.begin(), target_column_indices.end(), 0);
  auto data_query_result = get_render_data_template(
      result_targets,
      render_info.targets,
      target_column_indices,
      {},  // target columns to ignore
      {{raster_mesh_metadata.x_coord_name, "x"},
       {raster_mesh_metadata.y_coord_name, "y"}},  // extra target aliases
      QueryDataLayout::LayoutType::kVertexInterleaved,
      row_count,
      rowid_status,
      true,   // uses a result set
      true);  // allocate local row data buffer

  if (row_count > 0) {
    std::function<void(std::vector<TargetValue>&&, const size_t, const size_t)> do_work;
    if (force_singlethreaded) {
      do_work = [&](std::vector<TargetValue>&& crt_row,
                    const size_t row_idx,
                    const size_t resultrow_entry_idx) {
        set_non_in_situ_render_data_entry(data_query_result,
                                          crt_row,
                                          result_targets,
                                          row_idx,
                                          resultrow_entry_idx,
                                          rowid_status,
                                          data_query_result.align_bytes);
      };
    } else {
      do_work = [&](std::vector<TargetValue>&& crt_row,
                    const size_t row_idx,
                    const size_t resultrow_entry_idx) {
        // we know that resultrow_entry_idx is the row index of the result set
        set_non_in_situ_render_data_entry(data_query_result,
                                          crt_row,
                                          result_targets,
                                          resultrow_entry_idx,
                                          resultrow_entry_idx,
                                          rowid_status,
                                          data_query_result.align_bytes);
      };
    }

    executor_process_result_rows(*rows, do_work, force_singlethreaded);

    // Since we know that the input raster result set is in row-major order,
    // and we know the dimensions, the indices for triangulating the mesh will be in a
    // very predictable form and can be generated on its own.

    // TODO(croot): multithread this, and/or, generate it in parallel while building the
    // vbo above
    auto const num_triangles = (raster_width - 1) * (raster_height - 1) * 2;
    std::vector<uint32_t> indices(num_triangles * 3);

    uint32_t index_buffer_idx = 0;
    for (uint32_t y = 0; y < raster_height - 1; ++y) {
      auto const y_offset = y * raster_width;
      for (uint32_t x = 0; x < raster_width - 1; ++x) {
        auto const lower_left_idx = y_offset + x;
        auto const lower_right_idx = lower_left_idx + 1;
        auto const upper_left_idx = lower_left_idx + raster_width;
        auto const upper_right_idx = upper_left_idx + 1;
        indices[index_buffer_idx++] = lower_left_idx;
        indices[index_buffer_idx++] = lower_right_idx;
        indices[index_buffer_idx++] = upper_left_idx;
        indices[index_buffer_idx++] = upper_left_idx;
        indices[index_buffer_idx++] = lower_right_idx;
        indices[index_buffer_idx++] = upper_right_idx;
      }
    }

    // create the in-situ buffers
    auto& raster_mesh_mgr = render_manager.getRasterMeshMgr();
    auto buffer_ptrs = raster_mesh_mgr.createRasterMeshBuffers(
        name_loc.getString(),
        gpu_idx,
        {data_query_result.data.size(), indices.size() * sizeof(uint32_t)});

    auto buffer_memory_descriptors =
        raster_mesh_mgr.getBufferDescriptors(buffer_ptrs, gpu_idx);

#ifdef HAVE_CUDA
    auto const* cuda_mgr = render_manager.getCudaMgr();
    cuda_mgr->setContext(gpu_idx);
    cuMemcpyHtoD(
        reinterpret_cast<CUdeviceptr>(buffer_memory_descriptors.vbo_descriptor.handle),
        data_query_result.data.data(),
        buffer_memory_descriptors.vbo_descriptor.num_bytes);

    cuMemcpyHtoD(
        reinterpret_cast<CUdeviceptr>(buffer_memory_descriptors.ibo_descriptor.handle),
        indices.data(),
        buffer_memory_descriptors.ibo_descriptor.num_bytes);
#else   // !HAVE_CUDA
    std::memcpy(buffer_memory_descriptors.vbo_descriptor.handle,
                data_query_result.data.data(),
                buffer_memory_descriptors.vbo_descriptor.num_bytes);

    std::memcpy(buffer_memory_descriptors.ibo_descriptor.handle,
                indices.data(),
                buffer_memory_descriptors.ibo_descriptor.num_bytes);
#endif  // HAVE_CUDA

    buffer_ptrs.vbo->setQueryDataLayout(data_query_result.render_data_layout,
                                        data_query_result.data.size());
  }

  render_info.setQueryVboLayout(data_query_result.render_data_layout);

  return row_count;
}

}  // namespace QueryRenderer
