/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecuteRenderInterface/RenderQueryUtils/ProcessRowResults.h"

#include <numeric>

#include "ExecuteRenderInterface/RenderQueryUtils/ProcessResultsUtils.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/BufferLayout.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/GpuRtConstants.h"
#include "QueryRenderer/QueryRenderManager.h"

namespace QueryRenderer {

// POINT, MULTIPOINT, LINESTRING, MULTILINESTRING
uint64_t process_rows_in_situ(RenderInfo& render_info) {
  RENDER_LOG_SCOPE();
  // this path will ultimately support all in-situ direct geo rendering
  // POINT, MULTIPOINT, LINESTRING, MULTILINESTRING, POLYGON, MULTIPOLYGON
  // as well as in-situ scalar (x/y) point rendering
  CHECK(render_info.isInSitu());

  std::vector<QueryDataLayout::AttrAliasInfo> attr_info;
  attr_info.emplace_back("key", SQLTypeInfo(kBIGINT, true), -1, -1);
  for (const auto& te : render_info.targets) {
    const auto alias = te->get_resname();
    int table_id = -1, col_id = -1;
    const auto target_expr = te->get_expr();
    const auto& type_info = target_expr->get_type_info();
    std::tie(table_id, col_id) = get_table_id_col_id_from_target_expr(target_expr);
    attr_info.emplace_back(alias, type_info, table_id, col_id);
  }

  auto query_data_layout =
      std::make_shared<QueryDataLayout>(std::move(attr_info),
                                        QueryDataLayout::LayoutType::kVertexInterleaved,
                                        query_sql_type_to_render_type,
                                        EMPTY_KEY_64);
  render_info.setQueryVboLayout(query_data_layout);

  uint64_t total_used_bytes{0};
  for (size_t i = 0; i < render_info.render_allocator_map_ptr->size(); ++i) {
    auto render_allocator = render_info.render_allocator_map_ptr->getRenderAllocator(i);
    // only GPUs that have data
    if (render_allocator->getAllocatedSize()) {
      total_used_bytes += render_allocator->getCurrentChunkSize();
      if (!render_info.useCudaBuffers()) {
        // copy the render allocator buffer to the GPU
        auto ptr =
            render_allocator->getBasePtr() + render_allocator->getCurrentChunkOffset();
        render_info.render_allocator_map_ptr->bufferData(
            ptr, render_allocator->getCurrentChunkSize(), i);
      }
    }
  }

  RENDER_LOG() << "calling render_info.render_allocator_map_ptr->setDataLayout()";
  render_info.render_allocator_map_ptr->setDataLayout(query_data_layout);

  auto bytes_per_row = query_data_layout->getBufferLayout()->getNumBytesPerItem();
  CHECK_GT(bytes_per_row, 0ull);
  CHECK_EQ(total_used_bytes % bytes_per_row, 0ull);
  return total_used_bytes / bytes_per_row;
}

// scalar points only
uint64_t process_rows_non_in_situ(QueryRenderManager& render_manager,
                                  const ExecutionResult& results,
                                  RenderInfo& render_info) {
  RENDER_LOG_SCOPE();
  CHECK(render_info.render_allocator_map_ptr && !render_info.isInSitu());

  const int gpu_idx = render_manager.getLeastSubscribedGpuId();
  VLOG(1) << "Selected gpu " << gpu_idx << " for non-insitu point render";
  auto render_allocator =
      render_info.render_allocator_map_ptr->getRenderAllocator(gpu_idx);

  const auto& rows = results.getRows();
  const auto entry_count = rows->entryCount();
  const auto row_count = rows->rowCount(entry_count > kMinRowCountWorthMultiThreading);

  const auto& result_targets = results.getTargetsMeta();
  const auto rowid_status = get_rowid_status(result_targets, render_info);

  std::vector<unsigned int> target_column_indices(result_targets.size());
  std::iota(target_column_indices.begin(), target_column_indices.end(), 0);
  auto data_query_result =
      get_render_data_template(result_targets,
                               render_info.targets,
                               target_column_indices,
                               {},  // target columns to ignore
                               {},  // extra target aliases
                               QueryDataLayout::LayoutType::kVertexInterleaved,
                               row_count,
                               rowid_status,
                               true,   // uses a result set
                               true);  // allocate local row data buffer

  const auto do_work = [&data_query_result, &result_targets, &rowid_status](
                           std::vector<TargetValue>&& crt_row,
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

  executor_process_result_rows(*rows, do_work);

  render_allocator->alloc(data_query_result.data.size());

  render_info.render_allocator_map_ptr->bufferData(
      reinterpret_cast<int8_t*>(data_query_result.data.data()),
      data_query_result.data.size(),
      gpu_idx);

  render_info.render_allocator_map_ptr->setDataLayout(
      data_query_result.render_data_layout);

  render_info.setQueryVboLayout(data_query_result.render_data_layout);

  return row_count;
}

}  // namespace QueryRenderer
