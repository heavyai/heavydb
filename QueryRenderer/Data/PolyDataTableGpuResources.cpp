/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/PolyDataTableGpuResources.h"

#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/BaseDataTableGpuResources.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/GlobalRenderContext.h"

namespace QueryRenderer {

PolyDataTablePerGpuData::PolyDataTablePerGpuData(
    RootPerGpuData& rootData,
    const QueryVertexBufferShPtr& vbo,
    const QueryShaderStorageBufferShPtr& ssbo,
    const QueryIndirectVboShPtr& ivbo_lines,
    const QueryIndirectVboShPtr& ivbo_polys,
    const QueryShaderStorageBufferShPtr& poly_rowids)
    : BasePerGpuData{rootData}
    , vbo{vbo}
    , ssbo{ssbo}
    , ivbo_lines{ivbo_lines}
    , ivbo_polys{ivbo_polys}
    , poly_rowids{poly_rowids} {}

void PolyDataTablePerGpuData::resetPointers() {
  vbo = nullptr;
  ssbo = nullptr;
  ivbo_lines = nullptr;
  ivbo_polys = nullptr;
  poly_draw_batch_info = nullptr;
  poly_rowids = nullptr;
}

PolyDataTableGpuResources::PolyDataTableGpuResources(DataInputFormat input_format)
    : BaseDataTableGpuResources(), input_format_{input_format} {}

const gfx::IndirectDrawVertexBuffer*
PolyDataTableGpuResources::getIndirectDrawVertexBuffer_lines(const GpuId gpu_id) const {
  auto const& gpu_data = gpu_data_map_.getData(gpu_id);
  return (gpu_data.ivbo_lines ? gpu_data.ivbo_lines->unmapForDraw() : nullptr);
}

const gfx::IndirectDrawVertexBuffer*
PolyDataTableGpuResources::getIndirectDrawVertexBuffer_polys(const GpuId gpu_id) const {
  auto const& gpu_data = gpu_data_map_.getData(gpu_id);
  return (gpu_data.ivbo_polys ? gpu_data.ivbo_polys->unmapForDraw() : nullptr);
}

const gfx::BufferWrapper* PolyDataTableGpuResources::getShaderStorageBuffer_poly_rowids(
    const GpuId gpu_id) const {
  auto const& gpu_data = gpu_data_map_.getData(gpu_id);
  return (gpu_data.poly_rowids ? gpu_data.poly_rowids->unmapForDraw() : nullptr);
}

void PolyDataTableGpuResources::initGpuResourcesFromBuffers(
    const GlobalRenderContext& global_ctx,
    const std::string& vega_data_table_name) {
  // TODO(scb): All BaseDataTableGpuResources classes implement nearly identical versions
  // of this function, and should be refactored to remove duplication.
  // See same function in DataTableGpuResources.cpp for notes

  RENDER_LOG_SCOPE() << "table name: " << vega_data_table_name;
  std::vector<GpuId> unused_gpus;
  switch (input_format_) {
    case DataInputFormat::kEmbedded:
    case DataInputFormat::kURL: {
      // forcing these tables to always be on the first gpu
      auto& qrm_per_gpu_data = global_ctx.getRootPerGpuData();
      auto itr = qrm_per_gpu_data.begin();
      auto gpu_id = (*itr)->getGpuId();
      if (!gpu_data_map_.hasData(gpu_id)) {
        gpu_data_map_.try_emplace((*itr)->getGpuId(), **itr);
      }

      for (++itr; itr != qrm_per_gpu_data.end(); ++itr) {
        unused_gpus.push_back((*itr)->getGpuId());
      }

      break;
    }
    case DataInputFormat::kSQL: {
      RUNTIME_EX_ASSERT(vega_data_table_name.size() > 0,
                        "Unable to find name for poly data table when attempting to "
                        "initialize from buffers.");
      for (const auto& item : global_ctx.getRootPerGpuData()) {
        auto tmp_data_ptrs = item->getTmpPolyBuffersForDataTable(vega_data_table_name);
        if (tmp_data_ptrs) {
          auto rtn = gpu_data_map_.try_emplace(item->getGpuId(), *item);

          auto poly_vbo = tmp_data_ptrs->verts.lock();
          if (poly_vbo) {
            rtn.first->second.vbo = poly_vbo;
          }

          auto line_indirect_draw_vbo = tmp_data_ptrs->line_draw_struct.lock();
          if (line_indirect_draw_vbo) {
            rtn.first->second.ivbo_lines = line_indirect_draw_vbo;
          }

          auto poly_indirect_draw_vbo = tmp_data_ptrs->poly_draw_struct.lock();
          if (poly_indirect_draw_vbo) {
            rtn.first->second.ivbo_polys = poly_indirect_draw_vbo;
          }

          auto poly_ssbo = tmp_data_ptrs->per_row_data.lock();
          if (poly_ssbo) {
            rtn.first->second.ssbo = poly_ssbo;
          }

          auto poly_rowids_ssbo = tmp_data_ptrs->poly_rowids.lock();
          if (poly_rowids_ssbo) {
            rtn.first->second.poly_rowids = poly_rowids_ssbo;
          }

          // also find the matching temp PolyDrawBatchInfo for this data table
          if (item->hasPolyDrawBatchInfoForDataTable(vega_data_table_name)) {
            // and move it over (avoid copy)
            rtn.first->second.poly_draw_batch_info =
                item->extractPolyDrawBatchInfoForDataTable(vega_data_table_name);
          }
        } else {
          unused_gpus.push_back(item->getGpuId());
        }
      }
      break;
    }
    default:
      THROW_RUNTIME_EX("Unsupported data table type for multi-gpu configuration: " +
                       std::to_string(static_cast<int>(input_format_)));
  }

  gpu_data_map_.erase(unused_gpus);
}

void PolyDataTableGpuResources::resetDataPointers() {
  gpu_data_map_.visitData([](GpuId gpu_id, PolyDataTablePerGpuData& gpu_data) {
    gpu_data.resetPointers();
    return true;
  });
}

void PolyDataTableGpuResources::setPolyDrawBatchInfo(
    const GpuId gpu_id,
    PolyDrawBatchInfoUqPtr&& poly_draw_batch_info) {
  auto& gpu_data = gpu_data_map_.getData(gpu_id);
  gpu_data.poly_draw_batch_info = std::move(poly_draw_batch_info);
}

const PolyDrawBatchInfo& PolyDataTableGpuResources::getPolyDrawBatchInfo(
    const GpuId gpu_id) const {
  auto const& gpu_data = gpu_data_map_.getData(gpu_id);
  return *gpu_data.poly_draw_batch_info;
}

}  // namespace QueryRenderer
