/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/LineDataTableGpuResources.h"

#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/BaseDataTableGpuResources.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/GlobalRenderContext.h"

namespace QueryRenderer {

LineDataTablePerGpuData::LineDataTablePerGpuData(
    RootPerGpuData& root_data,
    const QueryVertexBufferShPtr& vbo,
    const QueryIndexBufferShPtr& ibo,
    const QueryShaderStorageBufferShPtr& ssbo,
    const QueryIndirectVboShPtr& ind_vbo,
    const QueryIndirectIboShPtr& ind_ibo)
    : BasePerGpuData(root_data)
    , vbo{vbo}
    , ibo{ibo}
    , ssbo{ssbo}
    , ind_vbo{ind_vbo}
    , ind_ibo{ind_ibo} {}

void LineDataTablePerGpuData::resetPointers() {
  vbo = nullptr;
  ibo = nullptr;
  ssbo = nullptr;
  ind_vbo = nullptr;
  ind_ibo = nullptr;
}

LineDataTableGpuResources::LineDataTableGpuResources(DataInputFormat input_format)
    : BaseDataTableGpuResources(), input_format_{input_format} {}

const gfx::IndexBuffer* LineDataTableGpuResources::getIndexBuffer(
    const GpuId gpu_id) const {
  auto const& gpu_data = gpu_data_map_.getData(gpu_id);
  return (gpu_data.ibo ? gpu_data.ibo->unmapForDraw() : nullptr);
}

const gfx::BufferWrapper* LineDataTableGpuResources::getShaderStorageBuffer(
    const GpuId gpu_id) const {
  auto const& gpu_data = gpu_data_map_.getData(gpu_id);
  return (gpu_data.ssbo ? gpu_data.ssbo->unmapForDraw() : nullptr);
}

const gfx::IndirectDrawVertexBuffer*
LineDataTableGpuResources::getIndirectDrawVertexBuffer(const GpuId gpu_id) const {
  auto const& gpu_data = gpu_data_map_.getData(gpu_id);
  return (gpu_data.ind_vbo ? gpu_data.ind_vbo->unmapForDraw() : nullptr);
}

const gfx::IndirectDrawIndexBuffer* LineDataTableGpuResources::getIndirectDrawIndexBuffer(
    const GpuId gpu_id) const {
  auto const& gpu_data = gpu_data_map_.getData(gpu_id);
  return (gpu_data.ind_ibo ? gpu_data.ind_ibo->unmapForDraw() : nullptr);
}

void LineDataTableGpuResources::initGpuResourcesFromBuffers(
    const GlobalRenderContext& global_ctx,
    bool use_index_buffer,
    const std::string& vega_data_table_name) {
  RENDER_LOG_SCOPE() << "table name: " << vega_data_table_name;
  // TODO(scb): All BaseDataTableGpuResources classes implement nearly identical versions
  // of this function, and should be refactored to remove duplication.
  // See same function in DataTableGpuResources.cpp for notes

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
                        "Unable to find name for line data table when attempting to "
                        "initialize from buffers.");
      for (const auto& item : global_ctx.getRootPerGpuData()) {
        auto tmp_data_ptrs = item->getTmpLineBuffersForDataTable(vega_data_table_name);
        if (tmp_data_ptrs) {
          auto rtn = gpu_data_map_.try_emplace(item->getGpuId(), *item);
          auto& gpu_data = rtn.first->second;
          auto line_vbo = tmp_data_ptrs->verts.lock();
          if (line_vbo) {
            gpu_data.vbo = line_vbo;
          }

          auto line_ssbo = tmp_data_ptrs->per_row_data.lock();
          if (line_ssbo) {
            gpu_data.ssbo = line_ssbo;
          }

          auto line_indirect_vbo = tmp_data_ptrs->indirect_vertex_struct.lock();
          if (line_indirect_vbo) {
            gpu_data.ind_vbo = line_indirect_vbo;
          }

          if (use_index_buffer) {
            auto line_ibo = tmp_data_ptrs->indices.lock();
            if (line_ibo) {
              gpu_data.ibo = line_ibo;
            }

            auto line_indirect_ibo = tmp_data_ptrs->indirect_index_struct.lock();
            if (line_indirect_ibo) {
              gpu_data.ind_ibo = line_indirect_ibo;
            }
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
  RENDER_LOG() << "unused gpus: " << render_logger::format_gpuid_vector(unused_gpus);
  gpu_data_map_.erase(unused_gpus);
}

void LineDataTableGpuResources::resetDataPointers() {
  gpu_data_map_.visitData([](GpuId gpu_id, LineDataTablePerGpuData& gpu_data) {
    gpu_data.resetPointers();
    return true;
  });
}

}  // namespace QueryRenderer
