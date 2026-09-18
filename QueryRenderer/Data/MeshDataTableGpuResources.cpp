/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/MeshDataTableGpuResources.h"

#include "QueryRenderer/GlobalRenderContext.h"

namespace QueryRenderer {

MeshDataTablePerGpuData::MeshDataTablePerGpuData(RootPerGpuData& root_data,
                                                 QueryVertexBufferShPtr in_vbo,
                                                 QueryIndexBufferShPtr in_ibo)
    : BasePerGpuData(root_data), vbo(std::move(in_vbo)), ibo(std::move(in_ibo)) {}

MeshDataTablePerGpuData::~MeshDataTablePerGpuData() {
  clear();
}

void MeshDataTablePerGpuData::clear() {
  RENDER_LOG_SCOPE_P(getGpuId());
  auto& root_per_gpu_data = getRootPerGpuData();
  QueryVertexBufferWkPtr vbo_wk = vbo;
  vbo = nullptr;
  RENDER_LOG() << "setting vbo inactive - use count: " << vbo_wk.use_count();
  root_per_gpu_data.getVboBufferPool().setRsrcInactive(vbo_wk);

  QueryIndexBufferWkPtr ibo_wk = ibo;
  ibo = nullptr;
  RENDER_LOG() << "setting ibo inactive - use count: " << ibo_wk.use_count();
  root_per_gpu_data.getIboBufferPool().setRsrcInactive(ibo_wk);
}

void MeshDataTableGpuResources::initGpuResourcesFromBuffers(
    const GlobalRenderContext& global_ctx,
    const std::string& vega_data_table_name) {
  // TODO(scb): All BaseDataTableGpuResources classes implement nearly identical versions
  // of this function, and should be refactored to remove duplication.
  // See same function in DataTableGpuResources.cpp for notes
  RENDER_LOG_SCOPE() << "table name: " << vega_data_table_name;

  std::vector<GpuId> unused_gpus;
  switch (input_format_) {
    case DataInputFormat::kSQL: {
      RUNTIME_EX_ASSERT(vega_data_table_name.size() > 0,
                        "Unable to find name for mesh data table when attempting to "
                        "initialize from buffers.");
      for (const auto& item : global_ctx.getRootPerGpuData()) {
        auto tmp_data_ptrs =
            item->getTmpRasterMeshBuffersForDataTable(vega_data_table_name);
        if (tmp_data_ptrs) {
          RENDER_LOG() << "adding: " << item->getGpuId();
          auto [itr, was_inserted] = gpu_data_map_.try_emplace(item->getGpuId(), *item);

          if (!was_inserted) {
            RENDER_LOG() << "clearing: " << item->getGpuId();
            itr->second.clear();
          }

          CHECK(itr->second.vbo == nullptr);
          CHECK(itr->second.ibo == nullptr);

          itr->second.vbo = tmp_data_ptrs->vbo.lock();
          itr->second.ibo = tmp_data_ptrs->ibo.lock();
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

}  // namespace QueryRenderer
