/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/RowDataTableGpuResources.h"

#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Interface/RenderQueryExecuteData.h"
#include "QueryRenderer/Interop/QueryBuffer.h"

namespace QueryRenderer {

QueryDataLayoutShPtr RowDataTableGpuResources::getDataLayout() const {
  if (gpu_data_map_.isEmpty()) {
    return nullptr;
  }

  auto* gpu_data = gpu_data_map_.getFirstData();
  CHECK(gpu_data);

  RUNTIME_EX_ASSERT(gpu_data->vbo != nullptr,
                    "Cannot get query data layout. The data has no query results defined "
                    "in the buffer.");

  auto result_buffer = dynamic_cast<QueryVertexBuffer*>(gpu_data->vbo.get());
  CHECK(result_buffer);

  return result_buffer->getQueryDataLayout();
}

void RowDataTableGpuResources::initGpuResourcesFromBuffers(
    const GlobalRenderContext& global_ctx,
    const QueryDataLayoutShPtr& layout) {
  // TODO(scb): All DataTableGpuResources classes implement nearly identical versions of
  // this function, and could be refactored to remove duplication.
  // There are 3 different call signatures where params after global_ctx vary,
  // complicating the refactor.
  //
  // kEmbedded / kURL case is identical in all implementations
  //  - not implemented in MeshDataTableGpuResources
  //
  // kSQL case varies and is the reason for varying parameters
  //  - The predicate controlling whether processing should continue varies
  //  - Populating the newly emplaced PerGpuData element also varies
  RENDER_LOG_SCOPE();
  std::vector<GpuId> unused_gpus;
  switch (input_format_) {
    case DataInputFormat::kEmbedded:
    case DataInputFormat::kURL: {
      // forcing these tables to always be on the first gpu
      auto& qrm_per_gpu_data = global_ctx.getRootPerGpuData();
      auto itr = qrm_per_gpu_data.begin();
      auto gpu_id = (*itr)->getGpuId();
      if (!gpu_data_map_.hasData(gpu_id)) {
        gpu_data_map_.try_emplace(gpu_id, **itr, nullptr);
      }

      for (++itr; itr != qrm_per_gpu_data.end(); ++itr) {
        unused_gpus.push_back((*itr)->getGpuId());
      }

      break;
    }
    case DataInputFormat::kSQL: {
      // Must have a layout. If not, there's no data from the query.
      if (layout) {
        for (const auto& item : global_ctx.getRootPerGpuData()) {
          auto used_bytes = item->getQueryResultBuffer()->getNumUsedBytes(*layout);
          RENDER_LOG() << "gpu " << item->getGpuId() << " result buffer " << used_bytes
                       << " bytes";
          if (used_bytes > 0) {
            if (!gpu_data_map_.hasData(item->getGpuId())) {
              auto rtn = gpu_data_map_.try_emplace(item->getGpuId(), *item, nullptr);
              rtn.first->second.vbo = item->getQueryResultBufferShPtr();
            }
          } else {
            unused_gpus.push_back(item->getGpuId());
          }
        }
      } else {
        RENDER_LOG() << "no layout, clearing per_gpu_data map";
        gpu_data_map_.clear();
      }
      break;
    }
    default:
      THROW_RUNTIME_EX("Unsupported data table type for mult-gpu configuration: " +
                       std::to_string(static_cast<int>(input_format_)));
  }

  RENDER_LOG() << "erasing query VBO data from "
               << render_logger::format_gpuid_vector(unused_gpus);
  gpu_data_map_.erase(unused_gpus);
}

bool RowDataTableGpuResources::updateOffsetBytesMap(const QueryDataLayoutShPtr& layout) {
  for (auto itr = curr_buf_offset_bytes_.begin(); itr != curr_buf_offset_bytes_.end();) {
    if (gpu_data_map_.hasData(itr->first)) {
      itr++;
    } else {
      itr = curr_buf_offset_bytes_.erase(itr);
    }
  }

  bool did_offset_change = false;
  gpu_data_map_.visitData([&](GpuId gpu_id, RowDataTablePerGpuData& gpu_data) {
    auto offset = gpu_data.vbo->getLayoutOffsetBytes(*layout);
    auto itr = curr_buf_offset_bytes_.find(gpu_id);
    if (itr == curr_buf_offset_bytes_.end()) {
      curr_buf_offset_bytes_.emplace(gpu_id, offset);
    } else if (itr->second != offset) {
      itr->second = offset;
      did_offset_change = true;
    }
    return true;
  });
  return did_offset_change;
}

void RowDataTableGpuResources::clearOffsetBytesMap() {
  curr_buf_offset_bytes_.clear();
}

}  // namespace QueryRenderer
