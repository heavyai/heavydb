/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/DataMgr.h"
#include "GfxDriver/Resources/BufferAllocator.h"

namespace QueryRenderer {

class QueryBufferManager;

class QueryBufferAllocator : public gfx::BufferAllocator {
 public:
  explicit QueryBufferAllocator(const gfx::DeviceContext& device_ctx,
                                QueryBufferManager& query_buffer_manager,
                                Data_Namespace::DataMgr* data_mgr,
                                const int gpu_id);
  QueryBufferAllocator() = delete;
  ~QueryBufferAllocator() override = default;

  gfx::BufferAllocationUqPtr alloc(const uint64_t num_bytes) final;
  void free(gfx::BufferAllocationUqPtr allocation) final;

  void validateCreateInfo(const gfx::BufferCreateInfo& create_info) final;

  const gfx::DeviceContext& getDeviceContext() const override { return device_ctx_; }

 private:
  const gfx::DeviceContext& device_ctx_;
  Data_Namespace::DataMgr* data_mgr_;
  std::map<uint64_t, AbstractBuffer*> abstract_buffers_;
#ifdef HAVE_CUDA
  QueryBufferManager& query_buffer_manager_;
  const int gpu_id_;
#endif
};

}  // namespace QueryRenderer
