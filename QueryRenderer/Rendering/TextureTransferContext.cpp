/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/TextureTransferContext.h"

#ifdef HAVE_CUDA
#include "QueryRenderer/Rendering/Vulkan/VulkanTextureTransferContext.h"
#endif

namespace QueryRenderer {

std::unique_ptr<TextureTransferContext> TextureTransferContext::create(
    const GlobalRenderContext& global_ctx,
    const CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const gfx::DeviceContext& device_ctx,
    std::string_view name) {
#ifdef HAVE_CUDA
  return std::make_unique<VulkanTextureTransferContext>(
      global_ctx, cuda_mgr, device_ctx, name);
#else
  LOG(FATAL) << "TextureTransferContext not supported in Cuda disabled builds";
#endif
  return nullptr;
}

}  // namespace QueryRenderer
