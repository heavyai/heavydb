/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include "QueryRenderer/Rendering/Vulkan/VulkanTextureTransferContext.h"

namespace QueryRenderer {

// State guard to automatially push/pop cuda contexts
class VulkanTextureTransferContext::CudaStateGuard {
 public:
  explicit CudaStateGuard(const CUcontext& cuda_ctx, const GpuId gpu_id);
  ~CudaStateGuard();

 private:
  GpuId gpu_id_;
};

// Utility to import Vulkan buffer memory into cuda and retrieve the deviceptr
void init_cuda_vulkan_buffer(const GpuId gpu_id,
                             const gfx::Buffer& buffer,
                             CudaImportedMemory& cuda_imported_memory,
                             const bool is_dest);

// Destroy imported memory resources (either image or buffer)
void destroy_cuda_imported_memory(CudaImportedMemory& cuda_imported_memory,
                                  const GpuId gpu_id);

// Create a vulkan semaphore object and name it
VkSemaphore init_vulkan_semaphore(const gfx::VulkanDeviceContext& device_ctx,
                                  const std::string& name);

// Utility to create and then import a Vulkan semaphore object into Cuda
// Used for synchronizing the two APIs. Currently supports binary semaphores only (no
// timeline semaphores)
void init_cuda_vulkan_semaphore(const gfx::VulkanDeviceContext& device_ctx,
                                VkSemaphore& vk_semaphore,
                                CUexternalSemaphore& cu_semaphore,
                                const std::string& vk_semaphore_name);

void destroy_cuda_vulkan_semaphore(const gfx::VulkanDeviceContext& device_ctx,
                                   VkSemaphore& vk_semaphore,
                                   CUexternalSemaphore& cu_semaphore);

}  // namespace QueryRenderer
