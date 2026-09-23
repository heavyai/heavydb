/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/Vulkan/VulkanTextureTransferContextCudaUtils.h"

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanBaseBuffer.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanMemoryMgr.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/RenderError.h"
#include "GfxInterop/Utils/CudaErrorCheck.h"

namespace QueryRenderer {

namespace {

void init_cuda_imported_memory(const GpuId gpu_id,
                               gfx::VulkanAllocation* vk_alloc,
                               CudaImportedMemory& cuda_imported_memory,
                               const std::string& buffer_name,
                               const bool is_dest) {
  // Export memory handle from Vulkan
  CHECK_EQ(static_cast<int>(vk_alloc->getAllocationType()),
           static_cast<int>(gfx::VulkanAllocation::AllocationType::kExportable));
  auto* vk_exp_alloc = static_cast<gfx::VulkanExportableAllocation*>(vk_alloc);
  CHECK(vk_exp_alloc);
  int fd = vk_exp_alloc->exportHandle(buffer_name, is_dest);
  CHECK(fd);

  // Verify the current cuda device context and gpu id match
  CUdevice cuda_device;
  CHECK_RENDER_CUDA_ERRORS(cuCtxGetDevice(&cuda_device), gpu_id);

  RUNTIME_EX_ASSERT(cuda_device == static_cast<int>(gpu_id),
                    "Invalid cuda context when registering a texture for Vulkan "
                    "compositing. Vulkan resource gpu id: " +
                        std::to_string(gpu_id) + " does not match Cuda device id: " +
                        std::to_string(cuda_device) + ".");

  // Import the memory handle into cuda
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC mem_handle_desc = {};
  mem_handle_desc.handle.fd = fd;
  mem_handle_desc.flags = 0;  // TODO: dedicated resource
  mem_handle_desc.size = vk_alloc->size();
  mem_handle_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  CHECK_RENDER_CUDA_ERRORS(
      cuImportExternalMemory(&cuda_imported_memory.ext_memory, &mem_handle_desc), gpu_id);

  // Get the Cuda deviceptr for the memory
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC ext_buffer_desc = {};
  ext_buffer_desc.size = vk_alloc->size();

  CHECK_RENDER_CUDA_ERRORS(
      cuExternalMemoryGetMappedBuffer(&cuda_imported_memory.device_ptr,
                                      cuda_imported_memory.ext_memory,
                                      &ext_buffer_desc),
      gpu_id);

  cuda_imported_memory.memory_size = vk_alloc->size();
}
}  // namespace

void init_cuda_vulkan_buffer(const GpuId gpu_id,
                             const gfx::Buffer& buffer,
                             CudaImportedMemory& cuda_imported_memory,
                             const bool is_dest) {
  auto const& vk_base_buffer = static_cast<const gfx::VulkanBaseBuffer&>(buffer);
  init_cuda_imported_memory(gpu_id,
                            vk_base_buffer.getMemoryAllocation(),
                            cuda_imported_memory,
                            buffer.getTrackingData().origin,
                            is_dest);
}

void destroy_cuda_imported_memory(CudaImportedMemory& cuda_imported_memory,
                                  const GpuId gpu_id) {
  if (cuda_imported_memory.ext_memory) {
    CHECK_RENDER_CUDA_ERRORS(cuMemFree(cuda_imported_memory.device_ptr), gpu_id);
    cuda_imported_memory.device_ptr = 0ULL;

    CHECK_RENDER_CUDA_ERRORS(cuDestroyExternalMemory(cuda_imported_memory.ext_memory),
                             gpu_id);
    cuda_imported_memory.ext_memory = nullptr;
    cuda_imported_memory.memory_size = 0ULL;
  }
}

VkSemaphore init_vulkan_semaphore(const gfx::VulkanDeviceContext& device_ctx,
                                  const std::string& name) {
  // Create the Vulkan semaphore
  VkExportSemaphoreCreateInfo semaphore_export_ci = {};
  semaphore_export_ci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
  semaphore_export_ci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkSemaphoreCreateInfo semaphore_ci = {};
  semaphore_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_ci.pNext = &semaphore_export_ci;

  VkSemaphore vk_semaphore{VK_NULL_HANDLE};
  CHECK_VKRESULT(
      vkCreateSemaphore(device_ctx.getHandle(), &semaphore_ci, nullptr, &vk_semaphore),
      "Creating copy complete semaphore");

  device_ctx.nameVulkanObject(
      VK_OBJECT_TYPE_SEMAPHORE, vk_semaphore, "Semaphore (" + std::string(name) + ")");

  return vk_semaphore;
}

void init_cuda_vulkan_semaphore(const gfx::VulkanDeviceContext& device_ctx,
                                VkSemaphore& vk_semaphore,
                                CUexternalSemaphore& cu_semaphore,
                                const std::string& vk_semaphore_name) {
  // Check that the correct cuda context is active
  GpuId gpu_id = device_ctx.getGpuId();
  CUdevice cuda_device;
  CHECK_RENDER_CUDA_ERRORS(cuCtxGetDevice(&cuda_device), gpu_id);
  RUNTIME_EX_ASSERT(cuda_device == static_cast<int>(gpu_id),
                    "Invalid cuda context when registering a texture for Vulkan "
                    "compositing. Vulkan resource gpu id: " +
                        std::to_string(gpu_id) + " does not match Cuda device id: " +
                        std::to_string(cuda_device) + ".");

  vk_semaphore = init_vulkan_semaphore(device_ctx, vk_semaphore_name);

  // Export handle
  int fd;
  VkSemaphoreGetFdInfoKHR get_fd_info = {};
  get_fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  get_fd_info.semaphore = vk_semaphore;
  get_fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  CHECK_VKRESULT(device_ctx.getFunctions().vkGetSemaphoreFdKHR(
                     device_ctx.getHandle(), &get_fd_info, &fd),
                 "Exporting semaphore");
  CHECK(fd);

  // Import into Cuda
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC cu_handle_desc = {};
  cu_handle_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
  cu_handle_desc.handle.fd = fd;
  CHECK_RENDER_CUDA_ERRORS(cuImportExternalSemaphore(&cu_semaphore, &cu_handle_desc),
                           gpu_id);
}

void destroy_cuda_vulkan_semaphore(const gfx::VulkanDeviceContext& device_ctx,
                                   VkSemaphore& vk_semaphore,
                                   CUexternalSemaphore& cu_semaphore) {
  if (cu_semaphore) {
    CHECK_RENDER_CUDA_ERRORS(cuDestroyExternalSemaphore(cu_semaphore),
                             device_ctx.getGpuId());
    cu_semaphore = nullptr;
  }
  if (vk_semaphore) {
    vkDestroySemaphore(device_ctx.getHandle(), vk_semaphore, nullptr);
    vk_semaphore = VK_NULL_HANDLE;
  }
}

VulkanTextureTransferContext::CudaStateGuard::CudaStateGuard(const CUcontext& cuda_ctx,
                                                             const GpuId gpu_id)
    : gpu_id_{gpu_id} {
  CHECK_RENDER_CUDA_ERRORS(cuCtxPushCurrent(cuda_ctx), gpu_id);
}

VulkanTextureTransferContext::CudaStateGuard::~CudaStateGuard() {
  CHECK_RENDER_CUDA_ERRORS(cuCtxPopCurrent(nullptr), gpu_id_);
}

}  // namespace QueryRenderer
