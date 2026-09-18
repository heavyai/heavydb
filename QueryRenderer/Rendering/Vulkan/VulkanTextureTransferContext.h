/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <vector>

#include <vulkan/vulkan.h>

#include "CudaMgr/CudaMgr.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Objects/TileBuilder.h"
#include "GfxDriver/Resources/Texture.h"
#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/Rendering/TextureTransferContext.h"

namespace QueryRenderer {

struct CudaImportedMemory {
  CUexternalMemory ext_memory = nullptr;
  CUdeviceptr device_ptr = 0ULL;
  uint64_t memory_size = 0ULL;
};

/**
 * VulkanTextureTransferContext
 *
 * Copies 1 or more Vulkan based textures into one or more texture arrays
 *
 * */
class VulkanTextureTransferContext : public TextureTransferContext {
 public:
  explicit VulkanTextureTransferContext(const GlobalRenderContext& global_ctx,
                                        const CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                        const gfx::DeviceContext& device_ctx,
                                        std::string_view name);
  VulkanTextureTransferContext() = delete;

  ~VulkanTextureTransferContext() override;

  void buildResources(uint32_t width, uint32_t height) final;
  void destroyResources();

  void updateTileQueue(uint32_t width, uint32_t height) final;

  // singleton API
  void registerSourceDevice(
      const RootPerGpuData& gpu_data,
      const std::vector<const gfx::Texture*>& src_id_textures) final;

  void copyTexturesFromDevice(
      GpuId source_gpu_id,
      SyncType sync_type,
      const std::vector<gfx::resource_ptr<gfx::Texture>>& dst_texture_arrays) final;

  void copyTextureArrayFromDevice(GpuId source_gpu_id,
                                  const gfx::Texture& src_texture_array,
                                  const gfx::Texture& dst_texture,
                                  uint32_t num_layers_to_copy,
                                  ArrayLayerTransferredCBFunc layer_transfer_cb) final;

  void waitComplete() final;

 private:
  const GlobalRenderContext& global_ctx_;

  class CudaStateGuard;
  const CudaMgr_Namespace::CudaMgr* cuda_mgr_;

  GpuId start_gpu_id_;

  const gfx::VulkanDeviceContext& dest_device_ctx_;
  const GpuId dest_gpu_id_;
  CUcontext dest_cuda_ctx_;

  uint32_t width_;
  uint32_t height_;

  // tiling
  std::vector<gfx::Rect2D> tile_queue_;
  uint32_t tile_queue_width_;
  uint32_t tile_queue_height_;
  uint64_t tile_buffer_mem_size_;

  // single / double buffering
  uint32_t num_buffers_;
  uint32_t active_buffer_index_;

  struct DestResources {
    // Destination buffer for peer copying
    gfx::resource_ptr<gfx::Buffer> peer_copy_buffer;
    VkBuffer peer_copy_vk_buffer{VK_NULL_HANDLE};

    // Imported cuda memory resources for the buffer
    CudaImportedMemory peer_copy_cuda_resources{};

    // Cuda events
    CUevent peer_copy_complete_event{nullptr};

    // Semaphores to allow synchronizing Cuda and Vulkan
    VkSemaphore vk_vulkan_done_semaphore{VK_NULL_HANDLE};  // Vulkan done
    CUexternalSemaphore cu_vulkan_done_semaphore{nullptr};
    VkSemaphore vk_cuda_done_semaphore{VK_NULL_HANDLE};  // Cuda done
    CUexternalSemaphore cu_cuda_done_semaphore{nullptr};
  };

  std::vector<DestResources> dst_resources_;

  // Cuda stream for issuing copies
  CUstream dst_cuda_stream_{nullptr};

  struct SourceResources {
    // Src buffer to hold src image data for peer copying
    gfx::resource_ptr<gfx::Buffer> peer_copy_buffer;
    VkBuffer peer_copy_vk_buffer{VK_NULL_HANDLE};

    // Imported memory resources for buffer
    CudaImportedMemory peer_copy_cuda_resources{};

    // Vulkan semaphore to be signaled by stream when peer copy is complete
    VkSemaphore vk_peer_copy_done_semaphore{VK_NULL_HANDLE};
    CUexternalSemaphore cu_peer_copy_done_semaphore{nullptr};

    // Vulkan semaphores and Cuda event to be signaled when src image to buffer copy
    // complete

    // Signal image->buffer copy complete and begin peer-to-peer copy
    VkSemaphore vk_src_begin_peer_copy_semaphore{VK_NULL_HANDLE};
    CUexternalSemaphore cu_src_begin_peer_copy_semaphore{nullptr};
    // Event to signal so destination stream can proceed. The destination
    // stream will wait on the event before beginning transfer
    CUevent src_begin_peer_copy_event{nullptr};
  };

  struct SourceGpuData {
    const gfx::VulkanDeviceContext& device_ctx;
    CUcontext cuda_ctx{nullptr};
    GpuId gpu_id{0u};

    // Src images
    VkImage rgba_vk_image{VK_NULL_HANDLE};
    std::vector<VkImage> id_vk_images;

    // Transfer resources
    std::vector<SourceResources> resources;

    // Signal image available for next RGBA sample extraction
    VkSemaphore vk_src_get_next_sample_semaphore{VK_NULL_HANDLE};

    // Cuda stream on the source gpu to allow use of cuStreamWaitEvent
    CUstream cuda_stream{nullptr};

    SourceGpuData() = delete;
    explicit SourceGpuData(const gfx::VulkanDeviceContext& device_ctx,
                           const CUcontext cuda_ctx)
        : device_ctx{device_ctx}, cuda_ctx{cuda_ctx}, gpu_id{device_ctx.getGpuId()} {}
  };
  std::map<GpuId, SourceGpuData> source_gpu_data_;

  // Command pool for Vulkan commands
  gfx::VulkanCommandPool& vk_cmd_pool_;

  void swapBuffers();

  void copyLocalTextures(SyncType sync_type,
                         const SourceGpuData& src_gpu_data,
                         const std::vector<VkImage>& dst_images);

  //
  // Buffer copy support
  //
  void copySrcImageToSrcBuffer(SourceGpuData& src_gpu_data,
                               VkImage src_image,
                               uint32_t layer_index,
                               const gfx::Rect2D& region,
                               VkImageLayout src_layout,
                               VkBuffer src_buffer,
                               VkSemaphore wait_semaphore,
                               const std::vector<VkSemaphore>& signal_semaphores,
                               const std::vector<VkPipelineStageFlags2>& signal_stages);

  void transitionSrcTextures(SourceGpuData& src_gpu_data,
                             const std::vector<VkImage>& src_images,
                             gfx::ImageLayout layout);

  void copyDstBufferToDstImage(VkImage image,
                               const gfx::Rect2D& region,
                               VkImageLayout dst_layout,
                               const DestResources& dst_resources,
                               bool do_signal_cuda);

  // Error logging
  void waitCallback(VkResult result);
};

}  // namespace QueryRenderer
