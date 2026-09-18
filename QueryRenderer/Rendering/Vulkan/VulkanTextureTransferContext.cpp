/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Rendering/Vulkan/VulkanTextureTransferContext.h"

#include <cuda.h>

#include "GfxDriver/Drivers/Vulkan/Resources/ImageLayoutManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxInterop/Utils/CudaErrorCheck.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Rendering/SeparateMultiSamplesPass.h"
#include "QueryRenderer/Rendering/Vulkan/VulkanTextureTransferContextCudaUtils.h"
#include "Shared/nvtx_helpers.h"

#define PROFILE_TRANSFER 0
#define PRINT_BUFFER_INFO 0

#if PROFILE_TRANSFER || PRINT_BUFFER_INFO
#include <iostream>
#include "Shared/measure.h"
#endif

#define SKIP_LAYOUT_TRANSITIONS 0

namespace QueryRenderer {

static constexpr bool kUseTileQueue = true;
static constexpr bool kUseDoubleBuffer = true;  // requires tile queue enabled
static constexpr uint32_t kBytesPerPixel = 4u;

// Param structs for Cuda external semaphore calls
// We don't need to set any values, just zero them out, so statics
// are fine
static CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS cuda_wait_params = {};
static CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS cuda_signal_params = {};

VulkanTextureTransferContext::VulkanTextureTransferContext(
    const GlobalRenderContext& global_ctx,
    const CudaMgr_Namespace::CudaMgr* cuda_mgr,
    const gfx::DeviceContext& device_ctx,
    std::string_view name)
    : TextureTransferContext(name)
    , global_ctx_{global_ctx}
    , cuda_mgr_{cuda_mgr}
    , start_gpu_id_{static_cast<GpuId>(cuda_mgr->getStartGpu())}
    , dest_device_ctx_{static_cast<const gfx::VulkanDeviceContext&>(device_ctx)}
    , dest_gpu_id_{device_ctx.getGpuId()}
    , dest_cuda_ctx_{nullptr}
    , width_{0u}
    , height_{0u}
    , tile_queue_width_{0u}
    , tile_queue_height_{0u}
    , tile_buffer_mem_size_{0u}
    , num_buffers_{1u}
    , active_buffer_index_{0u}
    , dst_cuda_stream_{nullptr}
    , vk_cmd_pool_{dest_device_ctx_.getCommandPool(
          gfx::VulkanDeviceContext::CommandPoolSelector::kMultiGpuTransfer)} {
  // Get cuda context for compositor gpu
  dest_cuda_ctx_ =
      cuda_mgr_->getDeviceContexts().at(dest_device_ctx_.getGpuId() - start_gpu_id_);

  // Create an async cuda stream to run all commands
  CudaStateGuard cuda_state_guard(dest_cuda_ctx_, dest_gpu_id_);

  CHECK_RENDER_CUDA_ERRORS(cuStreamCreate(&dst_cuda_stream_, CU_STREAM_NON_BLOCKING),
                           dest_gpu_id_);
  nvtx_helpers::name_cuda_stream(
      dst_cuda_stream_, (std::string("Compositor dest ") + std::string(name)).c_str());
}

VulkanTextureTransferContext::~VulkanTextureTransferContext() {
  destroyResources();

  CudaStateGuard cuda_state_guard(dest_cuda_ctx_, dest_gpu_id_);

  if (dst_cuda_stream_) {
    CHECK_RENDER_CUDA_ERRORS(cuStreamDestroy(dst_cuda_stream_), dest_gpu_id_);
  }
}

static auto compute_tile_size(uint32_t w, uint32_t h) {
  // 4 tiles with a 1 pixel pad if tile size would be odd
  return std::pair{w / 2 + (w % 2), h / 2 + (h % 2)};
}

void VulkanTextureTransferContext::updateTileQueue(uint32_t width, uint32_t height) {
  if (width != tile_queue_width_ || height != tile_queue_height_) {
    tile_queue_width_ = width;
    tile_queue_height_ = height;

    // check if entire image will fit in a single tile
    uint64_t image_mem_size = width * height * kBytesPerPixel;
    if (kUseTileQueue && image_mem_size > tile_buffer_mem_size_) {
      // Image too large, using tiling
      auto [tile_width, tile_height] = compute_tile_size(width, height);
      // ensure size fits buffer
      CHECK_LE(tile_width * tile_height * kBytesPerPixel, tile_buffer_mem_size_);
      gfx::build_tile_queue(tile_queue_, width, height, tile_width, tile_height);
#if PRINT_BUFFER_INFO
      std::cout << tile_queue_.size() << " tiles "
                << "w:" << tile_width << "  h:" << tile_height << std::endl;
#endif
    } else {
      // single tile of image size
      gfx::build_tile_queue(tile_queue_, width, height, width, height);
#if PRINT_BUFFER_INFO
      std::cout << "1 tile  "
                << "w:" << width << "  h:" << height << std::endl;
#endif
    }
  }
}

// memory equivalent of 2 x 512x512 tiles (2Mb)
static constexpr uint64_t kMemSizeTilingThreshold = 1024u * 512u * kBytesPerPixel;
void VulkanTextureTransferContext::buildResources(uint32_t width, uint32_t height) {
  destroyResources();

  width_ = width;
  height_ = height;

  // Generate tile queue
  uint32_t image_mem_size = width * height * kBytesPerPixel;

  // Compute transfer buffer size
  // Check if image mem size > tiling threshold
  if (kUseTileQueue && image_mem_size > kMemSizeTilingThreshold) {
    // Large transfer buffer - use double buffer with tiling
    // Generate 4 tiles of equal size, adding 1 pixel pad for odd image dimensions
    auto [tile_width, tile_height] = compute_tile_size(width, height);
    tile_buffer_mem_size_ = tile_width * tile_height * kBytesPerPixel;
    num_buffers_ = kUseDoubleBuffer ? 2 : 1;
#if PRINT_BUFFER_INFO
    std::cout << "buffers:" << num_buffers_
              << "  total bufsize:" << tile_buffer_mem_size_ * num_buffers_ << std::endl;
#endif
  } else {
    // Small transfer buffer - use single buffer, no tiling
    tile_buffer_mem_size_ = image_mem_size;
    num_buffers_ = 1;
#if PRINT_BUFFER_INFO
    std::cout << "buffers:" << num_buffers_ << "  total bufsize:" << tile_buffer_mem_size_
              << std::endl;
#endif
  }

  // Destination / target resources
  CudaStateGuard cuda_state_guard(dest_cuda_ctx_, dest_device_ctx_.getGpuId());
  gfx::BufferCreateInfo buffer_ci = {};

  buffer_ci.access_type = gfx::BufferAccessType::kExternalApi;
  buffer_ci.size = tile_buffer_mem_size_;

  dst_resources_.resize(num_buffers_);
  for (auto& r : dst_resources_) {
    r.peer_copy_buffer = dest_device_ctx_.getResourceManager().createBaseBuffer(
        "TransferContext peer copy dest", buffer_ci);
    r.peer_copy_vk_buffer =
        reinterpret_cast<VkBuffer>(r.peer_copy_buffer->getResourceHandle());

    init_cuda_vulkan_buffer(
        dest_gpu_id_, *r.peer_copy_buffer, r.peer_copy_cuda_resources, true);

    init_cuda_vulkan_semaphore(dest_device_ctx_,
                               r.vk_vulkan_done_semaphore,
                               r.cu_vulkan_done_semaphore,
                               std::string(name_) + ", Vulkan Done");
    init_cuda_vulkan_semaphore(dest_device_ctx_,
                               r.vk_cuda_done_semaphore,
                               r.cu_cuda_done_semaphore,
                               std::string(name_) + ", CUDA Done");

    // Use Cuda to signal Vulkan semaphore indicating the destination buffer is available
    CHECK_RENDER_CUDA_ERRORS(
        cuSignalExternalSemaphoresAsync(
            &r.cu_vulkan_done_semaphore, &cuda_signal_params, 1, dst_cuda_stream_),
        dest_gpu_id_);

    CHECK_RENDER_CUDA_ERRORS(
        cuEventCreate(&r.peer_copy_complete_event, CU_EVENT_DISABLE_TIMING),
        dest_gpu_id_);
  }

  // Source resources
  // Only needed if src_gpu != dest_gpu
  for (auto& gpu_data_itr : source_gpu_data_) {
    auto& gpu_data = gpu_data_itr.second;
    // base name for source resources
    const std::string base_name("comp src " + std::to_string(gpu_data.gpu_id));

    if (gpu_data.gpu_id != dest_gpu_id_) {
      CudaStateGuard cuda_state_guard(gpu_data.cuda_ctx, gpu_data.gpu_id);

      // Cuda stream and Vulkan semaphore for synchronizing commands on the source gpu
      CHECK_RENDER_CUDA_ERRORS(
          cuStreamCreate(&gpu_data.cuda_stream, CU_STREAM_NON_BLOCKING), gpu_data.gpu_id);
      nvtx_helpers::name_cuda_stream(gpu_data.cuda_stream, base_name.c_str());

      // Create semaphore to signal src image ready for copy to buffer
      // e.g. SeparateMultiSamplesPass is finished extracting the sample
      gpu_data.vk_src_get_next_sample_semaphore =
          init_vulkan_semaphore(gpu_data.device_ctx, base_name + " src get next sample");

      gpu_data.resources.resize(num_buffers_);
      for (auto& r : gpu_data.resources) {
        r.peer_copy_buffer = gpu_data.device_ctx.getResourceManager().createBaseBuffer(
            "TransferContext peer copy src", buffer_ci);
        r.peer_copy_vk_buffer =
            reinterpret_cast<VkBuffer>(r.peer_copy_buffer->getResourceHandle());
        init_cuda_vulkan_buffer(
            gpu_data.gpu_id, *r.peer_copy_buffer, r.peer_copy_cuda_resources, false);

        // Semaphore to signal src gpu when peer copy is done
        init_cuda_vulkan_semaphore(gpu_data.device_ctx,
                                   r.vk_peer_copy_done_semaphore,
                                   r.cu_peer_copy_done_semaphore,
                                   base_name + " peer copy done");

        // Semaphore to signal when src image to buffer copy done
        init_cuda_vulkan_semaphore(gpu_data.device_ctx,
                                   r.vk_src_begin_peer_copy_semaphore,
                                   r.cu_src_begin_peer_copy_semaphore,
                                   base_name + " begin peer copy");

        CHECK_RENDER_CUDA_ERRORS(
            cuEventCreate(&r.src_begin_peer_copy_event, CU_EVENT_DISABLE_TIMING),
            gpu_data.gpu_id);
      }
    }
  }
}

void VulkanTextureTransferContext::destroyResources() {
  // Destination / target resources
  CudaStateGuard cuda_state_guard(dest_cuda_ctx_, dest_gpu_id_);

  for (auto& r : dst_resources_) {
    destroy_cuda_vulkan_semaphore(
        dest_device_ctx_, r.vk_vulkan_done_semaphore, r.cu_vulkan_done_semaphore);
    destroy_cuda_vulkan_semaphore(
        dest_device_ctx_, r.vk_cuda_done_semaphore, r.cu_cuda_done_semaphore);

    destroy_cuda_imported_memory(r.peer_copy_cuda_resources, dest_gpu_id_);
    if (r.peer_copy_buffer) {
      dest_device_ctx_.getResourceManager().destroyBaseBuffer(
          std::move(r.peer_copy_buffer));
    }
    if (r.peer_copy_complete_event) {
      CHECK_RENDER_CUDA_ERRORS(cuEventDestroy(r.peer_copy_complete_event), dest_gpu_id_);
      r.peer_copy_complete_event = nullptr;
    }
  }

  // Source resources
  for (auto& gpu_data_itr : source_gpu_data_) {
    auto& gpu_data = gpu_data_itr.second;
    auto gpu_id = gpu_data.gpu_id;
    CudaStateGuard cuda_state_guard(gpu_data.cuda_ctx, gpu_id);
    for (auto& r : gpu_data.resources) {
      if (r.peer_copy_buffer) {
        destroy_cuda_imported_memory(r.peer_copy_cuda_resources, gpu_id);
        gpu_data.device_ctx.getResourceManager().destroyBaseBuffer(
            std::move(r.peer_copy_buffer));
      }

      destroy_cuda_vulkan_semaphore(gpu_data.device_ctx,
                                    r.vk_peer_copy_done_semaphore,
                                    r.cu_peer_copy_done_semaphore);
      destroy_cuda_vulkan_semaphore(gpu_data.device_ctx,
                                    r.vk_src_begin_peer_copy_semaphore,
                                    r.cu_src_begin_peer_copy_semaphore);
      if (r.src_begin_peer_copy_event) {
        CHECK_RENDER_CUDA_ERRORS(cuEventDestroy(r.src_begin_peer_copy_event),
                                 gpu_data.gpu_id);
      }
    }
    if (gpu_data.vk_src_get_next_sample_semaphore) {
      vkDestroySemaphore(gpu_data.device_ctx.getHandle(),
                         gpu_data.vk_src_get_next_sample_semaphore,
                         nullptr);
      gpu_data.vk_src_get_next_sample_semaphore = VK_NULL_HANDLE;
    }

    if (gpu_data.cuda_stream) {
      CHECK_RENDER_CUDA_ERRORS(cuStreamDestroy(gpu_data.cuda_stream), gpu_id);
      gpu_data.cuda_stream = nullptr;
    }
  }
}

void VulkanTextureTransferContext::registerSourceDevice(
    const RootPerGpuData& gpu_data,
    const std::vector<const gfx::Texture*>& src_id_textures) {
  auto const& device_ctx =
      static_cast<const gfx::VulkanDeviceContext&>(gpu_data.getDeviceContext());
  auto gpu_id = device_ctx.getGpuId();
  auto cuda_ctx = cuda_mgr_->getDeviceContexts().at(gpu_id - start_gpu_id_);
  CHECK(cuda_ctx);

  auto [new_data_itr, did_emplace] =
      source_gpu_data_.try_emplace(gpu_id, device_ctx, cuda_ctx);
  CHECK(did_emplace) << "Failed to register source gpu";
  auto& new_data = new_data_itr->second;

  // Get SeparateMultiSamplesPass image
  auto const& rgba_tex = global_ctx_.getSeparateMultiSamplesPass()->getTexture(gpu_id);
  new_data.rgba_vk_image = reinterpret_cast<VkImage>(rgba_tex.getResourceHandle());

  // ID images
  new_data.id_vk_images.reserve(src_id_textures.size());
  for (auto const& tex : src_id_textures) {
    new_data.id_vk_images.push_back(reinterpret_cast<VkImage>(tex->getResourceHandle()));
  }
}

void VulkanTextureTransferContext::waitComplete() {
  vk_cmd_pool_.waitForPendingBuffers([this](VkResult result) { waitCallback(result); });
}

static void log_cuda_stream_status(CUstream stream, std::ostream& os) {
  auto status = cuStreamQuery(stream);
  switch (status) {
    case CUDA_SUCCESS:
      os << "No pending operations";
      break;
    case CUDA_ERROR_NOT_READY:
      os << "Operations are pending";
      break;
    default: {
      const char* cuda_error_string{nullptr};
      cuGetErrorString(status, &cuda_error_string);
      os << cuda_error_string;
    }
  }
}

static void log_cuda_event_status(CUevent event, std::ostream& os) {
  auto status = cuEventQuery(event);
  if (status == CUDA_SUCCESS) {
    os << "None recorded";
  } else if (status == CUDA_ERROR_NOT_READY) {
    os << "Pending";
  } else {
    const char* cuda_error_string{nullptr};
    cuGetErrorString(status, &cuda_error_string);
    os << cuda_error_string;
  }
}

void VulkanTextureTransferContext::waitCallback(VkResult result) {
  std::stringstream ss;
  ss << "Compositor command execution state\n";
  ss << "----------------------\n";

  ss << "Compositor GPU\n";
  // Command pools
  const gfx::VulkanCommandPool& pool =
      static_cast<const gfx::VulkanDeviceContext&>(dest_device_ctx_)
          .getCommandPool(gfx::VulkanDeviceContext::CommandPoolSelector::kExecutor);
  pool.logPendingBufferNames(ss);

  ss << "\nCuda resources\n";
  ss << "  Stream status: ";
  log_cuda_stream_status(dst_cuda_stream_, ss);
  ss << "\n";
  for (auto const& resources : dst_resources_) {
    ss << "  Copy complete event status: ";
    log_cuda_event_status(resources.peer_copy_complete_event, ss);
    ss << "\n";
  }

  ss << "\nPeer GPUs\n";
  for (auto const& [src_gpu_id, src_data] : source_gpu_data_) {
    if (src_gpu_id != dest_gpu_id_) {
      ss << "GPU " << src_gpu_id << "\n";
      // Command pools
      const gfx::VulkanCommandPool& pool =
          static_cast<const gfx::VulkanDeviceContext&>(src_data.device_ctx)
              .getCommandPool(gfx::VulkanDeviceContext::CommandPoolSelector::kExecutor);
      pool.logPendingBufferNames(ss);

      ss << "\nCuda resources\n";
      ss << "  stream status: ";
      log_cuda_stream_status(src_data.cuda_stream, ss);
      ss << "\n";
      for (auto const& resources : src_data.resources) {
        ss << "  Begin copy event status: ";
        log_cuda_event_status(resources.src_begin_peer_copy_event, ss);
        ss << "\n";
      }
      ss << "----------------------\n";
    }
  }
  VLOG(1) << ss.str();
}

// During and RGBA or ID composite, when the src gpu is the same as the compositor gpu,
// skip Cuda and just copy the textures into the compositor texture
// TODO(scb): skip the copy entirely similar to accumulation
void VulkanTextureTransferContext::copyLocalTextures(
    SyncType sync_type,
    const SourceGpuData& src_gpu_data,
    const std::vector<VkImage>& dst_images) {
  RENDER_LOG_SCOPE();

  uint32_t num_textures = 0;
#if SKIP_LAYOUT_TRANSITIONS == 0
  if (sync_type == SyncType::kID) {
    auto& image_layout_mgr =
        static_cast<gfx::VulkanResourceManager*>(&dest_device_ctx_.getResourceManager())
            ->getImageLayoutManager();

    auto* cmd_buffer = vk_cmd_pool_.acquireBuffer();
    image_layout_mgr.transitionToLayout(src_gpu_data.id_vk_images,
                                        1,
                                        std::nullopt,
                                        gfx::ImageLayout::kTransferSrc,
                                        *cmd_buffer,
                                        std::nullopt,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT);

    vk_cmd_pool_.submitBuffer(cmd_buffer, "TTC copy local - layout transition ID");
    num_textures = src_gpu_data.id_vk_images.size();
  } else {
    num_textures = global_ctx_.getNumSamples();
  }

#endif
  // CHECK_LE(num_textures, dst_images.size());

  VkImageCopy copy_region{};
  copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.srcSubresource.layerCount = 1;
  copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.dstSubresource.layerCount = 1;
  copy_region.extent.width = width_;
  copy_region.extent.height = height_;
  copy_region.extent.depth = 1;

  for (uint32_t i = 0; i < num_textures; ++i) {
    auto* cmd_buffer = vk_cmd_pool_.acquireBuffer();
    auto vk_image = (sync_type == SyncType::kRGBA) ? src_gpu_data.rgba_vk_image
                                                   : src_gpu_data.id_vk_images[i];
    if (sync_type == SyncType::kRGBA) {
      global_ctx_.getSeparateMultiSamplesPass()->runPass(
          src_gpu_data.gpu_id,
          src_gpu_data.device_ctx.getCommandList(),
          SeparateMultiSamplesPass::SourceFramebuffer::kRender,
          SeparateMultiSamplesPass::OutputUsage::kTransferSrc,
          i,
          {},
          {});
    }

    vkCmdCopyImage(cmd_buffer->getHandle(),
                   vk_image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst_images[i],
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1,
                   &copy_region);

    vk_cmd_pool_.submitBuffer(cmd_buffer, "TTC copy local - copy image");
  }
}

void VulkanTextureTransferContext::transitionSrcTextures(
    SourceGpuData& src_gpu_data,
    const std::vector<VkImage>& src_images,
    gfx::ImageLayout layout) {
  // Use source's executor command pool to transition
  auto& src_cmd_pool = src_gpu_data.device_ctx.getCommandPool(
      gfx::VulkanDeviceContext::CommandPoolSelector::kExecutor);
  auto& src_layout_mgr = static_cast<gfx::VulkanResourceManager*>(
                             &src_gpu_data.device_ctx.getResourceManager())
                             ->getImageLayoutManager();

  auto* cmd_buffer = src_cmd_pool.acquireBuffer();
  src_layout_mgr.transitionToLayout(src_images,
                                    1,
                                    std::nullopt,
                                    gfx::ImageLayout::kTransferSrc,
                                    *cmd_buffer,
                                    std::nullopt,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT);
  src_cmd_pool.submitBuffer(cmd_buffer, "TTC src transition layout");
}

void VulkanTextureTransferContext::copySrcImageToSrcBuffer(
    SourceGpuData& src_gpu_data,
    VkImage src_image,
    uint32_t src_layer_index,
    const gfx::Rect2D& src_region,
    VkImageLayout src_layout,
    VkBuffer src_buffer,
    VkSemaphore wait_semaphore,
    const std::vector<VkSemaphore>& signal_semaphores,
    const std::vector<VkPipelineStageFlags2>& signal_stages) {
  CHECK_EQ(signal_semaphores.size(), signal_stages.size());
  VkBufferImageCopy copy_region = {};
  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageSubresource.baseArrayLayer = src_layer_index;
  copy_region.imageOffset.x = src_region.x;
  copy_region.imageOffset.y = src_region.y;
  copy_region.imageExtent.width = src_region.w;
  copy_region.imageExtent.height = src_region.h;
  copy_region.imageExtent.depth = 1;

  auto& src_cmd_pool = src_gpu_data.device_ctx.getCommandPool(
      gfx::VulkanDeviceContext::CommandPoolSelector::kExecutor);
  auto* cmd_buffer = src_cmd_pool.acquireBuffer();

  vkCmdCopyImageToBuffer(
      cmd_buffer->getHandle(), src_image, src_layout, src_buffer, 1, &copy_region);

  static std::vector<VkPipelineStageFlags2> wait_stages{VK_PIPELINE_STAGE_2_TRANSFER_BIT};

  src_cmd_pool.submitBuffer(cmd_buffer,
                            "TTC src copy image to buffer",
                            1,
                            &wait_semaphore,
                            wait_stages,
                            signal_semaphores.size(),
                            signal_semaphores.data(),
                            signal_stages);
}

void VulkanTextureTransferContext::copyDstBufferToDstImage(
    VkImage dst_image,
    const gfx::Rect2D& dst_region,
    VkImageLayout dst_layout,
    const DestResources& dst_resources,
    bool do_signal_cuda) {
  auto* cmd_buffer = vk_cmd_pool_.acquireBuffer();

  VkBufferImageCopy copy_region = {};
  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageOffset.x = dst_region.x;
  copy_region.imageOffset.y = dst_region.y;
  copy_region.imageExtent.width = dst_region.w;
  copy_region.imageExtent.height = dst_region.h;
  copy_region.imageExtent.depth = 1;

  vkCmdCopyBufferToImage(cmd_buffer->getHandle(),
                         dst_resources.peer_copy_vk_buffer,
                         dst_image,
                         dst_layout,
                         1,
                         &copy_region);

  // Wait for cuda to complete peer to peer copy, then signal when the local
  // buffer to image copy is done so the next peer to peer can begin
  static std::vector<VkPipelineStageFlags2> stages{VK_PIPELINE_STAGE_2_TRANSFER_BIT};

  if (do_signal_cuda) {
    vk_cmd_pool_.submitBuffer(cmd_buffer,
                              "TTC dst copy buffer to image",
                              1,
                              &dst_resources.vk_cuda_done_semaphore,
                              stages,
                              1,
                              &dst_resources.vk_vulkan_done_semaphore,
                              stages);
  } else {
    vk_cmd_pool_.submitBuffer(cmd_buffer,
                              "TTC dst copy buffer to image",
                              1,
                              &dst_resources.vk_cuda_done_semaphore,
                              stages,
                              0,
                              nullptr,
                              {});
  }
}

std::vector<VkImage> get_images_from_textures(
    const std::vector<gfx::resource_ptr<gfx::Texture>>& textures) {
  std::vector<VkImage> images;
  for (auto const& texture : textures) {
    images.push_back(reinterpret_cast<VkImage>(texture->getResourceHandle()));
  }
  return images;
}

void VulkanTextureTransferContext::swapBuffers() {
  if (num_buffers_ == 2) {
    active_buffer_index_ = 1 - active_buffer_index_;
  }
}

// RGBA or ID composite
// For local textures on the compositor gpu, use Vulkan to copy into place
// For other gpus loop over textures, for each texture:
// - use vulkan to copy image to buffer on source
// - use cuda to memcpy the buffer to an equivalent buffer on the dest (comp) gpu
// - use vulkan to copy buffer to image for compositing
void VulkanTextureTransferContext::copyTexturesFromDevice(
    GpuId source_gpu_id,
    SyncType sync_type,
    const std::vector<gfx::resource_ptr<gfx::Texture>>& dst_textures_) {
  RENDER_LOG_SCOPE();
  auto src_data_itr = source_gpu_data_.find(source_gpu_id);
  CHECK(src_data_itr != source_gpu_data_.end());
  auto& src_gpu_data = src_data_itr->second;

  auto dst_vk_images = get_images_from_textures(dst_textures_);

// Rely on queue submission order guarantees. For safety we should wait on semaphores but
// I'd prefer to save that for a timeline semaphore for simplicity
// TODO(scb): remove this code in the tiling PR if everything remains reliable
#if 0
  if (sync_type == SyncType::kID) {
    // Ensure staging buffer has completed ID blit (this starts before RGBA/accum so
    // should return immediately)
    auto& staging = src_gpu_data.device_ctx.getStagingContext();
    staging.waitForCompletion();

    // Wait for RGBA / Accumulation transfers to complete
    vk_cmd_pool_.waitForPendingBuffers();
  }
#endif

#if PROFILE_TRANSFER
  auto clock_begin = timer_start();
#endif  // PROFILE_TRANSFER

  // Copy all ID textures or extract all image samples for RGBA
  auto num_textures = sync_type == SyncType::kID ? src_gpu_data.id_vk_images.size()
                                                 : global_ctx_.getNumSamples();
  CHECK_LE(num_textures, dst_textures_.size());

  CudaStateGuard state_guard(dest_cuda_ctx_, dest_gpu_id_);

#if SKIP_LAYOUT_TRANSITIONS == 0
  auto& dest_layout_mgr =
      static_cast<gfx::VulkanResourceManager*>(&dest_device_ctx_.getResourceManager())
          ->getImageLayoutManager();
  {
    // Transition compositor texture to transfer dst
    auto* cmd_buffer = vk_cmd_pool_.acquireBuffer();
    dest_layout_mgr.transitionToLayout(dst_vk_images,
                                       1,
                                       std::nullopt,
                                       gfx::ImageLayout::kTransferDst,
                                       *cmd_buffer,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT);
    static std::string_view kFlushName{"TTC dst transition layout pre copy"};
    // No semaphores required since the destination images will only be modified on the
    // same queue (submission order guarantee)
    vk_cmd_pool_.submitBuffer(cmd_buffer, kFlushName);
  }
#endif

  // Do the copies
  // Source or dest context can be active for this call
  if (src_gpu_data.gpu_id == dest_gpu_id_) {
    copyLocalTextures(sync_type, src_gpu_data, dst_vk_images);
  } else {
#if SKIP_LAYOUT_TRANSITIONS == 0
    if (sync_type == SyncType::kID) {
      // transition the ID buffers
      // For RGBA SeparateMultiSamplesPass will handle this in the RenderPass
      transitionSrcTextures(
          src_gpu_data, src_gpu_data.id_vk_images, gfx::ImageLayout::kTransferSrc);
    }
#endif

    static std::vector<gfx::SemaphoreHandle> no_semaphores;
    std::vector<gfx::SemaphoreHandle> extract_wait_semaphore{
        reinterpret_cast<gfx::SemaphoreHandle>(
            src_gpu_data.vk_src_get_next_sample_semaphore)};

    for (uint32_t i = 0; i < num_textures; ++i) {
      bool is_first_tile{true};
      VkImage src_vk_image{VK_NULL_HANDLE};
      for (const auto& tile : tile_queue_) {
        swapBuffers();

        auto const& dst_resources = dst_resources_[active_buffer_index_];
        auto const& src_resources = src_gpu_data.resources[active_buffer_index_];

        auto const& dst_memory = dst_resources.peer_copy_cuda_resources;
        auto const& src_memory = src_resources.peer_copy_cuda_resources;

        // wait for any previous copy to complete before starting src image to buffer copy
        // (NOTE: on the very first copy after construction this event will be empty and
        // the wait will pass immediately)
        CHECK_RENDER_CUDA_ERRORS(
            cuStreamWaitEvent(src_gpu_data.cuda_stream,
                              dst_resources.peer_copy_complete_event,
                              0u),  // use CU_EVENT_WAIT_DEFAULT in cuda 11.1+
            src_gpu_data.gpu_id);
        CHECK_RENDER_CUDA_ERRORS(
            cuSignalExternalSemaphoresAsync(&src_resources.cu_peer_copy_done_semaphore,
                                            &cuda_signal_params,
                                            1,
                                            src_gpu_data.cuda_stream),
            src_gpu_data.gpu_id);

        // --- Source GPU ---
        // Copy the src texture to its intermediate buffer
        // Wait on previous peer-to-peer copy if not first loop iteration
        // Signal cuda when done so it can set the event on the source gpu
        std::vector<VkSemaphore> signal_semaphores{
            src_resources.vk_src_begin_peer_copy_semaphore};
        std::vector<VkPipelineStageFlags2> signal_stages{
            VK_PIPELINE_STAGE_2_TRANSFER_BIT};

        if (is_first_tile) {
          if (sync_type == SyncType::kRGBA) {
            src_vk_image = src_gpu_data.rgba_vk_image;
            global_ctx_.getSeparateMultiSamplesPass()->runPass(
                src_gpu_data.gpu_id,
                src_gpu_data.device_ctx.getCommandList(),
                SeparateMultiSamplesPass::SourceFramebuffer::kRender,
                SeparateMultiSamplesPass::OutputUsage::kTransferSrc,
                i,
                // Wait until the previous image to buffer copy is done for each buffer
                i > 0 ? extract_wait_semaphore : no_semaphores,
                no_semaphores);

            // signal SeparateMultiSamplesPass to start next sample extraction
            if (i < (num_textures - 1)) {
              signal_semaphores.push_back(src_gpu_data.vk_src_get_next_sample_semaphore);
              signal_stages.push_back(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
            }
          } else {
            src_vk_image = src_gpu_data.id_vk_images[i];
          }
          is_first_tile = false;
        }

        copySrcImageToSrcBuffer(src_gpu_data,
                                src_vk_image,
                                0,
                                tile,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                src_resources.peer_copy_vk_buffer,
                                src_resources.vk_peer_copy_done_semaphore,
                                signal_semaphores,
                                signal_stages);

        // Wait for src image to copy to the buffer
        CHECK_RENDER_CUDA_ERRORS(
            cuWaitExternalSemaphoresAsync(&src_resources.cu_src_begin_peer_copy_semaphore,
                                          &cuda_wait_params,
                                          1,
                                          src_gpu_data.cuda_stream),
            src_gpu_data.gpu_id);

        // Signal that the source image to buffer is complete so peer-to-peer can begin
        CHECK_RENDER_CUDA_ERRORS(cuEventRecord(src_resources.src_begin_peer_copy_event,
                                               src_gpu_data.cuda_stream),
                                 src_gpu_data.gpu_id);

        // --- Destination GPU ---
        // Wait for source gpu image to buffer copy to complete on the destination stream
        CHECK_RENDER_CUDA_ERRORS(
            cuStreamWaitEvent(dst_cuda_stream_,
                              src_resources.src_begin_peer_copy_event,
                              0u),  // use CU_EVENT_WAIT_DEFAULT in cuda 11.1+
            dest_gpu_id_);

        // Wait for vulkan to copy the destination buffer to its final image target
        CHECK_RENDER_CUDA_ERRORS(
            cuWaitExternalSemaphoresAsync(&dst_resources.cu_vulkan_done_semaphore,
                                          &cuda_wait_params,
                                          1,
                                          dst_cuda_stream_),
            dest_gpu_id_);

        // Start the copy
        uint64_t tile_mem_size = tile.w * tile.h * 4u;
        CHECK_RENDER_CUDA_ERRORS(
            cuMemcpyDtoDAsync(dst_memory.device_ptr,
                              src_memory.device_ptr,
                              std::min(tile_mem_size, dst_memory.memory_size),
                              dst_cuda_stream_),
            dest_gpu_id_);

        // Signal peer-to-peer event complete so the next source image to buffer copy can
        // begin
        CHECK_RENDER_CUDA_ERRORS(
            cuEventRecord(dst_resources.peer_copy_complete_event, dst_cuda_stream_),
            dest_gpu_id_);

        // Signal Vulkan when done
        CHECK_RENDER_CUDA_ERRORS(
            cuSignalExternalSemaphoresAsync(&dst_resources.cu_cuda_done_semaphore,
                                            &cuda_signal_params,
                                            1,
                                            dst_cuda_stream_),
            dest_gpu_id_);

        // Copy the buffer or texture into the destination texture
        // Signals Cuda when done
        copyDstBufferToDstImage(dst_vk_images[i],
                                tile,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                dst_resources,
                                true);
      }
    }
  }

#if SKIP_LAYOUT_TRANSITIONS == 0
  {
    auto* cmd_buffer = vk_cmd_pool_.acquireBuffer();
    dest_layout_mgr.transitionToLayout(dst_vk_images,
                                       1,
                                       std::nullopt,
                                       gfx::ImageLayout::kShaderReadOnly,
                                       *cmd_buffer,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vk_cmd_pool_.submitBuffer(cmd_buffer, "TTC dst transition layout post copy");
  }
#endif

#if PROFILE_TRANSFER
  vk_cmd_pool_.waitForPendingBuffers();

  std::cout << "[" << std::setw(4) << std::left << name_ << "] "
            << "Copying from gpu " << src_gpu_data.gpu_id << " to gpu " << dest_gpu_id_;

  auto const wall_time =
      timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
          clock_begin);
  std::cout << " - Textures (" << num_textures << ") transfer time: " << wall_time
            << std::endl;
#endif
}

// Accumulation composite
// Local textures are handled directly by compositor
// For other gpus loop over texture array layers, for each layer:
// - use vulkan to copy image layer to buffer on source
// - use cuda to memcpy the buffer to an equivalent buffer on the dest (comp) gpu
// - use vulkan to copy buffer to image layer for compositing into main accum image
void VulkanTextureTransferContext::copyTextureArrayFromDevice(
    GpuId src_gpu_id,
    const gfx::Texture& src_texture_array,
    const gfx::Texture& dst_texture,
    uint32_t num_layers_to_copy,
    ArrayLayerTransferredCBFunc layer_transfer_cb) {
  RENDER_LOG_SCOPE();

  CHECK_LE(src_texture_array.getWidth(), width_);
  CHECK_LE(src_texture_array.getHeight(), height_);
  CHECK_LE(num_layers_to_copy, src_texture_array.getDepth());
  CHECK_EQ(src_texture_array.getPixelFormat(), dst_texture.getPixelFormat());
  CHECK(layer_transfer_cb);

  // Get source gpu data
  auto src_data_itr = source_gpu_data_.find(src_gpu_id);
  CHECK(src_data_itr != source_gpu_data_.end());
  auto& src_gpu_data = src_data_itr->second;

  // Don't try and do a local copy, just use the src array directly for local
  // accumulation
  CHECK_NE(src_gpu_data.gpu_id, dest_gpu_id_)
      << "Attempting local device copy of accumulation TextureArray";

#if PROFILE_TRANSFER
  auto clock_begin = timer_start();
#endif  // PROFILE_TRANSFER

  auto src_image = reinterpret_cast<VkImage>(src_texture_array.getResourceHandle());
  auto dst_image = reinterpret_cast<VkImage>(dst_texture.getResourceHandle());

  CudaStateGuard state_guard(dest_cuda_ctx_, dest_gpu_id_);

  // Loop over each layer in the TextureArray
  // Copy the source layer into the source buffer using Vulkan
  // Copy the source buffer to the dest buffer using Cuda
  // Copy the dest buffer into the dest layer using Vulkan
  std::vector<gfx::SemaphoreHandle> vulkan_done_handles;
  for (auto& r : dst_resources_) {
    vulkan_done_handles.push_back(
        reinterpret_cast<gfx::SemaphoreHandle>(r.vk_vulkan_done_semaphore));
  }

  for (uint32_t i = 0; i < num_layers_to_copy; ++i) {
    uint32_t num_tiles_remaining = tile_queue_.size();
    for (const auto& tile : tile_queue_) {
      num_tiles_remaining--;
      swapBuffers();

      auto const& dst_resources = dst_resources_[active_buffer_index_];
      auto const& src_resources = src_gpu_data.resources[active_buffer_index_];

      auto const& dst_memory = dst_resources.peer_copy_cuda_resources;
      auto const& src_memory = src_resources.peer_copy_cuda_resources;

      // Wait for previous copy to complete before starting src image to buffer copy
      // Have the source stream wait on the event on the destination gpu, then
      // signal the vulkan semaphore on the source so it can start copying to
      // the buffer
      // (NOTE: on the very first copy after construction this event will be empty and the
      // wait will pass immediately)
      CHECK_RENDER_CUDA_ERRORS(
          cuStreamWaitEvent(src_gpu_data.cuda_stream,
                            dst_resources.peer_copy_complete_event,
                            0u),  // use CU_EVENT_WAIT_DEFAULT in cuda 11.1+
          src_gpu_data.gpu_id);

      CHECK_RENDER_CUDA_ERRORS(
          cuSignalExternalSemaphoresAsync(&src_resources.cu_peer_copy_done_semaphore,
                                          &cuda_signal_params,
                                          1,
                                          src_gpu_data.cuda_stream),
          src_gpu_data.gpu_id);

      // --- Source GPU ---
      // Copy the src texture to its intermediate buffer
      // Wait on previous peer-to-peer copy if not first loop iteration
      // Signal cuda when done so it can set the event on the source gpu
      copySrcImageToSrcBuffer(src_gpu_data,
                              src_image,
                              i,
                              tile,
                              VK_IMAGE_LAYOUT_GENERAL,
                              src_resources.peer_copy_vk_buffer,
                              src_resources.vk_peer_copy_done_semaphore,
                              {src_resources.vk_src_begin_peer_copy_semaphore},
                              {VK_PIPELINE_STAGE_2_TRANSFER_BIT});

      // Wait for src image to copy to the buffer
      CHECK_RENDER_CUDA_ERRORS(
          cuWaitExternalSemaphoresAsync(&src_resources.cu_src_begin_peer_copy_semaphore,
                                        &cuda_wait_params,
                                        1,
                                        src_gpu_data.cuda_stream),
          src_gpu_data.gpu_id);

      // Signal that the source image to buffer is complete so peer-to-peer can begin
      CHECK_RENDER_CUDA_ERRORS(cuEventRecord(src_resources.src_begin_peer_copy_event,
                                             src_gpu_data.cuda_stream),
                               src_gpu_data.gpu_id);

      // --- Destination GPU ---
      // Wait for source gpu image to buffer copy to complete on the destination stream
      CHECK_RENDER_CUDA_ERRORS(
          cuStreamWaitEvent(dst_cuda_stream_,
                            src_resources.src_begin_peer_copy_event,
                            0u),  // use CU_EVENT_WAIT_DEFAULT in cuda 11.1+
          dest_gpu_id_);

      // Wait for vulkan to copy the destination buffer to its final image target
      CHECK_RENDER_CUDA_ERRORS(
          cuWaitExternalSemaphoresAsync(&dst_resources.cu_vulkan_done_semaphore,
                                        &cuda_wait_params,
                                        1,
                                        dst_cuda_stream_),
          dest_gpu_id_);

      // Start the peer-to-peer copy
      uint64_t tile_mem_size = tile.w * tile.h * 4u;
      CHECK_RENDER_CUDA_ERRORS(
          cuMemcpyDtoDAsync(dst_memory.device_ptr,
                            src_memory.device_ptr,
                            std::min(tile_mem_size, dst_memory.memory_size),
                            dst_cuda_stream_),
          dest_gpu_id_);

      // Signal peer-to-peer event complete so the next source image to buffer copy can
      // begin
      CHECK_RENDER_CUDA_ERRORS(
          cuEventRecord(dst_resources.peer_copy_complete_event, dst_cuda_stream_),
          dest_gpu_id_);

      // Signal Vulkan on the destination when peer-to-peer is done so it can copy to the
      // final image
      CHECK_RENDER_CUDA_ERRORS(
          cuSignalExternalSemaphoresAsync(&dst_resources.cu_cuda_done_semaphore,
                                          &cuda_signal_params,
                                          1,
                                          dst_cuda_stream_),
          dest_gpu_id_);

      // Copy the buffer or texture into the destination texture. Signal the vulkan
      // done semaphore on all but the final tile(s). Once the final 1 or 2 tiles finish
      // the layer will be complete, and we need to call back to the compositor to comp
      // in the layer. Once the compositor finishes it will signal the semaphores,
      // allowing the next layer to begin copying.

      // This blocks the next cuda copy from starting since the same semaphore
      // (dst_vulkan_done) is serving dual purposes. Rather than adding ANOTHER semaphore
      // we'll just have a tiny inefficiency until timeline semaphores, or the compositor
      // works with tiles
      copyDstBufferToDstImage(dst_image,
                              tile,
                              VK_IMAGE_LAYOUT_GENERAL,
                              dst_resources,
                              num_tiles_remaining >= num_buffers_);
    }
    // Process the layer in the compositor
    // Signals Cuda when done so the next layer can be transferred
    layer_transfer_cb(i, vulkan_done_handles);
  }

#if PROFILE_TRANSFER
  vk_cmd_pool_.waitForPendingBuffers();

  std::cout << "[" << std::setw(4) << std::left << name_ << "] "
            << "Copying from gpu " << src_gpu_data.gpu_id << " to gpu " << dest_gpu_id_;

  auto const wall_time =
      timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(
          clock_begin);
  std::cout << " - Layers (" << num_layers_to_copy << ") transfer time: " << wall_time
            << std::endl;
#endif
}

}  // namespace QueryRenderer
