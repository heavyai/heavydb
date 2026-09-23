/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge_DefaultBackend.h"

#include <imgui/backends/imgui_impl_vulkan.h>

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandExecutor.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanQueue.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Drivers/Vulkan/WSI/VulkanWSI.h"

namespace gfx {

ImGuiBridge_DefaultBackend::ImGuiBridge_DefaultBackend(const DeviceContext& device,
                                                       const WindowSystemIntegration& wsi)
    : is_initialized_{false}, device_{device}, wsi_{wsi} {}

void ImGuiBridge_DefaultBackend::init(const RenderPass& render_pass,
                                      const uint32_t num_samples) {
  auto const& vk_device = static_cast<const VulkanDeviceContext&>(device_);

  // InitInfo for the ImGui backend
  // Note that many of these params are not used in practice, but are there to support
  // helper functions leveraged by a Vulkan demo app, however they must be valid
  // They also exist to support multiple viewports in the ImGui "Docking" branch
  // which we are not currently using
  // More info on multiple viewports here: https://github.com/ocornut/imgui/issues/1542
  ImGui_ImplVulkan_InitInfo info = {};
  info.Instance = static_cast<const VulkanWSI&>(wsi_).getVkInstance();
  info.PhysicalDevice = vk_device.getPhysicalDeviceHandle();
  info.Device = vk_device.getHandle();

  auto const& graphics_queue = vk_device.getGraphicsQueue();
  info.QueueFamily = graphics_queue.getFamilyIndex();
  info.Queue = graphics_queue.getHandle();
  info.PipelineCache = VK_NULL_HANDLE;
  info.Subpass = 0;
  info.MinImageCount = 2;  // unused
  info.ImageCount = 2;     // unused
  info.MSAASamples = num_samples_to_vk_sample_flag_bits(num_samples);
  info.Allocator = nullptr;
  info.CheckVkResultFn = nullptr;  // TODO: pass function wrapper for CHECK_VKRESULT

  // Create the Descriptor pool
  // Only require a single combined image sampler for the fragment shader
  // to use with the font atlas image
  VkDescriptorPoolSize pool_size = {};
  pool_size.descriptorCount = 1;
  pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

  VkDescriptorPoolCreateInfo pool_ci = {};
  pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_ci.poolSizeCount = 1;
  pool_ci.pPoolSizes = &pool_size;
  pool_ci.maxSets = 1;

  CHECK_VKRESULT(vkCreateDescriptorPool(
                     vk_device.getHandle(), &pool_ci, nullptr, &vk_descriptor_pool_),
                 "creating imgui descriptor pool");

  vk_device.nameVulkanObject(
      VK_OBJECT_TYPE_DESCRIPTOR_POOL, vk_descriptor_pool_, "ImGui descriptor pool");

  info.DescriptorPool = vk_descriptor_pool_;

  // Init the ImGui Vulkan backend
  ImGui_ImplVulkan_Init(&info,
                        reinterpret_cast<VkRenderPass>(render_pass.getResourceHandle()));

  // Create and upload fonts texture
  // Requires a valid VkCommandBuffer
  VulkanCommandExecutor& executor =
      static_cast<VulkanCommandExecutor&>(device_.getCommandExecutor());
  executor.beginCommandSequence();
#if IMGUI_VERSION_NUM < 19000
  ImGui_ImplVulkan_CreateFontsTexture(executor.getActiveCommandBuffer().getHandle());
#else
  ImGui_ImplVulkan_CreateFontsTexture();
#endif
  executor.submitCommandSequence(
      "ImGui fonts", CommandList::SubmitType::kWaitComplete, {}, {});
  is_initialized_ = true;
}

void ImGuiBridge_DefaultBackend::shutdown() {
  if (is_initialized_) {
    ImGui_ImplVulkan_Shutdown();
  }
  if (vk_descriptor_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(static_cast<const VulkanDeviceContext&>(device_).getHandle(),
                            vk_descriptor_pool_,
                            nullptr);
    vk_descriptor_pool_ = VK_NULL_HANDLE;
  }
}

void ImGuiBridge_DefaultBackend::newFrame() {
  ImGui_ImplVulkan_NewFrame();
}

void ImGuiBridge_DefaultBackend::draw(ImDrawData* draw_data,
                                      RenderPass& render_pass,
                                      Framebuffer& framebuffer) {
  VulkanCommandExecutor& executor =
      static_cast<VulkanCommandExecutor&>(device_.getCommandExecutor());
  executor.beginCommandSequence();
  auto vk_cmd_buffer = executor.getActiveCommandBuffer().getHandle();

  executor.beginRenderPass(render_pass, framebuffer);
  ImGui_ImplVulkan_RenderDrawData(draw_data, vk_cmd_buffer);
  executor.endRenderPass();

  executor.submitCommandSequence(
      "ImGui render", CommandList::SubmitType::kWaitComplete, {}, {});
}

}  // namespace gfx
