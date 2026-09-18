/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanHostVisibleBufferWrapper.h"

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

VulkanHostVisibleBufferWrapper::VulkanHostVisibleBufferWrapper(
    BufferWrapperUqPtr source_buffer_wrapper)
    : HostVisibleBufferWrapper(std::move(source_buffer_wrapper)) {}

VulkanHostVisibleBufferWrapper::~VulkanHostVisibleBufferWrapper() {
  if (is_mapped_) {
    unmap();
  }
}

void VulkanHostVisibleBufferWrapper::map(void** data) {
  CHECK(!is_mapped_);
  CHECK(source_buffer_wrapper_);
  mapImpl(static_cast<VulkanBaseBuffer&>(getSourceBuffer()), data);
  is_mapped_ = true;
}

void* VulkanHostVisibleBufferWrapper::map() {
  void* data{nullptr};
  map(&data);
  return data;
}

void VulkanHostVisibleBufferWrapper::unmap() {
  CHECK(is_mapped_);
  unmapImpl(static_cast<VulkanBaseBuffer&>(getSourceBuffer()));
  is_mapped_ = false;
}

void VulkanHostVisibleBufferWrapper::mapImpl(VulkanBaseBuffer& vk_buffer, void** data) {
  CHECK(data);
  CHECK(vk_buffer.vulkan_allocation_);
  CHECK(buffer_access_type_to_memory_properties_bits(vk_buffer.getAccessType()) &
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  auto const* vk_device =
      static_cast<const VulkanDeviceContext*>(&vk_buffer.getDeviceContext());

  VkResult result = vkMapMemory(vk_device->getHandle(),
                                vk_buffer.vulkan_allocation_->getHandle(),
                                0,
                                vk_buffer.vk_memory_requirements_.size,
                                0,
                                data);
  CHECK_VKRESULT(result, "mapping buffer");
}

void VulkanHostVisibleBufferWrapper::unmapImpl(VulkanBaseBuffer& vk_buffer) {
  CHECK(vk_buffer.vulkan_allocation_);
  auto const* vk_device =
      static_cast<const VulkanDeviceContext*>(&vk_buffer.getDeviceContext());
  vkUnmapMemory(vk_device->getHandle(), vk_buffer.vulkan_allocation_->getHandle());
}

}  // namespace gfx
