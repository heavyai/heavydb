/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanBaseBuffer.h"

#include "GfxDriver/Drivers/Vulkan/Commands/StagingContext.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanHostVisibleBufferWrapper.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanMemoryMgr.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "Shared/scope.h"

namespace gfx {

namespace {

void apply_usage_bits_to_flags(VkFlags& flags, const BufferUsageBits usage) {
  if (any_bits_set(usage & BufferUsageBits::kUniformBufferBit)) {
    flags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }

  if (any_bits_set(usage & BufferUsageBits::kStorageBufferBit)) {
    flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }

  if (any_bits_set(usage & BufferUsageBits::kDeviceAddressBit)) {
    flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  if (any_bits_set(usage & BufferUsageBits::kAccelerationStructureReadOnlyBit)) {
    flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
}

VkFlags get_usage_bits_from_buffer_state(const Buffer::BufferState& state) {
  VkFlags flags{0};
  switch (state.buffer_type) {
    case BufferType::kVertexBuffer:
      flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      break;
    case BufferType::kIndexBuffer:
      flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      break;
    case BufferType::kIndirectDrawVertexBuffer:
    case BufferType::kIndirectDrawIndexBuffer:
      flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      break;
    case BufferType::kPixelBuffer:
      break;
    case BufferType::kAccelerationStructureBuffer:
      CHECK(any_bits_set(state.usage & BufferUsageBits::kDeviceAddressBit))
          << "Vulkan RayAccelerationStructureBuffers must also have DeviceAddress usage";
      flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
      break;
    case BufferType::kShaderBindingTableBuffer:
      CHECK(any_bits_set(state.usage & BufferUsageBits::kDeviceAddressBit))
          << "Vulkan ShaderBindingBuffers must also have DeviceAddress usage";
      flags |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
      break;
    case BufferType::kSlabWrapperBuffer:
      CHECK(any_bits_set(state.usage & BufferUsageBits::kDeviceAddressBit))
          << "SlabWrapperBuffers must also have DeviceAddress usage";
      // enable all current QueryBuffer usage
      flags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
      flags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
      flags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      flags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      break;
    case BufferType::kUnspecified:
      // noop
      break;
    case BufferType::kCOUNT:
      break;
  }

  apply_usage_bits_to_flags(flags, state.usage);
  return flags;
}

bool buffer_access_type_to_enable_export(BufferAccessType access_type) {
  switch (access_type) {
    case BufferAccessType::kHostVisible:
    case BufferAccessType::kHostVisibleCached:
    case BufferAccessType::kDeviceLocal:
      return false;
    case BufferAccessType::kExternalApi:
      return true;
  }
  UNREACHABLE();
  return false;
}

}  // namespace

VkFlags buffer_access_type_to_memory_properties_bits(BufferAccessType access_type) {
  switch (access_type) {
    case BufferAccessType::kDeviceLocal:
    case BufferAccessType::kExternalApi:
      return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    case BufferAccessType::kHostVisible:
      return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    case BufferAccessType::kHostVisibleCached:
      return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
             VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  UNREACHABLE();
  return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

VulkanBaseBuffer::VulkanBaseBuffer(const DeviceContext& device_ctx,
                                   std::string_view resource_tracking_string,
                                   const BufferCreateInfo& create_info,
                                   std::optional<LoggingCallback> oom_logging_cb)
    : Buffer(device_ctx, resource_tracking_string, create_info)
    , vk_memory_requirements_({}) {
  if (create_info.size > 0) {
    createInternal(
        nullptr, create_info.size, create_info.import_allocation_fd, oom_logging_cb);
  }
}

VulkanBaseBuffer::~VulkanBaseBuffer() {
  cleanupResource();
}

void VulkanBaseBuffer::cleanupResourceBase() {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  // destroy the buffer object
  if (resource_handle_) {
    vkDestroyBuffer(
        vk_device.getHandle(), reinterpret_cast<VkBuffer>(resource_handle_), nullptr);
    resource_handle_ = 0;
  }

  // free up the memory it was using
  if (vulkan_allocation_) {
    vk_device.getMemoryManager().free(std::move(vulkan_allocation_));
    vk_memory_requirements_ = {};
  }

  makeEmpty();
}

void VulkanBaseBuffer::makeEmpty() {
  resource_handle_ = 0;
  setSize(0ull);
}

uint64_t VulkanBaseBuffer::getGpuAllocationSize() const {
  return vulkan_allocation_ ? vulkan_allocation_->size() : 0ULL;
}

void VulkanBaseBuffer::create(const void* data,
                              uint64_t num_bytes,
                              std::optional<LoggingCallback> oom_logging_cb) {
  createInternal(data, num_bytes, -1, oom_logging_cb);
}

void VulkanBaseBuffer::rebuild(const void* data,
                               uint64_t num_bytes,
                               std::optional<LoggingCallback> oom_logging_cb) {
  cleanupResourceBase();
  create(data, num_bytes, oom_logging_cb);
}

void VulkanBaseBuffer::createInternal(const void* data,
                                      uint64_t num_bytes,
                                      int32_t import_allocation_fd,
                                      std::optional<LoggingCallback> oom_logging_cb) {
  if (getNumBytes() != 0) {
    THROW_RUNTIME_EX(
        "Attempting to create an already allocated VulkanBaseBuffer, use rebuild() "
        "instead.")
  }

  CHECK(num_bytes > 0ULL);

  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  const auto access_type = getAccessType();

  // create info
  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = num_bytes;
  // Always set TRANSFER_SRC and TRANSFER_DST as there is no downside to doing so
  buffer_info.usage = get_usage_bits_from_buffer_state(state_) |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  // determine if we need BDA and/or export
  bool enable_buffer_device_address =
      (buffer_info.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
  bool enable_export = buffer_access_type_to_enable_export(state_.access_type);

  // check that the device can do what is needed
  if (enable_buffer_device_address) {
    CHECK(any_bits_set(vk_device.getCapabilityBits() &
                       DeviceCapabilityBits::kBufferDeviceAddress));
  }
  if (enable_export) {
    CHECK(any_bits_set(vk_device.getCapabilityBits() &
                       DeviceCapabilityBits::kBufferMemoryExport));
  }
  if (import_allocation_fd >= 0) {
    CHECK(any_bits_set(vk_device.getCapabilityBits() &
                       DeviceCapabilityBits::kBufferMemoryImport));
  }

  // import or export requires external memory config
  VkExternalMemoryBufferCreateInfo external_memory_buffer_ci = {};
  if (import_allocation_fd >= 0 || enable_export) {
    external_memory_buffer_ci.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_memory_buffer_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    buffer_info.pNext = &external_memory_buffer_ci;
  }

  // create it
  CHECK_OOM_VKRESULT(vkCreateBuffer(vk_device.getHandle(),
                                    &buffer_info,
                                    nullptr,
                                    reinterpret_cast<VkBuffer*>(&resource_handle_)),
                     "creating buffer",
                     getTrackingData().origin,
                     buffer_info.size,
                     vk_device,
                     oom_logging_cb);

  // name it
  VkBuffer vk_buffer = reinterpret_cast<VkBuffer>(resource_handle_);
  vk_device.nameVulkanObject(VK_OBJECT_TYPE_BUFFER, vk_buffer, getTrackingData().origin);

  // memory requirements
  vk_memory_requirements_ = {};
  vkGetBufferMemoryRequirements(vk_device.getHandle(),
                                reinterpret_cast<VkBuffer>(resource_handle_),
                                &vk_memory_requirements_);

  // allocate memory
  vulkan_allocation_ = vk_device.getMemoryManager().alloc(
      getTrackingData().origin,
      vk_memory_requirements_,
      buffer_access_type_to_memory_properties_bits(access_type),
      enable_export,
      enable_buffer_device_address,
      import_allocation_fd,
      oom_logging_cb);

  // bind memory to buffer
  CHECK_VKRESULT(vkBindBufferMemory(vk_device.getHandle(),
                                    reinterpret_cast<VkBuffer>(resource_handle_),
                                    vulkan_allocation_->getHandle(),
                                    0),
                 "binding buffer memory");

  setSize(num_bytes);

  setUsable();

  if (data) {
    updateSubData(data, num_bytes, 0);
  }
}

void VulkanBaseBuffer::updateSubData(const void* data,
                                     uint64_t num_bytes,
                                     uint64_t byte_offset) {
  RUNTIME_EX_ASSERT(byte_offset + num_bytes <= getNumBytes(),
                    "Cannot update VulkanBaseBuffer with " + std::to_string(num_bytes) +
                        " bytes starting at byteOffset: " + std::to_string(byte_offset) +
                        " as it would overrun the buffer of " +
                        std::to_string(getNumBytes()) + " bytes.")

  VulkanDeviceContext& vk_device =
      static_cast<VulkanDeviceContext&>(*const_cast<DeviceContext*>(&getDeviceContext()));

  // map buffer
  void* buffer_data = nullptr;

  StagingContext::LockedStagingBuffer locked_staging_buffer;
  if (isMappable()) {
    // @TODO(se) we may need a way to lock HV buffers but not yet
    // StagingContext uses one internally for the "small" buffer
    // so beware of nested locks when this is done
    VulkanHostVisibleBufferWrapper::mapImpl(*this, &buffer_data);
    buffer_data = (void*)((unsigned char*)buffer_data + byte_offset);
  } else {
    locked_staging_buffer = vk_device.getStagingContext().acquireStagingBuffer(num_bytes);
    buffer_data = locked_staging_buffer.buffer;
  }
  CHECK(buffer_data);

  ScopeGuard cleanup_mapping = [&] {
    // unmap buffer
    if (isMappable()) {
      VulkanHostVisibleBufferWrapper::unmapImpl(*this);
    } else {
      vk_device.getStagingContext().releaseStagingBuffer(
          std::move(locked_staging_buffer), this, byte_offset);
    }
  };

  // copy data
  memcpy(buffer_data, data, num_bytes);
}

void VulkanBaseBuffer::getData(void* data,
                               const uint64_t num_bytes,
                               const uint64_t byte_offset) {
  CHECK_LE(num_bytes, state_.size) << "Buffer size mismatch";
  if (isMappable()) {
    // unmap on exit
    ScopeGuard cleanup_mapping = [&] {
      VulkanHostVisibleBufferWrapper::unmapImpl(*this);
    };

    // map buffer
    void* buffer_data = nullptr;
    VulkanHostVisibleBufferWrapper::mapImpl(*this, &buffer_data);
    CHECK(buffer_data);

    // copy data
    void* offset_buffer_data =
        reinterpret_cast<void*>(reinterpret_cast<int8_t*>(buffer_data) + byte_offset);
    memcpy(data, offset_buffer_data, num_bytes);
  } else {
    VulkanDeviceContext& vk_device = static_cast<VulkanDeviceContext&>(
        *const_cast<DeviceContext*>(&getDeviceContext()));

    vk_device.getStagingContext().getBufferData(*this, data, num_bytes);
  }
}

DeviceAddress VulkanBaseBuffer::getDeviceAddress() const {
  CHECK(resource_handle_);
  CHECK(any_bits_set(state_.usage & BufferUsageBits::kDeviceAddressBit));
  auto const& vk_device = static_cast<const VulkanDeviceContext&>(getDeviceContext());
  CHECK(any_bits_set(vk_device.getCapabilityBits() &
                     DeviceCapabilityBits::kBufferDeviceAddress));
  VkBufferDeviceAddressInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  info.buffer = reinterpret_cast<VkBuffer>(resource_handle_);
  return vk_device.getFunctions().vkGetBufferDeviceAddress(vk_device.getHandle(), &info);
}

}  // namespace gfx
