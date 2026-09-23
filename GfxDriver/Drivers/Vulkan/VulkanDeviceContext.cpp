/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/Commands/FenceManager.h"
#include "GfxDriver/Drivers/Vulkan/Commands/StagingContext.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandExecutionContext.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandExecutor.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDriver.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

bool VulkanDeviceContext::is_device_lost_ = false;

VulkanDeviceContext::VulkanDeviceContext(const BaseDriver& driver,
                                         const DeviceId gpu_id,
                                         const VulkanPhysicalDevice& physical_device)
    : DeviceContext{driver}
    , gpu_id_{gpu_id}
    , physical_device_{physical_device}
    , vk_device_{VK_NULL_HANDLE}
    , enabled_capabilities_{DeviceCapabilityBits::kNone}
    , command_timeout_ms_{}
    , wsi_{nullptr}
    , destructor_locked_{true} {
  // reset global device_lost_ flag
  is_device_lost_ = false;
}

VulkanDeviceContext::~VulkanDeviceContext() {
  CHECK(!destructor_locked_)
      << "DeviceContext must be destroyed via DriverInstance interface";
  destructBase();
  destroyDevice();
}

namespace {

std::vector<VkFormat> pixel_format_to_preferred_vk_formats(PixelFormat pixel_format) {
  // in descending order of preference, if multiple
  switch (pixel_format) {
    case PixelFormat::kR8:
      return {VK_FORMAT_R8_UNORM};
    case PixelFormat::kRG8:
      return {VK_FORMAT_R8G8_UNORM};
    case PixelFormat::kRGBA8:
      return {VK_FORMAT_R8G8B8A8_UNORM};
    case PixelFormat::kBGRA8:
      return {VK_FORMAT_B8G8R8A8_UNORM};
    case PixelFormat::kR32UI:
      return {VK_FORMAT_R32_UINT};
    case PixelFormat::kR32I:
      return {VK_FORMAT_R32_SINT};
    case PixelFormat::kDepth:
      return {VK_FORMAT_D24_UNORM_S8_UINT,
              VK_FORMAT_X8_D24_UNORM_PACK32,
              VK_FORMAT_D32_SFLOAT};
    case PixelFormat::kDepthHighP:
      return {VK_FORMAT_D32_SFLOAT};
    case PixelFormat::kDepthStencil:
      return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
    case PixelFormat::kDepthStencilHighP:
      return {VK_FORMAT_D32_SFLOAT_S8_UINT};
    case PixelFormat::kCOUNT:
      CHECK(false);
  }
  UNREACHABLE();
  return {};
}

}  // namespace

void VulkanDeviceContext::createDevice(const CStrVector& enabled_extensions,
                                       const VkPhysicalDeviceFeatures2& enabled_features,
                                       DeviceCapabilityBits use_capabilities,
                                       uint32_t command_timeout_ms,
                                       VulkanWSI* wsi) {
  command_timeout_ms_ = command_timeout_ms;
  wsi_ = wsi;
  enabled_capabilities_ = use_capabilities;
  //
  // Init queue create infos
  //
  const QueueFamilyIndices& qf = physical_device_.getQueueFamilyIndices();
  CHECK_NE(qf.graphics, QueueFamilyIndices::kUnitialized);  // required
  std::set<int> unique_queue_families{qf.graphics};
  if (qf.compute != QueueFamilyIndices::kUnitialized) {
    unique_queue_families.insert(qf.compute);
  }
  if (qf.transfer != QueueFamilyIndices::kUnitialized) {
    unique_queue_families.insert(qf.transfer);
  }

  // Create a single queue for each unique family index
  float queue_priority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  for (int queue_family : unique_queue_families) {
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

  // Create the device
  // enabled_features is expected to be complete by this point
  VkDeviceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pNext = &enabled_features;
  create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos = queue_create_infos.data();

  // Enable device extensions
  create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
  create_info.ppEnabledExtensionNames = enabled_extensions.data();

  CHECK_VKRESULT(
      vkCreateDevice(physical_device_.getHandle(), &create_info, nullptr, &vk_device_),
      "Error creating VkDevice.");

  // name the device
  nameVulkanObject(VK_OBJECT_TYPE_DEVICE, vk_device_, "Device");

  // Double check that all required capabilities are supported by the device
  CHECK((physical_device_.getCapabilityBits() & use_capabilities) == use_capabilities);

  // Load device dispatch function table
  device_funcs_ = std::make_unique<VulkanDeviceFunctions>(vk_device_, use_capabilities);

  // create and init queue wrappers
  graphics_queue_ = std::make_unique<VulkanQueue>(*this);
  graphics_queue_->init(vk_device_, qf.graphics, 0, "Graphics Queue");
  if (qf.compute != QueueFamilyIndices::kUnitialized) {
    compute_queue_ = std::make_unique<VulkanQueue>(*this);
    compute_queue_->init(vk_device_, qf.compute, 0, "Compute Queue");
  }
  if (qf.transfer != QueueFamilyIndices::kUnitialized) {
    transfer_queue_ = std::make_unique<VulkanQueue>(*this);
    transfer_queue_->init(vk_device_, qf.transfer, 0, "Transfer Queue");
  }

  // create helper classes
  memory_mgr_ = std::make_unique<VulkanMemoryMgr>(*this);
  fence_mgr_ = std::make_unique<FenceManager>(*this);
  for (uint32_t i = 0; i < static_cast<uint32_t>(CommandPoolSelector::kCOUNT); ++i) {
    command_pools_.push_back(
        std::make_unique<VulkanCommandPool>(*this,
                                            *graphics_queue_,
                                            static_cast<CommandPoolSelector>(i),
                                            command_timeout_ms));
    // TODO(scb) support dedicated transfer and compute queues
  }

  // Swapchain if supported
  if (wsi_) {
    if (any_bits_set(use_capabilities & DeviceCapabilityBits::kCanPresent)) {
      swapchain_ = std::make_unique<VulkanSwapchain>(*this, *wsi_, command_timeout_ms);
    }
    wsi_->setPresentDevice(*this);
  }

  // populate pixel format to vk format cache
  // we only create textures with TILING_OPTIMAL so only need to check those features
  for (uint32_t i = 0; i < static_cast<uint32_t>(PixelFormat::kCOUNT); i++) {
    auto const pixel_format = static_cast<PixelFormat>(i);
    auto const preferred_formats = pixel_format_to_preferred_vk_formats(pixel_format);
    bool found_suitable_format = false;
    for (auto const& format : preferred_formats) {
      auto const props = physical_device_.getFormatProperties(format);
      auto const min_required_feature =
          is_color_pixel_format(pixel_format)
              ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT
              : VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
      if (props.optimalTilingFeatures & min_required_feature) {
        pixel_format_to_vk_format_cache_[i] = format;
        found_suitable_format = true;
        break;
      }
    }
    CHECK(found_suitable_format)
        << "Failed to find suitable VkFormat for PixelFormat '" << to_string(pixel_format)
        << "' on device " << getGpuUUID();
  }
}

bool VulkanDeviceContext::destroyDevice() {
  // Destroy everything in reverse order

  // Shutdown window system first
  if (wsi_) {
    wsi_->shutdown();
  }
  swapchain_ = nullptr;
  staging_context_ = nullptr;

  // Explicitly call cleanupResources, as setting nullptr clears the
  // pointer before calling the ResourceManager destructor, and any
  // resources that are dangling may try and access ResourceManager
  // via the pointer here
  bool did_leak_resources = false;
  if (resource_manager_) {
    did_leak_resources = resource_manager_->hasResources();
    resource_manager_->cleanupResources();
    resource_manager_ = nullptr;
  }
  command_pools_.clear();
  fence_mgr_ = nullptr;
  memory_mgr_ = nullptr;

  if (vk_device_ != nullptr) {
    vkDestroyDevice(vk_device_, nullptr);
    vk_device_ = nullptr;
  }
  return did_leak_resources;
}

uint64_t VulkanDeviceContext::getPeakMemoryUsage() const {
  CHECK(memory_mgr_);
  return memory_mgr_->getPeakMemoryUsage();
}

MemoryBudgetInfo VulkanDeviceContext::getMemoryBudget() const {
  return memory_mgr_->getMemoryBudget();
}

void VulkanDeviceContext::logMemoryBudgetInfo(std::ostream& os) const {
  memory_mgr_->logMemoryBudgetInfo(os);
}

void VulkanDeviceContext::waitIdle() {
  CHECK(vk_device_ != VK_NULL_HANDLE);
  auto result = vkDeviceWaitIdle(vk_device_);
  if (result == VK_ERROR_DEVICE_LOST) {
    handleDeviceLost(false);
  } else {
    CHECK_VKRESULT(result, "Error waiting for device idle");
  }
}

void VulkanDeviceContext::handleDeviceLost(bool is_timeout) {
  // Check if we're already in a bad state
  if (!is_device_lost_) {
    // Set device_lost_ global flag to prevent any further throws from happening
    is_device_lost_ = true;
    try {
      graphics_queue_->setDeviceLostState();
      if (transfer_queue_) {
        transfer_queue_->setDeviceLostState();
      }
      if (compute_queue_) {
        compute_queue_->setDeviceLostState();
      }
      fence_mgr_->setDeviceLostState();
    } catch (...) {
    }
    // do not place in ScopeGuard as it breaks exception catching in gtest
    throw DeviceLostError(is_timeout ? "Vulkan command timeout" : "Vulkan device lost");
  }
}

const VulkanPhysicalDevice& VulkanDeviceContext::getPhysicalDevice() const {
  return physical_device_;
}

const VkPhysicalDevice VulkanDeviceContext::getPhysicalDeviceHandle() const {
  return physical_device_.getHandle();
}

VkFormat VulkanDeviceContext::pixelFormatToVkFormat(PixelFormat pixel_format) const {
  return pixel_format_to_vk_format_cache_[static_cast<uint32_t>(pixel_format)];
}

void VulkanDeviceContext::createResourceManager(const ShaderManager& shader_mgr) {
  resource_manager_ = std::make_unique<VulkanResourceManager>(*this, shader_mgr);
  staging_context_ = std::make_unique<StagingContext>(*this);
}

void VulkanDeviceContext::createCommandList() {
  command_executor_ = std::make_unique<VulkanCommandExecutor>(
      *this, getCommandPool(CommandPoolSelector::kExecutor));
  command_list_ = std::make_unique<CommandList>(*command_executor_);
}

CommandExecutionContextUqPtr VulkanDeviceContext::createCommandExecutionContext() const {
  return std::make_unique<VulkanCommandExecutionContext>(*this, command_timeout_ms_);
}

VulkanCommandPool& VulkanDeviceContext::getCommandPool(
    CommandPoolSelector selector) const {
  CHECK(selector != CommandPoolSelector::kCOUNT);
  return *command_pools_[static_cast<uint32_t>(selector)];
}

void VulkanDeviceContext::resetCommandPools() const {
  for (auto& pool : command_pools_) {
    pool->waitForPendingBuffers();
    pool->resetPool();
  }
}

VulkanDebugUtils& VulkanDeviceContext::getDebugUtils() const {
  auto const& vulkan_driver = static_cast<const VulkanDriver&>(driver_);
  return vulkan_driver.getDebugUtils();
}

VulkanDeviceFunctions::VulkanDeviceFunctions(
    const VkDevice vk_device,
    const DeviceCapabilityBits& capability_bits) {
  // clang-format off
  if (any_bits_set(capability_bits & (DeviceCapabilityBits::kBufferMemoryExport | DeviceCapabilityBits::kImageMemoryExport))) {
    // Required functions
    vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vk_device, "vkGetMemoryFdKHR");
    CHECK(vkGetMemoryFdKHR);
  }
  if (any_bits_set(capability_bits & DeviceCapabilityBits::kSemaphoreExport)) {
    vkGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(vk_device, "vkGetSemaphoreFdKHR");
    CHECK(vkGetSemaphoreFdKHR);
  }

  // Optional functions
  if (any_bits_set(capability_bits & DeviceCapabilityBits::kBufferDeviceAddress)) {
    vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(vk_device, "vkGetBufferDeviceAddress");
  }

  if (any_bits_set(capability_bits & DeviceCapabilityBits::kRaytracing)) {
    vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(vk_device, "vkCreateAccelerationStructureKHR");
    vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(vk_device, "vkDestroyAccelerationStructureKHR");
    vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(vk_device, "vkGetAccelerationStructureBuildSizesKHR");
    vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(vk_device, "vkGetAccelerationStructureDeviceAddressKHR");
    vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(vk_device, "vkCmdBuildAccelerationStructuresKHR");
    vkBuildAccelerationStructuresKHR = (PFN_vkBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(vk_device, "vkBuildAccelerationStructuresKHR");
    vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(vk_device, "vkCmdTraceRaysKHR");
    vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(vk_device, "vkGetRayTracingShaderGroupHandlesKHR");
    vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(vk_device, "vkCreateRayTracingPipelinesKHR");

    CHECK(vkGetBufferDeviceAddress);
    CHECK(vkCreateAccelerationStructureKHR);
    CHECK(vkDestroyAccelerationStructureKHR);
    CHECK(vkGetAccelerationStructureBuildSizesKHR);
    CHECK(vkGetAccelerationStructureDeviceAddressKHR);
    CHECK(vkCmdBuildAccelerationStructuresKHR);
    CHECK(vkBuildAccelerationStructuresKHR);
    CHECK(vkCmdTraceRaysKHR);
    CHECK(vkGetRayTracingShaderGroupHandlesKHR);
    CHECK(vkCreateRayTracingPipelinesKHR);
  }

  if (any_bits_set(capability_bits & DeviceCapabilityBits::kMeshShaders)) {
    vkCmdDrawMeshTasksEXT = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(vk_device, "vkCmdDrawMeshTasksEXT");
    vkCmdDrawMeshTasksIndirectEXT = (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(vk_device, "vkCmdDrawMeshTasksIndirectEXT");
    vkCmdDrawMeshTasksIndirectCountEXT = (PFN_vkCmdDrawMeshTasksIndirectCountEXT)vkGetDeviceProcAddr(vk_device, "vkCmdDrawMeshTasksIndirectEXT");

    CHECK(vkCmdDrawMeshTasksEXT);
    CHECK(vkCmdDrawMeshTasksIndirectEXT);
    CHECK(vkCmdDrawMeshTasksIndirectCountEXT);
  }
  // clang-format on
}

void VulkanDeviceContext::nameVulkanObject(const VkObjectType object_type,
                                           const void* object_handle,
                                           const std::string& object_name) const {
  getDebugUtils().nameVulkanObject(
      vk_device_, getGpuId(), object_type, object_handle, object_name);
}

std::string to_string(const VulkanDeviceContext::CommandPoolSelector value) {
  switch (value) {
    case VulkanDeviceContext::CommandPoolSelector::kExecutor:
      return "Executor";
    case VulkanDeviceContext::CommandPoolSelector::kStagingGraphics:
      return "Staging Graphics";
    case VulkanDeviceContext::CommandPoolSelector::kStagingTransfer:
      return "Staging Transfer";
    case VulkanDeviceContext::CommandPoolSelector::kMultiGpuTransfer:
      return "Multi-GPU Transfer";
    case VulkanDeviceContext::CommandPoolSelector::kCOUNT:
      CHECK(false);
  }
  return "";
}

}  // namespace gfx
