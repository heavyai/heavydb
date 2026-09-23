/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/DeviceContext.h"

#include "GfxDriver/Drivers/Vulkan/VulkanMemoryMgr.h"
#include "GfxDriver/Drivers/Vulkan/VulkanPhysicalDevice.h"
#include "GfxDriver/Drivers/Vulkan/VulkanQueue.h"
#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

class VulkanMemoryMgr;
class StagingContext;
class FenceManager;
class VulkanCommandPool;
class VulkanDebugUtils;
class VulkanSwapchain;
class VulkanWSI;
class VulkanPlatform;

struct VulkanDeviceFunctions {
  explicit VulkanDeviceFunctions(const VkDevice vk_device,
                                 const DeviceCapabilityBits& capability_bits);

  //
  // Required
  //
  PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
  PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = nullptr;

  // Buffer device address
  PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress;

  //
  // Ray tracing
  //
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
  PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
  PFN_vkGetAccelerationStructureDeviceAddressKHR
      vkGetAccelerationStructureDeviceAddressKHR;
  PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
  PFN_vkBuildAccelerationStructuresKHR vkBuildAccelerationStructuresKHR;
  PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
  PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
  PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;

  //
  // Optional
  //
  // Mesh and Task shaders
  PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT = nullptr;
  PFN_vkCmdDrawMeshTasksIndirectEXT vkCmdDrawMeshTasksIndirectEXT = nullptr;
  PFN_vkCmdDrawMeshTasksIndirectCountEXT vkCmdDrawMeshTasksIndirectCountEXT = nullptr;
};

class VulkanDeviceContext : public DeviceContext {
 public:
  explicit VulkanDeviceContext(const BaseDriver& driver,
                               const DeviceId gpu_id,
                               const VulkanPhysicalDevice& physical_device);
  VulkanDeviceContext() = delete;
  ~VulkanDeviceContext() override;

  //
  // DeviceContext
  //

  const DeviceId getGpuId() const override { return gpu_id_; }
  const heavyai::UUID& getGpuUUID() const override { return physical_device_.getUUID(); }
  DeviceVendor getVendor() const override { return physical_device_.getVendor(); }
  const DeviceLimits& getLimits() const override { return physical_device_.getLimits(); }
  DeviceCapabilityBits getCapabilityBits() const override {
    return enabled_capabilities_;
  };
  ResourceManager& getResourceManager() const override { return *resource_manager_; }
  uint64_t getPeakMemoryUsage() const override;
  void logMemoryBudgetInfo(std::ostream& os) const override;
  MemoryBudgetInfo getMemoryBudget() const override;

  //
  // VulkanDeviceContext
  //

  // Create the logical device and get queue handles
  void createDevice(const CStrVector& enabled_extensions,
                    const VkPhysicalDeviceFeatures2& enabled_features,
                    DeviceCapabilityBits use_capabilities,
                    uint32_t command_timeout_ms,
                    VulkanWSI* wsi = nullptr);

  // Destroy the logical device and clean up
  // returns true if resources were leaked, otherwise false
  bool destroyDevice();

  // Only valid after createDevice
  void waitIdle() override;

  // Set device lost state for all Queues and Fences then throw
  void handleDeviceLost(bool is_timeout);

  const VkDevice getHandle() const { return vk_device_; }

  const VulkanPhysicalDevice& getPhysicalDevice() const;
  const VkPhysicalDevice getPhysicalDeviceHandle() const;

  // Device dispatch functions
  const VulkanDeviceFunctions& getFunctions() const { return *device_funcs_; };

  // Support classes
  VulkanMemoryMgr& getMemoryManager() const { return *memory_mgr_; }
  StagingContext& getStagingContext() const { return *staging_context_; }
  FenceManager& getFenceManager() const { return *fence_mgr_; }

  //
  // Command queues
  //
  // Graphics queue is required
  VulkanQueue& getGraphicsQueue() const { return *graphics_queue_; }
  // Optional dedicated compute and transfer queues
  VulkanQueue* getComputeQueue() const { return compute_queue_.get(); }
  VulkanQueue* getTransferQueue() const { return transfer_queue_.get(); }

  // Swapchain if presentation is supported on device (may be null)
  VulkanSwapchain* getSwapchain() const { return swapchain_.get(); }

  //
  // Command pools
  //
  enum class CommandPoolSelector {
    kExecutor,
    kStagingGraphics,
    kStagingTransfer,
    kMultiGpuTransfer,
    kCOUNT
  };
  VulkanCommandPool& getCommandPool(CommandPoolSelector selector) const;
  void resetCommandPools() const override;

  CommandExecutionContextUqPtr createCommandExecutionContext() const override;

  VulkanDebugUtils& getDebugUtils() const;

  // Utility functions
  VkFormat pixelFormatToVkFormat(PixelFormat pixel_format) const;

  void nameVulkanObject(const VkObjectType object_type,
                        const void* object_handle,
                        const std::string& object_name) const;

 private:
  void createResourceManager(const ShaderManager& shader_mgr) override;
  void createCommandList() override;

  const DeviceId gpu_id_;
  const VulkanPhysicalDevice& physical_device_;
  VkDevice vk_device_;
  DeviceCapabilityBits enabled_capabilities_;
  std::unique_ptr<VulkanDeviceFunctions> device_funcs_;
  std::unique_ptr<VulkanMemoryMgr> memory_mgr_;
  ResourceManagerUqPtr resource_manager_;
  std::unique_ptr<StagingContext> staging_context_;
  std::unique_ptr<FenceManager> fence_mgr_;

  std::unique_ptr<VulkanQueue> graphics_queue_;
  std::unique_ptr<VulkanQueue> compute_queue_;
  std::unique_ptr<VulkanQueue> transfer_queue_;

  uint32_t command_timeout_ms_;

  VulkanWSI* wsi_;
  std::unique_ptr<VulkanSwapchain> swapchain_;

  std::vector<std::unique_ptr<VulkanCommandPool>> command_pools_;

  std::array<VkFormat, static_cast<uint32_t>(PixelFormat::kCOUNT)>
      pixel_format_to_vk_format_cache_;

  bool destructor_locked_;

  static bool is_device_lost_;

  friend class VulkanPlatform;
};

std::string to_string(const VulkanDeviceContext::CommandPoolSelector value);

}  // namespace gfx
