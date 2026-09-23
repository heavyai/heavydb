/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <unordered_set>

#include <vulkan/vulkan.h>
#include <boost/noncopyable.hpp>

#include "GfxDriver/Drivers/Vulkan/VulkanDebugUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/WSI/VulkanWSI.h"
#include "GfxDriver/Enums.h"
#include "GfxDriver/Types.h"
#include "Shared/uuid.h"

namespace gfx {

class VulkanDriver;

void begin_renderdoc_vulkan_capture();
void end_renderdoc_vulkan_capture();

/**
 * Wrapper around the platform specific loading code. This class is responsible for
 * initializing the standard Vulkan loader, resolving the handful of functions that
 * need resolving manually, and creating a VkInstance. It will also handle the
 * instantiation of the WSI wrapper appropriate for the current platform, handling
 * display surface creation.
 * */
class VulkanPlatform : boost::noncopyable {
 public:
  explicit VulkanPlatform(const BaseDriver& driver,
                          GfxUsage usage,
                          uint32_t commmand_timeout,
                          const WindowSystemCreateInfo* wsi_ci,
                          bool allow_raytracing_init);
  VulkanPlatform() = delete;
  ~VulkanPlatform();

  uint32_t getNumGpus() const;
  std::vector<heavyai::UUID> getUUIDs() const;

  DeviceCapabilityBits getHomogeneousCapabilityBits() const;
  const DeviceLimits& getHomogenousDeviceLimits() const;

  // Find the VulkanPhysicalDevice that corresponds to the specified UUID and configure it
  // for rendering (provided it is suitable)
  DeviceContextUqPtr createDeviceContext(const heavyai::UUID uuid, const DeviceId gpu_id);

  bool destroyDeviceContext(DeviceContextUqPtr device_ctx);

  WindowSystemIntegration* getWSI() const;

  // Debugging (public for access by callback)
  void debugCallback() const;

  // Validation message suppression
  // Primarily useful for suppression validation message handling during tests which
  // are known to create validation warning or error
  static void pushSuppressValidationMessages();
  static void popSuppressValidationMessages();
  static bool areValidationMessagesSuppressed();

  // Validation control
  enum class ValidationMode { kDisable, kReport, kFailIfError, kFailAlways };
  static ValidationMode getValidationMode();

  VulkanDebugUtils& getDebugUtils() const;

  MemoryUsageInfo getPeakMemoryUsage() const;

 private:
  struct {
    void* handle;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
  } loader_;

  const BaseDriver& driver_;
  VkInstance vk_instance_;
  std::map<heavyai::UUID, std::unique_ptr<VulkanPhysicalDevice>> physical_devices_;
  static int32_t suppress_validation_count_;
  static ValidationMode validation_mode_;

  DeviceCapabilityBits capability_bits_;
  DeviceLimits limits_;
  CStrVector enable_extensions_;
  uint32_t command_timeout_;

  std::unique_ptr<VulkanDebugUtils> debug_utils_;
  std::unordered_set<VulkanDeviceContext*> all_device_contexts_;

  // Window system integration
  std::unique_ptr<VulkanWSI> wsi_;

  bool allow_raytracing_init_;

  // Vulkan loader
  void initLoader();
  void shutdownLoader();

  // Vulkan API instance
  void createInstance();
  void destroyInstance();

  // Debugging
  void createDebugUtils();
  void destroyDebugUtils();

  // Physical device management
  void enumerateDevices(GfxUsage usage, VkSurfaceKHR surface);
  bool isDeviceSuitable(const VulkanPhysicalDevice& device,
                        GfxUsage usage,
                        const CStrVector& required_extensions) const;

  // Device groups
  void logDeviceGroups();
};

}  // namespace gfx
