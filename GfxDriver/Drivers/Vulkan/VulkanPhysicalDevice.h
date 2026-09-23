/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <set>
#include <sstream>
#include <vector>

#include <vulkan/vulkan.h>
#include <boost/noncopyable.hpp>

#include "GfxDriver/DeviceLimits.h"
#include "GfxDriver/Enums.h"
#include "Shared/uuid.h"

using CStrVector = std::vector<const char*>;

namespace gfx {

struct QueueFamilyIndices {
  static constexpr int32_t kUnitialized = -1;
  int32_t graphics = kUnitialized;
  int32_t compute = kUnitialized;
  int32_t transfer = kUnitialized;
  int32_t present = kUnitialized;
};

class VulkanPhysicalDevice : boost::noncopyable {
 public:
  explicit VulkanPhysicalDevice(const VkPhysicalDevice device,
                                const VkSurfaceKHR surface,
                                const bool allow_raytracing_init);
  VulkanPhysicalDevice() = delete;
  ~VulkanPhysicalDevice() = default;

  // Identifiers and feature support
  const VkPhysicalDevice getHandle() const { return vk_physical_device_; }
  const heavyai::UUID& getUUID() const { return uuid_; }

  std::string getName() const { return {properties_.base.deviceName}; }
  DeviceType getType() const { return properties_.type; }
  DeviceVendor getVendor() const { return properties_.vendor; }
  uint32_t getApiVersion() const { return properties_.base.apiVersion; }
  uint32_t getDriverVersion() const { return properties_.base.driverVersion; }
  const VkPhysicalDeviceFeatures& getBaseFeatures() const {
    return properties_.device_features_2.features;
  }
  const VkPhysicalDeviceVulkan11Features& getVulkan11Features() const {
    return properties_.vk11_features;
  }
  const VkPhysicalDeviceVulkan12Features& getVulkan12Features() const {
    return properties_.vk12_features;
  }
  const VkPhysicalDeviceVulkan13Features& getVulkan13Features() const {
    return properties_.vk13_features;
  }
  const VkPhysicalDeviceVulkan11Properties& getVulkan11Properties() const {
    return properties_.vk11_props;
  }
  const VkPhysicalDeviceVulkan12Properties& getVulkan12Properties() const {
    return properties_.vk12_props;
  }
  const VkPhysicalDeviceVulkan13Properties& getVulkan13Properties() const {
    return properties_.vk13_props;
  }

  const DeviceLimits& getLimits() const { return limits_; }
  const VkPhysicalDeviceMeshShaderPropertiesEXT& getMeshShaderProps() const {
    return properties_.mesh_shader_props;
  }

  // Raytracing
  const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& getRaytracingPipelineProperties()
      const {
    return properties_.raytracing_pipeline_props;
  }
  const VkPhysicalDeviceRayTracingPipelineFeaturesKHR& getRaytracingPipelineFeatures()
      const {
    return properties_.raytracing_pipeline_features;
  }
  const VkPhysicalDeviceAccelerationStructureFeaturesKHR&
  getAccelerationStructureFeatures() const {
    return properties_.raytracing_accel_features;
  }

  bool supportsExtension(const std::string& extension);
  std::set<std::string> supportsExtensions(const CStrVector& required_extensions) const;
  DeviceCapabilityBits getCapabilityBits() const { return capability_bits_; }

  // Supported queue families
  const QueueFamilyIndices& getQueueFamilyIndices() const {
    return queue_family_indices_;
  }

  // Get current memory utilization for all memory heaps
  VkPhysicalDeviceMemoryBudgetPropertiesEXT queryMemoryBudget();
  const std::vector<VkMemoryPropertyFlags>& getMemoryHeapProperties() const {
    return memory_heap_aggregated_properties_;
  }

  // Image format support
  VkFormatProperties getFormatProperties(VkFormat format) const;

  // Detailed logging
  std::stringstream buildLog(bool do_log_queue_families, bool do_log_memory_props) const;

 private:
  VkPhysicalDevice vk_physical_device_;
  heavyai::UUID uuid_;
  heavyai::UUID driver_uuid_;

  // extensions
  bool are_extensions_enumerated_;
  std::vector<VkExtensionProperties> available_extensions_;
  DeviceCapabilityBits capability_bits_;
  DeviceLimits limits_;

  // queue families
  std::vector<VkQueueFamilyProperties> queue_family_props_;
  QueueFamilyIndices queue_family_indices_;

  // memory
  std::vector<VkMemoryPropertyFlags> memory_heap_aggregated_properties_;

  // features and properties
  struct {
    // General properties and features
    VkPhysicalDeviceProperties base;

    // Base features
    VkPhysicalDeviceFeatures2 device_features_2;
    VkPhysicalDeviceVulkan11Features vk11_features;
    VkPhysicalDeviceVulkan11Properties vk11_props;
    VkPhysicalDeviceVulkan12Features vk12_features;
    VkPhysicalDeviceVulkan12Properties vk12_props;
    VkPhysicalDeviceVulkan13Features vk13_features;
    VkPhysicalDeviceVulkan13Properties vk13_props;

    DeviceType type;
    DeviceVendor vendor;

    // Optional features
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragment_shader_interlock_features;
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features;
    VkPhysicalDeviceMeshShaderPropertiesEXT mesh_shader_props;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader_features;
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragment_shading_rate_features;

    // External API support
    VkExternalBufferProperties external_vertex_or_storage_buffer_props;
    VkExternalMemoryFeatureFlags external_image_feature_flags;

    // Raytracing support
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR raytracing_pipeline_props;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR raytracing_accel_features;
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR raytracing_pipeline_features;
  } properties_;

  void findQueueFamilies(const VkSurfaceKHR surface);

  void queryExtensionSupport();
  void queryExternalObjectSupport();
  bool supportsQueueFamilies() const;
};

}  // namespace gfx
