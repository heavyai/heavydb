/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanPhysicalDevice.h"

#include <set>
#include <string>
#include <vector>

#include "GfxDriver/Drivers/Vulkan/VulkanMemoryUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanPlatformUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

namespace {

// this must align with the VkPhyicalDeviceType enum:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceType.html
static std::array<std::string_view, 5> g_device_type_to_string = {"Other",
                                                                  "Integrated GPU",
                                                                  "Discrete GPU",
                                                                  "Virtual GPU",
                                                                  "CPU"};

#define VENDOR_ID_AMD 0x1002
#define VENDOR_ID_NVIDIA 0x10DE
#define VENDOR_ID_INTEL 0x8086
#define VENDOR_ID_MESA 0x10005

}  // namespace

VulkanPhysicalDevice::VulkanPhysicalDevice(const VkPhysicalDevice device,
                                           const VkSurfaceKHR surface,
                                           const bool allow_raytracing_init)
    : vk_physical_device_{device}
    , are_extensions_enumerated_{false}
    , capability_bits_{0}
    , properties_{} {
  vkGetPhysicalDeviceProperties(device, &properties_.base);
  queryExtensionSupport();

  // extract the device type
  switch (properties_.base.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      properties_.type = DeviceType::kIntegratedGpu;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      properties_.type = DeviceType::kDiscreetGpu;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      properties_.type = DeviceType::kVirtualGpu;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      properties_.type = DeviceType::kCPU;
      break;
    default:
      properties_.type = DeviceType::kOther;
      break;
  }

  // extract the vendor
  switch (properties_.base.vendorID) {
    case VENDOR_ID_NVIDIA:
      properties_.vendor = DeviceVendor::kNvidia;
      break;
    case VENDOR_ID_INTEL:
      properties_.vendor = DeviceVendor::kIntel;
      break;
    case VENDOR_ID_AMD:
      properties_.vendor = DeviceVendor::kAMD;
      break;
    case VENDOR_ID_MESA:
      properties_.vendor = DeviceVendor::kMesa;
      break;
    default:
      properties_.vendor = DeviceVendor::kOther;
      break;
  }

  // Require Vulkan 1.2 API support as a minimum
  bool is_vk_1_2 = (VK_VERSION_MAJOR(properties_.base.apiVersion) > 1 ||
                    VK_VERSION_MINOR(properties_.base.apiVersion) > 1);

  if (!is_vk_1_2) {
    // This is logged in VulkanPlatform
    return;
  }

  // Check for raytracing support
  bool supports_raytracing =
      allow_raytracing_init &&
      supportsExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
      supportsExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
  bool supports_ray_query =
      supports_raytracing && supportsExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME);

  // General properties
  VkPhysicalDeviceProperties2 props2;
  VkStructChainBuilder props_chain(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                   &props2);

  // Vulkan 1.1, 1.2, and 1.3 properties
  props_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
                  &properties_.vk11_props);
  props_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
                  &properties_.vk12_props);
  props_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
                  &properties_.vk13_props);

  // Raytracing properties
  if (supports_raytracing) {
    props_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
                    &properties_.raytracing_pipeline_props);
  }

  // Mesh and Task shaders
  if (supportsExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME)) {
    // actual support is in features
    props_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
                    &properties_.mesh_shader_props);
  }

  // populate props chain
  vkGetPhysicalDeviceProperties2(device, &props2);

  if (properties_.vk11_props.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) {
    capability_bits_ |= DeviceCapabilityBits::kSubgroupVote;
  }
  if (properties_.vk11_props.subgroupSupportedOperations &
      VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) {
    capability_bits_ |= DeviceCapabilityBits::kSubgroupArithmetic;
  }
  if (properties_.vk11_props.subgroupSupportedOperations &
      VK_SUBGROUP_FEATURE_BALLOT_BIT) {
    capability_bits_ |= DeviceCapabilityBits::kSubgroupBallot;
  }

  //
  // features
  //

  VkStructChainBuilder features_chain(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                      &properties_.device_features_2);
  features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                     &properties_.vk11_features);
  features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                     &properties_.vk12_features);
  features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                     &properties_.vk13_features);

  // Fragment shader interlock
  if (supportsExtension(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)) {
    features_chain.add(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
        &properties_.fragment_shader_interlock_features);
  }

  // Raytracing features
  if (supports_raytracing) {
    features_chain.add(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        &properties_.raytracing_pipeline_features);
    features_chain.add(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        &properties_.raytracing_accel_features);
  }

  // Ray query features
  if (supports_ray_query) {
    features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
                       &properties_.ray_query_features);
  }

  // Mesh and Task shaders
  if (supportsExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME) &&
      supportsExtension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) {
    features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
                       &properties_.mesh_shader_features);
    features_chain.add(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
        &properties_.fragment_shading_rate_features);
  }

  // Get features from device
  vkGetPhysicalDeviceFeatures2(device, &properties_.device_features_2);

  // Check for subgroup extended types
  if (properties_.vk12_features.shaderSubgroupExtendedTypes) {
    capability_bits_ |= DeviceCapabilityBits::kSubgroupExtendedTypes;
  }

  // Check fragment shader interlock features
  if (properties_.fragment_shader_interlock_features.fragmentShaderPixelInterlock) {
    capability_bits_ |= DeviceCapabilityBits::kFragmentShaderPixelInterlock;
  }
  if (properties_.fragment_shader_interlock_features.fragmentShaderSampleInterlock) {
    capability_bits_ |= DeviceCapabilityBits::kFragmentShaderSampleInterlock;
  }

  // Buffer device address
  if (properties_.vk12_features.bufferDeviceAddress) {
    capability_bits_ |= DeviceCapabilityBits::kBufferDeviceAddress;
  }

  uuid_ = heavyai::UUID(properties_.vk11_props.deviceUUID);
  driver_uuid_ = heavyai::UUID(properties_.vk11_props.driverUUID);

  if (properties_.raytracing_pipeline_features.rayTracingPipeline == VK_TRUE &&
      properties_.raytracing_accel_features.accelerationStructure == VK_TRUE) {
    // TODO: Check support for the advanced features in the structs
    capability_bits_ |= DeviceCapabilityBits::kRaytracing;
  }

  if (properties_.ray_query_features.rayQuery == VK_TRUE) {
    capability_bits_ |= DeviceCapabilityBits::kRayQuery;
  }

  if (properties_.mesh_shader_features.meshShader == VK_TRUE &&
      properties_.fragment_shading_rate_features.primitiveFragmentShadingRate ==
          VK_TRUE &&
      properties_.mesh_shader_features.meshShaderQueries == VK_TRUE) {
    capability_bits_ |= DeviceCapabilityBits::kMeshShaders;
  }
  if (properties_.mesh_shader_features.taskShader == VK_TRUE) {
    capability_bits_ |= DeviceCapabilityBits::kTaskShaders;
  }

  queryExternalObjectSupport();
  findQueueFamilies(surface);

  // Get memory properties for each heap by aggregating memory type information for the
  // heap
  VkPhysicalDeviceMemoryProperties mem_properties{};
  vkGetPhysicalDeviceMemoryProperties(vk_physical_device_, &mem_properties);
  memory_heap_aggregated_properties_ = get_aggregate_memory_heap_props(mem_properties);

  //
  // Limits
  //
  // clang-format off
  auto const& vk_limits = properties_.base.limits;
  limits_.max_uniform_buffer_size = vk_limits.maxUniformBufferRange;
  limits_.uniform_buffer_alignment = vk_limits.minUniformBufferOffsetAlignment;
  limits_.max_shader_storage_buffer_size = vk_limits.maxStorageBufferRange;
  limits_.shader_storage_buffer_alignment = vk_limits.minStorageBufferOffsetAlignment;
  for (int i=0; i<3; ++i) {
    limits_.max_compute_workgroup_count[i] = vk_limits.maxComputeWorkGroupCount[i];
  }

  limits_.subgroup_size = properties_.vk11_props.subgroupSize;
  limits_.max_compute_workgroup_subgroups = properties_.vk13_props.maxComputeWorkgroupSubgroups;
  limits_.max_compute_shared_memory_size = vk_limits.maxComputeSharedMemorySize;

  limits_.max_framebuffer_width = vk_limits.maxFramebufferWidth;
  limits_.max_framebuffer_height = vk_limits.maxFramebufferHeight;

  limits_.timestamp_period = vk_limits.timestampPeriod;

  if (any_bits_set(capability_bits_ & DeviceCapabilityBits::kRaytracing)) {
    auto const& rt_props = properties_.raytracing_pipeline_props;
    limits_.shader_group_handle_size = rt_props.shaderGroupHandleSize;
    limits_.shader_group_handle_alignment = rt_props.shaderGroupHandleAlignment;
    limits_.shader_group_base_alignment = rt_props.shaderGroupBaseAlignment;
  }
  if (any_bits_set(capability_bits_ &
                   DeviceCapabilityBits::kMeshShaders)) {
    auto const& mesh_props = properties_.mesh_shader_props;

    for (int i = 0; i < 3; i++) {
      limits_.max_task_workgroup_count[i] = mesh_props.maxTaskWorkGroupCount[i];
      limits_.max_task_workgroup_size[i] = mesh_props.maxTaskWorkGroupSize[i];
      limits_.max_mesh_workgroup_count[i] = mesh_props.maxMeshWorkGroupCount[i];
      limits_.max_mesh_workgroup_size[i] = mesh_props.maxMeshWorkGroupSize[i];
    }

    limits_.max_task_workgroup_total_count = mesh_props.maxTaskWorkGroupTotalCount;
    limits_.max_task_workgroup_invocations = mesh_props.maxTaskWorkGroupInvocations;
    limits_.max_task_payload_size = mesh_props.maxTaskPayloadSize;
    limits_.max_task_shared_memory_size = mesh_props.maxTaskSharedMemorySize;
    limits_.max_task_payload_and_shared_memory_size = mesh_props.maxTaskPayloadAndSharedMemorySize;
    limits_.max_mesh_workgroup_total_count = mesh_props.maxMeshWorkGroupTotalCount;
    limits_.max_mesh_workgroup_invocations = mesh_props.maxMeshWorkGroupInvocations;
    limits_.max_mesh_shared_memory_size = mesh_props.maxMeshSharedMemorySize;
    limits_.max_mesh_payload_and_shared_memory_size = mesh_props.maxMeshPayloadAndSharedMemorySize;
    limits_.max_mesh_output_memory_size = mesh_props.maxMeshOutputMemorySize;
    limits_.max_mesh_payload_and_output_memory_size = mesh_props.maxMeshPayloadAndOutputMemorySize;
    limits_.max_mesh_output_components = mesh_props.maxMeshOutputComponents;
    limits_.max_mesh_output_vertices = mesh_props.maxMeshOutputVertices;
    limits_.max_mesh_output_primitives = mesh_props.maxMeshOutputPrimitives;
    limits_.max_mesh_output_layers = mesh_props.maxMeshOutputLayers;
    limits_.max_mesh_multiview_view_count = mesh_props.maxMeshMultiviewViewCount;
    limits_.mesh_output_per_vertex_granularity = mesh_props.meshOutputPerVertexGranularity;
    limits_.mesh_output_per_primitive_granularity = mesh_props.meshOutputPerPrimitiveGranularity;
    limits_.max_preferred_task_workgroup_invocations = mesh_props.maxPreferredTaskWorkGroupInvocations;
    limits_.max_preferred_mesh_workgroup_invocations = mesh_props.maxPreferredMeshWorkGroupInvocations;
  }
  // clang-format on
}

void VulkanPhysicalDevice::queryExternalObjectSupport() {
  // Get support for exported vertex or storage buffer
  VkPhysicalDeviceExternalBufferInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
  buffer_info.flags = 0;
  buffer_info.usage =
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  properties_.external_vertex_or_storage_buffer_props.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;

  vkGetPhysicalDeviceExternalBufferProperties(
      vk_physical_device_,
      &buffer_info,
      &properties_.external_vertex_or_storage_buffer_props);

  if (properties_.external_vertex_or_storage_buffer_props.externalMemoryProperties
          .externalMemoryFeatures &
      VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) {
    capability_bits_ |= DeviceCapabilityBits::kBufferMemoryExport;
  }
  if (properties_.external_vertex_or_storage_buffer_props.externalMemoryProperties
          .externalMemoryFeatures &
      VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) {
    capability_bits_ |= DeviceCapabilityBits::kBufferMemoryImport;
  }

  // Images
  // Query all the required formats and generate homogenous ExternalMemoryFeatureFlags
  // so the device only reports exportable images if all formats are supported
  static constexpr std::array<VkFormat, 3> image_formats = {
      VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT};

  auto get_image_format_info = [this](VkFormat format) -> VkExternalMemoryFeatureFlags {
    VkPhysicalDeviceExternalImageFormatInfo format_info = {};
    format_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
    format_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkPhysicalDeviceImageFormatInfo2 image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    image_info.format = format;
    image_info.type = VK_IMAGE_TYPE_2D;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    image_info.pNext = &format_info;

    VkImageFormatProperties2 image_format_props = {};
    image_format_props.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;

    VkExternalImageFormatProperties external_image_props = {};

    external_image_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
    image_format_props.pNext = &external_image_props;

    vkGetPhysicalDeviceImageFormatProperties2(
        vk_physical_device_, &image_info, &image_format_props);

    return external_image_props.externalMemoryProperties.externalMemoryFeatures;
  };

  properties_.external_image_feature_flags = 0xFFFF;
  for (auto const format : image_formats) {
    properties_.external_image_feature_flags &= get_image_format_info(format);
  }

  if (properties_.external_image_feature_flags &
      VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) {
    capability_bits_ |= DeviceCapabilityBits::kImageMemoryExport;
  }
  if (properties_.external_image_feature_flags &
      VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) {
    capability_bits_ |= DeviceCapabilityBits::kImageMemoryImport;
  }

  // Semaphores
  VkPhysicalDeviceExternalSemaphoreInfo semaphore_info = {};
  semaphore_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
  semaphore_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkExternalSemaphoreProperties semaphore_props = {};
  semaphore_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;

  vkGetPhysicalDeviceExternalSemaphoreProperties(
      vk_physical_device_, &semaphore_info, &semaphore_props);

  if (semaphore_props.compatibleHandleTypes &
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
    if (semaphore_props.externalSemaphoreFeatures &
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) {
      capability_bits_ |= DeviceCapabilityBits::kSemaphoreExport;
    }
    if (semaphore_props.externalSemaphoreFeatures &
        VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) {
      capability_bits_ |= DeviceCapabilityBits::kSemaphoreImport;
    }
  }
}

VkPhysicalDeviceMemoryBudgetPropertiesEXT VulkanPhysicalDevice::queryMemoryBudget() {
  VkPhysicalDeviceMemoryBudgetPropertiesEXT mem_budget{};
  if (supportsExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

    mem_budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    mem_budget.pNext = nullptr;
    props2.pNext = &mem_budget;

    vkGetPhysicalDeviceProperties2(vk_physical_device_, &props2);
  }
  return mem_budget;
}

bool VulkanPhysicalDevice::supportsExtension(const std::string& extension_name) {
  if (!are_extensions_enumerated_) {
    queryExtensionSupport();
  }

  for (const auto& extension : available_extensions_) {
    if (extension.extensionName == extension_name) {
      return true;
    }
  }
  return false;
}

std::set<std::string> VulkanPhysicalDevice::supportsExtensions(
    const CStrVector& required_extensions) const {
  std::set<std::string> missing_extensions(required_extensions.begin(),
                                           required_extensions.end());
  for (const auto& extension : available_extensions_) {
    missing_extensions.erase(extension.extensionName);
  }
  return missing_extensions;
}

void VulkanPhysicalDevice::queryExtensionSupport() {
  uint32_t extension_count;
  CHECK_VKRESULT(vkEnumerateDeviceExtensionProperties(
                     vk_physical_device_, nullptr, &extension_count, nullptr),
                 "Error enumerating device extension properties");

  available_extensions_.resize(extension_count);
  CHECK_VKRESULT(
      vkEnumerateDeviceExtensionProperties(
          vk_physical_device_, nullptr, &extension_count, available_extensions_.data()),
      "Error enumerating device extension properties");

  are_extensions_enumerated_ = true;
}

void VulkanPhysicalDevice::findQueueFamilies(const VkSurfaceKHR surface) {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      vk_physical_device_, &queue_family_count, nullptr);

  queue_family_props_.resize(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      vk_physical_device_, &queue_family_count, queue_family_props_.data());

  int i = 0;
  static constexpr VkQueueFlags invalid_transfer_queue_flags =
      VK_QUEUE_VIDEO_DECODE_BIT_KHR | VK_QUEUE_OPTICAL_FLOW_BIT_NV;
  for (const auto& queue_family : queue_family_props_) {
    if (queue_family.queueCount > 0 && queue_family.timestampValidBits == 64) {
      if ((queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT)) {
        // Found conformant graphics queue
        queue_family_indices_.graphics = i;

        // Only allow present on the graphics queue for now
        VkBool32 present_supported = false;
        if (surface != VK_NULL_HANDLE) {
          vkGetPhysicalDeviceSurfaceSupportKHR(
              vk_physical_device_, i, surface, &present_supported);
          if (present_supported) {
            capability_bits_ |= DeviceCapabilityBits::kCanPresent;
          }
        }
        if (present_supported) {
          queue_family_indices_.present = i;
        }
      } else if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
        // Found dedicated compute queue
        queue_family_indices_.compute = i;
      } else if (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT &&
                 ((queue_family.queueFlags & invalid_transfer_queue_flags) == 0)) {
        // Found dedicated transfer queues
        queue_family_indices_.transfer = i;
      }
    }
    i++;
  }
}

VkFormatProperties VulkanPhysicalDevice::getFormatProperties(VkFormat format) const {
  VkFormatProperties props;
  vkGetPhysicalDeviceFormatProperties(vk_physical_device_, format, &props);
  return props;
}

static void logQueueFamilyProperties(std::stringstream& stream,
                                     const std::vector<VkQueueFamilyProperties>& props) {
  size_t num_families = props.size();
  stream << "Vulkan queue families:" << '\n';
  for (size_t i = 0; i < num_families; ++i) {
    const VkQueueFamilyProperties& p = props[i];
    stream << "  family: " << i;
    stream << "  queues: " << p.queueCount;
    if (p.queueCount < 10) {
      stream << " ";
    }
    stream << "  timestamp bits: " << p.timestampValidBits;
    stream << "  types: ";
    if ((p.queueFlags & VK_QUEUE_GRAPHICS_BIT) == VK_QUEUE_GRAPHICS_BIT) {
      stream << "[Gfx]";
    }
    if ((p.queueFlags & VK_QUEUE_COMPUTE_BIT) == VK_QUEUE_COMPUTE_BIT) {
      stream << "[Cmp]";
    }
    if ((p.queueFlags & VK_QUEUE_TRANSFER_BIT) == VK_QUEUE_TRANSFER_BIT) {
      stream << "[Xfr]";
    }
    if ((p.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) == VK_QUEUE_SPARSE_BINDING_BIT) {
      stream << "[SpB]";
    }
    if ((p.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) == VK_QUEUE_VIDEO_DECODE_BIT_KHR) {
      stream << "[VDc]";
    }
    if ((p.queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) == VK_QUEUE_OPTICAL_FLOW_BIT_NV) {
      stream << "[OFl]";
    }
    stream << '\n';
  }
}

static void logMemoryHeapProperties(std::stringstream& stream,
                                    const VkPhysicalDeviceMemoryProperties& mem_props) {
  stream << "Vulkan memory heap properties:\n";
  int num_usable_types = 0;
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if (mem_props.memoryTypes[i].propertyFlags) {
      num_usable_types++;
    }
  }
  stream << "  # types: " << mem_props.memoryTypeCount << " (" << num_usable_types
         << " usable)\n";
  stream << "  # heaps: " << mem_props.memoryHeapCount << '\n';
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    uint32_t heap_index = mem_props.memoryTypes[i].heapIndex;
    VkMemoryPropertyFlags flags = mem_props.memoryTypes[i].propertyFlags;
    if (flags) {
      stream << "    heap " << heap_index
             << ": size: " << mem_props.memoryHeaps[heap_index].size << " flags: ";
      if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        stream << "[DvcLocal]";
      }
      if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ==
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        stream << "[HostVis]";
      }
      if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ==
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        stream << "[HostCoherent]";
      }
      if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ==
          VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
        stream << "[HostCached]";
      }
      if ((flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ==
          VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) {
        stream << "[LazyAlloc]";
      }
      stream << '\n';
    }
  }
}

std::stringstream VulkanPhysicalDevice::buildLog(bool do_log_queue_families,
                                                 bool do_log_memory_props) const {
  std::stringstream stream;
  {
    // Base properties
    {
      auto const& p = properties_.base;
      stream << "Vulkan device properties:\n";
      stream << " Name: " << p.deviceName << '\n';
      stream << " Type: " << g_device_type_to_string[p.deviceType] << '\n';
      stream << " Vulkan API version: " << vulkan_version_to_string(p.apiVersion) << '\n';
    }

    // Driver properties
    auto const& p = properties_.vk12_props;
    stream << "Vulkan device driver Properties:" << '\n';
    stream << " Id: " << vk_driver_id_to_string(p.driverID) << '\n';
    stream << " Name: " << std::string(p.driverName) << '\n';
    stream << " Info: " << std::string(p.driverInfo) << '\n';
    auto const& v = p.conformanceVersion;
    stream << " Conformance Version: " << +v.major << "." << +v.minor << "."
           << +v.subminor << "." << +v.patch << '\n';
  }

  if (do_log_queue_families) {
    logQueueFamilyProperties(stream, queue_family_props_);
  }
  if (do_log_memory_props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(vk_physical_device_, &mem_props);
    logMemoryHeapProperties(stream, mem_props);
  }
  return stream;
}

}  // namespace gfx
