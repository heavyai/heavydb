/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanPlatform.h"

#include <dlfcn.h>
#include <sstream>
#include <vector>

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/VulkanPlatformUtils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/RenderDoc/RenderDoc.h"
#include "GfxDriver/RenderError.h"
#include "Shared/StringTransform.h"
#ifdef HAVE_GLFW
#include "GfxDriver/Drivers/Vulkan/WSI/VulkanWSI_GLFW.h"
#endif

//
// Validation behavior for release builds is now controlled by the
// environment variable
//
// OMNISCI_VULKAN_VALIDATION_MODE
//
// Values can be as follows (case-insensitive):
//   "disable" (or unset) - validation layers disabled, no reporting
//   "report" - layers active, reports all validation messages as warnings
//   "failiferror" - layers active, fails on error, warns for lesser severity
//   "failalways" - layers active, fails on ANY validation message
//
// Behavior for debug builds is controlled by the #define below
//

#define DEBUG_BUILD_VALIDATION_MODE ValidationMode::kFailAlways
#define VULKAN_API_VERSION VK_API_VERSION_1_3

namespace gfx {

//
// Instance extensions
//
static const CStrVector g_req_instance_extensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

static const CStrVector g_req_validation_layers = {"VK_LAYER_KHRONOS_validation"};

//
// Device extensions
//

// Required device extensions

static const CStrVector g_base_required_device_extensions = {};

static const CStrVector g_interop_required_device_extensions = {
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};

static const CStrVector g_present_required_device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// Struct to pair DeviceCapabilityBit with extensions required to support it
// These are used to enable a capability after support has been detected
struct CapabilityExtensionInfo {
  DeviceCapabilityBits capability_bit;
  CStrVector enable_extensions;
};

// Mesh shader extension info
static CapabilityExtensionInfo g_mesh_shader_capability_extension_info = {
    DeviceCapabilityBits::kMeshShaders,
    {VK_EXT_MESH_SHADER_EXTENSION_NAME, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME}};

// Fragment shader interlock extension info
static CapabilityExtensionInfo g_fragment_shader_interlock_extension_info = {
    DeviceCapabilityBits::kFragmentShaderPixelInterlock |
        DeviceCapabilityBits::kFragmentShaderSampleInterlock,
    {VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME}};

// Raytracing
static CapabilityExtensionInfo g_raytracing_capability_extension_info = {
    DeviceCapabilityBits::kRaytracing,
    {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
     VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
     VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME}};

// All CapabilityExtensionInfos
static std::vector<CapabilityExtensionInfo> g_optional_capability_extension_infos = {
    g_mesh_shader_capability_extension_info,
    g_fragment_shader_interlock_extension_info,
    g_raytracing_capability_extension_info};

//
// VulkanPlatform
//
VulkanPlatform::VulkanPlatform(const BaseDriver& driver,
                               GfxUsage usage,
                               uint32_t command_timeout,
                               const WindowSystemCreateInfo* wsi_ci,
                               bool allow_raytracing_init)
    : loader_{}
    , driver_{driver}
    , vk_instance_{VK_NULL_HANDLE}
    , capability_bits_{0}
    , command_timeout_{command_timeout}
    , allow_raytracing_init_{allow_raytracing_init} {
  // Initialize the Vulkan loader
  initLoader();

  // Create WSI, as we need the required Instance extensions immediately
  if (wsi_ci) {
#ifdef HAVE_GLFW
    wsi_ = std::make_unique<VulkanWSI_GLFW>(*wsi_ci);
#else
    CHECK(false) << "GLFW not found, WSI not supported";
#endif
  }

  // Create the VkInstance, initializing the Vulkan API
  createInstance();

  // If using WSI, create the VkSurface for the window, which is required
  // to properly enumerate the physical devices
  if (wsi_) {
    wsi_->createSurface(vk_instance_);
  }

  // Enumerate all the Vulkan physical devices
  enumerateDevices(usage, wsi_ ? wsi_->getSurface() : VK_NULL_HANDLE);

  // Initialize VulkanDebugUtils
  createDebugUtils();
}

VulkanPlatform::~VulkanPlatform() {
  wsi_ = nullptr;
  destroyDebugUtils();
  destroyInstance();
  shutdownLoader();
}

void VulkanPlatform::initLoader() {
  loader_.handle = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (loader_.handle == nullptr) {
    LOG(ERROR) << "Failed to load libvulkan.so.1";
    // TODO(scb): Log should indicate how to install the system vulkan
    // loader (e.g. apt install libvulkan on Ubuntu).
    THROW_RUNTIME_EX("Failed to load libVulkan.");
  }

  // Resolve vkGetInstanceProcAddr from the dynamic library, so we can
  // use it to load the other Instance functions we need.
  loader_.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader_.handle, "vkGetInstanceProcAddr");
  RUNTIME_EX_ASSERT(loader_.GetInstanceProcAddr,
                    "Vulkan: Loader does not export vkGetInstanceProcAddr");

  // Check for presence of vkEnumerateInstanceVersion which was not added until Vulkan 1.1
  // We don't bother caching this function pointer as we only call it here.
  auto vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(
      VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
  if (vkEnumerateInstanceVersion != nullptr) {
    // Log instance API version
    uint32_t api_version = 0;
    CHECK_VKRESULT(vkEnumerateInstanceVersion(&api_version),
                   "Error retrieving Vulkan API version");

    LOG(INFO) << "Vulkan Instance API version: " << VK_VERSION_MAJOR(api_version) << "."
              << VK_VERSION_MINOR(api_version) << "." << VK_VERSION_PATCH(api_version);
  } else {
    // System has a Vulkan 1.0 loader and needs to be updated.
    // TODO(scb): documentation on how to update on common platforms.
    THROW_RUNTIME_EX("System Vulkan loader is not Vulkan 1.1 capable. Please update.")
  }

  // Get function pointer for enumerating Instance extensions
  loader_.EnumerateInstanceExtensionProperties =
      (PFN_vkEnumerateInstanceExtensionProperties)vkGetInstanceProcAddr(
          VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
  RUNTIME_EX_ASSERT(loader_.EnumerateInstanceExtensionProperties,
                    "Vulkan: Failed to retrieve vkEnumerateInstanceExtensionProperties");
}

void VulkanPlatform::shutdownLoader() {
  if (loader_.handle != nullptr) {
    dlclose(loader_.handle);
    loader_.handle = nullptr;
  }
}

namespace {

bool validation_layers_available(const CStrVector& layer_names) {
  // how many layers are available?
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  if (layer_count == 0) {
    return false;
  }
  // which layers are available?
  std::vector<VkLayerProperties> available_layer_props(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layer_props.data());
  // check that all the ones being requested are available
  for (auto const& layer_name : layer_names) {
    bool layer_available = false;
    for (auto const& layer_props : available_layer_props) {
      if (strcmp(layer_name, layer_props.layerName) == 0) {
        layer_available = true;
        break;
      }
    }
    if (!layer_available) {
      return false;
    }
  }
  // we have all the requested layers
  return true;
}

}  // namespace

void VulkanPlatform::createInstance() {
  CHECK(vk_instance_ == VK_NULL_HANDLE);

  auto app_version = VK_MAKE_VERSION(
      VK_INSTANCE_APP_VERS_MAJOR, VK_INSTANCE_APP_VERS_MINOR, VK_INSTANCE_APP_VERS_PATCH);
  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  // TODO(scb): There is no actual core string for the application name. We need to decide
  // on app name versus engine name as well.
  app_info.pApplicationName = "heavydb";
  app_info.applicationVersion = app_version;
  app_info.pEngineName = "heavydb";
  app_info.engineVersion = app_version;
  app_info.apiVersion = VULKAN_API_VERSION;

  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  CStrVector instance_extensions = g_req_instance_extensions;
  if (wsi_) {
    auto wsi_extensions = wsi_->getRequiredInstanceExtensions();
    instance_extensions.insert(
        instance_extensions.end(), wsi_extensions.begin(), wsi_extensions.end());
  }

#if NDEBUG
  // validation mode defaults to disabled, override with env var
  validation_mode_ = ValidationMode::kDisable;
  if (auto* validation_mode_cstr = getenv("OMNISCI_VULKAN_VALIDATION_MODE")) {
    // parse mode
    auto validation_mode_str = to_lower(validation_mode_cstr);
    if (validation_mode_str == "report") {
      validation_mode_ = ValidationMode::kReport;
    } else if (validation_mode_str == "failiferror") {
      validation_mode_ = ValidationMode::kFailIfError;
    } else if (validation_mode_str == "failalways") {
      validation_mode_ = ValidationMode::kFailAlways;
    } else if (validation_mode_str != "" && validation_mode_str != "disable") {
      LOG(WARNING) << "Invalid OMNISCI_VULKAN_VALIDATION_MODE value ('"
                   << validation_mode_str << "'), defaulting to 'report'...";
      validation_mode_ = ValidationMode::kReport;
    }
  }
#else
  // validation mode set at compile time
  validation_mode_ = DEBUG_BUILD_VALIDATION_MODE;
#endif

  // enable validation?
  VkValidationFeaturesEXT features = {};
  std::vector<VkValidationFeatureEnableEXT> validation_features = {
    VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
  // VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
#if ENABLE_SHADER_DEBUG_PRINTF
    VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
#endif
  };

  if (validation_mode_ != ValidationMode::kDisable) {
    // @TODO(se)
    // get requested layer names from the env-var/config?
    if (validation_layers_available(g_req_validation_layers)) {
      for (auto const& layer_name : g_req_validation_layers) {
        LOG(INFO) << "Enabling Vulkan Validation Layer: " << layer_name;
      }
      create_info.enabledLayerCount =
          static_cast<uint32_t>(g_req_validation_layers.size());
      create_info.ppEnabledLayerNames = g_req_validation_layers.data();
    } else {
      LOG(FATAL) << "Requested Vulkan Validation Layers unavailable!";
    }

    if (validation_features.size()) {
      // Enable validation features extenion
      instance_extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);

      // Enabled / disable specific validation features
      features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
      features.enabledValidationFeatureCount = validation_features.size();
      features.pEnabledValidationFeatures = validation_features.data();
      create_info.pNext = &features;
    }
  }
  create_info.enabledExtensionCount = static_cast<uint32_t>(instance_extensions.size());
  create_info.ppEnabledExtensionNames = instance_extensions.data();

  CHECK_VKRESULT(vkCreateInstance(&create_info, nullptr, &vk_instance_),
                 "Error creating Vulkan instance");
}

void VulkanPlatform::destroyInstance() {
  all_device_contexts_.clear();
  physical_devices_.clear();
  vkDestroyInstance(vk_instance_, nullptr);
  vk_instance_ = VK_NULL_HANDLE;
}

void VulkanPlatform::createDebugUtils() {
  debug_utils_ = std::make_unique<VulkanDebugUtils>(*this, vk_instance_);
}

void VulkanPlatform::destroyDebugUtils() {
  debug_utils_ = nullptr;
}

uint32_t VulkanPlatform::getNumGpus() const {
  return static_cast<uint32_t>(physical_devices_.size());
}

std::vector<heavyai::UUID> VulkanPlatform::getUUIDs() const {
  std::vector<heavyai::UUID> ids;
  for (auto const& device : physical_devices_) {
    ids.emplace_back(device.first);
  }
  return ids;
}

DeviceCapabilityBits VulkanPlatform::getHomogeneousCapabilityBits() const {
  return capability_bits_;
}

const DeviceLimits& VulkanPlatform::getHomogenousDeviceLimits() const {
  return limits_;
}

// Experimental (unused)
void VulkanPlatform::logDeviceGroups() {
  uint32_t group_count = 0;
  auto result = vkEnumeratePhysicalDeviceGroups(vk_instance_, &group_count, nullptr);
  if (result == VK_SUCCESS && group_count) {
    std::stringstream stream;
    stream << "Found " << group_count << " physical device groups\n";
    std::vector<VkPhysicalDeviceGroupProperties> device_groups(group_count);
    vkEnumeratePhysicalDeviceGroups(vk_instance_, &group_count, device_groups.data());
    for (uint32_t i = 0; i < group_count; ++i) {
      stream << "  group " << i << " : "
             << "device count - " << device_groups[i].physicalDeviceCount << '\n';
    }
    VLOG(1) << stream.str();
  } else {
    VLOG(1) << "No device groups found";
  }
}

void VulkanPlatform::enumerateDevices(GfxUsage usage, VkSurfaceKHR surface) {
  CHECK(physical_devices_.empty());
  uint32_t device_count = 0;
  enable_extensions_ = g_base_required_device_extensions;
  if (usage == GfxUsage::kCudaInterop) {
    enable_extensions_.insert(enable_extensions_.end(),
                              g_interop_required_device_extensions.begin(),
                              g_interop_required_device_extensions.end());
  }

  // TODO(scb): PhysicalDeviceGroups
  CHECK_VKRESULT(vkEnumeratePhysicalDevices(vk_instance_, &device_count, nullptr),
                 "Error getting Vulkan device count");
  RUNTIME_EX_ASSERT(device_count > 0, "No Vulkan devices found");

  LOG(INFO) << "Found " << device_count << " Vulkan devices\n";

  std::vector<VkPhysicalDevice> vk_devices(device_count);
  CHECK_VKRESULT(
      vkEnumeratePhysicalDevices(vk_instance_, &device_count, vk_devices.data()),
      "Error enumerating Vulkan devices");

  // Initialize the platform capabilities. The final set of capability bits will be
  // a combination of these bits and optional features the used device(s) support that we
  // can take advantage of or are testing
  capability_bits_ = DeviceCapabilityBits::kAll;
  if (usage != GfxUsage::kCudaInterop) {
    // Currently only need to export memory or semaphores if using Cuda interop, so strip
    // these bits if not in interop mode
    capability_bits_ &= (~DeviceCapabilityBits::kBufferMemoryExport);
    capability_bits_ &= (~DeviceCapabilityBits::kImageMemoryExport);
    capability_bits_ &= (~DeviceCapabilityBits::kSemaphoreExport);
    capability_bits_ &= (~DeviceCapabilityBits::kBufferMemoryImport);
    capability_bits_ &= (~DeviceCapabilityBits::kImageMemoryImport);
    capability_bits_ &= (~DeviceCapabilityBits::kSemaphoreImport);
  }

  // Enumerate the physical devices, sorting them into the persistent device map keyed by
  // UUID (used for pairing), and a temporary multimap keyed by DeviceType for later
  // filtering by type
  std::multimap<DeviceType, heavyai::UUID> devices_by_type_map;
  for (auto const& vk_device : vk_devices) {
    auto physical_device_ptr = std::make_unique<VulkanPhysicalDevice>(
        vk_device, surface, allow_raytracing_init_);
    CHECK(physical_device_ptr);
    if (isDeviceSuitable(*physical_device_ptr, usage, enable_extensions_)) {
      auto uuid = physical_device_ptr->getUUID();
      auto [itr, result] =
          physical_devices_.try_emplace(uuid, std::move(physical_device_ptr));
      if (!result) {
        LOG(ERROR) << "Failed to insert Vulkan physical device into map:\n"
                   << physical_device_ptr->buildLog(true, true).str();
      } else {
        VLOG(2) << "Found usable Vulkan device:\n"
                << itr->second->buildLog(true, true).str();
        devices_by_type_map.emplace(itr->second->getType(), uuid);
      }
    } else {
      VLOG(2) << "Skipping Vulkan device: "
              << physical_device_ptr->buildLog(true, true).str();
    }
  }

  // Ensure at least suitable device was found
  // TODO (scb): should this throw and disable vulkan rendering in the server instead?
  LOG_IF(FATAL, physical_devices_.empty()) << "No suitable Vulkan devices found";

  // Filter the list based on usage
  if (usage != GfxUsage::kCudaInterop) {
    // Not using Cuda, so we only need one render device. Find the first avaiable device
    // of the preferred type, or fallback to other types
    // Currently only support descreet and integrated gpus
    std::vector<DeviceType> device_priority;
    if (usage == GfxUsage::kSingleGpuPreferIntegrated) {
      device_priority = {DeviceType::kIntegratedGpu, DeviceType::kDiscreetGpu};
    } else {
      device_priority = {DeviceType::kDiscreetGpu, DeviceType::kIntegratedGpu};
    }

    heavyai::UUID use_uuid{};
    bool found{false};
    for (auto type : device_priority) {
      // Get the first available device of type
      if (auto i = devices_by_type_map.lower_bound(type);
          i != devices_by_type_map.end()) {
        use_uuid = i->second;
        found = true;
        break;
      }
    }
    CHECK(found);

    // Extract the node from the device map, clear it, and reinsert the node.
    // Clients will only see the one device available and can create it by UUID
    auto node = physical_devices_.extract(use_uuid);
    CHECK(node);
    physical_devices_.clear();
    auto insert_result = physical_devices_.insert(std::move(node));
    CHECK(insert_result.inserted);
    auto const& device = *physical_devices_.begin()->second;
    LOG(INFO) << "Renderer using device: " << device.getName()
              << " [UUID:" << device.getUUID() << "]";
  } else {
    LOG(INFO) << "Renderer using devices:";
    for (auto const& itr : physical_devices_) {
      LOG(INFO) << "  " << itr.second->getName() << " [UUID:" << itr.second->getUUID()
                << "]";
      if (any_bits_set(itr.second->getCapabilityBits() &
                       DeviceCapabilityBits::kCanPresent)) {
        LOG(INFO) << "    supports presentation";
      }
    }
  }

  // Build final limits and capability bits including optional device bits (e.g. Mesh
  // shaders)
  auto device_itr = physical_devices_.begin();
  capability_bits_ &= device_itr->second->getCapabilityBits();
  limits_ = device_itr->second->getLimits();
  while (++device_itr != physical_devices_.end()) {
    capability_bits_ &= device_itr->second->getCapabilityBits();
    limits_.combine(device_itr->second->getLimits());
  }

  // Check optional capabilities and add required extensions for enabling
  for (auto const& capability_info : g_optional_capability_extension_infos) {
    // Check if all devices support the capability
    if (any_bits_set(capability_info.capability_bit & capability_bits_)) {
      enable_extensions_.insert(enable_extensions_.end(),
                                capability_info.enable_extensions.begin(),
                                capability_info.enable_extensions.end());
    }
  }
}

bool VulkanPlatform::isDeviceSuitable(const VulkanPhysicalDevice& device,
                                      GfxUsage usage,
                                      const CStrVector& required_extensions) const {
  VLOG(1) << "Checking device: " << device.getName();
  //
  // Check API version
  // This is theoretically redundant with the driver version check
  // but play it safe
  //
  static constexpr auto min_api_version = VULKAN_API_VERSION;
  if (device.getApiVersion() < min_api_version) {
    VLOG(1) << "  Incompatible Vulkan API version detected. Driver supports "
            << vulkan_version_to_string(device.getApiVersion()) << " but "
            << vulkan_version_to_string(min_api_version) << " is the minimum required";
    return false;
  }

  // Check for duplicate UUID (frequently occurs with T4 gcp nodes - 9-20-2020)
  // Should be after driver revision since dup IDs could be driver related
  auto const& uuid = device.getUUID();
  if (physical_devices_.count(uuid)) {
    VLOG(1) << "  Duplicate device UUID found: " << uuid;
    return false;
  }

  auto log_missing_extensions = [](const std::set<std::string>& extensions) {
    std::stringstream ss;
    ss << "  Required Vulkan device extensions not supported:";
    for (auto const& ext : extensions) {
      ss << "\n   " << ext;
    }
    VLOG(1) << ss.str();
  };

  // Check base required extensions
  if (auto missing_extensions = device.supportsExtensions(required_extensions);
      !missing_extensions.empty()) {
    log_missing_extensions(missing_extensions);
    return false;
  }

  // Check type, vendor, and driver version
  auto type = device.getType();
  auto vendor = device.getVendor();
  auto driver_version = device.getDriverVersion();

  switch (type) {
    case DeviceType::kDiscreetGpu:
    case DeviceType::kIntegratedGpu:
      if (vendor == DeviceVendor::kNvidia) {
        auto [major, minor, patch] = unpack_nvidia_driver_version(driver_version);
        // just check major number for now
        if (major < 535) {
          VLOG(1) << "  Incompatible Nvidia driver version. Found " << major << '.'
                  << +minor << '.' << +patch << " but require at least 535.0.0";
          return false;
        }
        // check it supports the required extensions for CUDA interop, if requested
        if (usage == GfxUsage::kCudaInterop) {
          if (auto missing_extensions =
                  device.supportsExtensions(g_interop_required_device_extensions);
              !missing_extensions.empty()) {
            log_missing_extensions(missing_extensions);
            return false;
          }
        }
      } else if (vendor == DeviceVendor::kIntel) {
        // no Intel GPU will support CUDA interop, so fail if that is requested
        if (usage == GfxUsage::kCudaInterop) {
          VLOG(1)
              << "  Intel integrated GPU not supported by current usage configuration";
          return false;
        }
        // Intel discreet GPUs are not supported
        // we support Intel integrated GPUs for basic rendering only
        if (type != DeviceType::kIntegratedGpu) {
          VLOG(1) << "  Intel non-integrated GPUs unsupported";
          return false;
        }
      } else {
        // no other vendors' GPUs of any type supported at all (yet)
        VLOG(1) << "  " << vendor << " GPUs unsupported";
        return false;
      }
      // continue to next step
      break;
    case DeviceType::kCPU:
      // only support mesa-llvmpipe (lavapipe)
      VLOG(1) << "  CPU rendering unsupported";
      return false;
    case DeviceType::kVirtualGpu:
      VLOG(1) << "  Virtual GPUs unsupported";
      // unknown
      return false;
    case DeviceType::kOther:
      VLOG(1) << "  Unknown device type";
      return false;
  }

  // Check required features
  // handle by device type!
  {
    bool all_supported = true;

    auto check_feature = [&](VkBool32 value, std::string_view name) {
      if (!value) {
        all_supported = false;
        VLOG(1) << "  Required feature not supported: " << name;
      }
    };

    // base
    auto const& base_features = device.getBaseFeatures();
    check_feature(base_features.geometryShader, "geometryShader");
    check_feature(base_features.multiDrawIndirect, "multiDrawIndirect");
    check_feature(base_features.largePoints, "largePoints");
    check_feature(base_features.fragmentStoresAndAtomics, "fragmentStoresAndAtomics");
    check_feature(base_features.shaderFloat64, "shaderFloat64");
    check_feature(base_features.shaderInt64, "shaderInt64");
    check_feature(base_features.independentBlend, "independentBlend");
    check_feature(base_features.sampleRateShading, "sampleRateShading");
    check_feature(base_features.shaderStorageImageMultisample,
                  "shaderStorageImageMultisample");
    check_feature(base_features.pipelineStatisticsQuery, "pipelineStatisticsQuery");
    check_feature(base_features.occlusionQueryPrecise, "occlusionQueryPrecise");

    // Vulkan 1.1 features
    auto const& vk11_features = device.getVulkan11Features();
    check_feature(vk11_features.shaderDrawParameters, "shaderDrawParameters");

    // Vulkan 1.2 features
    auto const& vk12_features = device.getVulkan12Features();
    check_feature(vk12_features.uniformBufferStandardLayout,
                  "uniformBufferStandardLayout");
    check_feature(vk12_features.shaderBufferInt64Atomics, "shaderBufferInt64Atomics");
    check_feature(vk12_features.scalarBlockLayout, "scalarBlockLayout");
    check_feature(vk12_features.hostQueryReset, "hostQueryReset");

    // Vulkan 1.3 features
    auto const& vk13_features = device.getVulkan13Features();
    check_feature(vk13_features.synchronization2, "synchronization2");

    if (!all_supported) {
      return false;
    }
  }

  return true;
}

DeviceContextUqPtr VulkanPlatform::createDeviceContext(const heavyai::UUID uuid,
                                                       const DeviceId gpu_id) {
  auto const& physical_device_itr = physical_devices_.find(uuid);
  if (physical_device_itr != physical_devices_.end()) {
    auto const& physical_device = *physical_device_itr->second;
    // Create DeviceContext and configure Vulkan logical device
    auto device = std::make_unique<VulkanDeviceContext>(driver_, gpu_id, physical_device);

    // Required features - support must be checked in isDeviceSuitable
    VkPhysicalDeviceFeatures2 enable_features = {};
    VkStructChainBuilder features_chain(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                        &enable_features);

    //
    // base
    //

    enable_features.features.geometryShader = VK_TRUE;
    enable_features.features.multiDrawIndirect = VK_TRUE;
    enable_features.features.largePoints = VK_TRUE;
    enable_features.features.fragmentStoresAndAtomics = VK_TRUE;
    enable_features.features.shaderFloat64 = VK_TRUE;
    enable_features.features.shaderInt64 = VK_TRUE;
    enable_features.features.independentBlend = VK_TRUE;
    enable_features.features.sampleRateShading = VK_TRUE;
    enable_features.features.shaderStorageImageMultisample = VK_TRUE;
    enable_features.features.pipelineStatisticsQuery = VK_TRUE;
    enable_features.features.occlusionQueryPrecise = VK_TRUE;

    // Vulkan 1.1 features
    VkPhysicalDeviceVulkan11Features vk11_features = {};
    features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                       &vk11_features);

    vk11_features.shaderDrawParameters = VK_TRUE;

    // Vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features vk12_features = {};
    features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                       &vk12_features);

    vk12_features.uniformBufferStandardLayout = VK_TRUE;
    vk12_features.shaderBufferInt64Atomics = VK_TRUE;
    if (physical_device.getVulkan12Features().shaderSubgroupExtendedTypes) {
      vk12_features.shaderSubgroupExtendedTypes = VK_TRUE;
    }
    vk12_features.scalarBlockLayout = VK_TRUE;
    vk12_features.hostQueryReset = VK_TRUE;

    // Vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features vk13_features = {};
    features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                       &vk13_features);
    vk13_features.synchronization2 = VK_TRUE;

    //
    // Optional extensions
    //
    auto const& capability_bits = physical_device.getCapabilityBits();

    // Mesh shaders (also requires FragmentShadingRate, but perhaps split out?)
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader_features = {};
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragment_shading_rate_features = {};
    if (any_bits_set(capability_bits & DeviceCapabilityBits::kMeshShaders)) {
      features_chain.add(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
                         &mesh_shader_features);
      mesh_shader_features.meshShader = VK_TRUE;
      mesh_shader_features.taskShader = VK_TRUE;
      mesh_shader_features.primitiveFragmentShadingRateMeshShader = VK_TRUE;
      mesh_shader_features.meshShaderQueries = VK_TRUE;
      features_chain.add(
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
          &fragment_shading_rate_features);
      fragment_shading_rate_features.primitiveFragmentShadingRate = VK_TRUE;
    }

    // Fragment shader interlock
    VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT
        fragment_shader_interlock_features = {};
    bool use_fragment_shader_interlock = false;
    if (any_bits_set(capability_bits &
                     DeviceCapabilityBits::kFragmentShaderPixelInterlock)) {
      fragment_shader_interlock_features.fragmentShaderPixelInterlock = VK_TRUE;
      use_fragment_shader_interlock = true;
    }
    if (any_bits_set(capability_bits &
                     DeviceCapabilityBits::kFragmentShaderSampleInterlock)) {
      fragment_shader_interlock_features.fragmentShaderSampleInterlock = VK_TRUE;
      use_fragment_shader_interlock = true;
    }
    if (use_fragment_shader_interlock) {
      features_chain.add(
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
          &fragment_shader_interlock_features);
    }

    // Raytracing
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR raytracing_pipeline_features = {};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features = {};
    if (any_bits_set(capability_bits & DeviceCapabilityBits::kRaytracing)) {
      raytracing_pipeline_features.rayTracingPipeline = VK_TRUE;
      raytracing_pipeline_features.rayTracingPipelineTraceRaysIndirect = VK_TRUE;
      raytracing_pipeline_features.rayTraversalPrimitiveCulling = VK_TRUE;
      features_chain.add(
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
          &raytracing_pipeline_features);

      accel_features.accelerationStructure = VK_TRUE;
      accel_features.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE;
      features_chain.add(
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
          &accel_features);
    }

    if (any_bits_set(capability_bits & DeviceCapabilityBits::kBufferDeviceAddress)) {
      vk12_features.bufferDeviceAddress = VK_TRUE;
    }

    // Create the VkDevice
    // Normally we use the homogenous capabilities to handle extension enabling,
    // Presentation is typically not homogenous so we need to handle it on a per
    // device basis
    CStrVector enable_extensions = enable_extensions_;
    DeviceCapabilityBits capability_bits_to_use = capability_bits_;

    bool using_presentation =
        any_bits_set(capability_bits & DeviceCapabilityBits::kCanPresent);

    if (using_presentation) {
      capability_bits_to_use |= DeviceCapabilityBits::kCanPresent;
      enable_extensions.insert(enable_extensions.end(),
                               g_present_required_device_extensions.begin(),
                               g_present_required_device_extensions.end());
    }

    try {
      device->createDevice(enable_extensions,
                           enable_features,
                           capability_bits_to_use,
                           command_timeout_,
                           using_presentation ? wsi_.get() : nullptr);
    } catch (const std::exception& e) {
      // If createDevice() throws, it means that CHECK_VKRESULT(vkCreateDevice())
      // caught a Vulkan error. This will leave the VulkanDeviceObject in an uninitialized
      // state. If it is allowed to self-destruct (when the unique_ptr goes out of scope)
      // there is a deliberate CHECK in its destructor because the destructor_locked_ flag
      // was not cleared, indicating that the object was not shut down correctly. This is
      // done by calling destroyDeviceContext() (below) which calls destroyDevice() on the
      // device to remove any low-level resources which may have been left dangling, and
      // then clears the flag which would otherwise have caused the CHECK in the
      // destructor, and finally explicitly invokes the destruction by clearing and
      // consuming the unique_ptr.
      //
      // A device creation failure is not a recoverable state, however. Vulkan is broken
      // and the server must exit. But at least do this neatly without an ugly crash.
      //
      // @TODO(simon) More investigation into what could possible leave Vulkan in a state
      // where a device will not re-create.
      LOG(ERROR) << "Exception thrown while creating Vulkan Device, destroying device "
                    "context cleanly";
      LOG(ERROR) << e.what();
      destroyDeviceContext(std::move(device));
      LOG(FATAL) << "Vulkan Driver is in an unrecoverable state. Server must exit.";
    }

#if ENABLE_RENDERDOC
    renderdoc::set_vulkan_device(vk_instance_);
#endif

    // capture this DC for debug callback
    all_device_contexts_.insert(device.get());

    LOG(INFO) << "Initializing render device: " << physical_device.getName();

    return device;
  } else {
    // TODO(scb): More detailed error message?
    THROW_RUNTIME_EX("Unable to find Vulkan device matching requested device UUID.");
    return nullptr;
  }
}

bool VulkanPlatform::destroyDeviceContext(DeviceContextUqPtr device_ctx) {
  CHECK(device_ctx);
  auto* vulkan_device = static_cast<VulkanDeviceContext*>(device_ctx.get());
  auto did_leak_resources = vulkan_device->destroyDevice();
  all_device_contexts_.erase(vulkan_device);
  vulkan_device->destructor_locked_ = false;
  device_ctx = nullptr;
  return did_leak_resources;
}

WindowSystemIntegration* VulkanPlatform::getWSI() const {
  return wsi_.get();
}

void VulkanPlatform::debugCallback() const {
// invoke required failure behaviour
#if LOG_IMAGE_INFO_ON_VALIDATION_ERROR
  for (auto const* device_context : all_device_contexts_) {
    device_context->getResourceManager().logImageResourceDetails();
  }
#endif
}

int VulkanPlatform::suppress_validation_count_ = 0;
VulkanPlatform::ValidationMode VulkanPlatform::validation_mode_ =
    VulkanPlatform::ValidationMode::kDisable;

void VulkanPlatform::pushSuppressValidationMessages() {
  suppress_validation_count_++;
}

void VulkanPlatform::popSuppressValidationMessages() {
  CHECK_GT(suppress_validation_count_, 0);
  suppress_validation_count_--;
}

bool VulkanPlatform::areValidationMessagesSuppressed() {
  return suppress_validation_count_ > 0;
}

VulkanPlatform::ValidationMode VulkanPlatform::getValidationMode() {
  return validation_mode_;
}

VulkanDebugUtils& VulkanPlatform::getDebugUtils() const {
  CHECK(debug_utils_);
  return *debug_utils_;
}

MemoryUsageInfo VulkanPlatform::getPeakMemoryUsage() const {
  MemoryUsageInfo rtn{0, 0ul};
  for (auto const* device : all_device_contexts_) {
    if (device) {
      if (auto size = device->getPeakMemoryUsage(); size > rtn.size) {
        rtn.device = device->getGpuId();
        rtn.size = size;
      }
    }
  }
  return rtn;
}

}  // namespace gfx
