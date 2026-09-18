/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanAccelerationStructure.h"

#include <iostream>

#include <glm/gtc/type_ptr.hpp>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/RenderError.h"
#include "Shared/DebugOutputStream.h"
#include "Shared/scope.h"

#define DEBUG_PRINT_ACCEL_INFO false
#define ACCEL_DEBUG_PRINT() DEBUG_OUTPUT_STREAM(DEBUG_PRINT_ACCEL_INFO, std::cout)

namespace gfx {

namespace {
VkAccelerationStructureBuildSizesInfoKHR get_build_sizes(
    const VulkanDeviceContext& device,
    VkAccelerationStructureTypeKHR accel_type,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometry_data,
    const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& range_infos) {
  VkAccelerationStructureBuildGeometryInfoKHR build_info{};
  build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  build_info.type = accel_type;
  build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  build_info.geometryCount = geometry_data.size();
  build_info.pGeometries = geometry_data.data();

  std::vector<uint32_t> max_primitive_counts(geometry_data.size());
  for (uint32_t i = 0; i < geometry_data.size(); ++i) {
    max_primitive_counts[i] = range_infos[i].primitiveCount;
  }

  VkAccelerationStructureBuildSizesInfoKHR build_sizes_info{};
  build_sizes_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  device.getFunctions().vkGetAccelerationStructureBuildSizesKHR(
      device.getHandle(),
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
      &build_info,
      max_primitive_counts.data(),
      &build_sizes_info);

  return build_sizes_info;
}
}  // namespace

//
// VulkanAccelerationStructureBuilder
//
VulkanAccelerationStructureBuilder::VulkanAccelerationStructureBuilder(
    const DeviceContext& device)
    : Builder()
    , device_{static_cast<const VulkanDeviceContext&>(device)}
    , num_primitives_{0u} {}

//
// addTriangleData
//
void VulkanAccelerationStructureBuilder::addTriangleData(
    const AccelerationStructure::TriangleData& data) {
  // Get device addresses first in case of error
  VkDeviceOrHostAddressConstKHR vbo_device_addr{};
  VkDeviceOrHostAddressConstKHR ibo_device_addr{};
  VkDeviceOrHostAddressConstKHR transform_device_addr{};

  vbo_device_addr.deviceAddress = data.vertex_buffer_address;
  CHECK_NE(vbo_device_addr.deviceAddress, 0u);
  ibo_device_addr.deviceAddress = data.index_buffer_address;
  CHECK_NE(ibo_device_addr.deviceAddress, 0u);
  transform_device_addr.deviceAddress = data.transform_address;

  // VkAccelerationStructureGeometryKHR
  geometry_data_.resize(geometry_data_.size() + 1);
  auto& vk_data = geometry_data_.back();

  vk_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  vk_data.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  vk_data.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;

  auto& vk_triangles = vk_data.geometry.triangles;
  vk_triangles.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  // vertex data
  vk_triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  vk_triangles.vertexData = vbo_device_addr;
  vk_triangles.maxVertex = data.num_vertices;
  vk_triangles.vertexStride = data.vertex_stride;
  // index data
  vk_triangles.indexType = index_buffer_data_type_to_vk_index_type(data.index_type);
  vk_triangles.indexData = ibo_device_addr;
  // transform data
  vk_triangles.transformData = transform_device_addr;

  // VkAccelerationStructureBuildRangeInfoKHR
  build_range_infos_.resize(build_range_infos_.size() + 1);
  auto& build_range_info = build_range_infos_.back();

  build_range_info.primitiveCount = data.num_triangles;
  build_range_info.primitiveOffset = 0;
  build_range_info.firstVertex = 0;
  build_range_info.transformOffset = 0;

  num_primitives_ += data.num_triangles;
}

//
// addAABBData
//
void VulkanAccelerationStructureBuilder::addAABBData(
    const AccelerationStructure::AABBData& data) {
  VkDeviceOrHostAddressConstKHR device_addr{};
  device_addr.deviceAddress = data.bounds_buffer.getDeviceAddress();
  CHECK_NE(device_addr.deviceAddress, 0u);

  geometry_data_.resize(geometry_data_.size() + 1);
  auto& vk_data = geometry_data_.back();

  vk_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  vk_data.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  vk_data.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;

  auto& vk_aabbs = vk_data.geometry.aabbs;
  vk_aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
  vk_aabbs.data = device_addr;
  vk_aabbs.stride = data.bounds_stride;
  vk_aabbs.pNext = nullptr;

  // VkAccelerationStructureBuildRangeInfoKHR
  build_range_infos_.resize(build_range_infos_.size() + 1);
  auto& build_range_info = build_range_infos_.back();

  build_range_info.primitiveCount = data.num_bounds;
  build_range_info.primitiveOffset = num_primitives_;
  build_range_info.firstVertex = 0;
  build_range_info.transformOffset = 0;

  num_primitives_ += data.num_bounds;
}

//
// buildBottomLevel
//
resource_ptr<AccelerationStructure> VulkanAccelerationStructureBuilder::buildBottomLevel(
    std::string_view resource_tracking_string) const {
  ACCEL_DEBUG_PRINT() << "Building bottom level acceleration structure" << std::endl;
  RUNTIME_EX_ASSERT(num_primitives_ > 0u,
                    "Attempting to build empty bottom-level acceleration structure");

  // Get structure size
  auto build_sizes_info = get_build_sizes(device_,
                                          VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                          geometry_data_,
                                          build_range_infos_);

  ACCEL_DEBUG_PRINT() << "num primitives: " << num_primitives_ << std::endl;
  ACCEL_DEBUG_PRINT() << "accel structure build size: "
                      << build_sizes_info.accelerationStructureSize << std::endl;
  ACCEL_DEBUG_PRINT() << "scratch buffer size: " << build_sizes_info.buildScratchSize
                      << std::endl;

  auto& resource_mgr = static_cast<VulkanResourceManager&>(device_.getResourceManager());

  // Create acceleration structure
  resource_ptr<AccelerationStructure> blas;

  try {
    blas = resource_mgr.createAccelerationStructure(
        resource_tracking_string, AccelerationStructureType::kBottomLevel);

    static_cast<VulkanAccelerationStructure&>(*blas).build(
        geometry_data_, build_range_infos_, build_sizes_info);
  } catch (...) {
    if (blas) {
      resource_mgr.destroyAccelerationStructure(std::move(blas));
    }
    std::rethrow_exception(std::current_exception());
  }
  ACCEL_DEBUG_PRINT() << "blas device addr: " << blas->getDeviceAddress() << std::endl;
  ACCEL_DEBUG_PRINT() << "blas ready" << std::endl;
  return blas;
}

//
// addInstanceData
//
void VulkanAccelerationStructureBuilder::addInstanceData(
    const AccelerationStructure::InstanceMatrixType& matrix,
    DeviceAddress blas_device_addr,
    uint8_t visibility_mask,
    uint32_t instance_custom_index,
    uint32_t sbt_record_offset) {
  CHECK_NE(blas_device_addr, 0u);
  instance_data_.resize(instance_data_.size() + 1);

  auto& new_instance = instance_data_.back();
  std::memcpy(
      &new_instance.transform, glm::value_ptr(matrix), sizeof(VkTransformMatrixKHR));
  new_instance.instanceCustomIndex = instance_custom_index;
  new_instance.mask = visibility_mask;
  new_instance.instanceShaderBindingTableRecordOffset = sbt_record_offset;
  new_instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  new_instance.accelerationStructureReference = blas_device_addr;
}

//
// getInstancesBufferSize
//
uint64_t VulkanAccelerationStructureBuilder::getInstancesBufferSize() const {
  return instance_data_.size() * sizeof(VkAccelerationStructureInstanceKHR);
}

//
// buildTopLevel
//
resource_ptr<AccelerationStructure> VulkanAccelerationStructureBuilder::buildTopLevel(
    std::string_view resource_tracking_string,
    BufferWrapper& instances_buffer) const {
  ACCEL_DEBUG_PRINT() << "Building top level acceleration structure" << std::endl;
  uint32_t num_instances = instance_data_.size();
  RUNTIME_EX_ASSERT(num_instances > 0u,
                    "Attempting to build empty top-level acceleration structure");
  auto const instances_buffer_size = getInstancesBufferSize();
  CHECK_GE(instances_buffer.getNumBytes(), instances_buffer_size);

  VkDeviceOrHostAddressConstKHR instances_device_addr;
  instances_device_addr.deviceAddress = instances_buffer.getDeviceAddress();
  CHECK_NE(instances_device_addr.deviceAddress, 0u);

  // Fill instances buffer
  instances_buffer.updateSubData(instance_data_.data(), instances_buffer_size, 0u);

  // Use a single geometry for all instances
  // TODO: multiple geometries
  std::vector<VkAccelerationStructureGeometryKHR> vk_geometries(1);
  auto& vk_geometry = vk_geometries.back();
  vk_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  vk_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  vk_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  vk_geometry.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  vk_geometry.geometry.instances.arrayOfPointers = VK_FALSE;
  vk_geometry.geometry.instances.data = instances_device_addr;

  // Build range info
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> range_infos(1);
  auto& build_range_info = range_infos.back();
  build_range_info.primitiveCount = num_instances;
  build_range_info.primitiveOffset = 0;
  build_range_info.firstVertex = 0;
  build_range_info.transformOffset = 0;

  // Get structure size
  auto build_sizes_info = get_build_sizes(
      device_, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, vk_geometries, range_infos);

  ACCEL_DEBUG_PRINT() << "num instances: " << num_instances << std::endl;
  ACCEL_DEBUG_PRINT() << "accel structure build size: "
                      << build_sizes_info.accelerationStructureSize << std::endl;
  ACCEL_DEBUG_PRINT() << "scratch buffer size: " << build_sizes_info.buildScratchSize
                      << std::endl;

  auto& resource_mgr = static_cast<VulkanResourceManager&>(device_.getResourceManager());

  resource_ptr<AccelerationStructure> tlas;

  try {
    // Create accel structure
    tlas = resource_mgr.createAccelerationStructure(resource_tracking_string,
                                                    AccelerationStructureType::kTopLevel);

    static_cast<VulkanAccelerationStructure&>(*tlas).build(
        vk_geometries, range_infos, build_sizes_info);
  } catch (...) {
    if (tlas) {
      resource_mgr.destroyAccelerationStructure(std::move(tlas));
    }
    std::rethrow_exception(std::current_exception());
  }
  ACCEL_DEBUG_PRINT() << "tlas device addr: " << tlas->getDeviceAddress() << std::endl;
  ACCEL_DEBUG_PRINT() << "tlas ready" << std::endl;
  return tlas;
}

void VulkanAccelerationStructureBuilder::clearBlasData() {
  geometry_data_.clear();
  build_range_infos_.clear();
  num_primitives_ = 0;
}

void VulkanAccelerationStructureBuilder::clearTlasData() {
  instance_data_.clear();
}

//
// VulkanAccelerationStructure
//
VulkanAccelerationStructure::VulkanAccelerationStructure(
    const DeviceContext& device_ctx,
    std::string_view resource_tracking_string,
    AccelerationStructureType type)
    : AccelerationStructure(device_ctx, resource_tracking_string, type) {}

VulkanAccelerationStructure::~VulkanAccelerationStructure() {
  cleanupResource();
}

void VulkanAccelerationStructure::cleanupResourceBase() {
  if (resource_handle_) {
    auto const& vk_device = static_cast<const VulkanDeviceContext&>(getDeviceContext());
    vk_device.getFunctions().vkDestroyAccelerationStructureKHR(
        vk_device.getHandle(),
        reinterpret_cast<VkAccelerationStructureKHR>(resource_handle_),
        nullptr);
  }
  if (buffer_) {
    getDeviceContext().getResourceManager().destroyBuffer(std::move(buffer_));
  }
}

void VulkanAccelerationStructure::makeEmpty() {}

void VulkanAccelerationStructure::build(
    const std::vector<VkAccelerationStructureGeometryKHR>& geometry_data,
    const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& build_ranges,
    VkAccelerationStructureBuildSizesInfoKHR build_sizes_info) {
  auto const& vk_device = static_cast<const VulkanDeviceContext&>(getDeviceContext());
  auto& resource_mgr = vk_device.getResourceManager();

  // Create scratch buffer
  auto scratch_buffer = resource_mgr.createBuffer(
      "Accel Scratch",
      {BufferType::kUnspecified,
       build_sizes_info.buildScratchSize,
       BufferUsageBits::kStorageBufferBit | BufferUsageBits::kDeviceAddressBit});

  ScopeGuard scratch_cleanup = [&]() {
    resource_mgr.destroyBuffer(std::move(scratch_buffer));
  };

  // Create acceleration structure buffer
  buffer_ = getDeviceContext().getResourceManager().createBuffer(
      getTrackingData().origin,
      {BufferType::kAccelerationStructureBuffer,
       build_sizes_info.accelerationStructureSize,
       BufferUsageBits::kDeviceAddressBit});

  // Create VkAccelerationStructure
  VkAccelerationStructureCreateInfoKHR accel_structure_ci{};
  accel_structure_ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
  accel_structure_ci.buffer =
      reinterpret_cast<VkBuffer>(buffer_->getBuffer().getResourceHandle());
  accel_structure_ci.size = build_sizes_info.accelerationStructureSize;
  accel_structure_ci.type = accel_struct_type_to_vk_accel_struct_type(type_);
  VkAccelerationStructureKHR vk_accel_handle;
  CHECK_VKRESULT(
      vk_device.getFunctions().vkCreateAccelerationStructureKHR(
          vk_device.getHandle(), &accel_structure_ci, nullptr, &vk_accel_handle),
      "Creating ray acceleration structure");
  resource_handle_ = reinterpret_cast<ResourceHandle>(vk_accel_handle);

  // Build geometry info
  VkAccelerationStructureBuildGeometryInfoKHR build_geometry_info{};
  build_geometry_info.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  build_geometry_info.type = accel_struct_type_to_vk_accel_struct_type(type_);
  build_geometry_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  build_geometry_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  build_geometry_info.dstAccelerationStructure = vk_accel_handle;
  build_geometry_info.geometryCount = geometry_data.size();
  build_geometry_info.pGeometries = geometry_data.data();
  build_geometry_info.scratchData.deviceAddress =
      scratch_buffer->getBuffer().getDeviceAddress();

  // Build range info
  std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> build_range_info_ptrs;
  for (auto& info : build_ranges) {
    build_range_info_ptrs.emplace_back(&info);
  }

  // Build structure on device
  vk_device.getCommandList()
      .buildAccelerationStructure(&build_geometry_info, build_range_info_ptrs.data())
      .flush("Build Accel", CommandList::SubmitType::kWaitComplete);

  // Get device address
  VkAccelerationStructureDeviceAddressInfoKHR device_addr_info{};
  device_addr_info.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
  device_addr_info.accelerationStructure = vk_accel_handle;
  device_address_ = vk_device.getFunctions().vkGetAccelerationStructureDeviceAddressKHR(
      vk_device.getHandle(), &device_addr_info);

  RUNTIME_EX_ASSERT(device_address_ > 0u,
                    "Failed to get AccelerationStructure DeviceAddress");

  setUsable();
}

}  // namespace gfx
