/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Resources/AccelerationStructure.h"

#include <vulkan/vulkan.h>

namespace gfx {

//
// VulkanAccelerationStructureBuilder
//
class VulkanAccelerationStructureBuilder : public AccelerationStructure::Builder {
 public:
  explicit VulkanAccelerationStructureBuilder(const DeviceContext& device);
  ~VulkanAccelerationStructureBuilder() override = default;

  void addTriangleData(const AccelerationStructure::TriangleData& data) override;
  void addAABBData(const AccelerationStructure::AABBData& data) override;
  resource_ptr<AccelerationStructure> buildBottomLevel(
      std::string_view resource_tracking_string) const;

  void addInstanceData(const AccelerationStructure::InstanceMatrixType& matrix,
                       DeviceAddress blas_device_address,
                       uint8_t visibility_mask = 0XFF,
                       uint32_t instance_custom_index = 0u,
                       uint32_t sbt_record_offset = 0u) override;
  uint64_t getInstancesBufferSize() const override;
  resource_ptr<AccelerationStructure> buildTopLevel(
      std::string_view resource_tracking_string,
      BufferWrapper& instances_buffer) const;

  void clearBlasData() override;
  void clearTlasData() override;

 private:
  const VulkanDeviceContext& device_;
  std::vector<VkAccelerationStructureGeometryKHR> geometry_data_;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> build_range_infos_;
  uint32_t num_primitives_;
  std::vector<VkAccelerationStructureInstanceKHR> instance_data_;
};

//
// VulkanAccelerationStructure
//
class VulkanAccelerationStructure : public AccelerationStructure {
 public:
  explicit VulkanAccelerationStructure(const DeviceContext& device_ctx,
                                       std::string_view resource_tracking_string,
                                       AccelerationStructureType type);
  ~VulkanAccelerationStructure() override;

  void build(const std::vector<VkAccelerationStructureGeometryKHR>& geometry_data,
             const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& build_ranges,
             VkAccelerationStructureBuildSizesInfoKHR build_sizes_info);

 protected:
  void cleanupResourceBase() override;
  void makeEmpty() override;
};

}  // namespace gfx
