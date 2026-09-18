/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <glm/mat3x4.hpp>
#include <glm/vec3.hpp>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Resource.h"
#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/Types.h"

namespace gfx {

class AccelerationStructure : public Resource {
 public:
  struct TriangleData {
    DeviceAddress vertex_buffer_address;
    uint32_t num_vertices{0};
    uint64_t vertex_stride{0};
    DeviceAddress index_buffer_address;
    IndexBufferDataType index_type;
    uint32_t num_indices{0};
    DeviceAddress transform_address;
    uint32_t num_triangles{0};
  };

  struct AABB {
    glm::vec3 min;
    glm::vec3 max;
    uint64_t pad;  // alignment must be multiple of 8
    AABB() = default;
    AABB(const glm::vec3& min, const glm::vec3& max) : min{min}, max{max}, pad{0} {}
  };

  struct AABBData {
    BufferWrapper& bounds_buffer;
    uint32_t num_bounds{0};
    uint64_t bounds_stride{0};
  };

  using InstanceMatrixType = glm::mat3x4;

  class Builder {
   public:
    virtual ~Builder() = default;

    // Bottom-level (blas)
    virtual void addTriangleData(const TriangleData& data) = 0;
    virtual void addAABBData(const AABBData& data) = 0;

    // Top-level (tlas)
    virtual void addInstanceData(const InstanceMatrixType& matrix,
                                 DeviceAddress blas_device_address,
                                 uint8_t visibility_mask = 0XFF,
                                 uint32_t instance_custom_index = 0u,
                                 uint32_t shader_binding_table_record_offset = 0u) = 0;
    virtual uint64_t getInstancesBufferSize() const = 0;

    virtual void clearBlasData() = 0;
    virtual void clearTlasData() = 0;
  };

  explicit AccelerationStructure(const DeviceContext& device_ctx,
                                 std::string_view resource_tracking_string,
                                 AccelerationStructureType type)
      : Resource(device_ctx,
                 resource_tracking_string,
                 ResourceType::kAccelerationStructure)
      , type_{type}
      , device_address_{0u} {}
  ~AccelerationStructure() override = default;

  using BuilderUqPtr = std::unique_ptr<Builder>;

  AccelerationStructureType getType() const { return type_; }

  ResourceHandle getResourceHandle() const override { return resource_handle_; }
  BufferWrapper& getBuffer() const { return *buffer_; }
  DeviceAddress getDeviceAddress() const { return device_address_; }

 protected:
  AccelerationStructureType type_;
  ResourceHandle resource_handle_;
  BufferWrapperUqPtr buffer_;
  DeviceAddress device_address_;
};

}  // namespace gfx
