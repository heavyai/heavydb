/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>

#include "GfxDriver/Pipeline/PrimitiveAssembly.h"
#include "GfxDriver/Pipeline/ShaderBindingTable.h"
#include "GfxDriver/Pipeline/Types.h"
#include "GfxDriver/Resources/Resource.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

// struct for defining a pipeline specialization constant to be passed to
// createPipeline. Currently only supported for compute pipelines
struct SpecializationMapEntry {
  uint32_t constant_id;  // id declared in the shader via the layout decorator
  uint32_t offset;       // offset into cpu data buffer used when creating the VkPipeline
  uint64_t size;         // size of the data for this constant
};

//
// Pipeline class
//
// Base class for all pipeline types, with functions common to all pipeline types
//
class Pipeline : public Resource {
 public:
  enum Type { kGraphics, kCompute, kRaytracing };

  explicit Pipeline(const DeviceContext& device_ctx,
                    std::string_view resource_tracking_string,
                    const Material& material)
      : Resource(device_ctx, resource_tracking_string, ResourceType::kPipeline)
      , material_{material} {};
  Pipeline() = delete;
  ~Pipeline() override = default;

  virtual Type getType() const = 0;

  const Material& getMaterial() const { return material_; }
  virtual ResourceHandle getLayout() const = 0;

  // Get ResourceHandle for a pipeline specialization
  // Vulkan compute pipelines only
  virtual ResourceHandle getPipelineHandle(const uint32_t specialization_id) const = 0;

  // Resource
  ResourceHandle getResourceHandle() const override { return getPipelineHandle(0u); }

 protected:
  const Material& material_;
};

//
// GraphicsPipeline class
//
// Base class for graphics pipeline implementations, defining additional
// API specific to graphics
class GraphicsPipeline : public Pipeline {
 public:
  enum DynamicStateBits { kNone = 0x00, kViewport = 0x01, kAll = 0xFFFF };

  explicit GraphicsPipeline(const DeviceContext& device_ctx,
                            std::string_view resource_tracking_string,
                            const Material& material,
                            const PipelineDescriptor& pipeline_descriptor,
                            const PrimitiveAssembly* primitive_assembly)
      : Pipeline(device_ctx, resource_tracking_string, material)
      , pipeline_descriptor_{pipeline_descriptor}
      , primitive_assembly_{primitive_assembly} {};
  ~GraphicsPipeline() override = default;

  Type getType() const override { return Type::kGraphics; }

  virtual DynamicStateBits getDynamicStateBits() const = 0;

  // Create API resource object (VkPipeline)
  virtual void create(const RenderPass& render_pass) = 0;

  PrimitiveTopology getTopology() const {
    if (primitive_assembly_) {
      return primitive_assembly_->getTopology();
    } else {
      return PrimitiveTopology::kTriangleList;
    }
  }

 protected:
  const PipelineDescriptor& pipeline_descriptor_;
  const PrimitiveAssembly* primitive_assembly_;
};

//
// ComputePipeline class
//
// Base class for compute pipeline implementations
// Adds API specific to compute
class ComputePipeline : public Pipeline {
 public:
  explicit ComputePipeline(const DeviceContext& device_ctx,
                           std::string_view resource_tracking_string,
                           const Material& material)
      : Pipeline(device_ctx, resource_tracking_string, material) {}
  ~ComputePipeline() override = default;

  Type getType() const override { return Type::kCompute; }

  // Create API resource object (VkPipeline)
  // Creates an unspecialized pipeline, using default values for any specialization
  // constants Destroys all existing pipeline resources
  virtual void create() = 0;

  // create a specialization of the pipeline
  // If a pipeline handle exists for `specialization_id`
  // it will be destroyed and replaced by a new pipeline
  virtual void createSpecialization(const uint32_t specialization_id,
                                    const void* specialization_data,
                                    const uint64_t data_size) = 0;
};

//
// RaytracingPipeline class
//
class RaytracingPipeline : public Pipeline {
 public:
  struct ShaderGroup {
    enum class Type { kGeneral, kTriangleHit, kProceduralHit };
    Type type{Type::kGeneral};
    ShaderBindingTable::Entry sbt_entry{ShaderBindingTable::Entry::kCount};
    std::vector<uint8_t> handle_data;

    explicit ShaderGroup(Type type, ShaderBindingTable::Entry sbt_entry)
        : type{type}, sbt_entry{sbt_entry} {}
  };

  explicit RaytracingPipeline(const DeviceContext& device_ctx,
                              std::string_view resource_tracking_string,
                              const Material& material)
      : Pipeline(device_ctx, resource_tracking_string, material) {}

  ~RaytracingPipeline() override = default;

  Type getType() const override { return Type::kRaytracing; }

  // TODO(scb): VulkanRaytracingPipelines support dynamic states, most notably
  // vkCmdSetRayTracingPipelineStackSizeKHR to set the max "depth" of a ray tree
  // For more information see:
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#ray-tracing-pipeline-stack
  // virtual DynamicStateBits getDynamicStateBits() const = 0;

  virtual void create(uint32_t max_ray_recursion_depth) = 0;

  // Shader group access
  // valid only after calling create()
  virtual const std::vector<ShaderGroup>& getShaderGroups() const = 0;
};

std::ostream& operator<<(std::ostream& os,
                         const RaytracingPipeline::ShaderGroup::Type value);

std::ostream& operator<<(std::ostream& os, const RaytracingPipeline::ShaderGroup& value);

}  // namespace gfx
