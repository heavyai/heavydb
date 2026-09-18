/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>
#include <vector>

#include "GfxDriver/Commands/QueryPool.h"
#include "GfxDriver/Pipeline/Enums.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PushConstantRanges.h"
#include "GfxDriver/Pipeline/Types.h"
#include "GfxDriver/Resources/AccelerationStructure.h"
#include "GfxDriver/Resources/Framebuffer.h"
#include "GfxDriver/Resources/HostVisibleBufferWrapper.h"
#include "GfxDriver/Resources/IndexBuffer.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/TextureSamplerState.h"
#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

class LocalBufferAllocator;
using DefaultBufferAllocator = LocalBufferAllocator;

class ResourceManager {
 public:
  explicit ResourceManager(const DeviceContext& device_ctx,
                           const ShaderManager& shader_mgr)
      : device_ctx_(device_ctx), shader_mgr_(shader_mgr) {}
  virtual ~ResourceManager() = default;

  ResourceManager() = delete;

  void cleanupResources();

  const DeviceContext& getDeviceContext() const { return device_ctx_; }
  const ShaderManager& getShaderManager() const { return shader_mgr_; }

  //
  // textures (to be renamed images)
  //

  virtual resource_ptr<Texture> createTexture(std::string_view resource_tracking_string,
                                              uint32_t width,
                                              uint32_t height,
                                              uint32_t depth,
                                              PixelFormat pixel_format,
                                              uint32_t num_samples,
                                              bool is_array_texture,
                                              ImageUsageBits extra_usage_bits,
                                              TextureSamplerState sampler_state,
                                              const void* pixel_data = nullptr) = 0;

  virtual void destroyTexture(resource_ptr<Texture> tex);

  //
  // renderpass
  //

  virtual resource_ptr<RenderPass> createRenderPass(
      std::string_view resource_tracking_string,
      const Framebuffer::Layout& framebuffer_layout,
      RenderPass::ClearBits clear_bits = RenderPass::ClearBits::kNone,
      ImageLayout initial_layout = ImageLayout::kAttachment,
      ImageLayout final_layout = ImageLayout::kAttachment,
      const std::vector<SubpassDescriptor>& subpass_descriptors = {},
      const AttachmentToImageLayoutMap& unused_attachment_layouts = {}) = 0;

  virtual void destroyRenderPass(resource_ptr<RenderPass> render_pass);

  //
  // framebuffer
  //

  virtual resource_ptr<Framebuffer> createFramebuffer(
      std::string_view resource_tracking_string,
      const RenderPass& render_pass,
      AttachmentManager& attachment_mgr,
      uint32_t width,
      uint32_t height,
      uint32_t num_samples) = 0;

  virtual void destroyFramebuffer(resource_ptr<Framebuffer> framebuffer) = 0;

  //
  // buffers
  //

  virtual resource_ptr<Buffer> createBaseBuffer(
      std::string_view resource_tracking_string,
      const BufferCreateInfo& create_info,
      std::optional<LoggingCallback> oom_logging_cb = std::nullopt) = 0;

  BufferWrapperUqPtr createBuffer(
      std::string_view resource_tracking_string,
      const BufferCreateInfo& create_info,
      std::optional<BufferAllocatorShPtr> buffer_allocator = std::nullopt,
      std::optional<LoggingCallback> oom_logging_cb = std::nullopt);

  HostVisibleBufferWrapperUqPtr createHostVisibleBuffer(
      std::string_view resource_tracking_string,
      const HostVisibleBufferCreateInfo& create_info);

  HostVisibleBufferWrapperUqPtr convertToHostVisibleBuffer(
      BufferWrapperUqPtr&& source_buffer);

  virtual BufferWrapperUqPtr createPixelBuffer(
      std::string_view resource_tracking_string,
      uint32_t width,
      uint32_t height,
      PixelFormat pixel_format,
      std::optional<BufferAllocatorShPtr> buffer_allocator = std::nullopt,
      std::optional<LoggingCallback> oom_logging_cb = std::nullopt) = 0;

  virtual BufferWrapperUqPtr createPixelBuffer(
      std::string_view resource_tracking_string,
      uint32_t width,
      uint32_t height,
      PixelFormat pixel_format,
      void* pixel_data,
      std::optional<BufferAllocatorShPtr> buffer_allocator = std::nullopt,
      std::optional<LoggingCallback> oom_logging_cb = std::nullopt) = 0;

  virtual void destroyBaseBuffer(resource_ptr<Buffer> buffer);
  void destroyBuffer(BufferWrapperUqPtr buffer_wrapper);
  void destroyHostVisibleBuffer(HostVisibleBufferWrapperUqPtr buffer_wrapper);

  //
  // pipelines
  //

  virtual resource_ptr<GraphicsPipeline> createGraphicsPipeline(
      std::string_view resource_tracking_string,
      const Material& material,
      const PipelineDescriptor& pipeline_descriptor,
      const PrimitiveAssembly* primitive_assembly = nullptr) = 0;

  virtual resource_ptr<ComputePipeline> createComputePipeline(
      std::string_view resource_tracking_string,
      const Material& material,
      const std::vector<SpecializationMapEntry>& specializations = {},
      const PushConstantRanges& push_constant_ranges = {}) = 0;

  virtual resource_ptr<RaytracingPipeline> createRaytracingPipeline(
      std::string_view resource_tracking_string,
      const Material& material,
      const PushConstantRanges& push_constant_ranges) = 0;

  virtual void destroyPipeline(resource_ptr<GraphicsPipeline> pipeline);
  virtual void destroyPipeline(resource_ptr<ComputePipeline> pipeline);
  virtual void destroyPipeline(resource_ptr<RaytracingPipeline> pipeline);

  //
  // pipeline components
  //

  virtual PrimitiveAssemblyUqPtr createPrimitiveAssembly(
      std::string_view resource_tracking_string,
      const PrimitiveTopology topology,
      const Material& material,
      const PrimitiveAssemblyAttrInfo& attr_info,
      const IndexBuffer* ibo = nullptr) const = 0;

  virtual PrimitiveAssemblyUqPtr createPrimitiveAssembly(
      std::string_view resource_tracking_string,
      const PrimitiveTopology topology,
      const Material& material,
      const PrimitiveAssemblyAttrInfo& instanced_attr_info,
      const PrimitiveAssemblyAttrInfo& instances_attr_info,
      const uint32_t num_instances_per_attr = 1,
      const IndexBuffer* instanced_ibo = nullptr) const = 0;

  virtual MaterialUqPtr createMaterial(
      std::string_view resource_tracking_string,
      ShaderCacheShPtrVector& caches,
      bool allow_duplicate_shader_stages = false) const = 0;

  virtual MaterialUqPtr cloneMaterial(const Material& material_to_clone) const = 0;

  //
  // Acceleration structures
  //

  virtual AccelerationStructure::BuilderUqPtr createAccelerationStructureBuilder()
      const = 0;
  virtual resource_ptr<AccelerationStructure> createTopLevelAccelerationStructure(
      std::string_view resource_tracking_string,
      const AccelerationStructure::Builder& builder,
      BufferWrapper& instances_buffer) = 0;
  virtual resource_ptr<AccelerationStructure> createBottomLevelAccelerationStructure(
      std::string_view resource_tracking_string,
      const AccelerationStructure::Builder& builder) = 0;
  virtual void destroyAccelerationStructure(resource_ptr<AccelerationStructure>) = 0;

  //
  // Shader Binding Table
  //

  virtual ShaderBindingTableUqPtr createShaderBindingTable(
      const RaytracingPipeline& pipeline) const = 0;

  //
  // Query Pools
  //
  virtual resource_ptr<QueryPool> createTimestampQueryPool(
      std::string_view resource_tracking_string,
      uint32_t size) = 0;
  virtual resource_ptr<QueryPool> createPipelineStatisticsQueryPool(
      std::string_view resource_tracking_string,
      Pipeline::Type pipeline_type) = 0;
  virtual resource_ptr<QueryPool> createOcclusionQueryPool(
      std::string_view resource_tracking_string) = 0;
  virtual void destroyQueryPool(resource_ptr<QueryPool> query_pool);

  //
  // stats
  //

  struct Stats {
    uint32_t num_shader_programs;
    uint32_t num_shader_modules;
    uint32_t num_framebuffers;
    uint32_t num_primitive_assemblies;
    uint32_t num_textures;
    uint32_t num_texture_arrays;
    uint32_t num_buffers[static_cast<uint32_t>(BufferType::kCOUNT)];
    uint32_t num_pipelines;
    uint32_t num_render_passes;
    uint32_t num_acceleration_structures;
    uint32_t num_query_pools;
    uint64_t bytes_used_textures;
    uint64_t bytes_used_buffers[static_cast<uint32_t>(BufferType::kCOUNT)];

    Stats() { std::memset(this, 0, sizeof(Stats)); }

    bool operator==(const Stats& rhs) const {
      return memcmp(this, &rhs, sizeof(Stats)) == 0;
    }
  };

  virtual bool hasResources() const;
  virtual const Stats getStats() const;
  virtual void logMemorySummary(std::ostream& os) const;

  virtual void logImageResourceDetails() = 0;

 protected:
  Resource* addResource(ResourceUqPtr&& resource);
  void removeResource(Resource* resource, ResourceType resourceType);

  const DeviceContext& device_ctx_;
  const ShaderManager& shader_mgr_;

  std::vector<ResourceUqPtr> resources_;
  std::vector<ResourceId> free_rsrc_ids_;

  // unlocks for resource_ptr
  template <typename RESOURCE_TYPE>
  RESOURCE_TYPE* unlockResourcePtr(resource_ptr<RESOURCE_TYPE>& p) {
    auto& deleter = p.get_deleter();
    CHECK(typeid(deleter) == typeid(SinkedDeleter<RESOURCE_TYPE, ResourceManager>));
    deleter.allow_delete_ = true;
    return p.get();
  }

 private:
  /**
   * This is called internally after validation of the source_buffer
   */
  virtual HostVisibleBufferWrapperUqPtr convertToHostVisibleBufferImpl(
      BufferWrapperUqPtr source_buffer) = 0;
};

}  // namespace gfx
