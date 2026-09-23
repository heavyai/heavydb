/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/ResourceManager.h"

#include "GfxDriver/Drivers/Vulkan/Resources/ImageLayoutManager.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"

namespace gfx {

class VulkanShaderModule;

class VulkanResourceManager : public ResourceManager {
 public:
  explicit VulkanResourceManager(const DeviceContext& device_ctx,
                                 const ShaderManager& shader_mgr);
  VulkanResourceManager() = delete;
  ~VulkanResourceManager() override = default;

  ImageLayoutManager& getImageLayoutManager() { return image_layout_mgr_; }

  resource_ptr<VulkanShaderModule> createShaderModule(
      std::string_view resource_tracking_string,
      ShaderCacheShPtr& cache);

  resource_ptr<VulkanShaderModule> cloneShaderModuleFrom(
      const VulkanShaderModule* src_shader_module);

  void destroyShaderModule(resource_ptr<VulkanShaderModule> shader);

  resource_ptr<Texture> createTexture(std::string_view resource_tracking_string,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t depth,
                                      PixelFormat pixel_format,
                                      uint32_t num_samples,
                                      bool is_array_texture,
                                      ImageUsageBits extra_usage_bits,
                                      TextureSamplerState sampler_state,
                                      const void* pixel_data) override;

  resource_ptr<RenderPass> createRenderPass(
      std::string_view resource_tracking_string,
      const Framebuffer::Layout& framebuffer_layout,
      RenderPass::ClearBits clear_bits,
      ImageLayout initial_layout,
      ImageLayout final_layout,
      const std::vector<SubpassDescriptor>& subpass_descriptors,
      const AttachmentToImageLayoutMap& unused_attachment_layouts) override;

  resource_ptr<Framebuffer> createFramebuffer(std::string_view resource_tracking_string,
                                              const RenderPass& render_pass,
                                              AttachmentManager& attachment_mgr,
                                              uint32_t width,
                                              uint32_t height,
                                              uint32_t num_samples) override;

  void destroyFramebuffer(resource_ptr<Framebuffer> framebuffer) override;

  resource_ptr<Buffer> createBaseBuffer(
      std::string_view resource_tracking_string,
      const BufferCreateInfo& create_info,
      std::optional<LoggingCallback> oom_logging_cb = std::nullopt) override;

  BufferWrapperUqPtr createPixelBuffer(
      std::string_view resource_tracking_string,
      uint32_t width,
      uint32_t height,
      PixelFormat pixel_format,
      std::optional<BufferAllocatorShPtr> buffer_allocator,
      std::optional<LoggingCallback> oom_logging_cb) override;

  BufferWrapperUqPtr createPixelBuffer(
      std::string_view resource_tracking_string,
      uint32_t width,
      uint32_t height,
      PixelFormat pixel_format,
      void* pixel_data,
      std::optional<BufferAllocatorShPtr> buffer_allocator,
      std::optional<LoggingCallback> oom_logging_cb) override;

  resource_ptr<GraphicsPipeline> createGraphicsPipeline(
      std::string_view resource_tracking_string,
      const Material& material,
      const PipelineDescriptor& pipeline_descriptor,
      const PrimitiveAssembly* primitive_assembly) override;

  resource_ptr<ComputePipeline> createComputePipeline(
      std::string_view resource_tracking_string,
      const Material& material,
      const std::vector<SpecializationMapEntry>& specializations,
      const PushConstantRanges& push_constants) override;

  resource_ptr<RaytracingPipeline> createRaytracingPipeline(
      std::string_view resource_tracking_string,
      const Material& material,
      const PushConstantRanges& push_constant_ranges) override;

  PrimitiveAssemblyUqPtr createPrimitiveAssembly(
      std::string_view resource_tracking_string,
      const PrimitiveTopology topology,
      const Material& material,
      const PrimitiveAssemblyAttrInfo& attr_info,
      const IndexBuffer* ibo = nullptr) const override;
  PrimitiveAssemblyUqPtr createPrimitiveAssembly(
      std::string_view resource_tracking_string,
      const PrimitiveTopology topology,
      const Material& material,
      const PrimitiveAssemblyAttrInfo& instanced_attr_info,
      const PrimitiveAssemblyAttrInfo& instances_attr_info,
      const uint32_t num_instances_per_attr = 1,
      const IndexBuffer* instanced_ibo = nullptr) const override;

  MaterialUqPtr createMaterial(std::string_view resource_tracking_string,
                               ShaderCacheShPtrVector& caches,
                               bool allow_duplicate_shader_stages = false) const override;
  MaterialUqPtr cloneMaterial(const Material& material_to_clone) const override;

  resource_ptr<AccelerationStructure> createAccelerationStructure(
      std::string_view resource_tracking_string,
      AccelerationStructureType type);

  AccelerationStructure::BuilderUqPtr createAccelerationStructureBuilder() const override;
  resource_ptr<AccelerationStructure> createTopLevelAccelerationStructure(
      std::string_view resource_tracking_string,
      const AccelerationStructure::Builder& builder,
      BufferWrapper& instances_buffer) override;
  resource_ptr<AccelerationStructure> createBottomLevelAccelerationStructure(
      std::string_view resource_tracking_string,
      const AccelerationStructure::Builder& builder) override;
  void destroyAccelerationStructure(resource_ptr<AccelerationStructure>) override;

  ShaderBindingTableUqPtr createShaderBindingTable(
      const RaytracingPipeline& pipeline) const override;

  resource_ptr<QueryPool> createTimestampQueryPool(
      std::string_view resource_tracking_string,
      uint32_t size) override;
  resource_ptr<QueryPool> createPipelineStatisticsQueryPool(
      std::string_view resource_tracking_string,
      Pipeline::Type pipeline_type) override;
  resource_ptr<QueryPool> createOcclusionQueryPool(
      std::string_view resource_tracking_string) override;

  void logImageResourceDetails() override;

 private:
  const VulkanDeviceContext& vulkan_device_ctx_;
  ImageLayoutManager image_layout_mgr_;

  HostVisibleBufferWrapperUqPtr convertToHostVisibleBufferImpl(
      BufferWrapperUqPtr source_buffer) override;
};

}  // namespace gfx
