/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"

#include <iostream>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanQueryPool.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanComputePipeline.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanGraphicsPipeline.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanPrimitiveAssembly.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanRaytracingPipeline.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanShaderBindingTable.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanAccelerationStructure.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanBaseBuffer.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanFramebuffer.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanHostVisibleBufferWrapper.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanPixelBuffer2d.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanRenderPass.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanShaderModule.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Resources/LocalBufferAllocator.h"

namespace gfx {

VulkanResourceManager::VulkanResourceManager(const DeviceContext& device_ctx,
                                             const ShaderManager& shader_mgr)
    : ResourceManager(device_ctx, shader_mgr)
    , vulkan_device_ctx_{*static_cast<const VulkanDeviceContext*>(&device_ctx)} {}

resource_ptr<VulkanShaderModule> VulkanResourceManager::createShaderModule(
    std::string_view resource_tracking_string,
    ShaderCacheShPtr& cache) {
  auto r(std::make_unique<VulkanShaderModule>(device_ctx_, resource_tracking_string));
  r->initResource(cache);
  return make_resource_ptr<VulkanShaderModule>(addResource(std::move(r)));
}

resource_ptr<VulkanShaderModule> VulkanResourceManager::cloneShaderModuleFrom(
    const VulkanShaderModule* src_shader_program) {
  auto src_vk_shader_module = static_cast<const VulkanShaderModule*>(src_shader_program);
  auto r(std::make_unique<VulkanShaderModule>(
      device_ctx_, src_vk_shader_module->getTrackingData().origin));
  src_vk_shader_module->cloneTo(*r);
  return make_resource_ptr<VulkanShaderModule>(addResource(std::move(r)));
}

void VulkanResourceManager::destroyShaderModule(
    resource_ptr<VulkanShaderModule> shader_module) {
  removeResource(unlockResourcePtr(shader_module), ResourceType::kShaderModule);
}

resource_ptr<RenderPass> VulkanResourceManager::createRenderPass(
    std::string_view resource_tracking_string,
    const Framebuffer::Layout& framebuffer_layout,
    RenderPass::ClearBits clear_bits,
    ImageLayout initial_layout,
    ImageLayout final_layout,
    const std::vector<SubpassDescriptor>& subpass_descriptors,
    const AttachmentToImageLayoutMap& unused_attachment_layouts) {
  auto r(std::make_unique<VulkanRenderPass>(device_ctx_,
                                            resource_tracking_string,
                                            framebuffer_layout,
                                            clear_bits,
                                            initial_layout,
                                            final_layout,
                                            subpass_descriptors,
                                            unused_attachment_layouts));
  return make_resource_ptr<RenderPass>(addResource(std::move(r)));
}

resource_ptr<Framebuffer> VulkanResourceManager::createFramebuffer(
    std::string_view resource_tracking_string,
    const RenderPass& render_pass,
    AttachmentManager& attachment_mgr,
    uint32_t width,
    uint32_t height,
    uint32_t num_samples) {
  auto r(std::make_unique<VulkanFramebuffer>(vulkan_device_ctx_,
                                             resource_tracking_string,
                                             render_pass,
                                             attachment_mgr,
                                             width,
                                             height,
                                             num_samples));
  return make_resource_ptr<Framebuffer>(addResource(std::move(r)));
}

void VulkanResourceManager::destroyFramebuffer(resource_ptr<Framebuffer> framebuffer) {
  removeResource(unlockResourcePtr(framebuffer), ResourceType::kFramebuffer);
}

resource_ptr<Buffer> VulkanResourceManager::createBaseBuffer(
    std::string_view resource_tracking_string,
    const BufferCreateInfo& create_info,
    std::optional<LoggingCallback> oom_logging_cb) {
  auto r = std::make_unique<VulkanBaseBuffer>(
      device_ctx_, resource_tracking_string, create_info, oom_logging_cb);
  return make_resource_ptr<Buffer>(addResource(std::move(r)));
}

BufferWrapperUqPtr VulkanResourceManager::createPixelBuffer(
    std::string_view resource_tracking_string,
    uint32_t width,
    uint32_t height,
    PixelFormat pixel_format,
    std::optional<BufferAllocatorShPtr> buffer_allocator_opt,
    std::optional<LoggingCallback> oom_logging_cb) {
  BufferCreateInfo create_info{BufferType::kPixelBuffer,
                               0,
                               BufferUsageBits::kNone,
                               BufferAccessType::kHostVisibleCached};
  BufferAllocatorShPtr buffer_allocator =
      buffer_allocator_opt
          ? buffer_allocator_opt.value()
          : std::make_shared<DefaultBufferAllocator>(device_ctx_,
                                                     resource_tracking_string,
                                                     *this,
                                                     create_info,
                                                     oom_logging_cb);
  buffer_allocator->validateCreateInfo(create_info);
  RUNTIME_EX_ASSERT(
      width > 0 && height > 0,
      "Invalid dimensions for the 2d pixel buffer:" + std::to_string(width) + "x" +
          std::to_string(height) + ". Dimensions must be > 0");
  auto buffer_wrapper = std::make_unique<VulkanPixelBuffer2d>(resource_tracking_string,
                                                              std::move(buffer_allocator),
                                                              width,
                                                              height,
                                                              pixel_format,
                                                              nullptr);
  CHECK(buffer_wrapper);
  return buffer_wrapper;
}

BufferWrapperUqPtr VulkanResourceManager::createPixelBuffer(
    std::string_view resource_tracking_string,
    uint32_t width,
    uint32_t height,
    PixelFormat pixel_format,
    void* pixel_data,
    std::optional<BufferAllocatorShPtr> buffer_allocator_opt,
    std::optional<LoggingCallback> oom_logging_cb) {
  BufferCreateInfo create_info{BufferType::kPixelBuffer,
                               0,
                               BufferUsageBits::kNone,
                               BufferAccessType::kHostVisibleCached};
  BufferAllocatorShPtr buffer_allocator =
      buffer_allocator_opt
          ? buffer_allocator_opt.value()
          : std::make_shared<DefaultBufferAllocator>(device_ctx_,
                                                     resource_tracking_string,
                                                     *this,
                                                     create_info,
                                                     oom_logging_cb);
  auto buffer_wrapper = std::make_unique<VulkanPixelBuffer2d>(resource_tracking_string,
                                                              std::move(buffer_allocator),
                                                              width,
                                                              height,
                                                              pixel_format,
                                                              pixel_data);
  CHECK(buffer_wrapper);
  return buffer_wrapper;
}

HostVisibleBufferWrapperUqPtr VulkanResourceManager::convertToHostVisibleBufferImpl(
    BufferWrapperUqPtr source_buffer) {
  return std::make_unique<VulkanHostVisibleBufferWrapper>(std::move(source_buffer));
}

resource_ptr<Texture> VulkanResourceManager::createTexture(
    std::string_view resource_tracking_string,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    PixelFormat pixel_format,
    uint32_t num_samples,
    bool is_array_texture,
    ImageUsageBits extra_usage_bits,
    TextureSamplerState sampler_state,
    const void* pixel_data) {
  auto r(std::make_unique<VulkanTexture>(device_ctx_,
                                         resource_tracking_string,
                                         width,
                                         height,
                                         depth,
                                         pixel_format,
                                         num_samples,
                                         is_array_texture,
                                         extra_usage_bits,
                                         std::move(sampler_state),
                                         pixel_data));
  return make_resource_ptr<Texture>(addResource(std::move(r)));
}

resource_ptr<GraphicsPipeline> VulkanResourceManager::createGraphicsPipeline(
    std::string_view resource_tracking_string,
    const Material& material,
    const PipelineDescriptor& pipeline_descriptor,
    const PrimitiveAssembly* primitive_assembly) {
  auto r(std::make_unique<VulkanGraphicsPipeline>(device_ctx_,
                                                  resource_tracking_string,
                                                  material,
                                                  pipeline_descriptor,
                                                  primitive_assembly));
  return make_resource_ptr<GraphicsPipeline>(addResource(std::move(r)));
}

resource_ptr<ComputePipeline> VulkanResourceManager::createComputePipeline(
    std::string_view resource_tracking_string,
    const Material& material,
    const std::vector<SpecializationMapEntry>& specializations,
    const PushConstantRanges& push_constant_ranges) {
  auto r(std::make_unique<VulkanComputePipeline>(device_ctx_,
                                                 resource_tracking_string,
                                                 material,
                                                 specializations,
                                                 push_constant_ranges));
  return make_resource_ptr<ComputePipeline>(addResource(std::move(r)));
}

resource_ptr<RaytracingPipeline> VulkanResourceManager::createRaytracingPipeline(
    std::string_view resource_tracking_string,
    const Material& material,
    const PushConstantRanges& push_constant_ranges) {
  auto r(std::make_unique<VulkanRaytracingPipeline>(
      device_ctx_, resource_tracking_string, material, push_constant_ranges));
  return make_resource_ptr<RaytracingPipeline>(addResource(std::move(r)));
}

PrimitiveAssemblyUqPtr VulkanResourceManager::createPrimitiveAssembly(
    std::string_view resource_tracking_string,
    const PrimitiveTopology topology,
    const Material& material,
    const PrimitiveAssemblyAttrInfo& attr_info,
    const IndexBuffer* ibo) const {
  return std::make_unique<VulkanPrimitiveAssembly>(
      device_ctx_, resource_tracking_string, topology, material, attr_info, ibo);
}

PrimitiveAssemblyUqPtr VulkanResourceManager::createPrimitiveAssembly(
    std::string_view resource_tracking_string,
    const PrimitiveTopology topology,
    const Material& material,
    const PrimitiveAssemblyAttrInfo& instanced_attr_info,
    const PrimitiveAssemblyAttrInfo& instances_attr_info,
    const uint32_t num_instances_per_attr,
    const IndexBuffer* instanced_ibo) const {
  return std::make_unique<VulkanPrimitiveAssembly>(device_ctx_,
                                                   resource_tracking_string,
                                                   topology,
                                                   material,
                                                   instanced_attr_info,
                                                   instances_attr_info,
                                                   num_instances_per_attr,
                                                   instanced_ibo);
}

MaterialUqPtr VulkanResourceManager::createMaterial(
    std::string_view resource_tracking_string,
    ShaderCacheShPtrVector& caches,
    bool allow_duplicate_shader_stages) const {
  return std::make_unique<VulkanMaterial>(
      device_ctx_, resource_tracking_string, caches, allow_duplicate_shader_stages);
}

MaterialUqPtr VulkanResourceManager::cloneMaterial(
    const Material& material_to_clone) const {
  return std::make_unique<VulkanMaterial>(device_ctx_, material_to_clone);
}

AccelerationStructure::BuilderUqPtr
VulkanResourceManager::createAccelerationStructureBuilder() const {
  return std::make_unique<VulkanAccelerationStructureBuilder>(device_ctx_);
}

resource_ptr<AccelerationStructure> VulkanResourceManager::createAccelerationStructure(
    std::string_view resource_tracking_string,
    AccelerationStructureType type) {
  auto r(std::make_unique<VulkanAccelerationStructure>(
      device_ctx_, resource_tracking_string, type));
  return make_resource_ptr<AccelerationStructure>(addResource(std::move(r)));
}

resource_ptr<AccelerationStructure>
VulkanResourceManager::createTopLevelAccelerationStructure(
    std::string_view resource_tracking_string,
    const AccelerationStructure::Builder& builder,
    BufferWrapper& instances_buffer) {
  auto const& vk_builder =
      static_cast<const VulkanAccelerationStructureBuilder&>(builder);
  return vk_builder.buildTopLevel(resource_tracking_string, instances_buffer);
}

resource_ptr<AccelerationStructure>
VulkanResourceManager::createBottomLevelAccelerationStructure(
    std::string_view resource_tracking_string,
    const AccelerationStructure::Builder& builder) {
  auto const& vk_builder =
      static_cast<const VulkanAccelerationStructureBuilder&>(builder);
  return vk_builder.buildBottomLevel(resource_tracking_string);
}

void VulkanResourceManager::destroyAccelerationStructure(
    resource_ptr<AccelerationStructure> accel_structure) {
  removeResource(unlockResourcePtr(accel_structure),
                 ResourceType::kAccelerationStructure);
}

ShaderBindingTableUqPtr VulkanResourceManager::createShaderBindingTable(
    const RaytracingPipeline& pipeline) const {
  return std::make_unique<VulkanShaderBindingTable>(pipeline);
}

resource_ptr<QueryPool> VulkanResourceManager::createTimestampQueryPool(
    std::string_view resource_tracking_string,
    uint32_t size) {
  auto r(std::make_unique<VulkanQueryPool>(device_ctx_,
                                           resource_tracking_string,
                                           QueryPool::Type::kTimestamp,
                                           std::nullopt,
                                           size));
  return make_resource_ptr<QueryPool>(addResource(std::move(r)));
}
resource_ptr<QueryPool> VulkanResourceManager::createPipelineStatisticsQueryPool(
    std::string_view resource_tracking_string,
    Pipeline::Type pipeline_type) {
  auto r(std::make_unique<VulkanQueryPool>(device_ctx_,
                                           resource_tracking_string,
                                           QueryPool::Type::kPipelineStatistics,
                                           pipeline_type,
                                           std::nullopt));
  return make_resource_ptr<QueryPool>(addResource(std::move(r)));
}
resource_ptr<QueryPool> VulkanResourceManager::createOcclusionQueryPool(
    std::string_view resource_tracking_string) {
  auto r(std::make_unique<VulkanQueryPool>(device_ctx_,
                                           resource_tracking_string,
                                           QueryPool::Type::kOcclusion,
                                           Pipeline::Type::kGraphics,
                                           std::nullopt));
  return make_resource_ptr<QueryPool>(addResource(std::move(r)));
}

void VulkanResourceManager::logImageResourceDetails() {
  std::cerr << "Vulkan Image Resources for GPU " << getDeviceContext().getGpuId() << ":"
            << std::endl;
  for (auto const& resource : resources_) {
    if (resource.get() && resource->getResourceType() == ResourceType::kTexture) {
      auto const* texture_array = static_cast<VulkanTexture*>(resource.get());
      auto const vk_image = texture_array->getImage();
      auto layout = getImageLayoutManager().getCurrentLayout(vk_image);
      std::cerr << "  VkImage " << std::hex << vk_image << std::dec << ", '"
                << texture_array->getTrackingData().origin << "' (array), format "
                << texture_array->getPixelFormat() << ", size "
                << texture_array->getWidth() << "x" << texture_array->getHeight() << "x"
                << texture_array->getDepth() << ", layout " << layout << std::endl;
    }
  }
}

}  // namespace gfx
