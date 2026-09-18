/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandExecutor.h"

#include <algorithm>

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanQueryPool.h"
#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanFramebuffer.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanRenderPass.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanQueue.h"
#include "GfxDriver/Resources/IndexBuffer.h"
#include "GfxDriver/Resources/IndirectDrawBuffer.h"
#include "GfxDriver/Resources/VertexBuffer.h"

namespace gfx {

VulkanCommandExecutor::VulkanCommandExecutor(const VulkanDeviceContext& device_context,
                                             VulkanCommandPool& command_pool)
    : device_context_{device_context}
    , device_funcs_{device_context.getFunctions()}
    , command_pool_{command_pool}
    , image_layout_mgr_{static_cast<VulkanResourceManager&>(
                            device_context.getResourceManager())
                            .getImageLayoutManager()}
    , active_buffer_{nullptr}
    , render_area_{}
    , default_viewport_{}
    , default_scissor_{}
    , current_render_pass_{nullptr}
    , subpass_index_{0u}
    , is_viewport_bound_{false}
    , bound_graphics_pipeline_{VK_NULL_HANDLE}
    , bound_compute_pipeline_{VK_NULL_HANDLE}
    , bound_vertex_buffers_{VK_NULL_HANDLE, VK_NULL_HANDLE}
    , bound_vertex_buffers_offset_bytes_{0ULL, 0ULL}
    , bound_index_buffer_{VK_NULL_HANDLE}
    , bound_index_buffer_offset_bytes_{0ULL} {}

const DeviceContext& VulkanCommandExecutor::getDeviceContext() const {
  return device_context_;
}

void VulkanCommandExecutor::setRenderArea(uint32_t x,
                                          uint32_t y,
                                          uint32_t width,
                                          uint32_t height) {
  render_area_ = {{static_cast<int32_t>(x), static_cast<int32_t>(y)}, {width, height}};
  // Expand render area width by 1 pixel to avoid artifacts when multisampling along the
  // right edge of the image.
  render_area_.extent.width++;
}

void VulkanCommandExecutor::beginCommandSequence() {
  CHECK(active_buffer_ == nullptr);
  active_buffer_ = command_pool_.acquireBuffer();
}

const VulkanCommandBuffer& VulkanCommandExecutor::getActiveCommandBuffer() const {
  CHECK(active_buffer_);
  return *active_buffer_;
}

void VulkanCommandExecutor::submitCommandSequence(
    const std::string_view name,
    CommandList::SubmitType submit_type,
    const std::vector<SemaphoreHandle>& wait_semaphores,
    const std::vector<SemaphoreHandle>& signal_semaphores) {
  CHECK(active_buffer_);
  if (wait_semaphores.size() || signal_semaphores.size()) {
    // Wait stages should be passed via the API, but we don't have an abstraction yet.
    // Current usage only deals with either FRAGMENT_SHADER or TRANSFER so we just set
    // both stages. This vector is not cleared, it just grows until it's at the max size
    // we ever need (typically 1 or 2)
    static std::vector<VkPipelineStageFlags2> vk_semaphore_wait_stages;
    if (vk_semaphore_wait_stages.size() < wait_semaphores.size()) {
      vk_semaphore_wait_stages.resize(wait_semaphores.size());
      std::fill(
          vk_semaphore_wait_stages.begin(),
          vk_semaphore_wait_stages.end(),
          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    }

    static std::vector<VkPipelineStageFlags2KHR> vk_semaphore_signal_stages;
    if (vk_semaphore_signal_stages.size() < signal_semaphores.size()) {
      vk_semaphore_signal_stages.resize(signal_semaphores.size());
      std::fill(vk_semaphore_signal_stages.begin(),
                vk_semaphore_signal_stages.end(),
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR |
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR);
    }

    command_pool_.submitBuffer(
        active_buffer_,
        name,
        wait_semaphores.size(),
        reinterpret_cast<const VkSemaphore*>(wait_semaphores.data()),
        vk_semaphore_wait_stages,
        signal_semaphores.size(),
        reinterpret_cast<const VkSemaphore*>(signal_semaphores.data()),
        vk_semaphore_signal_stages);
  } else {
    command_pool_.submitBuffer(active_buffer_, name);
  }
  if (submit_type == CommandList::SubmitType::kWaitComplete) {
    command_pool_.waitForPendingBuffers();
  }
  clearBindings();
}

void VulkanCommandExecutor::submitCommandBatch() {}

void VulkanCommandExecutor::waitForCompletion(bool is_profiling_run) {
  if (!command_pool_.waitForPendingBuffers()) {
    resetState(false);
    THROW_RUNTIME_EX("Pending command buffer(s) failed to complete");
  }
}

void VulkanCommandExecutor::resetState(bool abort_on_exception) {
  try {
    command_pool_.resetPool();
    clearBindings();
  } catch (...) {
    if (abort_on_exception) {
      LOG(FATAL)
          << "A fatal error occured attempting to reset VulkanCommandExecutor state";
    }
  }
}

void VulkanCommandExecutor::beginRenderPass(RenderPass& render_pass,
                                            Framebuffer& framebuffer) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kRecording);
  VkRenderPassBeginInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass =
      reinterpret_cast<VkRenderPass>(render_pass.getResourceHandle());
  render_pass_info.framebuffer =
      reinterpret_cast<VkFramebuffer>(framebuffer.getResourceHandle());

  // Ensure render area doesn't exceed the Framebuffer dimensions
  // This occurs due to the +1 width padding required to avoid artifacts
  // when multisampling. Some framebuffers (such as those in tests),
  // do not have this padding
  render_pass_info.renderArea.offset = render_area_.offset;
  render_pass_info.renderArea.extent.width =
      std::min(render_area_.extent.width, framebuffer.getWidth() - render_area_.offset.x);
  render_pass_info.renderArea.extent.height = std::min(
      render_area_.extent.height, framebuffer.getHeight() - render_area_.offset.y);

  auto* vulkan_render_pass = reinterpret_cast<VulkanRenderPass*>(&render_pass);
  auto const& clear_values = vulkan_render_pass->getClearValues();
  render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
  render_pass_info.pClearValues = clear_values.data();

  active_buffer_->beginRenderPass(&render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
  current_render_pass_ = vulkan_render_pass;
  subpass_index_ = 0;
}

void VulkanCommandExecutor::endRenderPass() {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  active_buffer_->endRenderPass();
  current_render_pass_ = nullptr;
  subpass_index_ = 0;
}

void VulkanCommandExecutor::nextSubpass() {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  // No secondary command buffer support, so all subpasses are inline
  // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS support will require a new state
  // transition in VulkanCommandBuffer to restrict it to only accepting
  // VkCmdExecuteCommands until a new subpass begins or we exit the renderpass
  vkCmdNextSubpass(active_buffer_->getHandle(), VK_SUBPASS_CONTENTS_INLINE);
  subpass_index_++;
}

static void make_vk_viewport(VkViewport& vp,
                             VkRect2D& scissor,
                             uint32_t x,
                             uint32_t y,
                             uint32_t width,
                             uint32_t height) {
  static constexpr bool kInvertY = false;
  if constexpr (kInvertY) {
    vp.x = static_cast<float>(x),
    vp.y = static_cast<float>(height) - static_cast<float>(y);
    vp.width = static_cast<float>(width);
    vp.height = -static_cast<float>(height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
  } else {
    vp.x = static_cast<float>(x);
    vp.y = static_cast<float>(y);
    vp.width = static_cast<float>(width);
    vp.height = static_cast<float>(height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
  }

  scissor.offset.x = static_cast<int32_t>(x);
  scissor.offset.y = static_cast<int32_t>(y);
  scissor.extent.width = static_cast<uint32_t>(width);
  scissor.extent.height = static_cast<uint32_t>(height);
}

void VulkanCommandExecutor::setDefaultViewportAndRenderArea(uint32_t x,
                                                            uint32_t y,
                                                            uint32_t width,
                                                            uint32_t height) {
  make_vk_viewport(default_viewport_, default_scissor_, x, y, width, height);
  setRenderArea(x, y, width, height);
}

// CommandList command
void VulkanCommandExecutor::setViewport(uint32_t x,
                                        uint32_t y,
                                        uint32_t width,
                                        uint32_t height) {
  CHECK(active_buffer_);
  VkViewport viewport = {};
  VkRect2D scissor = {};
  make_vk_viewport(viewport, scissor, x, y, width, height);
  vkCmdSetViewport(active_buffer_->getHandle(), 0, 1, &viewport);

  // Viewport and Scissor need to match or we get a validation error
  vkCmdSetScissor(active_buffer_->getHandle(), 0, 1, &scissor);

  is_viewport_bound_ = true;
}

void VulkanCommandExecutor::setScissor(int32_t x,
                                       int32_t y,
                                       uint32_t width,
                                       uint32_t height) {
  CHECK(active_buffer_);
  VkRect2D scissor{{x, y}, {width, height}};
  vkCmdSetScissor(active_buffer_->getHandle(), 0, 1, &scissor);
}

// Internal only
void VulkanCommandExecutor::autobindViewport() {
  if (!is_viewport_bound_) {
    vkCmdSetViewport(active_buffer_->getHandle(), 0, 1, &default_viewport_);

    // Viewport and Scissor need to match or we get a validation error
    vkCmdSetScissor(active_buffer_->getHandle(), 0, 1, &default_scissor_);

    is_viewport_bound_ = true;
  }
}

// command buffer must be recording, and must be inside render pass
void VulkanCommandExecutor::bindPipeline(const VkCommandBuffer vk_cmd_buffer,
                                         const Pipeline::Type type,
                                         const Pipeline& pipeline,
                                         const uint32_t specialization_id) {
  CHECK_EQ(pipeline.getType(), type) << "Invalid Pipeline Type!";
  auto const& material = static_cast<const VulkanMaterial&>(pipeline.getMaterial());
  auto descriptor_set = material.getDescriptorSet();
  auto vk_pipeline =
      reinterpret_cast<VkPipeline>(pipeline.getPipelineHandle(specialization_id));
  CHECK(vk_pipeline != VK_NULL_HANDLE);

  VkPipeline* current_bound_pipeline{nullptr};
  VkPipelineBindPoint bind_point{VK_PIPELINE_BIND_POINT_MAX_ENUM};
  switch (type) {
    case Pipeline::Type::kCompute:
      current_bound_pipeline = &bound_compute_pipeline_;
      bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
      break;
    case Pipeline::Type::kGraphics:
      current_bound_pipeline = &bound_graphics_pipeline_;
      bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
      break;
    case Pipeline::Type::kRaytracing:
      current_bound_pipeline = &bound_raytracing_pipeline_;
      bind_point = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
      break;
  }

  // check if already bound foo
  if (vk_pipeline != *current_bound_pipeline) {
    // bind descriptor sets
    if (descriptor_set) {
      vkCmdBindDescriptorSets(vk_cmd_buffer,
                              bind_point,
                              reinterpret_cast<VkPipelineLayout>(pipeline.getLayout()),
                              0,
                              1,
                              &descriptor_set,
                              0,
                              nullptr);
    }

    // bind pipeline
    vkCmdBindPipeline(vk_cmd_buffer, bind_point, vk_pipeline);
    *current_bound_pipeline = vk_pipeline;
  }
}

void VulkanCommandExecutor::bindVertexBuffer(const VkCommandBuffer vk_cmd_buffer,
                                             const VertexBuffer& vertex_buffer,
                                             const uint32_t binding,
                                             const uint64_t additional_offset_bytes) {
  auto vk_vertex_buffer = reinterpret_cast<VkBuffer>(vertex_buffer.getResourceHandle());
  CHECK_LT(binding, bound_vertex_buffers_.size());
  VkDeviceSize offset_bytes =
      vertex_buffer.getAllocationOffsetBytes() + additional_offset_bytes;
  if (vk_vertex_buffer != bound_vertex_buffers_[binding] ||
      offset_bytes != bound_vertex_buffers_offset_bytes_[binding]) {
    vkCmdBindVertexBuffers(vk_cmd_buffer, binding, 1, &vk_vertex_buffer, &offset_bytes);
    bound_vertex_buffers_[binding] = vk_vertex_buffer;
    bound_vertex_buffers_offset_bytes_[binding] = offset_bytes;
  }
  CHECK(bound_vertex_buffers_[binding] != VK_NULL_HANDLE);
}

void VulkanCommandExecutor::bindIndexBuffer(const VkCommandBuffer vk_cmd_buffer,
                                            const IndexBuffer& index_buffer) {
  auto vk_index_buffer = reinterpret_cast<VkBuffer>(index_buffer.getResourceHandle());
  VkDeviceSize offset_bytes = index_buffer.getAllocationOffsetBytes();
  if (vk_index_buffer != bound_index_buffer_ ||
      offset_bytes != bound_index_buffer_offset_bytes_) {
    vkCmdBindIndexBuffer(
        vk_cmd_buffer,
        vk_index_buffer,
        offset_bytes,
        index_buffer_data_type_to_vk_index_type(index_buffer.getIndexDataType()));
    bound_index_buffer_ = vk_index_buffer;
    bound_index_buffer_offset_bytes_ = offset_bytes;
  }
  CHECK(bound_index_buffer_ != VK_NULL_HANDLE);
}

void VulkanCommandExecutor::drawFullScreenQuad(Pipeline& pipeline) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  // Bind default viewport if no viewport is bound to command buffer
  autobindViewport();

  // bind pipeline and descriptor sets
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kGraphics, pipeline);

  // draw
  vkCmdDraw(vk_cmd_buffer, 3, 1, 0, 0);
}

void VulkanCommandExecutor::drawVertices(Pipeline& pipeline,
                                         const VertexBuffer& vertex_buffer,
                                         uint32_t vertex_count,
                                         uint64_t offset_bytes,
                                         uint32_t first_vertex,
                                         uint32_t instance_count,
                                         uint32_t start_instance) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  // Bind default viewport if no viewport is bound to command buffer
  autobindViewport();

  // bind pipeline and descriptor sets
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kGraphics, pipeline);

  // bind vertex buffer
  bindVertexBuffer(vk_cmd_buffer, vertex_buffer, 0, offset_bytes);

  // draw
  vkCmdDraw(vk_cmd_buffer, vertex_count, instance_count, first_vertex, start_instance);
}

void VulkanCommandExecutor::drawVertices(Pipeline& pipeline,
                                         const VertexBufferRefs& vertex_buffer_refs,
                                         uint32_t vertex_count,
                                         uint32_t first_vertex,
                                         uint32_t instance_count,
                                         uint32_t start_instance) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  // Bind default viewport if no viewport is bound to command buffer
  autobindViewport();

  // bind pipeline and descriptor sets
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kGraphics, pipeline);

  // bind vertex buffers
  uint32_t vertex_buffer_binding{0U};
  for (auto const& vertex_buffer_ref : vertex_buffer_refs) {
    bindVertexBuffer(vk_cmd_buffer,
                     vertex_buffer_ref.vertex_buffer,
                     vertex_buffer_binding++,
                     vertex_buffer_ref.offset_bytes);
  }

  // draw
  vkCmdDraw(vk_cmd_buffer, vertex_count, instance_count, first_vertex, start_instance);
}

void VulkanCommandExecutor::drawIndirect(Pipeline& pipeline,
                                         const VertexBuffer& vertex_buffer,
                                         const IndirectDrawVertexBuffer& indirect_buffer,
                                         uint32_t draw_count,
                                         uint32_t first_index) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  // Bind default viewport if no viewport is bound to command buffer
  autobindViewport();

  // bind pipeline and descriptor sets
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kGraphics, pipeline);

  // bind vertex buffer
  bindVertexBuffer(vk_cmd_buffer, vertex_buffer, 0, 0u);

  // draw
  auto buffer_handle = reinterpret_cast<VkBuffer>(indirect_buffer.getResourceHandle());
  VkDeviceSize offset_bytes = indirect_buffer.getAllocationOffsetBytes() +
                              (first_index * sizeof(VkDrawIndirectCommand));
  vkCmdDrawIndirect(vk_cmd_buffer,
                    buffer_handle,
                    offset_bytes,
                    draw_count,
                    sizeof(VkDrawIndirectCommand));
}

void VulkanCommandExecutor::drawIndirectIndexed(
    Pipeline& pipeline,
    const VertexBuffer& vertex_buffer,
    const IndexBuffer& index_buffer,
    const IndirectDrawIndexBuffer& indirect_buffer,
    uint32_t draw_count,
    uint32_t first_index) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  // Bind default viewport if no viewport is bound to command buffer
  autobindViewport();

  // bind pipeline and descriptor sets
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kGraphics, pipeline);

  // bind vertex and index buffers
  bindVertexBuffer(vk_cmd_buffer, vertex_buffer, 0, 0u);
  bindIndexBuffer(vk_cmd_buffer, index_buffer);

  // draw
  auto buffer_handle = reinterpret_cast<VkBuffer>(indirect_buffer.getResourceHandle());
  VkDeviceSize offset_bytes = indirect_buffer.getAllocationOffsetBytes() +
                              (first_index * sizeof(VkDrawIndexedIndirectCommand));
  vkCmdDrawIndexedIndirect(vk_cmd_buffer,
                           buffer_handle,
                           offset_bytes,
                           draw_count,
                           sizeof(VkDrawIndexedIndirectCommand));
}

void VulkanCommandExecutor::drawIndexed(Pipeline& pipeline,
                                        const VertexBuffer& vertex_buffer,
                                        const IndexBuffer& index_buffer,
                                        uint32_t index_count,
                                        uint32_t first_index,
                                        int32_t vertex_offset,
                                        uint32_t instance_count,
                                        uint32_t first_instance) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  // Bind default viewport if no viewport is bound to command buffer
  autobindViewport();

  // bind pipeline and descriptor sets
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kGraphics, pipeline);

  // bind vertex and index buffers
  bindVertexBuffer(vk_cmd_buffer, vertex_buffer, 0, 0u);
  bindIndexBuffer(vk_cmd_buffer, index_buffer);

  // draw
  vkCmdDrawIndexed(vk_cmd_buffer,
                   index_count,
                   instance_count,
                   first_index,
                   vertex_offset,
                   first_instance);
}

void VulkanCommandExecutor::drawIndexed(Pipeline& pipeline,
                                        const VertexBufferRefs& vertex_buffer_refs,
                                        const IndexBuffer& index_buffer,
                                        uint32_t index_count,
                                        uint32_t first_index,
                                        uint32_t instance_count,
                                        uint32_t first_instance) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  // Bind default viewport if no viewport is bound to command buffer
  autobindViewport();

  // bind pipeline and descriptor sets
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kGraphics, pipeline);

  // bind vertex and index buffers
  uint32_t vertex_buffer_binding{0U};
  for (auto const& vertex_buffer_ref : vertex_buffer_refs) {
    bindVertexBuffer(vk_cmd_buffer,
                     vertex_buffer_ref.vertex_buffer,
                     vertex_buffer_binding++,
                     vertex_buffer_ref.offset_bytes);
  }
  bindIndexBuffer(vk_cmd_buffer, index_buffer);

  // draw
  vkCmdDrawIndexed(
      vk_cmd_buffer, index_count, instance_count, first_index, 0, first_instance);
}

void VulkanCommandExecutor::drawMeshTasks(Pipeline& pipeline,
                                          uint32_t group_count_x,
                                          uint32_t group_count_y,
                                          uint32_t group_count_z) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  // Bind default viewport if no viewport is bound to command buffer
  autobindViewport();

  // bind pipeline and descriptor sets
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kGraphics, pipeline);

  device_funcs_.vkCmdDrawMeshTasksEXT(
      vk_cmd_buffer, group_count_x, group_count_y, group_count_z);
}

void VulkanCommandExecutor::fillBuffer(const Buffer& buffer,
                                       uint32_t data,
                                       uint64_t num_bytes,
                                       uint64_t offset) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() != VulkanCommandBuffer::State::kInRenderpass);
  vkCmdFillBuffer(active_buffer_->getHandle(),
                  reinterpret_cast<VkBuffer>(buffer.getResourceHandle()),
                  offset,
                  num_bytes,
                  data);
}

void VulkanCommandExecutor::copyBuffer(const Buffer& src_buffer,
                                       const Buffer& dst_buffer,
                                       uint64_t num_bytes,
                                       uint64_t src_offset,
                                       uint64_t dst_offset) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() != VulkanCommandBuffer::State::kInRenderpass);
  VkBufferCopy copy_params;
  copy_params.srcOffset = src_offset;
  copy_params.dstOffset = dst_offset;
  copy_params.size = num_bytes;
  vkCmdCopyBuffer(active_buffer_->getHandle(),
                  reinterpret_cast<VkBuffer>(src_buffer.getResourceHandle()),
                  reinterpret_cast<VkBuffer>(dst_buffer.getResourceHandle()),
                  1,
                  &copy_params);
}

void VulkanCommandExecutor::transitionFramebufferLayout(Framebuffer& framebuffer,
                                                        ImageLayout layout) {
  CHECK(active_buffer_);
  static_cast<VulkanFramebuffer*>(&framebuffer)
      ->transitionToImageLayout(*active_buffer_, layout);
}

void VulkanCommandExecutor::clearFramebufferAttachment(
    Framebuffer& framebuffer,
    Framebuffer::Attachment attachment) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kInRenderpass);
  CHECK(current_render_pass_);
  current_render_pass_->clearAttachment(
      framebuffer, attachment, *active_buffer_, render_area_, subpass_index_);
}

void VulkanCommandExecutor::clearTexture(Texture& texture, ImageLayout final_layout) {
  static constexpr VkClearColorValue black{};
  static constexpr VkClearDepthStencilValue zero{};
  clearTextureToValue(texture, black, zero, final_layout);
}

void VulkanCommandExecutor::clearTextureToValue(Texture& texture,
                                                ClearTextureValue value,
                                                ImageLayout final_layout) {
  switch (texture.getPixelFormat()) {
    case PixelFormat::kRGBA8: {
      const VkClearColorValue color_value{{value.r, value.g, value.b, value.a}};
      static constexpr VkClearDepthStencilValue zero{};
      clearTextureToValue(texture, color_value, zero, final_layout);
    } break;
    case PixelFormat::kBGRA8: {
      const VkClearColorValue color_value{{value.b, value.g, value.r, value.a}};
      static constexpr VkClearDepthStencilValue zero{};
      clearTextureToValue(texture, color_value, zero, final_layout);
    } break;
    case PixelFormat::kDepth:
    case PixelFormat::kDepthHighP:
    case PixelFormat::kDepthStencil:
    case PixelFormat::kDepthStencilHighP: {
      static constexpr VkClearColorValue black{};
      const VkClearDepthStencilValue depth_stencil_value{value.d, value.s};
      clearTextureToValue(texture, black, depth_stencil_value, final_layout);
    } break;
    case PixelFormat::kR32UI: {
      VkClearColorValue color_value{};
      color_value.uint32[0] = value.u;
      static constexpr VkClearDepthStencilValue zero{};
      clearTextureToValue(texture, color_value, zero, final_layout);
    } break;
    case PixelFormat::kR32I: {
      VkClearColorValue color_value{};
      color_value.int32[0] = value.i;
      static constexpr VkClearDepthStencilValue zero{};
      clearTextureToValue(texture, color_value, zero, final_layout);
    } break;
    case PixelFormat::kR8:
    case PixelFormat::kRG8:
      CHECK(false) << "ClearTextureValue not yet implemented";
    case PixelFormat::kCOUNT:
      UNREACHABLE();
  }
}

void VulkanCommandExecutor::clearTextureToValue(
    Texture& texture,
    const VkClearColorValue& color_value,
    const VkClearDepthStencilValue& depth_stencil_value,
    ImageLayout final_layout) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() != VulkanCommandBuffer::State::kInRenderpass);

  auto const& vulkan_texture = static_cast<const VulkanTexture&>(texture);

  auto vk_cmd_buffer = active_buffer_->getHandle();
  auto vk_image = vulkan_texture.getImage();
  auto const vk_image_aspect =
      pixel_format_to_vk_image_aspect_flags(texture.getPixelFormat());
  VkImageSubresourceRange subresource_range{vk_image_aspect, 0, 1, 0, texture.getDepth()};

  auto const original_layout = image_layout_mgr_.getCurrentLayout(vk_image);

  // Clear commands are technically a transfer operation so we must be in transfer dest
  // layout and barriers should bracket on PIPELINE_STAGE_TRANSFER_BIT
  if (original_layout != ImageLayout::kTransferDst) {
    image_layout_mgr_.transitionToLayout(vk_image,
                                         ImageLayout::kTransferDst,
                                         *active_buffer_,
                                         std::nullopt,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         subresource_range);
  }

  bool is_color_format = true;  // common case
  if (vk_image_aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
    vkCmdClearColorImage(vk_cmd_buffer,
                         vk_image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &color_value,
                         1,
                         &subresource_range);
  } else {
    vkCmdClearDepthStencilImage(vk_cmd_buffer,
                                vk_image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                &depth_stencil_value,
                                1,
                                &subresource_range);
    is_color_format = false;
  }

  // what layout do we need to leave the image in?
  if (final_layout == ImageLayout::kUndefined) {
    // Final layout is not specified
    // If original layout IS NOT undefined, restore it
    // If original layout IS undefined, try to autodetect from usage bits since we can't
    // set it back
    final_layout = original_layout == ImageLayout::kUndefined
                       ? image_usage_bits_to_final_layout(vulkan_texture.getUsageBits())
                       : original_layout;
  }

  // restore layout if necessary
  if (final_layout != ImageLayout::kTransferDst) {
    image_layout_mgr_.transitionToLayout(
        vk_image,
        final_layout,
        *active_buffer_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        get_stage_mask_for_layout(final_layout, is_color_format),
        subresource_range);
  }
}

void VulkanCommandExecutor::imageMemoryBarrier(Texture& texture,
                                               ImageMemoryBarrierType barrier_type,
                                               std::optional<ImageLayout> to_layout) {
  CHECK(active_buffer_);

  auto pixel_format = texture.getPixelFormat();
  auto image_aspect = pixel_format_to_vk_image_aspect_flags(pixel_format);
  auto vk_image = reinterpret_cast<VkImage>(texture.getResourceHandle());
  VkImageSubresourceRange subresource_range{image_aspect, 0, 1, 0, texture.getDepth()};

  // Let ImageLayoutManager handle pure layout transitions
  if (barrier_type == ImageMemoryBarrierType::kImageLayout) {
    CHECK(to_layout) << "Layout required for kImageLayout barrier";
    CHECK_NE(*to_layout, ImageLayout::kUndefined)
        << "Cannot transition to ImageLayout::kUndefined";
    image_layout_mgr_.transitionToLayout(
        vk_image,
        *to_layout,
        *active_buffer_,
        std::nullopt,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        subresource_range);
  } else {
    // Allow folding layout transitions into other barriers
    bool is_color_format = is_color_pixel_format(pixel_format);
    VkImageLayout old_layout = image_layout_to_vk_image_layout(
        image_layout_mgr_.getCurrentLayout(vk_image), is_color_format);
    VkImageLayout new_layout = old_layout;  // default to no layout change
    if (to_layout) {
      new_layout = image_layout_to_vk_image_layout(*to_layout, is_color_format);
      image_layout_mgr_.addOrSetLayout(vk_image, *to_layout);
    }

    // Populate the memory barrier
    VkImageMemoryBarrier vk_barrier = {};
    vk_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    vk_barrier.image = vk_image;
    vk_barrier.oldLayout = old_layout;
    vk_barrier.newLayout = new_layout;
    vk_barrier.subresourceRange = subresource_range;
    vk_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vk_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    VkPipelineStageFlags src_stage_flags = 0u;
    VkPipelineStageFlags dst_stage_flags = 0u;

    // Stage and access flags for fragment shader barriers (attachment vs storage image)
    auto get_fragment_stage_flags =
        [&](VkImageLayout layout) -> std::pair<VkPipelineStageFlags, VkAccessFlags> {
      if (layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        return {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT};
      } else {  // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        return {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT};
      }
    };
    switch (barrier_type) {
      case ImageMemoryBarrierType::kImageLayout:
        UNREACHABLE();
      case ImageMemoryBarrierType::kStorageImageWriteRead:
        vk_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage_flags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dst_stage_flags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        break;

      case ImageMemoryBarrierType::kComputeToFragmentShader:
        vk_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        src_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        std::tie(dst_stage_flags, vk_barrier.dstAccessMask) =
            get_fragment_stage_flags(new_layout);
        break;

      case ImageMemoryBarrierType::kShaderReadToTransfer:
        vk_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vk_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src_stage_flags =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dst_stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;

      case ImageMemoryBarrierType::kFragmentShaderToCompute:
        std::tie(src_stage_flags, vk_barrier.srcAccessMask) =
            get_fragment_stage_flags(old_layout);

        vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dst_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;

      case ImageMemoryBarrierType::kTransferToCompute:
        vk_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src_stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;

        vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dst_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;

      case ImageMemoryBarrierType::kTransferToFragmentShader:
        vk_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src_stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;

        std::tie(dst_stage_flags, vk_barrier.dstAccessMask) =
            get_fragment_stage_flags(new_layout);
        break;
    }

    vkCmdPipelineBarrier(active_buffer_->getHandle(),
                         src_stage_flags,
                         dst_stage_flags,
                         VK_DEPENDENCY_BY_REGION_BIT,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &vk_barrier);
  }
}

void VulkanCommandExecutor::bufferMemoryBarrier(const Buffer& buffer,
                                                BufferMemoryBarrierType barrier_type) {
  VkBufferMemoryBarrier vk_barrier = {};
  vk_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  vk_barrier.buffer = reinterpret_cast<VkBuffer>(buffer.getResourceHandle());
  vk_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  vk_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  vk_barrier.offset = 0u;
  vk_barrier.size = buffer.getNumBytes();

  VkPipelineStageFlags src_stage_flags = 0u;
  VkPipelineStageFlags dst_stage_flags = 0u;

  switch (barrier_type) {
    case BufferMemoryBarrierType::kTransferToFragmentShader:
      vk_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      src_stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      break;
    case BufferMemoryBarrierType::kTransferToCompute:
      vk_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      src_stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    case BufferMemoryBarrierType::kFragmentShaderToCompute:
      vk_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      src_stage_flags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    case BufferMemoryBarrierType::kComputeToMeshShader:
      vk_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      src_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT;
      break;
    case BufferMemoryBarrierType::kComputeWriteToComputeRead:
      vk_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      src_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      dst_stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
  }

  vkCmdPipelineBarrier(active_buffer_->getHandle(),
                       src_stage_flags,
                       dst_stage_flags,
                       VK_DEPENDENCY_BY_REGION_BIT,
                       0,
                       nullptr,
                       1,
                       &vk_barrier,
                       0,
                       nullptr);
}

void VulkanCommandExecutor::setPushConstantUInt32(Pipeline& pipeline,
                                                  std::string_view name,
                                                  ShaderStageBits shader_stages,
                                                  uint32_t value,
                                                  uint32_t offset) {
  CHECK(active_buffer_);
  vkCmdPushConstants(active_buffer_->getHandle(),
                     reinterpret_cast<VkPipelineLayout>(pipeline.getLayout()),
                     shader_stage_bits_to_vk_shader_stage_flag(shader_stages),
                     offset,
                     sizeof(uint32_t),
                     &value);
}

void VulkanCommandExecutor::setPushConstants(Pipeline& pipeline,
                                             std::string_view name,
                                             ShaderStageBits shader_stages,
                                             const void* values,
                                             uint32_t num_bytes,
                                             uint32_t offset) {
  CHECK(active_buffer_);
  vkCmdPushConstants(active_buffer_->getHandle(),
                     reinterpret_cast<VkPipelineLayout>(pipeline.getLayout()),
                     shader_stage_bits_to_vk_shader_stage_flag(shader_stages),
                     offset,
                     num_bytes,
                     values);
}

void VulkanCommandExecutor::dispatchCompute(Pipeline& pipeline,
                                            uint32_t specialization_id,
                                            uint32_t group_count_x,
                                            uint32_t group_count_y,
                                            uint32_t group_count_z) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kRecording);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  bindPipeline(vk_cmd_buffer, Pipeline::Type::kCompute, pipeline, specialization_id);

  vkCmdDispatch(vk_cmd_buffer, group_count_x, group_count_y, group_count_z);
}

void VulkanCommandExecutor::buildAccelerationStructure(const void* build_geometry_info,
                                                       const void* build_ranges_info) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kRecording);
  auto vk_cmd_buffer = active_buffer_->getHandle();

  device_funcs_.vkCmdBuildAccelerationStructuresKHR(
      vk_cmd_buffer,
      1,
      static_cast<const VkAccelerationStructureBuildGeometryInfoKHR*>(
          build_geometry_info),
      static_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(
          build_ranges_info));
}

void VulkanCommandExecutor::traceRays(RaytracingPipeline& pipeline,
                                      const ShaderBindingTable& sbt,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t depth) {
  CHECK(active_buffer_);
  CHECK(active_buffer_->getState() == VulkanCommandBuffer::State::kRecording);

  auto vk_cmd_buffer = active_buffer_->getHandle();
  bindPipeline(vk_cmd_buffer, Pipeline::Type::kRaytracing, pipeline);
  device_funcs_.vkCmdTraceRaysKHR(
      vk_cmd_buffer,
      reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(
          &sbt.getRegion(ShaderBindingTable::Entry::kRayGen)),
      reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(
          &sbt.getRegion(ShaderBindingTable::Entry::kMiss)),
      reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(
          &sbt.getRegion(ShaderBindingTable::Entry::kHit)),
      reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(
          &sbt.getRegion(ShaderBindingTable::Entry::kCallable)),
      width,
      height,
      depth);
}

void VulkanCommandExecutor::insertLabel(const std::string_view name) {
  CHECK(active_buffer_);
  active_buffer_->insertLabel(name);
}

void VulkanCommandExecutor::pushLabel(const std::string_view name) {
  CHECK(active_buffer_);
  active_buffer_->pushLabel(name);
}

void VulkanCommandExecutor::popLabel() {
  CHECK(active_buffer_);
  active_buffer_->popLabel();
}

void VulkanCommandExecutor::resetQueryPool(QueryPool& query_pool,
                                           std::optional<uint32_t> first_query,
                                           std::optional<uint32_t> query_count) {
  CHECK(active_buffer_);
  uint32_t first = first_query ? *first_query : 0;
  uint32_t count = query_count ? *query_count : query_pool.getNumQueries();
  CHECK_LE(first + count, query_pool.getNumQueries());
  vkCmdResetQueryPool(active_buffer_->getHandle(),
                      reinterpret_cast<VkQueryPool>(query_pool.getResourceHandle()),
                      first,
                      count);
}

void VulkanCommandExecutor::writeTimestamp(QueryPool& query_pool,
                                           PipelineStageBits pipeline_stage,
                                           uint32_t query_id) {
  CHECK(active_buffer_);
  vkCmdWriteTimestamp(active_buffer_->getHandle(),
                      static_cast<VkPipelineStageFlagBits>(pipeline_stage),
                      reinterpret_cast<VkQueryPool>(query_pool.getResourceHandle()),
                      query_id);
}

void VulkanCommandExecutor::beginQuery(QueryPool& query_pool) {
  CHECK(active_buffer_);
  VkQueryControlFlags flags = query_pool.getType() == QueryPool::Type::kOcclusion
                                  ? VK_QUERY_CONTROL_PRECISE_BIT
                                  : 0u;
  vkCmdBeginQuery(active_buffer_->getHandle(),
                  reinterpret_cast<VkQueryPool>(query_pool.getResourceHandle()),
                  0u,
                  flags);
}

void VulkanCommandExecutor::endQuery(QueryPool& query_pool) {
  CHECK(active_buffer_);
  vkCmdEndQuery(active_buffer_->getHandle(),
                reinterpret_cast<VkQueryPool>(query_pool.getResourceHandle()),
                0u);
}

}  // namespace gfx
