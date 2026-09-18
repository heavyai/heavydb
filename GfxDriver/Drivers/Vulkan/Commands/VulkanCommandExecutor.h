/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Commands/CommandExecutor.h"

#include <vector>

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/Resources/ImageLayoutManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanRenderPass.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Pipeline/Pipeline.h"

namespace gfx {

class VulkanCommandExecutor : public CommandExecutor {
 public:
  explicit VulkanCommandExecutor(const VulkanDeviceContext& device,
                                 VulkanCommandPool& command_pool);
  ~VulkanCommandExecutor() override = default;

  const DeviceContext& getDeviceContext() const override;

  // Attempt to clean up internal state. Should generally only be called
  // in exceptional cases, such as when an error occurs mid execution
  void resetState(bool abort_on_exception) override;

  // Persistent render area used for setting RenderPass render regions
  void setRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override;

  void setDefaultViewportAndRenderArea(uint32_t x,
                                       uint32_t y,
                                       uint32_t width,
                                       uint32_t height) override;

  // Command sequence
  void beginCommandSequence() override;
  void submitCommandSequence(
      const std::string_view name,
      CommandList::SubmitType submit_type,
      const std::vector<SemaphoreHandle>& wait_semaphores,
      const std::vector<SemaphoreHandle>& signal_semaphores) override;
  void submitCommandBatch() override;
  void waitForCompletion(bool is_profiling_run) override;

  // Direct command buffer recording
  // For use with third party APIs like ImGui
  // call beginCommandSequence() then
  //  call getActiveCommandBuffer() to check state and record vkCmd* commands
  //  call VulkanCommandExecutor command functions directly
  // call submitCommandSequence() when done to submit commands to the queue
  const VulkanCommandBuffer& getActiveCommandBuffer() const;

  //
  // Commands
  //
  void beginRenderPass(RenderPass& render_pass, Framebuffer& framebuffer) override;
  void endRenderPass() override;
  void nextSubpass() override;

  void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) override;

  void setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) override;

  void drawFullScreenQuad(Pipeline& pipeline) override;

  void drawVertices(Pipeline& pipeline,
                    const VertexBuffer& vertex_buffer,
                    uint32_t vertex_count,
                    uint64_t offset_bytes,
                    uint32_t first_vertex,
                    uint32_t instance_count,
                    uint32_t start_instance) override;

  void drawVertices(Pipeline& pipeline,
                    const VertexBufferRefs& vertex_buffer_refs,
                    uint32_t vertex_count,
                    uint32_t first_vertex,
                    uint32_t instance_count,
                    uint32_t start_instance) override;

  void drawIndirect(Pipeline& pipeline,
                    const VertexBuffer& vertex_buffer,
                    const IndirectDrawVertexBuffer& indirect_buffer,
                    uint32_t draw_count,
                    uint32_t first_index) override;

  void drawIndirectIndexed(Pipeline& pipeline,
                           const VertexBuffer& vertex_buffer,
                           const IndexBuffer& index_buffer,
                           const IndirectDrawIndexBuffer& buffer,
                           uint32_t draw_count,
                           uint32_t first_index) override;

  void drawIndexed(Pipeline& pipeline,
                   const VertexBuffer& vertex_buffer,
                   const IndexBuffer& index_buffer,
                   uint32_t index_count,
                   uint32_t start_index,
                   int32_t vertex_offset,
                   uint32_t instance_count,
                   uint32_t start_instance) override;

  void drawIndexed(Pipeline& pipeline,
                   const VertexBufferRefs& vertex_buffer_refs,
                   const IndexBuffer& index_buffer,
                   uint32_t index_count,
                   uint32_t start_index,
                   uint32_t instance_count,
                   uint32_t start_instance) override;

  void drawMeshTasks(Pipeline& pipeline,
                     uint32_t group_count_x,
                     uint32_t group_count_y,
                     uint32_t group_count_z) override;

  void fillBuffer(const Buffer& buffer,
                  uint32_t data,
                  uint64_t num_bytes,
                  uint64_t offset) override;

  void copyBuffer(const Buffer& src_buffer,
                  const Buffer& dst_buffer,
                  uint64_t num_bytes,
                  uint64_t src_offset,
                  uint64_t dst_offset) override;

  void transitionFramebufferLayout(Framebuffer& framebuffer, ImageLayout layout) override;
  void clearFramebufferAttachment(Framebuffer& framebuffer,
                                  Framebuffer::Attachment attachment) override;

  void clearTexture(Texture& texture, ImageLayout final_layout) override;
  void clearTextureToValue(Texture& texture,
                           ClearTextureValue value,
                           ImageLayout final_layout) override;

  void imageMemoryBarrier(Texture& texture,
                          ImageMemoryBarrierType barrier_type,
                          std::optional<ImageLayout> to_layout) override;

  void bufferMemoryBarrier(const Buffer& buffer,
                           BufferMemoryBarrierType barrier_type) override;

  void setPushConstantUInt32(Pipeline& pipeline,
                             std::string_view name,
                             ShaderStageBits shader_stages,
                             uint32_t value,
                             uint32_t offset) override;

  void setPushConstants(Pipeline& pipeline,
                        std::string_view name,
                        ShaderStageBits shader_stages,
                        const void* values,
                        uint32_t num_bytes,
                        uint32_t offset) override;

  void dispatchCompute(Pipeline& pipeline,
                       uint32_t specialization_id,
                       uint32_t group_count_x,
                       uint32_t group_count_y,
                       uint32_t group_count_z) override;

  void buildAccelerationStructure(const void* build_geometry_info,
                                  const void* build_ranges_info) override;

  void traceRays(RaytracingPipeline& pipeline,
                 const ShaderBindingTable& sbt,
                 uint32_t width,
                 uint32_t height,
                 uint32_t depth) override;

  void insertLabel(const std::string_view name) override;
  void pushLabel(const std::string_view name) override;
  void popLabel() override;

  void resetQueryPool(QueryPool& query_pool,
                      std::optional<uint32_t> first_query,
                      std::optional<uint32_t> query_count) override;

  void writeTimestamp(QueryPool& query_pool,
                      PipelineStageBits pipeline_stage,
                      uint32_t query_id) override;

  void beginQuery(QueryPool& query_pool) override;
  void endQuery(QueryPool& query_pool) override;

 private:
  const VulkanDeviceContext& device_context_;
  const VulkanDeviceFunctions& device_funcs_;
  VulkanCommandPool& command_pool_;
  ImageLayoutManager& image_layout_mgr_;

  VulkanCommandBuffer* active_buffer_;

  VkRect2D render_area_;
  VkViewport default_viewport_;
  VkRect2D default_scissor_;
  VulkanRenderPass* current_render_pass_;
  uint32_t subpass_index_;

  // Pipeline and buffer bindings for active_buffer_
  bool is_viewport_bound_;
  VkPipeline bound_graphics_pipeline_;
  VkPipeline bound_compute_pipeline_;
  VkPipeline bound_raytracing_pipeline_;
  std::array<VkBuffer, 2> bound_vertex_buffers_;
  std::array<uint64_t, 2> bound_vertex_buffers_offset_bytes_;
  VkBuffer bound_index_buffer_;
  uint64_t bound_index_buffer_offset_bytes_;

  void autobindViewport();
  void bindPipeline(const VkCommandBuffer vk_cmd_buffer,
                    const Pipeline::Type type,
                    const Pipeline& pipeline,
                    const uint32_t specialization_id = 0u);
  void bindVertexBuffer(const VkCommandBuffer vk_cmd_buffer,
                        const VertexBuffer& vertex_buffer,
                        const uint32_t binding,
                        const uint64_t additional_offset_bytes);
  void bindIndexBuffer(const VkCommandBuffer vk_cmd_buffer,
                       const IndexBuffer& index_buffer);
  inline void clearBindings() {
    is_viewport_bound_ = false;
    bound_graphics_pipeline_ = VK_NULL_HANDLE;
    bound_compute_pipeline_ = VK_NULL_HANDLE;
    bound_raytracing_pipeline_ = VK_NULL_HANDLE;
    bound_vertex_buffers_ = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    bound_index_buffer_ = VK_NULL_HANDLE;
    current_render_pass_ = nullptr;
    active_buffer_ = nullptr;
  }

  void clearTextureToValue(Texture& texture,
                           const VkClearColorValue& color_value,
                           const VkClearDepthStencilValue& depth_stencil_value,
                           ImageLayout final_layout);
};

}  // namespace gfx
