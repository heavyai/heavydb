/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

#include <boost/noncopyable.hpp>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Resources/Framebuffer.h"
#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/ShaderCompiler/Types.h"
namespace gfx {

/**
 * CommandExecutor class
 *
 * Provides the execution interface to be called by Commands from a CommandList
 * GfxDrivers must implement a command executor that handles translating and
 * dispatching the commands to the graphics API in use. A single Command may
 * result in many API calls within the CommandExecutor
 * */
class CommandExecutor : boost::noncopyable {
 public:
  virtual ~CommandExecutor() = default;

  virtual const DeviceContext& getDeviceContext() const = 0;

  // Attempt to clean up internal state. Should generally only be called
  // in exceptional cases, such as when an error occurs mid execution
  virtual void resetState(bool abort_on_exception) = 0;

  // Persistent render area used for setting RenderPass render regions
  // and viewports in Vulkan
  virtual void setRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height) = 0;

  // Default viewport to bind to command buffer if no viewport has been set
  virtual void setDefaultViewportAndRenderArea(uint32_t x,
                                               uint32_t y,
                                               uint32_t width,
                                               uint32_t height) = 0;

  // Command sequence
  virtual void beginCommandSequence() = 0;
  virtual void submitCommandSequence(
      const std::string_view name,
      CommandList::SubmitType submit_type,
      const std::vector<SemaphoreHandle>& wait_semaphores,
      const std::vector<SemaphoreHandle>& signal_semaphores) = 0;

  // Batched commands
  // (Currently just calls glFlush. Does nothing in Vulkan)
  virtual void submitCommandBatch() = 0;

  // Wait for all pending commands
  // For Vulkan, this waits on all pending fences
  virtual void waitForCompletion(bool is_profiling_run) = 0;

  // Commands
  virtual void beginRenderPass(RenderPass& render_pass, Framebuffer& framebuffer) = 0;
  virtual void endRenderPass() = 0;
  virtual void nextSubpass() = 0;

  virtual void setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) = 0;

  virtual void setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) = 0;

  virtual void drawFullScreenQuad(Pipeline& pipeline) = 0;

  virtual void drawVertices(Pipeline& pipeline,
                            const VertexBuffer& vertex_buffer,
                            uint32_t count,
                            uint64_t offset_bytes,
                            uint32_t first_vertex,
                            uint32_t instance_count,
                            uint32_t start_instance) = 0;

  virtual void drawVertices(Pipeline& pipeline,
                            const VertexBufferRefs& vertex_buffer_refs,
                            uint32_t count,
                            uint32_t first_vertex,
                            uint32_t instance_count,
                            uint32_t start_instance) = 0;

  virtual void drawIndirect(Pipeline& pipeline,
                            const VertexBuffer& vertex_buffer,
                            const IndirectDrawVertexBuffer& indirect_buffer,
                            uint32_t draw_count,
                            uint32_t start_index) = 0;

  virtual void drawIndirectIndexed(Pipeline& pipeline,
                                   const VertexBuffer& vertex_buffer,
                                   const IndexBuffer& index_buffer,
                                   const IndirectDrawIndexBuffer& indirect_buffer,
                                   uint32_t draw_count,
                                   uint32_t start_index) = 0;

  virtual void drawIndexed(Pipeline& pipeline,
                           const VertexBuffer& vertex_buffer,
                           const IndexBuffer& index_buffer,
                           uint32_t index_count,
                           uint32_t start_index,
                           int32_t vertex_offset,
                           uint32_t instance_count,
                           uint32_t start_instance) = 0;

  virtual void drawIndexed(Pipeline& pipeline,
                           const VertexBufferRefs& vertex_buffer_refs,
                           const IndexBuffer& index_buffer,
                           uint32_t index_count,
                           uint32_t start_index,
                           uint32_t instance_count,
                           uint32_t start_instance) = 0;

  virtual void fillBuffer(const Buffer& buffer,
                          uint32_t data,
                          uint64_t num_bytes,
                          uint64_t offset) = 0;

  virtual void drawMeshTasks(Pipeline& pipeline,
                             uint32_t group_count_x,
                             uint32_t group_count_y,
                             uint32_t group_count_z) = 0;

  virtual void copyBuffer(const Buffer& src_buffer,
                          const Buffer& dst_buffer,
                          uint64_t num_bytes,
                          uint64_t src_offset,
                          uint64_t dst_offset) = 0;

  virtual void transitionFramebufferLayout(Framebuffer& framebuffer,
                                           ImageLayout layout) = 0;
  virtual void clearFramebufferAttachment(Framebuffer& framebuffer,
                                          Framebuffer::Attachment attachment) = 0;

  virtual void clearTexture(Texture& texture, ImageLayout final_layout) = 0;
  virtual void clearTextureToValue(Texture& texture,
                                   ClearTextureValue value,
                                   ImageLayout final_layout) = 0;

  virtual void imageMemoryBarrier(Texture& texture,
                                  ImageMemoryBarrierType barrier_type,
                                  std::optional<ImageLayout> to_layout) = 0;

  virtual void bufferMemoryBarrier(const Buffer& buffer,
                                   BufferMemoryBarrierType barrier_type) = 0;

  virtual void setPushConstantUInt32(Pipeline& pipeline,
                                     std::string_view name,
                                     ShaderStageBits shader_stages,
                                     uint32_t value,
                                     uint32_t offset) = 0;

  virtual void setPushConstants(Pipeline& pipeline,
                                std::string_view name,
                                ShaderStageBits shader_stages,
                                const void* values,
                                uint32_t num_bytes,
                                uint32_t offset) = 0;

  virtual void dispatchCompute(Pipeline& pipeline,
                               uint32_t specialization_id,
                               uint32_t group_count_x,
                               uint32_t group_count_y,
                               uint32_t group_count_z) = 0;

  virtual void buildAccelerationStructure(const void* build_geometry_info,
                                          const void* build_ranges_info) = 0;

  virtual void traceRays(RaytracingPipeline& pipeline,
                         const ShaderBindingTable& sbt,
                         uint32_t width,
                         uint32_t height,
                         uint32_t depth) = 0;

  virtual void insertLabel(const std::string_view name) = 0;
  virtual void pushLabel(const std::string_view name) = 0;
  virtual void popLabel() = 0;

  virtual void resetQueryPool(QueryPool& query_pool,
                              std::optional<uint32_t> first_query,
                              std::optional<uint32_t> query_count) = 0;
  virtual void writeTimestamp(QueryPool& query_pool,
                              PipelineStageBits pipeline_stage,
                              uint32_t query_id) = 0;
  virtual void beginQuery(QueryPool& query_pool) = 0;
  virtual void endQuery(QueryPool& query_pool) = 0;
};

using CommandExecutorUqPtr = std::unique_ptr<CommandExecutor>;

}  // namespace gfx
