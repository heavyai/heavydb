/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Commands/CommandList.h"

#include <exception>

#include "GfxDriver/Commands/CommandExecutor.h"
#include "GfxDriver/Commands/CommandImpls.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/RenderPass.h"
#include "Logger/Logger.h"

namespace gfx {

#ifdef NDEBUG
static constexpr bool kEnableCommandLabels = false;
#else
static constexpr bool kEnableCommandLabels = true;
#endif

//
// CommandList class
//
CommandList::CommandList(CommandExecutor& executor)
    : executor_{executor}
    , command_count_{0}
    , max_command_count_{0}
    , root_{nullptr}
    , next_link_{nullptr}
    , is_inside_renderpass_{false} {}

BaseCommand* CommandList::alloc(uint32_t size, size_t alignment) {
  BaseCommand* rtn = static_cast<BaseCommand*>(arena_.alloc(size, alignment));

  if (next_link_ != nullptr) {
    *next_link_ = rtn;
  } else if (root_ == nullptr) {
    root_ = rtn;
  }
  next_link_ = &rtn->next;
  command_count_++;
  return rtn;
}

void CommandList::clear() {
  max_command_count_ = std::max(max_command_count_, command_count_);
  command_count_ = 0;
  root_ = nullptr;
  next_link_ = nullptr;
  is_inside_renderpass_ = false;
  arena_.clear();
}

uint64_t CommandList::getMemoryArenaSize() const {
  return arena_.size();
}

int32_t CommandList::getMaxCount() {
  // ensure current count is included
  max_command_count_ = std::max(max_command_count_, command_count_);
  return max_command_count_;
}

void CommandList::flush(std::string_view name, SubmitType submit_type) {
  RENDER_LOG_SCOPE();
  CHECK_NE(is_inside_renderpass_, true) << "beginRenderPass must be followed by "
                                           "endRenderPass before flushing command buffer";
  static std::vector<SemaphoreHandle> no_semaphores;
  execute(name, submit_type, no_semaphores, no_semaphores);
  clear();
}

void CommandList::flush(std::string_view name,
                        SubmitType submit_type,
                        const std::vector<SemaphoreHandle>& wait_semaphores,
                        const std::vector<SemaphoreHandle>& signal_semaphores) {
  RENDER_LOG_SCOPE();
  CHECK_NE(is_inside_renderpass_, true) << "beginRenderPass must be followed by "
                                           "endRenderPass before flushing command buffer";
  execute(name, submit_type, wait_semaphores, signal_semaphores);
  clear();
}

void CommandList::execute(std::string_view name,
                          SubmitType submit_type,
                          const std::vector<SemaphoreHandle>& wait_semaphores,
                          const std::vector<SemaphoreHandle>& signal_semaphores) {
  CommandListIterator itr(*this);
  executor_.beginCommandSequence();
  try {
    while (itr.hasCommandsRemaining()) {
      auto* command = itr.getNextCommand();
      command->execute(executor_);
    }
    executor_.submitCommandSequence(
        name, submit_type, wait_semaphores, signal_semaphores);
  } catch (OutOfGpuMemoryError& e) {
    auto current_exception = std::current_exception();
    clear();
    // don't abort if an exception throws during state cleanup
    // to allow out of gpu memory to propagate
    executor_.resetState(false);
    std::rethrow_exception(current_exception);
  } catch (...) {
    auto current_exception = std::current_exception();
    clear();
    // abort if exceptions are thrown as we are in an unknown state
    executor_.resetState(true);
    std::rethrow_exception(current_exception);
  }
}

//
// CommandList command insertion methods
//
CommandList& CommandList::setRenderArea(uint32_t x,
                                        uint32_t y,
                                        uint32_t width,
                                        uint32_t height) {
  addCommand<CmdSetRenderArea>(x, y, width, height);
  return *this;
}

CommandList& CommandList::setViewport(uint32_t x,
                                      uint32_t y,
                                      uint32_t width,
                                      uint32_t height) {
  addCommand<CmdSetViewport>(x, y, width, height);
  return *this;
}

CommandList& CommandList::setScissor(int32_t x,
                                     int32_t y,
                                     uint32_t width,
                                     uint32_t height) {
  addCommand<CmdSetScissor>(x, y, width, height);
  return *this;
}

CommandList& CommandList::beginRenderPass(RenderPass& render_pass,
                                          Framebuffer& framebuffer) {
  CHECK_EQ(is_inside_renderpass_, false)
      << "beginRenderPass must not be called while inside a render pass";
  addCommand<CmdBeginRenderPass>(render_pass, framebuffer);
  render_pass.updateImageLayouts(framebuffer);
  is_inside_renderpass_ = true;
  return *this;
}

CommandList& CommandList::endRenderPass() {
  CHECK(is_inside_renderpass_);
  addCommand<CmdEndRenderPass>();
  is_inside_renderpass_ = false;
  return *this;
}

CommandList& CommandList::nextSubpass() {
  CHECK(is_inside_renderpass_);
  addCommand<CmdNextSubpass>();
  return *this;
}

CommandList& CommandList::drawFullscreen(Pipeline& pipeline) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "drawFullscreen command must be recording inside a render pass";
  addCommand<CmdDrawFullscreenQuad>(pipeline);
  return *this;
}

CommandList& CommandList::drawVertices(Pipeline& pipeline,
                                       const VertexBuffer& vertex_buffer,
                                       uint32_t vertex_count,
                                       uint64_t offset_bytes,
                                       uint32_t first_vertex,
                                       uint32_t instance_count,
                                       uint32_t start_instance) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "drawVertices command must be recording inside a render pass";
  CHECK_GT(vertex_count, 0u);
  CHECK_GT(instance_count, 0u);
  addCommand<CmdDrawVerticesSingle>(pipeline,
                                    vertex_buffer,
                                    vertex_count,
                                    offset_bytes,
                                    first_vertex,
                                    instance_count,
                                    start_instance);
  return *this;
}

CommandList& CommandList::drawVertices(Pipeline& pipeline,
                                       const VertexBufferRefs& vertex_buffer_refs,
                                       uint32_t vertex_count,
                                       uint32_t first_vertex,
                                       uint32_t instance_count,
                                       uint32_t start_instance) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "drawVertices command must be recording inside a render pass";
  CHECK_GT(vertex_count, 0u);
  CHECK_GT(instance_count, 0u);
  addCommand<CmdDrawVerticesMulti>(pipeline,
                                   vertex_buffer_refs,
                                   vertex_count,
                                   first_vertex,
                                   instance_count,
                                   start_instance);
  return *this;
}

CommandList& CommandList::drawIndirect(Pipeline& pipeline,
                                       const VertexBuffer& vertex_buffer,
                                       const IndirectDrawVertexBuffer& indirect_buffer,
                                       uint32_t draw_count,
                                       uint32_t first_index) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "drawIndirect command must be recording inside a render pass";
  addCommand<CmdDrawIndirect>(
      pipeline, vertex_buffer, indirect_buffer, draw_count, first_index);
  CHECK_GT(draw_count, 0u);
  return *this;
}

CommandList& CommandList::drawIndirectIndexed(
    Pipeline& pipeline,
    const VertexBuffer& vertex_buffer,
    const IndexBuffer& index_buffer,
    const IndirectDrawIndexBuffer& indirect_buffer,
    uint32_t draw_count,
    uint32_t first_index) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "drawIndirectIndexed command must be recording inside a render pass";
  CHECK_GT(draw_count, 0u);
  addCommand<CmdDrawIndirectIndexed>(
      pipeline, vertex_buffer, index_buffer, indirect_buffer, draw_count, first_index);
  return *this;
}

CommandList& CommandList::drawIndexed(Pipeline& pipeline,
                                      const VertexBuffer& vertex_buffer,
                                      const IndexBuffer& index_buffer,
                                      uint32_t index_count,
                                      uint32_t start_index,
                                      int32_t vertex_offset,
                                      uint32_t instance_count,
                                      uint32_t start_instance) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "drawIndexed command must be recording inside a render pass";
  CHECK_GT(index_count, 0u);
  CHECK_GT(instance_count, 0u);
  addCommand<CmdDrawIndexedSingle>(pipeline,
                                   vertex_buffer,
                                   index_buffer,
                                   index_count,
                                   start_index,
                                   vertex_offset,
                                   instance_count,
                                   start_instance);
  return *this;
}

CommandList& CommandList::drawIndexed(Pipeline& pipeline,
                                      const VertexBufferRefs& vertex_buffer_refs,
                                      const IndexBuffer& index_buffer,
                                      uint32_t index_count,
                                      uint32_t start_index,
                                      uint32_t instance_count,
                                      uint32_t start_instance) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "drawIndexed command must be recording inside a render pass";
  CHECK_GT(index_count, 0u);
  CHECK_GT(instance_count, 0u);
  addCommand<CmdDrawIndexedMulti>(pipeline,
                                  vertex_buffer_refs,
                                  index_buffer,
                                  index_count,
                                  start_index,
                                  instance_count,
                                  start_instance);
  return *this;
}

CommandList& CommandList::drawMeshTasks(Pipeline& pipeline,
                                        uint32_t group_count_x,
                                        uint32_t group_count_y,
                                        uint32_t group_count_z) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "drawMeshTasks command must be recorded inside a render pass";
  // removed count validation, per dispatchCompute
  addCommand<CmdDrawMeshTasks>(pipeline, group_count_x, group_count_y, group_count_z);
  return *this;
}

CommandList& CommandList::fillBuffer(const Buffer& buffer,
                                     uint32_t data,
                                     uint64_t num_bytes,
                                     uint64_t offset) {
  // size and offset must be multiples of 4
  CHECK_EQ(num_bytes % 4u, 0u);
  CHECK_EQ(offset % 4u, 0u);
  addCommand<CmdFillBuffer>(buffer, data, num_bytes, offset);
  return *this;
}

CommandList& CommandList::copyBuffer(const Buffer& src_buffer,
                                     const Buffer& dst_buffer,
                                     uint64_t num_bytes,
                                     uint64_t src_offset,
                                     uint64_t dst_offset) {
  addCommand<CmdCopyBuffer>(src_buffer, dst_buffer, num_bytes, src_offset, dst_offset);
  return *this;
}

CommandList& CommandList::transitionFramebufferLayout(Framebuffer& framebuffer,
                                                      ImageLayout layout) {
  addCommand<CmdTransitionFramebufferLayout>(framebuffer, layout);
  return *this;
}

CommandList& CommandList::clearFramebufferAttachment(Framebuffer& framebuffer,
                                                     Framebuffer::Attachment attachment) {
  CHECK_EQ(is_inside_renderpass_, true)
      << "clearFramebufferAttachment command must be recording inside a render pass";
  addCommand<CmdClearFramebufferAttachment>(framebuffer, attachment);
  return *this;
}

CommandList& CommandList::clearTexture(Texture& texture, ImageLayout final_layout) {
  CHECK_EQ(is_inside_renderpass_, false)
      << "clearTexture command must not be recording inside a render pass";
  addCommand<CmdClearTexture>(texture, final_layout);
  return *this;
}

CommandList& CommandList::clearTextureToValue(Texture& texture,
                                              ClearTextureValue value,
                                              ImageLayout final_layout) {
  CHECK_EQ(is_inside_renderpass_, false)
      << "clearTextureToValue command must not be recording inside a render pass";
  addCommand<CmdClearTextureToValue>(texture, value, final_layout);
  return *this;
}

CommandList& CommandList::imageMemoryBarrier(Texture& texture,
                                             ImageMemoryBarrierType barrier_type,
                                             std::optional<ImageLayout> to_layout) {
  addCommand<CmdImageMemoryBarrier>(texture, barrier_type, to_layout);
  return *this;
}

CommandList& CommandList::bufferMemoryBarrier(const Buffer& buffer,
                                              BufferMemoryBarrierType barrier_type) {
  addCommand<CmdBufferMemoryBarrier>(buffer, barrier_type);
  return *this;
}

CommandList& CommandList::setPushConstantUInt32(Pipeline& pipeline,
                                                std::string_view name,
                                                ShaderStageBits shader_stages,
                                                uint32_t value,
                                                uint32_t offset) {
  addCommand<CmdSetPushConstantUInt32>(pipeline, name, shader_stages, value, offset);
  return *this;
}

CommandList& CommandList::setPushConstants(Pipeline& pipeline,
                                           std::string_view name,
                                           ShaderStageBits shader_stages,
                                           const void* values,
                                           uint32_t num_bytes,
                                           uint32_t offset) {
  addCommand<CmdSetPushConstants>(
      pipeline, name, shader_stages, values, num_bytes, offset);
  return *this;
}

CommandList& CommandList::dispatchCompute(Pipeline& pipeline,
                                          uint32_t specialization_id,
                                          uint32_t group_count_x,
                                          uint32_t group_count_y,
                                          uint32_t group_count_z) {
  addCommand<CmdDispatchCompute>(
      pipeline, specialization_id, group_count_x, group_count_y, group_count_z);
  return *this;
}

CommandList& CommandList::buildAccelerationStructure(const void* build_geometry_info,
                                                     const void* build_ranges_info) {
  addCommand<CmdBuildAccelerationStructure>(build_geometry_info, build_ranges_info);
  return *this;
}

CommandList& CommandList::traceRays(RaytracingPipeline& pipeline,
                                    const ShaderBindingTable& sbt,
                                    uint32_t width,
                                    uint32_t height,
                                    uint32_t depth) {
  addCommand<CmdTraceRays>(pipeline, sbt, width, height, depth);
  return *this;
}

CommandList& CommandList::insertLabel(const std::string_view name) {
  if constexpr (kEnableCommandLabels) {
    addCommand<CmdInsertLabel>(name);
  }
  return *this;
}

CommandList& CommandList::pushLabel(const std::string_view name) {
  if constexpr (kEnableCommandLabels) {
    addCommand<CmdPushLabel>(name);
  }
  return *this;
}

CommandList& CommandList::popLabel() {
  if constexpr (kEnableCommandLabels) {
    addCommand<CmdPopLabel>();
  }
  return *this;
}

CommandList& CommandList::resetQueryPool(QueryPool& query_pool,
                                         std::optional<uint32_t> first_query,
                                         std::optional<uint32_t> query_count) {
  addCommand<CmdResetQueryPool>(query_pool, first_query, query_count);
  return *this;
}

CommandList& CommandList::writeTimestamp(QueryPool& query_pool,
                                         PipelineStageBits pipeline_stage,
                                         uint32_t query_id) {
  addCommand<CmdWriteTimestamp>(query_pool, pipeline_stage, query_id);
  return *this;
}

CommandList& CommandList::beginQuery(QueryPool& query_pool) {
  addCommand<CmdBeginQuery>(query_pool);
  return *this;
}

CommandList& CommandList::endQuery(QueryPool& query_pool) {
  addCommand<CmdEndQuery>(query_pool);
  return *this;
}

//
// CommandListIterator class
//
CommandListIterator::CommandListIterator(const CommandList& list)
    : current_command_{list.getRootCommand()}
    , num_list_commands_{list.getCount()}
    , num_iterated_commands_{0} {}

CommandListIterator::~CommandListIterator() {
  LOG_IF(WARNING, hasCommandsRemaining())
      << "Destroying gfx::CommandListIterator with pending commands";
}

bool CommandListIterator::hasCommandsRemaining() const {
  return num_iterated_commands_ != num_list_commands_;
}

BaseCommand* CommandListIterator::getNextCommand() {
  BaseCommand* rtn = current_command_;
  if (current_command_ != nullptr) {
    current_command_ = current_command_->next;
    num_iterated_commands_++;
  }
  return rtn;
}

}  // namespace gfx
