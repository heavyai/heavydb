/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/noncopyable.hpp>

#include <optional>
#include <string_view>
#include <vector>

#include "GfxDriver/Commands/MemoryArena.h"
#include "GfxDriver/Commands/QueryPool.h"
#include "GfxDriver/Pipeline/Enums.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/ShaderBindingTable.h"
#include "GfxDriver/Resources/Framebuffer.h"
#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/Resources/VertexBufferRef.h"
#include "GfxDriver/ShaderCompiler/Types.h"
#include "Logger/Logger.h"

namespace gfx {

class CommandExecutor;

/**
 * Image memory barrier types
 * */
enum class ImageMemoryBarrierType {
  kImageLayout,  // Transition the image to a new layout. This barriers on ALL_COMMANDS so
                 // is not optimal
  kStorageImageWriteRead,    // Shader write to shader read barrier to protect
                             // storage images from Read after Write hazards
  kComputeToFragmentShader,  // Compute shader write to fragment shader read
  kFragmentShaderToCompute,  // Fragment shader write to compute shader read
  kShaderReadToTransfer,     // Compute / fragment shader read to transfer (usually for
                             // read->clear)
  kTransferToCompute,        // Transfer write to compute shader read
  kTransferToFragmentShader  // Transfer write to fragment shader read
};

/**
 * Buffer memory barrier types
 * */
enum class BufferMemoryBarrierType {
  kTransferToFragmentShader,  // Transfer write (fill cmd) to fragment shader read
  kTransferToCompute,         // Transfer write (fill cmd) to compute shader read
  kFragmentShaderToCompute,   // Fragment shader write to compute shader read
  kComputeToMeshShader,       // Compute shader write to mesh shader read
  kComputeWriteToComputeRead  // Compute shader write to compute shader read
};

/**
 * BaseCommand and Command
 *
 * Base and implementation template classes for command types
 * Commands are typically parameter containers that can be fed to
 * a graphics api dependent CommandExecutor to handle that command's action
 * */
struct BaseCommand {
  virtual void execute(CommandExecutor& executor) = 0;
  BaseCommand* next = nullptr;
};

template <typename TCommand>
struct Command : public BaseCommand {
  void execute(CommandExecutor& executor) final {
    TCommand* cmd = static_cast<TCommand*>(this);
    cmd->executeImpl(executor);
    cmd->~TCommand();
  }
};

/**
 * CommandList class
 *
 * Manages a linked list of commands allocated from a MemoryArena
 * Internally stores a CommandExecutor that is used to execute the
 * individual commands during a flush()
 * */
class CommandList : boost::noncopyable {
 public:
  enum class SubmitType { kImmediateReturn, kWaitComplete };

  // target limit for max command list size. This is not a hard limit
  static constexpr int32_t kMaxCommandsPerSubmit = 500;

  explicit CommandList(CommandExecutor& executor);
  CommandList() = delete;
  ~CommandList() = default;

  int32_t getCount() const { return command_count_; }
  bool hasCommands() const { return command_count_ > 0; }
  BaseCommand* getRootCommand() const { return root_; }

  // Stats
  int32_t getMaxCount();
  uint64_t getMemoryArenaSize() const;

  // Iterate command list, executing and destructing the commands.
  // The list is cleared upon completion.
  // name MUST remain valid until the resulting CommandBuffer has completed execution
  // SubmitType::kImmediateReturn will not wait for command completion
  // SubmitType::kWaitComplete will wait on completion of submission and
  // any other pending command buffers
  void flush(const std::string_view name, SubmitType type = SubmitType::kWaitComplete);

  // Command submit with optional wait and signal semaphores
  // Submitted commands will not execute until all wait_semaphores are signaled
  // When complete, all signal_semaphores will be signaled
  void flush(const std::string_view name,
             SubmitType type,
             const std::vector<SemaphoreHandle>& wait_semaphores,
             const std::vector<SemaphoreHandle>& signal_semaphores);

  // Clear all commands in the list. Does NOT destruct the commands
  void clear();

  //
  // Command insertion
  // Create new commands in the command list, allocated from the arena
  //

  //
  // RenderArea and Viewport
  //

  // Set the viewport for a specific set of commands. Reset by flush as it must be
  // bound to every CommandBuffer in vulkan (dynamic pipeline state)
  CommandList& setViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

  // Set the RenderArea to use in beginRenderPass calls. Must be manually reset
  CommandList& setRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

  CommandList& setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height);

  //
  // RenderPass commands
  // begin/end and commands that must be issued inside a render pass
  // beginRenderPass must be followed by an endRenderPass or flush will fail
  //

  // RenderArea must be set (either the default or via explicit command)
  CommandList& beginRenderPass(RenderPass& render_pass, Framebuffer& framebuffer);

  CommandList& endRenderPass();

  // Advance to the next subpass in the RenderPass
  CommandList& nextSubpass();

  //
  // Draw commands
  //
  CommandList& drawFullscreen(Pipeline& pipeline);

  CommandList& drawVertices(Pipeline& pipeline,
                            const VertexBuffer& vertex_buffer,
                            uint32_t vertex_count,
                            uint64_t offset_bytes = 0u,
                            uint32_t first_vertex = 0u,
                            uint32_t instance_count = 1u,
                            uint32_t start_instance = 0u);

  CommandList& drawVertices(Pipeline& pipeline,
                            const VertexBufferRefs& vertex_buffer_refs,
                            uint32_t vertex_count,
                            uint32_t first_vertex = 0u,
                            uint32_t instance_count = 1u,
                            uint32_t start_instance = 0u);

  CommandList& drawIndirect(Pipeline& pipeline,
                            const VertexBuffer& vertex_buffer,
                            const IndirectDrawVertexBuffer& indirect_buffer,
                            uint32_t draw_count,
                            uint32_t first_index = 0u);

  CommandList& drawIndirectIndexed(Pipeline& pipeline,
                                   const VertexBuffer& vertex_buffer,
                                   const IndexBuffer& index_buffer,
                                   const IndirectDrawIndexBuffer& indirect_buffer,
                                   uint32_t draw_count,
                                   uint32_t first_index = 0u);

  CommandList& drawIndexed(Pipeline& pipeline,
                           const VertexBuffer& vertex_buffer,
                           const IndexBuffer& index_buffer,
                           uint32_t index_count,
                           uint32_t start_index = 0u,
                           int32_t vertex_offset = 0u,
                           uint32_t instance_count = 1u,
                           uint32_t start_instance = 0u);

  CommandList& drawIndexed(Pipeline& pipeline,
                           const VertexBufferRefs& vertex_buffer_refs,
                           const IndexBuffer& index_buffer,
                           uint32_t index_count,
                           uint32_t start_index = 0u,
                           uint32_t instance_count = 1u,
                           uint32_t start_instance = 0u);

  // Mesh Shader task dispatch
  CommandList& drawMeshTasks(Pipeline& pipeline,
                             uint32_t group_count_x,
                             uint32_t group_count_y,
                             uint32_t group_count_z);

  //
  // Buffer commands
  //

  // fillBuffer
  // num_bytes and offset must be a multiple of 4
  CommandList& fillBuffer(const Buffer& buffer,
                          uint32_t data,
                          uint64_t num_bytes,
                          uint64_t offset = 0ull);

  CommandList& copyBuffer(const Buffer& src_buffer,
                          const Buffer& dst_buffer,
                          uint64_t num_bytes,
                          uint64_t src_offset = 0ull,
                          uint64_t dst_offset = 0ull);

  //
  // Image layouts and texture clearing
  //
  // Transition all color attachments for the Framebuffer
  // depth / stencil are currently never read so are always left in
  // Attachment optimal layout. Unpacking depth can be expensive so if
  // depth readback becomes necessary it should be made optional
  CommandList& transitionFramebufferLayout(Framebuffer& framebuffer, ImageLayout layout);
  CommandList& clearFramebufferAttachment(Framebuffer& framebuffer,
                                          Framebuffer::Attachment attachment);

  // If final_layout is left as kUndefined, the Image will be transitioned based on usage
  // bits. Note that ShaderReadOnly and Storage usage are mutually exclusive in this case
  CommandList& clearTexture(Texture& texture,
                            ImageLayout final_layout = ImageLayout::kUndefined);
  CommandList& clearTextureToValue(Texture& texture,
                                   ClearTextureValue value,
                                   ImageLayout final_layout = ImageLayout::kUndefined);

  //
  // Pipeline barriers
  //

  // For image barriers, passing to_layout is required for kLayoutTransition barrier type,
  // but is optional for other barriers
  CommandList& imageMemoryBarrier(Texture& texture,
                                  ImageMemoryBarrierType barrier_type,
                                  std::optional<ImageLayout> to_layout = std::nullopt);

  CommandList& bufferMemoryBarrier(const Buffer& buffer,
                                   BufferMemoryBarrierType barrier_type);

  //
  // Push constants
  //
  CommandList& setPushConstantUInt32(Pipeline& pipeline,
                                     std::string_view name,
                                     ShaderStageBits shader_stages,
                                     uint32_t value,
                                     uint32_t offset = 0);

  // WARNING: data is NOT copied until CommandList::flush is called
  // values MUST remain valid until flush
  CommandList& setPushConstants(Pipeline& pipeline,
                                std::string_view name,
                                ShaderStageBits shader_stages,
                                const void* values,
                                uint32_t num_bytes,
                                uint32_t offset = 0);

  //
  // Compute
  //
  CommandList& dispatchCompute(Pipeline& pipeline,
                               uint32_t specialization_id,
                               uint32_t group_count_x,
                               uint32_t group_count_y,
                               uint32_t group_count_z);

  //
  // Raytracing
  //
  CommandList& buildAccelerationStructure(const void* build_geometry_info,
                                          const void* build_ranges_info);

  CommandList& traceRays(RaytracingPipeline& pipeline,
                         const ShaderBindingTable& sbt,
                         uint32_t width,
                         uint32_t height,
                         uint32_t depth);

  //
  // Debug Labels (annotations)
  //
  CommandList& insertLabel(const std::string_view name);
  CommandList& pushLabel(const std::string_view name);
  CommandList& popLabel();

  //
  // Queries
  //
  CommandList& resetQueryPool(QueryPool& query_pool,
                              std::optional<uint32_t> first_query = std::nullopt,
                              std::optional<uint32_t> query_count = std::nullopt);
  CommandList& writeTimestamp(QueryPool& query_pool,
                              PipelineStageBits pipeline_stage,
                              uint32_t query_id);
  CommandList& beginQuery(QueryPool& query_pool);
  CommandList& endQuery(QueryPool& query_pool);

 private:
  CommandExecutor& executor_;
  MemoryArena arena_;
  int32_t command_count_;
  int32_t max_command_count_;
  BaseCommand* root_;        // list head pointer
  BaseCommand** next_link_;  // pointer to the tail command's next pointer

  bool is_inside_renderpass_;

  // Allocate a new command from the MemoryArena
  // Set root command if first command otherwise set next_link_
  BaseCommand* alloc(uint32_t size, size_t alignment);

  // Traverse the command list and execute and destruct them
  // Does not call clear on the memory arena
  void execute(const std::string_view name,
               SubmitType submit_type,
               const std::vector<SemaphoreHandle>& wait_semaphores,
               const std::vector<SemaphoreHandle>& signal_semaphores);

  // Allocate a new command of type COMMAND_TYPE with constructor arguments ARGS
  // Command is allocated from the memory arena and constructed with placement new
  template <typename COMMAND_TYPE, typename... ARGS>
  BaseCommand* addCommand(ARGS&&... args) {
    void* ptr = alloc(sizeof(COMMAND_TYPE), alignof(COMMAND_TYPE));
    BaseCommand* cmd =
        static_cast<BaseCommand*>(new (ptr) COMMAND_TYPE(std::forward<ARGS>(args)...));
    CHECK(cmd);
    return cmd;
  }
};

using CommandListUqPtr = std::unique_ptr<CommandList>;

/**
 * CommandListIterator class
 *
 * Utility class to simplify iterating the linked list of commands during
 * command list execution and testing
 * */
class CommandListIterator {
 public:
  explicit CommandListIterator(const CommandList& list);
  ~CommandListIterator();

  bool hasCommandsRemaining() const;
  BaseCommand* getNextCommand();

 private:
  BaseCommand* current_command_;
  int32_t num_list_commands_;
  int32_t num_iterated_commands_;
};

}  // namespace gfx
