/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Commands/CommandList.h"

#include <string_view>

namespace gfx {

struct CmdBeginRenderPass final : public Command<CmdBeginRenderPass> {
  RenderPass& render_pass;
  Framebuffer& framebuffer;
  CmdBeginRenderPass(RenderPass& render_pass, Framebuffer& framebuffer)
      : render_pass{render_pass}, framebuffer{framebuffer} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdEndRenderPass final : public Command<CmdEndRenderPass> {
  CmdEndRenderPass() = default;
  void executeImpl(CommandExecutor& executor);
};

struct CmdNextSubpass final : public Command<CmdNextSubpass> {
  CmdNextSubpass() = default;
  void executeImpl(CommandExecutor& executor);
};

struct CmdSetViewport final : public Command<CmdSetViewport> {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  CmdSetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
      : x{x}, y{y}, width{width}, height{height} {};
  void executeImpl(CommandExecutor& executor);
};

struct CmdSetRenderArea final : public Command<CmdSetRenderArea> {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  CmdSetRenderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
      : x{x}, y{y}, width{width}, height{height} {};
  void executeImpl(CommandExecutor& executor);
};

struct CmdSetScissor final : public Command<CmdSetScissor> {
  int32_t x;
  int32_t y;
  uint32_t width;
  uint32_t height;
  CmdSetScissor(int32_t x, int32_t y, uint32_t width, uint32_t height)
      : x{x}, y{y}, width{width}, height{height} {};
  void executeImpl(CommandExecutor& executor);
};

struct CmdDrawFullscreenQuad final : public Command<CmdDrawFullscreenQuad> {
  Pipeline& pipeline;
  CmdDrawFullscreenQuad(Pipeline& pipeline) : pipeline{pipeline} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdDrawVerticesSingle final : public Command<CmdDrawVerticesSingle> {
  Pipeline& pipeline;
  const VertexBuffer& vertex_buffer;
  uint32_t vertex_count;
  uint64_t offset_bytes;
  uint32_t first_vertex;
  uint32_t instance_count;
  uint32_t start_instance;
  CmdDrawVerticesSingle(Pipeline& pipeline,
                        const VertexBuffer& vertex_buffer,
                        uint32_t vertex_count,
                        uint64_t offset_bytes,
                        uint32_t first_vertex,
                        uint32_t instance_count,
                        uint32_t start_instance)
      : pipeline{pipeline}
      , vertex_buffer{vertex_buffer}
      , vertex_count{vertex_count}
      , offset_bytes{offset_bytes}
      , first_vertex{first_vertex}
      , instance_count{instance_count}
      , start_instance{start_instance} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdDrawVerticesMulti final : public Command<CmdDrawVerticesMulti> {
  Pipeline& pipeline;
  const VertexBufferRefs& vertex_buffer_refs;
  uint32_t vertex_count;
  uint32_t first_vertex;
  uint32_t instance_count;
  uint32_t start_instance;
  CmdDrawVerticesMulti(Pipeline& pipeline,
                       const VertexBufferRefs& vertex_buffer_refs,
                       uint32_t vertex_count,
                       uint32_t first_vertex,
                       uint32_t instance_count,
                       uint32_t start_instance)
      : pipeline{pipeline}
      , vertex_buffer_refs{vertex_buffer_refs}
      , vertex_count{vertex_count}
      , first_vertex{first_vertex}
      , instance_count{instance_count}
      , start_instance{start_instance} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdDrawIndirect final : public Command<CmdDrawIndirect> {
  Pipeline& pipeline;
  const VertexBuffer& vertex_buffer;
  const IndirectDrawVertexBuffer& indirect_buffer;
  uint32_t draw_count;
  uint32_t first_index;
  CmdDrawIndirect(Pipeline& pipeline,
                  const VertexBuffer& vertex_buffer,
                  const IndirectDrawVertexBuffer& indirect_buffer,
                  uint32_t draw_count,
                  uint32_t first_index)
      : pipeline{pipeline}
      , vertex_buffer{vertex_buffer}
      , indirect_buffer{indirect_buffer}
      , draw_count{draw_count}
      , first_index{first_index} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdDrawIndirectIndexed final : public Command<CmdDrawIndirectIndexed> {
  Pipeline& pipeline;
  const VertexBuffer& vertex_buffer;
  const IndexBuffer& index_buffer;
  const IndirectDrawIndexBuffer& indirect_buffer;
  uint32_t draw_count;
  uint32_t first_index;
  CmdDrawIndirectIndexed(Pipeline& pipeline,
                         const VertexBuffer& vertex_buffer,
                         const IndexBuffer& index_buffer,
                         const IndirectDrawIndexBuffer& indirect_buffer,
                         uint32_t draw_count,
                         uint32_t first_index)
      : pipeline{pipeline}
      , vertex_buffer{vertex_buffer}
      , index_buffer{index_buffer}
      , indirect_buffer{indirect_buffer}
      , draw_count{draw_count}
      , first_index{first_index} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdDrawIndexedSingle final : public Command<CmdDrawIndexedSingle> {
  Pipeline& pipeline;
  const VertexBuffer& vertex_buffer;
  const IndexBuffer& index_buffer;
  uint32_t index_count;
  uint32_t start_index;
  int32_t vertex_offset;
  uint32_t instance_count;
  uint32_t start_instance;
  CmdDrawIndexedSingle(Pipeline& pipeline,
                       const VertexBuffer& vertex_buffer,
                       const IndexBuffer& index_buffer,
                       uint32_t index_count,
                       uint32_t start_index,
                       int32_t vertex_offset,
                       uint32_t instance_count,
                       uint32_t start_instance)
      : pipeline{pipeline}
      , vertex_buffer{vertex_buffer}
      , index_buffer{index_buffer}
      , index_count{index_count}
      , start_index{start_index}
      , vertex_offset{vertex_offset}
      , instance_count{instance_count}
      , start_instance{start_instance} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdDrawIndexedMulti final : public Command<CmdDrawIndexedMulti> {
  Pipeline& pipeline;
  const VertexBufferRefs& vertex_buffer_refs;
  const IndexBuffer& index_buffer;
  uint32_t index_count;
  uint32_t start_index;
  uint32_t instance_count;
  uint32_t start_instance;
  CmdDrawIndexedMulti(Pipeline& pipeline,
                      const VertexBufferRefs& vertex_buffer_refs,
                      const IndexBuffer& index_buffer,
                      uint32_t index_count,
                      uint32_t start_index,
                      uint32_t instance_count,
                      uint32_t start_instance)
      : pipeline{pipeline}
      , vertex_buffer_refs{vertex_buffer_refs}
      , index_buffer{index_buffer}
      , index_count{index_count}
      , start_index{start_index}
      , instance_count{instance_count}
      , start_instance{start_instance} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdDrawMeshTasks final : public Command<CmdDrawMeshTasks> {
  Pipeline& pipeline;
  uint32_t group_count_x;
  uint32_t group_count_y;
  uint32_t group_count_z;
  CmdDrawMeshTasks(Pipeline& pipeline,
                   uint32_t group_count_x,
                   uint32_t group_count_y,
                   uint32_t group_count_z)
      : pipeline{pipeline}
      , group_count_x{group_count_x}
      , group_count_y{group_count_y}
      , group_count_z{group_count_z} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdFillBuffer final : public Command<CmdFillBuffer> {
  const Buffer& buffer;
  uint32_t data;
  uint64_t num_bytes;
  uint64_t offset;
  CmdFillBuffer(const Buffer& buffer, uint32_t data, uint64_t num_bytes, uint64_t offset)
      : buffer{buffer}, data{data}, num_bytes{num_bytes}, offset{offset} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdCopyBuffer final : public Command<CmdCopyBuffer> {
  const Buffer& src_buffer;
  const Buffer& dst_buffer;
  uint64_t num_bytes;
  uint64_t src_offset;
  uint64_t dst_offset;
  CmdCopyBuffer(const Buffer& src_buffer,
                const Buffer& dst_buffer,
                uint64_t num_bytes,
                uint64_t src_offset,
                uint64_t dst_offset)
      : src_buffer{src_buffer}
      , dst_buffer{dst_buffer}
      , num_bytes{num_bytes}
      , src_offset{src_offset}
      , dst_offset{dst_offset} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdTransitionFramebufferLayout final
    : public Command<CmdTransitionFramebufferLayout> {
  Framebuffer& framebuffer;
  ImageLayout layout;
  CmdTransitionFramebufferLayout(Framebuffer& framebuffer, ImageLayout layout)
      : framebuffer{framebuffer}, layout{layout} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdClearFramebufferAttachment final
    : public Command<CmdClearFramebufferAttachment> {
  Framebuffer& framebuffer;
  Framebuffer::Attachment attachment;
  CmdClearFramebufferAttachment(Framebuffer& framebuffer,
                                Framebuffer::Attachment attachment)
      : framebuffer{framebuffer}, attachment{attachment} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdClearTexture final : public Command<CmdClearTexture> {
  Texture& texture;
  ImageLayout final_layout;
  CmdClearTexture(Texture& texture, ImageLayout final_layout)
      : texture{texture}, final_layout{final_layout} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdClearTextureToValue final : public Command<CmdClearTextureToValue> {
  Texture& texture;
  ClearTextureValue value;
  ImageLayout final_layout;
  CmdClearTextureToValue(Texture& texture,
                         ClearTextureValue value,
                         ImageLayout final_layout)
      : texture{texture}, value{value}, final_layout{final_layout} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdImageMemoryBarrier final : public Command<CmdImageMemoryBarrier> {
  Texture& texture;
  ImageMemoryBarrierType barrier_type;
  std::optional<ImageLayout> to_layout;
  CmdImageMemoryBarrier(Texture& texture,
                        ImageMemoryBarrierType barrier_type,
                        std::optional<ImageLayout> to_layout)
      : texture{texture}, barrier_type{barrier_type}, to_layout{to_layout} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdBufferMemoryBarrier final : public Command<CmdBufferMemoryBarrier> {
  const Buffer& buffer;
  BufferMemoryBarrierType barrier_type;
  CmdBufferMemoryBarrier(const Buffer& buffer, BufferMemoryBarrierType barrier_type)
      : buffer{buffer}, barrier_type{barrier_type} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdSetPushConstantUInt32 final : public Command<CmdSetPushConstantUInt32> {
  Pipeline& pipeline;
  std::string_view name;
  ShaderStageBits shader_stages;
  uint32_t value;
  uint32_t offset;
  CmdSetPushConstantUInt32(Pipeline& pipeline,
                           std::string_view name,
                           ShaderStageBits shader_stages,
                           uint32_t value,
                           uint32_t offset)
      : pipeline{pipeline}
      , name{name}
      , shader_stages{shader_stages}
      , value{value}
      , offset{offset} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdSetPushConstants final : public Command<CmdSetPushConstants> {
  Pipeline& pipeline;
  std::string_view name;
  ShaderStageBits shader_stages;
  const void* values;
  uint32_t num_bytes;
  uint32_t offset;
  CmdSetPushConstants(Pipeline& pipeline,
                      std::string_view name,
                      ShaderStageBits shader_stages,
                      const void* values,
                      uint32_t num_bytes,
                      uint32_t offset)
      : pipeline{pipeline}
      , name{name}
      , shader_stages{shader_stages}
      , values{values}
      , num_bytes{num_bytes}
      , offset{offset} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdDispatchCompute final : public Command<CmdDispatchCompute> {
  Pipeline& pipeline;
  uint32_t specialization_id;
  uint32_t group_count_x;
  uint32_t group_count_y;
  uint32_t group_count_z;
  CmdDispatchCompute(Pipeline& pipeline,
                     uint32_t specialization_id,
                     uint32_t group_count_x,
                     uint32_t group_count_y,
                     uint32_t group_count_z)
      : pipeline{pipeline}
      , specialization_id{specialization_id}
      , group_count_x{group_count_x}
      , group_count_y{group_count_y}
      , group_count_z{group_count_z} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdBuildAccelerationStructure final
    : public Command<CmdBuildAccelerationStructure> {
  const void* build_geometry_info;
  const void* build_ranges_info;
  CmdBuildAccelerationStructure(const void* build_geometry_info,
                                const void* build_ranges_info)
      : build_geometry_info{build_geometry_info}, build_ranges_info{build_ranges_info} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdTraceRays final : public Command<CmdTraceRays> {
  RaytracingPipeline& pipeline;
  const ShaderBindingTable& sbt;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  CmdTraceRays(RaytracingPipeline& pipeline,
               const ShaderBindingTable& sbt,
               uint32_t width,
               uint32_t height,
               uint32_t depth)
      : pipeline{pipeline}, sbt{sbt}, width{width}, height{height}, depth{depth} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdInsertLabel final : public Command<CmdInsertLabel> {
  const std::string_view name;
  CmdInsertLabel(const std::string_view name) : name{name} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdPushLabel final : public Command<CmdPushLabel> {
  const std::string_view name;
  CmdPushLabel(const std::string_view name) : name{name} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdPopLabel final : public Command<CmdPopLabel> {
  void executeImpl(CommandExecutor& executor);
};

struct CmdResetQueryPool final : public Command<CmdResetQueryPool> {
  QueryPool& query_pool;
  std::optional<uint32_t> first_query;
  std::optional<uint32_t> query_count;
  CmdResetQueryPool(QueryPool& query_pool,
                    std::optional<uint32_t> first_query,
                    std::optional<uint32_t> query_count)
      : query_pool{query_pool}, first_query{first_query}, query_count{query_count} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdWriteTimestamp final : public Command<CmdWriteTimestamp> {
  QueryPool& query_pool;
  PipelineStageBits pipeline_stage;
  uint32_t query_id;
  CmdWriteTimestamp(QueryPool& query_pool,
                    PipelineStageBits pipeline_stage,
                    uint32_t query_id)
      : query_pool{query_pool}, pipeline_stage{pipeline_stage}, query_id{query_id} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdBeginQuery final : public Command<CmdBeginQuery> {
  QueryPool& query_pool;
  CmdBeginQuery(QueryPool& query_pool) : query_pool{query_pool} {}
  void executeImpl(CommandExecutor& executor);
};

struct CmdEndQuery final : public Command<CmdEndQuery> {
  QueryPool& query_pool;
  CmdEndQuery(QueryPool& query_pool) : query_pool{query_pool} {}
  void executeImpl(CommandExecutor& executor);
};

}  // namespace gfx
