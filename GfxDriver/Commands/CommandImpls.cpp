/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Commands/CommandImpls.h"
#include "GfxDriver/Commands/CommandExecutor.h"

namespace gfx {

void CmdBeginRenderPass::executeImpl(CommandExecutor& executor) {
  executor.beginRenderPass(render_pass, framebuffer);
}

void CmdEndRenderPass::executeImpl(CommandExecutor& executor) {
  executor.endRenderPass();
}

void CmdNextSubpass::executeImpl(CommandExecutor& executor) {
  executor.nextSubpass();
}

void CmdSetViewport::executeImpl(CommandExecutor& executor) {
  executor.setViewport(x, y, width, height);
}

void CmdSetRenderArea::executeImpl(CommandExecutor& executor) {
  executor.setRenderArea(x, y, width, height);
}

void CmdSetScissor::executeImpl(CommandExecutor& executor) {
  executor.setScissor(x, y, width, height);
}

void CmdDrawFullscreenQuad::executeImpl(CommandExecutor& executor) {
  executor.drawFullScreenQuad(pipeline);
}

void CmdDrawVerticesSingle::executeImpl(CommandExecutor& executor) {
  executor.drawVertices(pipeline,
                        vertex_buffer,
                        vertex_count,
                        offset_bytes,
                        first_vertex,
                        instance_count,
                        start_instance);
}

void CmdDrawVerticesMulti::executeImpl(CommandExecutor& executor) {
  executor.drawVertices(pipeline,
                        vertex_buffer_refs,
                        vertex_count,
                        first_vertex,
                        instance_count,
                        start_instance);
}

void CmdDrawIndirect::executeImpl(CommandExecutor& executor) {
  executor.drawIndirect(
      pipeline, vertex_buffer, indirect_buffer, draw_count, first_index);
}

void CmdDrawIndirectIndexed::executeImpl(CommandExecutor& executor) {
  executor.drawIndirectIndexed(
      pipeline, vertex_buffer, index_buffer, indirect_buffer, draw_count, first_index);
}

void CmdDrawIndexedSingle::executeImpl(CommandExecutor& executor) {
  executor.drawIndexed(pipeline,
                       vertex_buffer,
                       index_buffer,
                       index_count,
                       start_index,
                       vertex_offset,
                       instance_count,
                       start_instance);
}

void CmdDrawIndexedMulti::executeImpl(CommandExecutor& executor) {
  executor.drawIndexed(pipeline,
                       vertex_buffer_refs,
                       index_buffer,
                       index_count,
                       start_index,
                       instance_count,
                       start_instance);
}

void CmdDrawMeshTasks::executeImpl(CommandExecutor& executor) {
  executor.drawMeshTasks(pipeline, group_count_x, group_count_y, group_count_z);
}

void CmdFillBuffer::executeImpl(CommandExecutor& executor) {
  executor.fillBuffer(buffer, data, num_bytes, offset);
}

void CmdCopyBuffer::executeImpl(CommandExecutor& executor) {
  executor.copyBuffer(src_buffer, dst_buffer, num_bytes, src_offset, dst_offset);
}

void CmdTransitionFramebufferLayout::executeImpl(CommandExecutor& executor) {
  executor.transitionFramebufferLayout(framebuffer, layout);
}

void CmdClearFramebufferAttachment::executeImpl(CommandExecutor& executor) {
  executor.clearFramebufferAttachment(framebuffer, attachment);
}

void CmdClearTexture::executeImpl(CommandExecutor& executor) {
  executor.clearTexture(texture, final_layout);
}

void CmdClearTextureToValue::executeImpl(CommandExecutor& executor) {
  executor.clearTextureToValue(texture, value, final_layout);
}

void CmdImageMemoryBarrier::executeImpl(CommandExecutor& executor) {
  executor.imageMemoryBarrier(texture, barrier_type, to_layout);
}

void CmdBufferMemoryBarrier::executeImpl(CommandExecutor& executor) {
  executor.bufferMemoryBarrier(buffer, barrier_type);
}

void CmdSetPushConstantUInt32::executeImpl(CommandExecutor& executor) {
  executor.setPushConstantUInt32(pipeline, name, shader_stages, value, offset);
}

void CmdSetPushConstants::executeImpl(CommandExecutor& executor) {
  executor.setPushConstants(pipeline, name, shader_stages, values, num_bytes, offset);
}

void CmdDispatchCompute::executeImpl(CommandExecutor& executor) {
  executor.dispatchCompute(
      pipeline, specialization_id, group_count_x, group_count_y, group_count_z);
}

void CmdBuildAccelerationStructure::executeImpl(CommandExecutor& executor) {
  executor.buildAccelerationStructure(build_geometry_info, build_ranges_info);
}

void CmdTraceRays::executeImpl(CommandExecutor& executor) {
  executor.traceRays(pipeline, sbt, width, height, depth);
}

void CmdInsertLabel::executeImpl(CommandExecutor& executor) {
  executor.insertLabel(name);
}

void CmdPushLabel::executeImpl(CommandExecutor& executor) {
  executor.pushLabel(name);
}

void CmdPopLabel::executeImpl(CommandExecutor& executor) {
  executor.popLabel();
}

void CmdResetQueryPool::executeImpl(CommandExecutor& executor) {
  executor.resetQueryPool(query_pool, first_query, query_count);
}

void CmdWriteTimestamp::executeImpl(CommandExecutor& executor) {
  executor.writeTimestamp(query_pool, pipeline_stage, query_id);
}

void CmdBeginQuery::executeImpl(CommandExecutor& executor) {
  executor.beginQuery(query_pool);
}

void CmdEndQuery::executeImpl(CommandExecutor& executor) {
  executor.endQuery(query_pool);
}

}  // namespace gfx
