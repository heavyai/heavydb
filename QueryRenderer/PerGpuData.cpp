/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PerGpuData.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/Texture.h"
#include "QueryRenderer/Rendering/QueryIdMapPboPool.h"
#include "Rendering/QueryFramebuffer.h"
#include "Rendering/QueryIdMapPboPool.h"

#define PROFILE_CLEAR_QUERY_OUTPUT_BUFFER 0

#if PROFILE_CLEAR_QUERY_OUTPUT_BUFFER
#include "Shared/measure.h"
#endif

namespace QueryRenderer {

void AccumExtentsPipelines::destroyPipelines(gfx::ResourceManager& resource_mgr) {
  if (extents_pipeline) {
    resource_mgr.destroyPipeline(std::move(extents_pipeline));
  }
  if (std_dev_pipeline) {
    resource_mgr.destroyPipeline(std::move(std_dev_pipeline));
  }
}

void AccumIdPassResources::destroyResources(gfx::ResourceManager& resource_mgr) {
  material = nullptr;
  pipeline_desc = nullptr;
  framebuffer = nullptr;
  if (pipeline) {
    resource_mgr.destroyPipeline(std::move(pipeline));
  }
  if (renderpass) {
    resource_mgr.destroyRenderPass(std::move(renderpass));
  }
}

void ClearQueryOutputBufferResources::destroyResources(
    gfx::ResourceManager& resource_mgr) {
  material = nullptr;
  if (pipeline) {
    resource_mgr.destroyPipeline(std::move(pipeline));
  }
}

RootPerGpuData::RootPerGpuData(const gfx::DeviceContext& device_ctx,
                               Data_Namespace::DataMgr* data_mgr,
                               const bool renderer_enable_slab_allocation)
    : device_ctx_{device_ctx}
    , query_buffer_mgr_{
          std::make_unique<QueryBufferManager>(device_ctx_,
                                               data_mgr,
                                               renderer_enable_slab_allocation)} {}

RootPerGpuData::~RootPerGpuData() {
  // Explicitly delete PPLLResources and QueryBufferManager classes before destroying
  // device
  ppll_resources_ = nullptr;
  query_buffer_mgr_ = nullptr;
}

GpuId RootPerGpuData::getGpuId() const {
  return getDeviceContext().getGpuId();
}

const gfx::DeviceContext& RootPerGpuData::getDeviceContext() const {
  return device_ctx_;
}

gfx::ResourceManager& RootPerGpuData::getResourceManager() const {
  return device_ctx_.getResourceManager();
}

gfx::CommandList& RootPerGpuData::getCommandList() const {
  return device_ctx_.getCommandList();
}

void RootPerGpuData::prepareRenderTargets(uint32_t width, uint32_t height) {
  RENDER_LOG_SCOPE_P(getGpuId());
  CHECK(ms_framebuffer_);
  ms_framebuffer_->resize(width, height);

  if (aa_framebuffer_) {
    aa_framebuffer_->resize(width, height);
  }

  empty_framebuffer_->resize(width, height);
}

void RootPerGpuData::prepareAccumTextureArray(uint32_t width,
                                              uint32_t height,
                                              uint32_t depth) {
  RENDER_LOG_SCOPE_P(getGpuId());
  if (accum_tx_array_) {
    accum_tx_array_->resize(width, height, depth);
  } else {
    accum_tx_array_ = getResourceManager().createTexture(
        "Accum Texture Array",
        width,
        height,
        depth,
        gfx::PixelFormat::kR32UI,
        1,
        true,
        gfx::ImageUsageBits::kStorageBit | gfx::ImageUsageBits::kExternalApiBit,
        gfx::TextureSamplerState(gfx::SamplerFilterMode::kNearest,
                                 gfx::SamplerFilterMode::kNearest,
                                 gfx::SamplerWrapMode::kClampEdge,
                                 gfx::SamplerWrapMode::kClampEdge));
  }
}

void RootPerGpuData::destroyAccumTextureArray() {
  RENDER_LOG_SCOPE_P(getGpuId());
  if (accum_tx_array_) {
    getResourceManager().destroyTexture(std::move(accum_tx_array_));
  }
}

void RootPerGpuData::updateAccumIDPassResources() {
  auto& r = *accum_id_pass_resources_;
  r.framebuffer =
      getRenderFramebuffer()->getOrCreateFramebufferForRenderPass(*r.renderpass);

  // bind SS FBO ID buffers as textures
  auto const* ss_fbo = getAntiAliasingFramebuffer()->getFramebuffer();
  auto const& am = ss_fbo->getAttachmentManager();
  auto const* id1A_tx = am.getAttachmentTexture(gfx::Framebuffer::Attachment::kColor1);
  auto const* id1B_tx = am.getAttachmentTexture(gfx::Framebuffer::Attachment::kColor2);
  auto const* id2_tx = am.getAttachmentTexture(gfx::Framebuffer::Attachment::kColor3);
  CHECK(id1A_tx);
  CHECK(id1B_tx);
  CHECK(id2_tx);
  r.material->setSamplerAttribute("id1ASampler", *id1A_tx);
  r.material->setSamplerAttribute("id1BSampler", *id1B_tx);
  r.material->setSamplerAttribute("id2Sampler", *id2_tx);
  r.material->updateDescriptorSets();
}

gfx::RenderPass& RootPerGpuData::getCommonRenderPass(CommonRenderPassType type,
                                                     bool multisampled) const {
  CHECK(type != CommonRenderPassType::kCount);
  static constexpr int kCount = static_cast<int>(CommonRenderPassType::kCount);
  int i = multisampled ? static_cast<int>(type) : (static_cast<int>(type) + kCount);
  auto* render_pass = common_render_passes_[i].get();
  CHECK(render_pass);
  return *render_pass;
}

gfx::Framebuffer& RootPerGpuData::getFramebufferForCommonRenderPass(
    CommonRenderPassType type,
    bool multisampled) const {
  CHECK(type != CommonRenderPassType::kAllAttachments)
      << "Use primary FBO for kAllAttachments";
  CHECK(type != CommonRenderPassType::kAllAttachmentsClear)
      << "Use primary FBO for kAllAttachmentsClear";
  CHECK(type != CommonRenderPassType::kCount) << "Invalid type kCount";
  if (multisampled) {
    auto* rtn = ms_framebuffer_->getOrCreateFramebufferForRenderPass(
        getCommonRenderPass(type, true));
    return *rtn;
  }
  auto* rtn = aa_framebuffer_->getOrCreateFramebufferForRenderPass(
      getCommonRenderPass(type, false));
  return *rtn;
}

void RootPerGpuData::prepareCommonFramebuffers(CommonRenderPassTypeBits type_bits) {
  // Ensure common framebuffers are created for non-compatible render passes
  if (any_bits_set(type_bits & CommonRenderPassTypeBits::kDepthStencilThenAll)) {
    ms_framebuffer_->getOrCreateFramebufferForRenderPass(
        getCommonRenderPass(CommonRenderPassType::kDepthStencilThenAll, true));
    aa_framebuffer_->getOrCreateFramebufferForRenderPass(
        getCommonRenderPass(CommonRenderPassType::kDepthStencilThenAll, false));
  }
}

const gfx::RenderPass& RootPerGpuData::getEmptyRenderPass() const {
  return *empty_renderpass_;
}

std::pair<gfx::RenderPass&, gfx::Framebuffer&>
RootPerGpuData::getEmptyRenderPassAndFramebuffer() const {
  return {*empty_renderpass_, *empty_framebuffer_};
}

gfx::RenderPass& RootPerGpuData::getAccumRenderPass() const {
  return *accum_render_pass_;
}

QueryIdMapPixelBufferWkPtr RootPerGpuData::getInactiveIdMapPbo(uint32_t width,
                                                               uint32_t height) {
  CHECK(id_pbo_pool_);
  return id_pbo_pool_->getInactiveRsrc(width, height);
}

void RootPerGpuData::setIdMapPboInactive(QueryIdMapPixelBufferWkPtr& pbo) {
  CHECK(id_pbo_pool_);
  id_pbo_pool_->setRsrcInactive(pbo);
}

TmpLineBufferWkPtrs* RootPerGpuData::getTmpLineBuffersForDataTable(
    const std::string& data_table_name) {
  auto itr = tmp_line_buffer_wk_ptr_map_.find(data_table_name);
  if (itr != tmp_line_buffer_wk_ptr_map_.end()) {
    return &(itr->second);
  }
  return nullptr;
}

TmpLineBufferWkPtrs& RootPerGpuData::createTmpLineBuffersForDataTable(
    const std::string& data_table_name) {
  auto itr = tmp_line_buffer_wk_ptr_map_.find(data_table_name);
  RUNTIME_EX_ASSERT(itr == tmp_line_buffer_wk_ptr_map_.end(),
                    "Line buffers for data table \"" + data_table_name +
                        "\" already allocated. Possible failed line render.");
  tmp_line_buffer_wk_ptr_map_.insert(
      std::make_pair(data_table_name, TmpLineBufferWkPtrs()));
  return tmp_line_buffer_wk_ptr_map_[data_table_name];
}

TmpPolyBufferWkPtrs* RootPerGpuData::getTmpPolyBuffersForDataTable(
    const std::string& data_table_name) {
  auto itr = tmp_poly_buffer_wk_ptr_map_.find(data_table_name);
  if (itr != tmp_poly_buffer_wk_ptr_map_.end()) {
    return &(itr->second);
  }
  return nullptr;
}

TmpPolyBufferWkPtrs& RootPerGpuData::createTmpPolyBuffersForDataTable(
    const std::string& data_table_name) {
  // Create the temporary wk pointers and get allocations
  auto itr = tmp_poly_buffer_wk_ptr_map_.find(data_table_name);
  RUNTIME_EX_ASSERT(itr == tmp_poly_buffer_wk_ptr_map_.end(),
                    "Poly buffers for data table \"" + data_table_name +
                        "\" already allocated. Possible failed poly render.");
  tmp_poly_buffer_wk_ptr_map_.insert(
      std::make_pair(data_table_name, TmpPolyBufferWkPtrs()));
  return tmp_poly_buffer_wk_ptr_map_[data_table_name];
}

void RootPerGpuData::createPolyDrawBatchInfoForDataTable(
    const std::string& data_table_name,

    PolyDrawBatchInfoUqPtr&& poly_draw_batch_info) {
  // capture the poly_draw_batch_info data for this data table
  auto itr = tmp_poly_draw_batch_info_map_.find(data_table_name);
  RUNTIME_EX_ASSERT(itr == tmp_poly_draw_batch_info_map_.end(),
                    "PolyDrawBatchInfo for data table \"" + data_table_name +
                        "\" already allocated. Possible failed poly render.");
  // store in the temp table; this takes a copy but it will be std::move'd from here into
  // the BQPDT PerGpuData later
  tmp_poly_draw_batch_info_map_.emplace(
      std::make_pair(data_table_name, std::move(poly_draw_batch_info)));
}

bool RootPerGpuData::hasPolyDrawBatchInfoForDataTable(
    const std::string& data_table_name) {
  return tmp_poly_draw_batch_info_map_.find(data_table_name) !=
         tmp_poly_draw_batch_info_map_.end();
}

PolyDrawBatchInfoUqPtr RootPerGpuData::extractPolyDrawBatchInfoForDataTable(
    const std::string& data_table_name) {
  auto itr = tmp_poly_draw_batch_info_map_.find(data_table_name);
  RUNTIME_EX_ASSERT(
      itr != tmp_poly_draw_batch_info_map_.end(),
      "PolyDrawBatchInfo for data table \"" + data_table_name + "\" does not exist.");
  auto tmp = std::move(itr->second);
  tmp_poly_draw_batch_info_map_.erase(itr);
  return tmp;
}

TmpRasterMeshBufferWkPtrs* RootPerGpuData::getTmpRasterMeshBuffersForDataTable(
    const std::string& data_table_name) {
  auto itr = tmp_raster_mesh_buffer_wk_ptr_map_.find(data_table_name);
  if (itr != tmp_raster_mesh_buffer_wk_ptr_map_.end()) {
    return &(itr->second);
  }
  return nullptr;
}

TmpRasterMeshBufferWkPtrs& RootPerGpuData::createTmpRasterMeshBuffersForDataTable(
    const std::string& data_table_name) {
  // Create the temporary wk pointers and get allocations
  auto itr = tmp_raster_mesh_buffer_wk_ptr_map_.find(data_table_name);
  RUNTIME_EX_ASSERT(
      itr == tmp_raster_mesh_buffer_wk_ptr_map_.end(),
      "Raster mesh buffers for data table \"" + data_table_name +
          "\" already allocated. This may be due to a previously failed render");
  return tmp_raster_mesh_buffer_wk_ptr_map_
      .insert(std::make_pair(data_table_name, TmpRasterMeshBufferWkPtrs()))
      .first->second;
}

void RootPerGpuData::releaseLineAndPolyBuffers() {
  // for each temp map, inactivate the buffer resources
  // note that line indices are optional
  for (auto& entry : tmp_line_buffer_wk_ptr_map_) {
    vbo_buffer_pool_->setRsrcInactive(entry.second.verts);
    ibo_buffer_pool_->setRsrcInactive(entry.second.indices);
    ssbo_buffer_pool_->setRsrcInactive(entry.second.per_row_data);
    indvbo_buffer_pool_->setRsrcInactive(entry.second.indirect_vertex_struct);
    indibo_buffer_pool_->setRsrcInactive(entry.second.indirect_index_struct);
  }
  for (auto& entry : tmp_poly_buffer_wk_ptr_map_) {
    vbo_buffer_pool_->setRsrcInactive(entry.second.verts);
    indvbo_buffer_pool_->setRsrcInactive(entry.second.line_draw_struct);
    indvbo_buffer_pool_->setRsrcInactive(entry.second.poly_draw_struct);
    ssbo_buffer_pool_->setRsrcInactive(entry.second.per_row_data);
    ssbo_buffer_pool_->setRsrcInactive(entry.second.poly_rowids);
  }

  // empty the temp maps
  tmp_line_buffer_wk_ptr_map_.clear();
  tmp_poly_buffer_wk_ptr_map_.clear();
  tmp_poly_draw_batch_info_map_.clear();

  // NOTE: not releasing/setting inactive the raster mesh buffers because the
  // SqlQueryMeshDataTableJSON instance will take ownership and be responsible for
  // releasing back to the pool. As such the tmp map here is only used as a conduit to
  // transfer the buffers between the ExecuteRenderInterface and the
  // SqlQueryMeshDataTableJSON instance. As such, this tmp map should be empty after the
  // render. This enforces that it is empty.
  tmp_raster_mesh_buffer_wk_ptr_map_.clear();
}

void RootPerGpuData::clearResources() {
  RENDER_LOG_SCOPE_P(getGpuId());
  tmp_line_buffer_wk_ptr_map_.clear();
  tmp_poly_buffer_wk_ptr_map_.clear();
  indibo_buffer_pool_.reset();
  indvbo_buffer_pool_.reset();
  ssbo_buffer_pool_.reset();
  ibo_buffer_pool_.reset();
  vbo_buffer_pool_.reset();
  id_pbo_pool_.reset();
  aa_framebuffer_.reset();
  ms_framebuffer_.reset();
  query_result_buffer_.reset();

  auto& resource_mgr = getResourceManager();

  if (empty_framebuffer_) {
    resource_mgr.destroyFramebuffer(std::move(empty_framebuffer_));
  }
  if (empty_renderpass_) {
    resource_mgr.destroyRenderPass(std::move(empty_renderpass_));
  }

  if (accum_extents_pipelines_) {
    accum_extents_pipelines_->destroyPipelines(resource_mgr);
    accum_extents_pipelines_ = nullptr;
  }
  if (accum_tx_array_) {
    resource_mgr.destroyTexture(std::move(accum_tx_array_));
  }
  if (accum_extents_buffer_) {
    resource_mgr.destroyBuffer(std::move(accum_extents_buffer_));
  }

  if (accum_id_pass_resources_) {
    accum_id_pass_resources_->destroyResources(resource_mgr);
  }

  if (geo_count_resources_) {
    geo_count_resources_->destroyResources();
    geo_count_resources_ = nullptr;
  }

  if (ppll_resources_) {
    ppll_resources_->destroyResources();
  }

  for (auto& render_pass : common_render_passes_) {
    if (render_pass) {
      resource_mgr.destroyRenderPass(std::move(render_pass));
    }
  }
  if (accum_render_pass_) {
    resource_mgr.destroyRenderPass(std::move(accum_render_pass_));
  }

  if (clear_query_output_buffer_resources_) {
    clear_query_output_buffer_resources_->destroyResources(resource_mgr);
  }

  if (query_buffer_mgr_) {
    query_buffer_mgr_->destroyResources();
  }
}

QueryBufferManager& RootPerGpuData::getQueryBufferManager() const {
  CHECK(query_buffer_mgr_);
  return *query_buffer_mgr_;
}

void RootPerGpuData::clearQueryOutputBuffer() {
  RENDER_LOG_SCOPE();
  CHECK(query_result_buffer_);

  auto clear_region = [&](const uint64_t offset_bytes, const uint64_t num_bytes) {
#if PROFILE_CLEAR_QUERY_OUTPUT_BUFFER
    VLOG(1) << "GPU " << getGpuId() << ": Clearing QOB region, offset = " << offset_bytes
            << ", range = " << num_bytes;
#endif

    constexpr uint32_t kWorkgroupSize = 32u;
    constexpr uint32_t kWorkgroupCountLimitX = 65535u;  // vulkan spec minimum required

    // we need enough workgroups to cover the buffer size, even if not a multiple of 32
    const uint32_t num_values = num_bytes / sizeof(int64_t);
    const uint32_t num_values_rounded_up =
        (num_values + (kWorkgroupSize - 1)) & ~(kWorkgroupSize - 1);

    const uint32_t total_num_workgroups = num_values_rounded_up / kWorkgroupSize;
    uint32_t num_workgroups[2];
    if (total_num_workgroups > kWorkgroupCountLimitX) {
      num_workgroups[0] = kWorkgroupCountLimitX;
      num_workgroups[1] =
          (total_num_workgroups + num_workgroups[0] - 1) / num_workgroups[0];  // round up

      // y is usually limited at 65535 (spec min), so this assert will only trigger if
      // num_values > 137,434,759,200 (275 gb buffer)
      auto y_limit = getDeviceContext().getLimits().max_compute_workgroup_count[1];
      RUNTIME_EX_ASSERT(num_workgroups[1] <= y_limit,
                        "Unable to clear query output buffer. Y workgroup count " +
                            std::to_string(num_workgroups[1]) +
                            " exceeds device limit of " + std::to_string(y_limit));
    } else {
      num_workgroups[0] = total_num_workgroups;
      num_workgroups[1] = 1u;
    }

    auto const* query_output_buffer =
        static_cast<const gfx::VertexBuffer*>(query_result_buffer_->getBufferWrapper());
    CHECK(query_output_buffer);

    clear_query_output_buffer_resources_->material->bindShaderStorageBufferToBlock(
        "query_output_buffer", *query_output_buffer, offset_bytes, num_bytes);
    clear_query_output_buffer_resources_->material->setUniformAttribute("num_values",
                                                                        num_values);
    clear_query_output_buffer_resources_->material->updateDescriptorSets();

    getDeviceContext()
        .getCommandList()
        .dispatchCompute(*clear_query_output_buffer_resources_->pipeline,
                         0u,
                         num_workgroups[0],
                         num_workgroups[1],
                         1u)
        .flush("Clear query output buffer", gfx::CommandList::SubmitType::kWaitComplete);
  };

#if PROFILE_CLEAR_QUERY_OUTPUT_BUFFER
  getDeviceContext().getCommandExecutor().waitForCompletion(true);
  auto timer = timer_start();
#endif

  // find the largest binary power region size within the device limit (which may itself
  // not be a binary power, e.g. nVidia returns 2^32 - 1) so that offset_bytes remains a
  // multiple of the SSBO alignment
  uint64_t max_num_bytes = getDeviceContext().getLimits().shader_storage_buffer_alignment;
  const uint64_t max_ssbo_size =
      getDeviceContext().getLimits().max_shader_storage_buffer_size;
  while (max_num_bytes * 2 < max_ssbo_size) {
    max_num_bytes *= 2;
  }

  // clear in regions of that size
  const uint64_t num_bytes = query_result_buffer_->getNumBytes();
  uint64_t offset_bytes = 0ULL;
  while (offset_bytes < num_bytes) {
    const uint64_t range_bytes = std::min(num_bytes - offset_bytes, max_num_bytes);
    clear_region(offset_bytes, range_bytes);
    offset_bytes += range_bytes;
  }

#if PROFILE_CLEAR_QUERY_OUTPUT_BUFFER
  getDeviceContext().getCommandExecutor().waitForCompletion(true);
  auto const us =
      timer_stop<std::chrono::steady_clock::time_point, std::chrono::microseconds>(timer);
  LOG(INFO) << "GPU " << getGpuId() << ": Clear QOB took " << us << "us";
#endif
}

const gfx::BufferWrapper& RootPerGpuData::getSlabAddressTableBuffer() const {
  CHECK(query_buffer_mgr_);
  return query_buffer_mgr_->getSlabAddressTableBuffer();
}

gfx::BufferAllocatorShPtr RootPerGpuData::getBufferAllocator() const {
  CHECK(query_buffer_mgr_);
  return query_buffer_mgr_->getBufferAllocator();
}

GpuId BasePerGpuData::getGpuId() const {
  return root_per_gpu_data_.getGpuId();
}

const gfx::DeviceContext& BasePerGpuData::getDeviceContext() const {
  return root_per_gpu_data_.getDeviceContext();
}

gfx::ResourceManager& BasePerGpuData::getResourceManager() const {
  return root_per_gpu_data_.getResourceManager();
}

QueryIdMapPixelBufferWkPtr BasePerGpuData::getInactiveIdMapPbo(uint32_t width,
                                                               uint32_t height) {
  return root_per_gpu_data_.getInactiveIdMapPbo(width, height);
}

void BasePerGpuData::setIdMapPboInactive(QueryIdMapPixelBufferWkPtr& pbo) {
  root_per_gpu_data_.setIdMapPboInactive(pbo);
}

}  // namespace QueryRenderer
