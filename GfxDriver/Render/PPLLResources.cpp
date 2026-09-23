/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Render/PPLLResources.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Objects/TileBuilder.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Render/PPLLConstants.h"
#include "GfxDriver/Render/shaders/PPLL/ppllCommon.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/Texture.h"

namespace gfx {

//
// PPLLResources::LocalOrExternalTexture class
//
class PPLLResources::LocalOrExternalTexture {
 public:
  bool setExternalTexture(Texture& texture, uint32_t view_id);
  bool didExternalTextureChange();

  bool createOrResizeTexture(ResourceManager& resource_mgr,
                             uint32_t width,
                             uint32_t height);
  void destroyTexture(ResourceManager& resource_mgr);

  Texture* getTexture() const;
  uint32_t getViewId() const;

 private:
  resource_ptr<Texture> local_texture_;
  Texture* external_texture_{nullptr};
  uint32_t view_id_{0};

  // used to check if external resource changed
  ResourceHandle external_image_handle_{0};
  ResourceHandle external_image_view_handle_{0};
};

bool PPLLResources::LocalOrExternalTexture::setExternalTexture(
    Texture& external_texture,
    uint32_t external_view_id) {
  external_texture_ = &external_texture;
  bool resource_changed = didExternalTextureChange();
  if (view_id_ != external_view_id) {
    view_id_ = external_view_id;
    resource_changed = true;
  }
  return resource_changed;
}

bool PPLLResources::LocalOrExternalTexture::didExternalTextureChange() {
  bool resource_changed = false;
  if (auto new_image_handle = external_texture_->getResourceHandle();
      new_image_handle != external_image_handle_) {
    external_image_handle_ = new_image_handle;
    resource_changed = true;
  }

  if (auto new_image_view_handle = external_texture_->getViewHandle(view_id_);
      new_image_view_handle != external_image_view_handle_) {
    external_image_view_handle_ = new_image_view_handle;
    resource_changed = true;
  }
  return resource_changed;
}

Texture* PPLLResources::LocalOrExternalTexture::getTexture() const {
  if (external_texture_) {
    return external_texture_;
  } else {
    return local_texture_.get();
  }
}

uint32_t PPLLResources::LocalOrExternalTexture::getViewId() const {
  return view_id_;
}

bool PPLLResources::LocalOrExternalTexture::createOrResizeTexture(
    ResourceManager& resource_mgr,
    uint32_t width,
    uint32_t height) {
  auto* texture = getTexture();
  if (texture && width <= texture->getWidth() && height <= texture->getHeight()) {
    return didExternalTextureChange();
  } else if (!local_texture_) {
    local_texture_ = resource_mgr.createTexture(
        "Batch Fragment Counts",
        width,
        height,
        1,
        PixelFormat::kR32UI,
        1,
        false,
        ImageUsageBits::kStorageBit,
        get_default_sampler_state_for_format(PixelFormat::kR32UI));
    view_id_ = 0;
    texture = local_texture_.get();
  } else {
    texture->resize(width, height, 1);
  }
  return true;
}

void PPLLResources::LocalOrExternalTexture::destroyTexture(
    ResourceManager& resource_mgr) {
  if (local_texture_) {
    resource_mgr.destroyTexture(std::move(local_texture_));
  }
}

//
// PPLLResources class
//
PPLLResources::PPLLResources(
    const DeviceContext& device,
    PipelineResources&& stats_pipeline_resources,
    PipelineResources&& stats_tiled_pipeline_resources,
    PipelineResources&& stats_tiled_batches_stage1_pipeline_resources,
    PipelineResources&& stats_tiled_batches_stage2_pipeline_resources,
    PipelineResources&& debug_vis_pipeline_resources)
    : device_{device}
    , num_pixels_in_largest_tile_{0u}
    , counts_texture_{std::make_unique<LocalOrExternalTexture>()}
    , batch_counts_texture_{std::make_unique<LocalOrExternalTexture>()} {
  pipeline_resources_[kStats] = std::move(stats_pipeline_resources);
  pipeline_resources_[kStatsTiled] = std::move(stats_tiled_pipeline_resources);
  pipeline_resources_[kStatsTiledBatches_Stage1] =
      std::move(stats_tiled_batches_stage1_pipeline_resources);
  pipeline_resources_[kStatsTiledBatches_Stage2] =
      std::move(stats_tiled_batches_stage2_pipeline_resources);
  pipeline_resources_[kDebugVis] = std::move(debug_vis_pipeline_resources);
  auto gpu_id = device_.getGpuId();
  for (auto const& resource : pipeline_resources_) {
    CHECK(resource.material);
    CHECK(resource.pipeline);
    CHECK_EQ(resource.material->getDeviceContext().getGpuId(), gpu_id);
    CHECK_EQ(resource.pipeline->getDeviceContext().getGpuId(), gpu_id);
  }

  auto& resource_mgr = device.getResourceManager();

  // create image data uniform buffer
  image_info_buffer_ = resource_mgr.createBuffer(
      "PPLL Image Data UBO",
      {BufferType::kUnspecified, sizeof(ImageInfo), BufferUsageBits::kUniformBufferBit});

  // create tiles uniform buffer
  tile_info_buffer_ = resource_mgr.createBuffer("PPLL Tiles UBO",
                                                {gfx::BufferType::kUnspecified,
                                                 sizeof(gfx::Rect2D) * kPPLLTilesUBOSize,
                                                 BufferUsageBits::kUniformBufferBit});

  // create stats buffers
  stats_buffer_ = resource_mgr.createBuffer("PPLL Fragment Stats",
                                            {gfx::BufferType::kUnspecified,
                                             sizeof(PPLLFragmentStats) * kNumPPLLTiles,
                                             BufferUsageBits::kStorageBufferBit});

  batch_stats_buffer_ = resource_mgr.createBuffer(
      "PPLL Batch Fragment Stats",
      {gfx::BufferType::kUnspecified,
       sizeof(PPLLFragmentStats) * kPPLLStatsUBOPrimitiveBatches * kPPLLTilesUBOSize,
       BufferUsageBits::kStorageBufferBit});

  // create fragment alloc atomic buffer
  alloc_buffer_ = resource_mgr.createBuffer("PPLL Alloc",
                                            {gfx::BufferType::kUnspecified,
                                             sizeof(uint32_t),
                                             BufferUsageBits::kStorageBufferBit});

  // Set static resource descriptors
  auto const& stats_buffer = getStatsBuffer();
  auto const& image_info_buffer = getImageInfoBuffer();
  auto* material = getPipelineMaterial(PPLLResources::PipelineType::kStats);
  material->bindShaderStorageBufferToBlock("PPLL_STAT_COUNTERS_SSBO", stats_buffer);
  material->bindExternalUniformBufferToBlock("IMAGE_INFO_UBO", image_info_buffer);

  material = getPipelineMaterial(PPLLResources::PipelineType::kStatsTiled);
  material->bindShaderStorageBufferToBlock("PPLL_STAT_COUNTERS_SSBO", stats_buffer);
  material->bindExternalUniformBufferToBlock("IMAGE_TILES_UBO", getTileInfoBuffer());

  material = getPipelineMaterial(PPLLResources::PipelineType::kStatsTiledBatches_Stage1);
  material->bindShaderStorageBufferToBlock("PPLL_BATCH_STAT_COUNTERS_SSBO",
                                           getBatchStatsBuffer());
  material->bindExternalUniformBufferToBlock("IMAGE_TILES_UBO", getTileInfoBuffer());

  material = getPipelineMaterial(PPLLResources::PipelineType::kStatsTiledBatches_Stage2);
  material->bindShaderStorageBufferToBlock("PPLL_BATCH_STAT_COUNTERS_SSBO",
                                           getBatchStatsBuffer());
  material->bindExternalUniformBufferToBlock("IMAGE_TILES_UBO", getTileInfoBuffer());

  material = getPipelineMaterial(PPLLResources::PipelineType::kDebugVis);
  material->bindShaderStorageBufferToBlock("PPLL_STAT_COUNTERS_SSBO", stats_buffer);
  material->bindExternalUniformBufferToBlock("IMAGE_INFO_UBO", image_info_buffer);
}

PPLLResources::~PPLLResources() {
  destroyResources();
}

void PPLLResources::destroyResources() {
  RENDER_LOG_SCOPE();
  auto& resource_mgr = device_.getResourceManager();

  if (image_info_buffer_) {
    resource_mgr.destroyBuffer(std::move(image_info_buffer_));
  }
  if (tile_info_buffer_) {
    resource_mgr.destroyBuffer(std::move(tile_info_buffer_));
  }
  if (stats_buffer_) {
    resource_mgr.destroyBuffer(std::move(stats_buffer_));
  }
  if (batch_stats_buffer_) {
    resource_mgr.destroyBuffer(std::move(batch_stats_buffer_));
  }
  if (alloc_buffer_) {
    resource_mgr.destroyBuffer(std::move(alloc_buffer_));
  }

  destroyDynamicBuffersAndImages();
  destroyPipelineResources();
}

void PPLLResources::destroyDynamicBuffersAndImages() {
  RENDER_LOG_SCOPE();
  auto& resource_mgr = device_.getResourceManager();

  counts_texture_->destroyTexture(resource_mgr);
  batch_counts_texture_->destroyTexture(resource_mgr);

  if (list_buffer_) {
    resource_mgr.destroyBuffer(std::move(list_buffer_));
  }
  if (payload_buffer_) {
    resource_mgr.destroyBuffer(std::move(payload_buffer_));
  }
}

void PPLLResources::destroyPipelineResources() {
  RENDER_LOG_SCOPE();
  auto& resource_mgr = device_.getResourceManager();
  for (auto& resource : pipeline_resources_) {
    if (resource.pipeline) {
      resource_mgr.destroyPipeline(std::move(resource.pipeline));
    }
    resource.material = nullptr;
  }
}

void PPLLResources::prepareRenderTargets(uint32_t width, uint32_t height) {
  if (image_info_.width != width || image_info_.height != height) {
    updateImageDataUniformBuffer(width, height);
    updateTileUniformBuffer(width, height);

    auto& resource_mgr = device_.getResourceManager();
    bool texture_updated =
        counts_texture_->createOrResizeTexture(resource_mgr, width, height);
    texture_updated |=
        batch_counts_texture_->createOrResizeTexture(resource_mgr, width, height);

    if (texture_updated) {
      updateCountsTextureDependencies();
    }
  }
}

void PPLLResources::updateImageDataUniformBuffer(uint32_t width, uint32_t height) {
  CHECK(image_info_buffer_);
  image_info_.width = width;
  image_info_.height = height;
  image_info_.num_pixels = width * height;
  image_info_buffer_->updateSubData(&image_info_, sizeof(ImageInfo), 0ul);
}

void PPLLResources::setExternalCountsTexture(Texture& texture, uint32_t view_id) {
  if (counts_texture_->setExternalTexture(texture, view_id)) {
    updateCountsTextureDependencies();
  }
}

void PPLLResources::setExternalBatchCountsTexture(Texture& texture, uint32_t view_id) {
  if (batch_counts_texture_->setExternalTexture(texture, view_id)) {
    updateCountsTextureDependencies();
  }
}

void PPLLResources::updateTileUniformBuffer(uint32_t width, uint32_t height) {
  // base tile size, ignoring edge pad with odd size images
  uint32_t tile_width = width / 2;
  uint32_t tile_height = height / 2;

  // TODO: [GFX-110] Add TileBuilder support for tile count driven queue generation
  // gfx::build_tile_queue(
  //     tile_rects_, render_width, render_height, tile_width, tile_height);
  // CHECK_EQ(tile_rects_.size(), 4u);

  uint32_t edge_tile_width = tile_width + (width % 2);
  uint32_t edge_tile_height = tile_height + (height % 2);
  int32_t itile_width = static_cast<int32_t>(tile_width);
  int32_t itile_height = static_cast<int32_t>(tile_height);

  tile_rects_ = {{itile_width, itile_height, edge_tile_width, edge_tile_height},
                 {itile_width, 0, edge_tile_width, tile_height},
                 {0, itile_height, tile_width, edge_tile_height},
                 {0, 0, tile_width, tile_height}};

  // Sort tiles largest to smallest
  // The fragment buffer is not cleared between tiles, instead head buffer pixels are set
  // to 0 by the composite shader.
  // If small tiles render before large tiles, then small tile list records can
  // contaminate the head region of the larger tiles
  std::sort(
      tile_rects_.begin(),
      tile_rects_.end(),
      [](const gfx::Rect2D& a, const gfx::Rect2D& b) { return a.w * a.h > b.w * b.h; });

  // Append full image as a "tile" to use when tiling is disabled by heuristics
  tile_rects_.emplace_back(0, 0, width, height);

  // TODO (scb): Tile generation should be global
  // Update tiles shared UBO
  tile_info_buffer_->updateSubData(
      tile_rects_.data(), sizeof(gfx::Rect2D) * kPPLLTilesUBOSize, 0);

  num_pixels_in_largest_tile_ = edge_tile_width * edge_tile_height;
}

uint32_t PPLLResources::getNumPixelsInLargestTile() const {
  return num_pixels_in_largest_tile_;
}

void PPLLResources::updateCountsTextureDependencies() {
  // Update material descriptors
  auto* counts_texture = getCountsTexture();
  auto view_id = getCountsTextureViewId();

  auto* material = getPipelineMaterial(PipelineType::kStats);
  material->setImageLoadStoreAttribute("fragment_count_image", *counts_texture, view_id);
  material->updateDescriptorSets();

  material = getPipelineMaterial(PipelineType::kStatsTiled);
  material->setImageLoadStoreAttribute("fragment_count_image", *counts_texture, view_id);
  material->updateDescriptorSets();

  auto* batch_counts_texture = getBatchCountsTexture();
  if (batch_counts_texture) {
    auto* material = getPipelineMaterial(PipelineType::kStatsTiledBatches_Stage1);
    material->setImageLoadStoreAttribute("batch_fragment_count_image",
                                         *batch_counts_texture,
                                         batch_counts_texture_->getViewId());
    material->setImageLoadStoreAttribute(
        "total_fragment_count_image", *getCountsTexture(), counts_texture_->getViewId());
    material->updateDescriptorSets();
  }

  material = getPipelineMaterial(PipelineType::kStatsTiledBatches_Stage2);
  material->setImageLoadStoreAttribute(
      "total_fragment_count_image", *counts_texture, view_id);
  material->updateDescriptorSets();

  material = getPipelineMaterial(PipelineType::kDebugVis);
  material->setImageLoadStoreAttribute("fragment_count_image", *counts_texture, view_id);
  // Debug visualizer requires the PPLLRender output_image descriptor
  // so skip updateDescriptorSets

  // Notify dependents
  notifyChanged(NotifyChangedCallback::kCountsImage);
}

void PPLLResources::createOrResizeListBuffers(
    uint64_t list_buffer_size,
    uint64_t payload_buffer_size,
    std::optional<LoggingCallback> oom_logging_cb) {
  auto& resource_mgr = device_.getResourceManager();
  // Fragment list
  if (list_buffer_) {
    if (list_buffer_->getNumBytes() < list_buffer_size) {
      list_buffer_->rebuild(nullptr, list_buffer_size, oom_logging_cb);
      notifyChanged(NotifyChangedCallback::kRecordBuffer);
    }
  } else {
    list_buffer_ = resource_mgr.createBuffer(
        "PPLL Records",
        {BufferType::kUnspecified, list_buffer_size, BufferUsageBits::kStorageBufferBit},
        std::nullopt,  // @TODO(se) buffer allocator
        oom_logging_cb);
    notifyChanged(NotifyChangedCallback::kRecordBuffer);
  }

  // Fragment payload
  if (payload_buffer_size > 0u) {
    if (payload_buffer_) {
      if (payload_buffer_->getNumBytes() < payload_buffer_size) {
        payload_buffer_->rebuild(nullptr, payload_buffer_size, oom_logging_cb);
        notifyChanged(NotifyChangedCallback::kPayloadBuffer);
      }
    } else {
      payload_buffer_ =
          resource_mgr.createBuffer("PPLL Payload",
                                    {BufferType::kUnspecified,
                                     payload_buffer_size,
                                     BufferUsageBits::kStorageBufferBit},
                                    std::nullopt,  // @TODO(se) buffer allocator
                                    oom_logging_cb);
      notifyChanged(NotifyChangedCallback::kPayloadBuffer);
    }
  } else {
    // TODO: Shrink payload if not in use? Combine buffers?
  }
}

const std::vector<Rect2D>& PPLLResources::getTileRects() const {
  return tile_rects_;
}

Texture* PPLLResources::getCountsTexture() const {
  return counts_texture_->getTexture();
}

uint32_t PPLLResources::getCountsTextureViewId() const {
  return counts_texture_->getViewId();
}

Texture* PPLLResources::getBatchCountsTexture() const {
  return batch_counts_texture_->getTexture();
}

uint32_t PPLLResources::getBatchCountsTextureViewId() const {
  return batch_counts_texture_->getViewId();
}

const BufferWrapper& PPLLResources::getImageInfoBuffer() const {
  return *image_info_buffer_;
}

const BufferWrapper& PPLLResources::getTileInfoBuffer() const {
  return *tile_info_buffer_;
}

BufferWrapper& PPLLResources::getStatsBuffer() const {
  return *stats_buffer_;
}

BufferWrapper& PPLLResources::getBatchStatsBuffer() const {
  return *batch_stats_buffer_;
}

BufferWrapper& PPLLResources::getAllocBuffer() const {
  return *alloc_buffer_;
}

BufferWrapper* PPLLResources::getListBuffer() const {
  return list_buffer_.get();
}

BufferWrapper* PPLLResources::getPayloadBuffer() const {
  return payload_buffer_.get();
}

Material* PPLLResources::getPipelineMaterial(PipelineType type) const {
  return pipeline_resources_[type].material.get();
}

Pipeline* PPLLResources::getPipeline(PipelineType type) const {
  return pipeline_resources_[type].pipeline.get();
}

std::pair<Material&, Pipeline&> PPLLResources::getPipelineResources(
    PipelineType type) const {
  auto& r = pipeline_resources_[type];
  return {*r.material, *r.pipeline};
}

void PPLLResources::addNotifyChangedCallback(NotifyChangedCallback* callback) {
  notify_changed_callbacks_.insert(callback);
}

void PPLLResources::removeNotifyChangedCallback(NotifyChangedCallback* callback) {
  notify_changed_callbacks_.erase(callback);
}

void PPLLResources::notifyChanged(NotifyChangedCallback::Type type) const {
  for (auto* callback : notify_changed_callbacks_) {
    (*callback)(type);
  }
}

}  // namespace gfx
