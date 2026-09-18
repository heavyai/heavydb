/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_set>

#include <boost/noncopyable.hpp>

#include "GfxDriver/Objects/TileBuilder.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/Types.h"

namespace gfx {

/**
 * PPLLResources class
 *
 * Manages shared resources for per-pixel linked list (PPLL) rendering:
 * - Image data UBO (width, height, num pixels)
 * - Tile data UBO (per tile x, y, width, height)
 * - PPLLFragmentStats SSBO (can accomodate stats per tile)
 * - List record SSBO
 * - Optional payload SSBO
 *
 * Generates the tile vector for tile rendering (TODO: extract and generalize)
 *
 * Manages fragment counts image for accumulating counts
 *  - Can be set to an externally supplied texture with view index for aliasing
 *    existing resources
 *  - Created automatically if no external texture is provided
 *
 * Takes ownership of the Stats, TileStats, and DebugVisualization Materials and Pipelines
 *  - Must be supplied by the application (TODO: Gfx shader library)
 *  - Updates descriptors as needed
 *  - Handles resource destruction
 *
 * Descriptor update notification callbacks
 *  - Applications can register a notification callback for resource size changes
 *    This is necessary to update resource bindings in descriptor sets, and is used
 *    by PPLLRender to automatically update the count, capture, and composite
 *    Materials
 **/

class PPLLResources : private boost::noncopyable {
 public:
  enum PipelineType {
    kStats,
    kStatsTiled,
    kStatsTiledBatches_Stage1,
    kStatsTiledBatches_Stage2,
    kDebugVis,
    kCOUNT
  };

  struct PipelineResources {
    std::unique_ptr<Material> material;
    resource_ptr<ComputePipeline> pipeline;
  };

  // Callback invoked when a resource changes such that a descriptor binding
  // becomes invalid (eg resizing the counts texture)
  class NotifyChangedCallback {
   public:
    enum Type { kCountsImage, kRecordBuffer, kPayloadBuffer };
    NotifyChangedCallback(std::function<void(Type)> fn) : fn_{fn} {}
    void operator()(Type r) { fn_(r); }

   private:
    std::function<void(Type)> fn_;
  };

  explicit PPLLResources(
      const DeviceContext& device,
      PipelineResources&& stats_pipeline_resources,
      PipelineResources&& stats_tiled_pipeline_resources,
      PipelineResources&& stats_tiled_batches_stage1_pipeline_resources,
      PipelineResources&& stats_tiled_batches_stage2_pipeline_resources,
      PipelineResources&& debug_vis_pipeline_resources);
  ~PPLLResources();

  // Update ImageInfo, Tiles, and counts texture
  // If an external counts texture is not set create a local one
  void prepareRenderTargets(uint32_t width, uint32_t height);

  const std::vector<Rect2D>& getTileRects() const;
  uint32_t getNumPixelsInLargestTile() const;

  // Provide a texture to use for counts image
  // view_id enables aliasing an RGBA image
  void setExternalCountsTexture(Texture& texture, uint32_t view_id);

  void setExternalBatchCountsTexture(Texture& texture, uint32_t view_id);

  // Allocate storage for lists (payload is optional)
  void createOrResizeListBuffers(uint64_t list_buffer_size,
                                 uint64_t payload_buffer_size,
                                 std::optional<LoggingCallback> oom_logging_cb);

  // Destroy all resources (pipelines, materials, buffers, images)
  void destroyResources();

  // Destroy dynamically sized expensive resources
  // fragment and payload buffer
  void destroyDynamicBuffersAndImages();

  //
  // Global static resources
  //
  Texture* getCountsTexture() const;
  uint32_t getCountsTextureViewId() const;
  Texture* getBatchCountsTexture() const;
  uint32_t getBatchCountsTextureViewId() const;
  const BufferWrapper& getTileInfoBuffer() const;
  const BufferWrapper& getImageInfoBuffer() const;
  BufferWrapper& getStatsBuffer() const;
  BufferWrapper& getBatchStatsBuffer() const;

  // for global atomic to acquire list index during capture
  BufferWrapper& getAllocBuffer() const;

  //
  // Global dynamic resources (payload is optional)
  //
  BufferWrapper* getListBuffer() const;
  BufferWrapper* getPayloadBuffer() const;

  //
  // Material and Pipeline access
  //
  Material* getPipelineMaterial(PipelineType pipeline_type) const;
  Pipeline* getPipeline(PipelineType pipeline_type) const;
  std::pair<Material&, Pipeline&> getPipelineResources(PipelineType pipeline_type) const;

  //
  // Notification callbacks
  //
  void addNotifyChangedCallback(NotifyChangedCallback* callback);
  void removeNotifyChangedCallback(NotifyChangedCallback* callback);

 private:
  class LocalOrExternalTexture;
  const DeviceContext& device_;

  // image tiles
  std::vector<Rect2D> tile_rects_;
  uint32_t num_pixels_in_largest_tile_;

  // static resources (created in constructor)
  BufferWrapperUqPtr tile_info_buffer_;
  BufferWrapperUqPtr stats_buffer_;
  BufferWrapperUqPtr batch_stats_buffer_;
  BufferWrapperUqPtr alloc_buffer_;

  // dynamic resources (resolution / count dependent)
  std::unique_ptr<LocalOrExternalTexture> counts_texture_;
  std::unique_ptr<LocalOrExternalTexture> batch_counts_texture_;

  BufferWrapperUqPtr list_buffer_;
  BufferWrapperUqPtr payload_buffer_;

  // materials and pipelines
  std::array<PipelineResources, PipelineType::kCOUNT> pipeline_resources_;

  // Image info shared UBO
  struct ImageInfo {
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t num_pixels = 0u;
  };
  ImageInfo image_info_;
  BufferWrapperUqPtr image_info_buffer_;

  std::unordered_set<NotifyChangedCallback*> notify_changed_callbacks_;

  void destroyPipelineResources();
  void updateImageDataUniformBuffer(uint32_t width, uint32_t height);
  void updateTileUniformBuffer(uint32_t width, uint32_t height);

  void updateCountsTextureDependencies();

  void notifyChanged(NotifyChangedCallback::Type type) const;
};

}  // namespace gfx
