/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Render/PPLLRender.h"

#include <iostream>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Objects/TileBuilder.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Render/PPLLConstants.h"
#include "GfxDriver/Render/shaders/PPLL/ppllCommon.h"
#include "GfxDriver/RenderError.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/Utils/LoggingUtils.h"
#include "Logger/Logger.h"
#include "Shared/measure.h"

namespace gfx {

// Dev mode compile constants
constexpr bool kPrintPPLLStats = false;
constexpr bool kPrintTimestampStats = false;
constexpr bool kPrintCaptureAllocMaxima = false;

constexpr bool kAllowBatching = true;
constexpr bool kAllowTiling = true;
constexpr bool kAllowShallowPixelOptimization = false;

static std::vector<uint32_t> tile_indices = {0, 1, 2, 3};
static std::vector<uint32_t> image_tile_index = {kNumPPLLTiles};

// Timestamp pool needs to accomodate start and end timing pairs
// for capture and composite (4 timestamps per batch)
constexpr uint32_t kQueryPoolSize = kMaxNumPPLLPrimitiveBatches * 4;

// Stats formatter
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#endif
auto& stats_stream = std::cout;
static StreamStatFormatter print_stat(stats_stream, 30);
#ifdef __clang__
#pragma clang diagnostic pop
#endif

PPLLRender::PPLLRender(const DeviceContext& device,
                       PPLLResources& global_resources,
                       const bool enable_tiling,
                       const bool enable_batching,
                       const bool enable_2_stage_comp,
                       const uint32_t initial_comp_array_size)
    : device_{device}
    , global_resources_{global_resources}
    , enable_tiling_{kAllowTiling && enable_tiling}
    , enable_batching_{kAllowBatching && enable_batching}
    , enable_2_stage_comp_{enable_2_stage_comp}
    , initial_comp_array_size_{initial_comp_array_size}
    // limit composite shader sort (ID) array size to half shared memory size
    , comp_array_size_limit_{(device.getLimits().max_compute_shared_memory_size / 8) / 2}
    , render_width_{0u}
    , render_height_{0u}
    , output_rgb_texture_{nullptr}
    , num_primitives_{0u}
    , color_output_mode_{ColorOutputMode::kNormal}
    , using_tiles_{false}
    , using_batches_{false}
    , current_comp_specialization_size_{0u} {
  if (enable_2_stage_comp_) {
    CHECK_GT(initial_comp_array_size_, 0u);
  }
  query_pool_ = device_.getResourceManager().createTimestampQueryPool(
      "PPLLRender Timestamps", kQueryPoolSize);

  resources_changed_cb_ = std::make_unique<PPLLResources::NotifyChangedCallback>(
      [this](PPLLResources::NotifyChangedCallback::Type t) {
        using Type = PPLLResources::NotifyChangedCallback::Type;
        switch (t) {
          case Type::kCountsImage:
            is_descriptor_dirty_.setAll();
            break;
          case Type::kRecordBuffer:
          case Type::kPayloadBuffer:
            is_descriptor_dirty_[DescriptorBits::kCapture] = true;
            is_descriptor_dirty_[DescriptorBits::kComposite] = true;
            break;
        }
      });
  global_resources_.addNotifyChangedCallback(resources_changed_cb_.get());
}

PPLLRender::~PPLLRender() {
  if (query_pool_) {
    device_.getResourceManager().destroyQueryPool(std::move(query_pool_));
  }
  global_resources_.removeNotifyChangedCallback(resources_changed_cb_.get());
}

void PPLLRender::setPipelineResources(PipelineResources&& count_resources,
                                      PipelineResources&& capture_resources,
                                      PipelineResources&& composite_resources) {
  pipeline_resources_[PipelineType::kCount] = std::move(count_resources);
  pipeline_resources_[PipelineType::kCapture] = std::move(capture_resources);
  pipeline_resources_[PipelineType::kComposite] = std::move(composite_resources);

  auto gpu_id = device_.getGpuId();
  for (auto const& resource : pipeline_resources_) {
    CHECK(resource.material);
    CHECK(resource.pipeline);
    CHECK_EQ(resource.material->getDeviceContext().getGpuId(), gpu_id);
    CHECK_EQ(resource.pipeline->getDeviceContext().getGpuId(), gpu_id);
  }

  // Bind static resources
  capture_resources.material->bindShaderStorageBufferToBlock(
      "PPLL_ALLOC_SSBO", global_resources_.getAllocBuffer());

  if (!enable_tiling_) {
    auto const& image_info_buffer = global_resources_.getImageInfoBuffer();
    composite_resources.material->bindExternalUniformBufferToBlock("IMAGE_INFO_UBO",
                                                                   image_info_buffer);
  }
  is_descriptor_dirty_.setAll();
  current_comp_specialization_size_ = 0u;
}

void PPLLRender::renderBegin(const uint32_t render_width,
                             const uint32_t render_height,
                             Texture& output_rgb_texture,
                             DrawPrimitivesCB draw_callback,
                             uint32_t num_primitives,
                             ColorOutputMode color_ouput_mode) {
  VLOG(1) << "Begin PPLL render for " << num_primitives << " primitives";
  CHECK(draw_callback);
  for (auto const& resource : pipeline_resources_) {
    CHECK(resource.material);
    CHECK(resource.pipeline);
  }

  draw_callback_ = draw_callback;

  render_width_ = render_width;
  render_height_ = render_height;
  output_rgb_texture_ = &output_rgb_texture;
  num_primitives_ = num_primitives;
  color_output_mode_ = color_ouput_mode;
}

void PPLLRender::updateCountDescriptorSets() {
  // Stats material (legacy, disregards batching)
  // TODO(scb): Determine the fate of this (requires OITSandbox work)
  if (enable_tiling_) {
    global_resources_.getPipelineMaterial(PPLLResources::PipelineType::kStatsTiled)
        ->updateDescriptorSets();
  } else {
    global_resources_.getPipelineMaterial(PPLLResources::PipelineType::kStats)
        ->updateDescriptorSets();
  }

  // Stats materials (supports batching and tiling - always used by PolyMark)
  global_resources_
      .getPipelineMaterial(PPLLResources::PipelineType::kStatsTiledBatches_Stage1)
      ->updateDescriptorSets();
  global_resources_
      .getPipelineMaterial(PPLLResources::PipelineType::kStatsTiledBatches_Stage2)
      ->updateDescriptorSets();

  // Count material
  auto* material = pipeline_resources_[PipelineType::kCount].material;
  if (is_descriptor_dirty_[DescriptorBits::kCount]) {
    material->setImageLoadStoreAttribute("fragment_count_image",
                                         *global_resources_.getCountsTexture(),
                                         global_resources_.getCountsTextureViewId());
    is_descriptor_dirty_[DescriptorBits::kCount] = false;
  }
  material->updateDescriptorSets();
}

void PPLLRender::updateCaptureAndCompositeDescriptorSets() {
  auto const& stats_buffer = global_resources_.getStatsBuffer();
  auto const& image_info_buffer = global_resources_.getImageInfoBuffer();
  auto const& tile_info_buffer = global_resources_.getTileInfoBuffer();
  auto* counts_texture = global_resources_.getCountsTexture();
  auto counts_view_id = global_resources_.getCountsTextureViewId();
  auto* list_buffer = global_resources_.getListBuffer();
  auto* payload_buffer = global_resources_.getPayloadBuffer();

  // Capture material
  auto* material = pipeline_resources_[PipelineType::kCapture].material;
  if (is_descriptor_dirty_[DescriptorBits::kCapture]) {
    material->setImageLoadStoreAttribute(
        "fragment_count_image", *counts_texture, counts_view_id);
    if (enable_tiling_) {
      material->bindExternalUniformBufferToBlock("IMAGE_TILES_UBO", tile_info_buffer);
    } else {
      material->bindExternalUniformBufferToBlock("IMAGE_INFO_UBO", image_info_buffer);
    }
    material->bindShaderStorageBufferToBlock("PPLL_RECORDS_SSBO", *list_buffer);
    if (payload_buffer) {
      material->bindShaderStorageBufferToBlock("PPLL_PAYLOAD_SSBO", *payload_buffer);
    }
    is_descriptor_dirty_[DescriptorBits::kCapture] = false;
  }
  material->updateDescriptorSets();

  // Composite material
  material = pipeline_resources_[PipelineType::kComposite].material;
  if (is_descriptor_dirty_[DescriptorBits::kComposite]) {
    material->setImageLoadStoreAttribute(
        "fragment_count_image", *counts_texture, counts_view_id);
    material->bindShaderStorageBufferToBlock("PPLL_STATS_SSBO", stats_buffer);
    material->bindExternalUniformBufferToBlock("IMAGE_INFO_UBO", image_info_buffer);
    if (enable_tiling_) {
      material->bindExternalUniformBufferToBlock("IMAGE_TILES_UBO", tile_info_buffer);
    }
    material->bindShaderStorageBufferToBlock("PPLL_RECORDS_SSBO", *list_buffer);
    if (payload_buffer) {
      material->bindShaderStorageBufferToBlock("PPLL_PAYLOAD_SSBO", *payload_buffer);
    }
    is_descriptor_dirty_[DescriptorBits::kComposite] = false;
  }
  material->updateDescriptorSets();
}

void PPLLRender::drawStatsVisualization() {
  auto* material =
      global_resources_.getPipelineMaterial(PPLLResources::PipelineType::kDebugVis);
  auto* pipeline = global_resources_.getPipeline(PPLLResources::PipelineType::kDebugVis);

  material->setUniformAttribute(
      "visualize_mode",
      static_cast<int>(color_output_mode_) -
          static_cast<int>(PPLLRender::ColorOutputMode::kBeginStatsVis));
  material->setImageLoadStoreAttribute("output_image", *output_rgb_texture_);

  material->updateDescriptorSets();

  if (enable_tiling_) {
    // Update stats buffer with aggregated tile stats for normalized count display
    global_resources_.getStatsBuffer().updateSubData(
        &fragment_stats_, sizeof(PPLLFragmentStats), 0u);
  }

  auto subgroup_size = device_.getLimits().subgroup_size;

  device_.getCommandList()
      .pushLabel("Debug Viz")
      .imageMemoryBarrier(*output_rgb_texture_,
                          ImageMemoryBarrierType::kImageLayout,
                          ImageLayout::kGeneral)
      .dispatchCompute(*pipeline,
                       0u,
                       (render_width_ + subgroup_size - 1) / subgroup_size,
                       render_height_,
                       1u)
      .imageMemoryBarrier(*output_rgb_texture_,
                          ImageMemoryBarrierType::kImageLayout,
                          ImageLayout::kAttachment)
      .popLabel()
      .flush("Debug Vis", CommandList::SubmitType::kWaitComplete);
}

bool PPLLRender::countFragmentsAndComputeStats() {
  VLOG(1) << "Count fragments and compute stats";
  updateCountDescriptorSets();

  // clear the per-pixel fragment counts image
  // TODO: more granular layout / barrier control. clearTexture can be heavy and
  // doesn't _guarantee_ a safe and efficient barrier setup
  auto& cmd_list = device_.getCommandList();
  auto& counts_texture = *global_resources_.getCountsTexture();

  cmd_list.clearTexture(counts_texture, ImageLayout::kGeneral)
      .flush("Clear counts", CommandList::SubmitType::kWaitComplete);

  draw_callback_(cmd_list,
                 "Count fragments",
                 *pipeline_resources_[PipelineType::kCount].pipeline,
                 num_primitives_,
                 0u);

#if 1
  cmd_list.flush("Draw primitives (counting pass)",
                 CommandList::SubmitType::kWaitComplete);
#else
  cmd_list.imageMemoryBarrier(counts_texture,
                              gfx::ImageMemoryBarrierType::kFragmentShaderToCompute);
  if constexpr (kPrintPPLLStats) {
    // wait for draw to complete before grabbing start time
    cmd_list.flush("Stats flush");
  }
#endif

  auto subgroup_size = device_.getLimits().subgroup_size;
  auto stats_clock_start = timer_start();
  auto const& stats_buffer = global_resources_.getStatsBuffer().getBuffer();

  if (enable_tiling_) {
    //
    // Get fragment count stats per tile
    //
    auto& tile_rects = global_resources_.getTileRects();
    auto& tile_stats_pipeline =
        *global_resources_.getPipeline(PPLLResources::PipelineType::kStatsTiled);

    cmd_list.fillBuffer(stats_buffer, 0u, sizeof(PPLLFragmentStats) * kNumPPLLTiles);

    cmd_list.pushLabel("tile stats");
    for (auto tile_index : tile_indices) {
      cmd_list
          .setPushConstantUInt32(tile_stats_pipeline,
                                 "tileIndex",
                                 gfx::ShaderStageBits::kCompute,
                                 tile_index)
          .dispatchCompute(tile_stats_pipeline,
                           0u,
                           (tile_rects[tile_index].w + subgroup_size - 1) / subgroup_size,
                           tile_rects[tile_index].h,
                           1u);
    }
    cmd_list.popLabel().flush("Stats compute",
                              gfx::CommandList::SubmitType::kWaitComplete);

    //
    // Read tile stats and aggregate into full image stats
    //
    std::vector<PPLLFragmentStats> tile_stats(kNumPPLLTiles);
    global_resources_.getStatsBuffer().getData(tile_stats.data(),
                                               sizeof(PPLLFragmentStats) * kNumPPLLTiles);

    fragment_stats_.reset();
    max_fragments_stats_.reset();
    for (auto const& stats : tile_stats) {
      max_fragments_stats_.per_tile_all =
          std::max(max_fragments_stats_.per_tile_all, stats.total_fragment_count);
      max_fragments_stats_.per_tile_deep =
          std::max(max_fragments_stats_.per_tile_deep, stats.deep_pixel_fragment_count);
      fragment_stats_.total_fragment_count += stats.total_fragment_count;
      fragment_stats_.deep_pixel_fragment_count += stats.deep_pixel_fragment_count;
      fragment_stats_.max_per_pixel_fragment_count =
          std::max(fragment_stats_.max_per_pixel_fragment_count,
                   stats.max_per_pixel_fragment_count);
    }
  } else {
    auto& stats_pipeline =
        *global_resources_.getPipeline(PPLLResources::PipelineType::kStats);
    cmd_list.fillBuffer(stats_buffer, 0u, sizeof(PPLLFragmentStats))
        .pushLabel("stats")
        .dispatchCompute(stats_pipeline,
                         0u,
                         (render_width_ + subgroup_size - 1) / subgroup_size,
                         render_height_,
                         1u)
        .popLabel();
    cmd_list.flush("Stats compute", gfx::CommandList::SubmitType::kWaitComplete);

    global_resources_.getStatsBuffer().getData(&fragment_stats_,
                                               sizeof(PPLLFragmentStats));
  }

  uint64_t stats_time = 0;
  if constexpr (kPrintPPLLStats) {
    stats_time = timer_stop_microseconds(stats_clock_start);
  }

  if constexpr (kPrintPPLLStats) {
    stats_stream << "num primitives: " << num_primitives_ << std::endl;
    stats_stream << "\nFragment stats:\n";
    print_stat("  compute and cpu time", stats_time, " us");
  }

  // Check for stats count visualization before 0 fragments to
  // ensure the image is cleared
  if ((color_output_mode_ >= ColorOutputMode::kBeginStatsVis) &&
      (color_output_mode_ < ColorOutputMode::kBeginCompVis)) {
    drawStatsVisualization();
    return false;
  }

  // Check for 0 fragments
  if (fragment_stats_.total_fragment_count == 0u) {
    return false;
  }

  return true;
}

bool PPLLRender::countFragmentsAndComputeStats(
    const std::vector<uint32_t>& num_primitives_per_batch) {
  VLOG(1) << "Count fragments and compute stats";
  auto& batch_counts_texture = *global_resources_.getBatchCountsTexture();
  auto batch_counts_view_id = global_resources_.getBatchCountsTextureViewId();

  // Use the per batch count texture for count accumulation
  auto* count_material = pipeline_resources_[PipelineType::kCount].material;
  count_material->setImageLoadStoreAttribute(
      "fragment_count_image", batch_counts_texture, batch_counts_view_id);
  count_material->updateDescriptorSets();
  is_descriptor_dirty_[DescriptorBits::kCount] = true;

  auto const& [stats_material, stats_pipeline] = global_resources_.getPipelineResources(
      PPLLResources::PipelineType::kStatsTiledBatches_Stage1);
  stats_material.updateDescriptorSets();

  auto subgroup_size = device_.getLimits().subgroup_size;
  auto num_batches = num_primitives_per_batch.size();
  auto& tile_rects = global_resources_.getTileRects();
  auto& batch_stats_buffer = global_resources_.getBatchStatsBuffer();

  // We're going to submit all draw and compute commands in one command buffer.
  // Since push contants are not copied into the CommandList memory arena we need to
  // store them in a local vector so they persist until the CommandList is flushed
  struct PushConstants {
    uint32_t tile_index;
    uint32_t batch_index;
  };
  std::vector<PushConstants> push_constants(num_batches * kNumPPLLTiles);

  //
  // Record commands
  //
  auto& cmd_list = device_.getCommandList();
  auto stats_clock_start = timer_start();
  uint32_t start_primitive = 0;

  // Timestamps
  enum Timestamps { kDrawStart, kDrawEnd, kComputeStart, kComputeEnd, kNumTimestamps };

  // Reset the timestamps in the query pool
  cmd_list.resetQueryPool(*query_pool_, 0, kNumTimestamps);

  cmd_list.pushLabel("Frags per batch");
  cmd_list.writeTimestamp(*query_pool_, PipelineStageBits::kTopOfPipeBit, kDrawStart);

  // Clear the (unbatched) per-pixel fragment counts image. The memory barrier is
  // necessary as clearTexture only barriers if there is a layout transition
  auto& counts_texture = *global_resources_.getCountsTexture();
  cmd_list.clearTexture(counts_texture, ImageLayout::kGeneral)
      .imageMemoryBarrier(counts_texture, ImageMemoryBarrierType::kTransferToCompute);

  // Clear the stats buffer used to accumulate stats per tile per batch
  cmd_list.fillBuffer(
      batch_stats_buffer.getBuffer(), 0, batch_stats_buffer.getNumBytes());

  // Loop over the batches
  //  - clear batch counts texture
  //  - draw polys in batch accumulating counts in batch counts texture
  //  - loop over tiles dispatching stats compute shader per tile
  //  - stats compute also accumulates batch count into total fragment count texture
  //
  // After batch loop, run stage 2 stats shader on total fragment count texture
  // to determine deep pixel count and max fragments per pixel which cannot be determined
  // from the per batch counts

  for (size_t batch_index = 0; batch_index < num_batches; ++batch_index) {
    auto num_primitives_in_batch = num_primitives_per_batch[batch_index];
    if (batch_index > 0) {
      // Ensure previous batch texture reads are complete before clearing
      cmd_list.imageMemoryBarrier(batch_counts_texture,
                                  ImageMemoryBarrierType::kShaderReadToTransfer);
    }
    // Clear batch counts texture and barrier
    cmd_list.clearTexture(batch_counts_texture, ImageLayout::kGeneral)
        .imageMemoryBarrier(batch_counts_texture,
                            ImageMemoryBarrierType::kTransferToFragmentShader,
                            ImageLayout::kGeneral);

    // Draw primitives and accumulate counts
    draw_callback_(cmd_list,
                   "Count fragments",
                   *pipeline_resources_[PipelineType::kCount].pipeline,
                   num_primitives_in_batch,
                   start_primitive);

    // Ensure fragment shader operations are complete
    cmd_list.imageMemoryBarrier(batch_counts_texture,
                                ImageMemoryBarrierType::kFragmentShaderToCompute);

    // Ensure previous compute writes are done (pretty much guaranteed, but better
    // to be safe!)
    if (batch_index > 0) {
      cmd_list.bufferMemoryBarrier(batch_stats_buffer.getBuffer(),
                                   BufferMemoryBarrierType::kComputeWriteToComputeRead);
    }

    // Run stats shader on each tile to accumulate PPLLFragmentStats
    // This will also accumulate the per-pixel fragment counts into the main fragment
    // count texture
    for (auto tile_index : tile_indices) {
      uint32_t push_constant_index = batch_index * kNumPPLLTiles + tile_index;
      push_constants[push_constant_index].tile_index = tile_index;
      push_constants[push_constant_index].batch_index = batch_index;
      cmd_list
          .setPushConstants(stats_pipeline,
                            "pushConstants",
                            ShaderStageBits::kCompute,
                            &push_constants[push_constant_index],
                            sizeof(PushConstants))
          .dispatchCompute(stats_pipeline,
                           0u,
                           (tile_rects[tile_index].w + subgroup_size - 1) / subgroup_size,
                           tile_rects[tile_index].h,
                           1u);
    }
    start_primitive += num_primitives_in_batch;
  }

  // Now that the unbatched per-pixel fragment count texture is complete we can get the
  // deep pixel fragment count and max fragments per pixel for each tile and the full
  // image
  auto& unbatched_stats_pipeline = *global_resources_.getPipeline(
      PPLLResources::PipelineType::kStatsTiledBatches_Stage2);
  cmd_list.writeTimestamp(*query_pool_, PipelineStageBits::kBottomOfPipeBit, kDrawEnd);
  cmd_list.imageMemoryBarrier(counts_texture,
                              ImageMemoryBarrierType::kStorageImageWriteRead);
  cmd_list.writeTimestamp(*query_pool_, PipelineStageBits::kTopOfPipeBit, kComputeStart);
  for (auto tile_index : tile_indices) {
    cmd_list
        .setPushConstantUInt32(unbatched_stats_pipeline,
                               "pushConstants",
                               ShaderStageBits::kCompute,
                               tile_index)
        .dispatchCompute(*global_resources_.getPipeline(
                             PPLLResources::PipelineType::kStatsTiledBatches_Stage2),
                         0u,
                         (tile_rects[tile_index].w + subgroup_size - 1) / subgroup_size,
                         tile_rects[tile_index].h,
                         1u);
  }

  //
  // Submit commands and wait
  //
  cmd_list.writeTimestamp(
      *query_pool_, PipelineStageBits::kComputeShaderBit, kComputeEnd);
  cmd_list.popLabel();
  cmd_list.flush("Stats compute", gfx::CommandList::SubmitType::kWaitComplete);

  //
  // Read batched and tiled stats and aggregate tile stats into full image stats
  // per batch. We need to read back the entire stats buffer, not just the batches in use,
  // to ensure we get the extra batch at the end where the unbatched deeps pixel and max
  // fragments per pixel stats were accumulated by the stage 2 compute shader
  //
  std::vector<PPLLFragmentStats> read_stats(kPPLLTilesUBOSize *
                                            kPPLLStatsUBOPrimitiveBatches);
  batch_stats_buffer.getData(read_stats.data(),
                             sizeof(PPLLFragmentStats) * read_stats.size());

  // Reset aggregate structs
  fragment_stats_.reset();
  max_fragments_stats_.reset();
  std::vector<uint64_t> total_fragments_per_tile(kNumPPLLTiles, 0);
  for (uint32_t batch_index = 0; batch_index < num_batches; ++batch_index) {
    uint64_t batch_sum{0};  // total fragments for this batch (all tiles)
    for (uint32_t tile_index = 0; tile_index < kNumPPLLTiles; ++tile_index) {
      auto const& stats = read_stats[(batch_index * kPPLLTilesUBOSize) + tile_index];
      // Add batch count to total for this batch
      total_fragments_per_tile[tile_index] += stats.total_fragment_count;

      // Add tile fragment count to batch total
      batch_sum += stats.total_fragment_count;

      // Find the per tile maximum for any tile in any batch - governs buffer
      // sizing when both batches and tiles are in use
      max_fragments_stats_.per_batch_tiled =
          std::max(stats.total_fragment_count, max_fragments_stats_.per_batch_tiled);

      // Add to global sum (no batches or tiles)
      fragment_stats_.total_fragment_count += stats.total_fragment_count;

      // Find the max fragments per pixel in any batch, which is used to guide sizing
      // of the unique_id array in the overflow pass of the fragment compositor
      max_fragments_stats_.per_pixel_batched = std::max(
          max_fragments_stats_.per_pixel_batched, stats.max_per_pixel_fragment_count);
    }
    // Store the maximum fragments in any batch (no tiles), governs buffer sizing when
    // batches are in use, but not tiles
    max_fragments_stats_.per_batch = std::max(max_fragments_stats_.per_batch, batch_sum);
  }

  // Get the stats vector offset to the extra batch used to hold unbatched per tile stats
  uint32_t per_tile_stats_offset = kMaxNumPPLLPrimitiveBatches * kPPLLTilesUBOSize;

  // Get the max fragments per pixel for the image (no tiles or batches)
  // TODO(scb): This formerly controlled if batches would be used, since they were
  // originally just used to limit the unique_id array size in the compute shader. That is
  // no longer part of the heuristic so this stat can potentially be dropped
  fragment_stats_.max_per_pixel_fragment_count =
      read_stats[per_tile_stats_offset + kNumPPLLTiles].max_per_pixel_fragment_count;

  // Accumulate needed per tile stats in the event batching is not used, but tiling is
  for (uint32_t tile_index = 0; tile_index < kNumPPLLTiles; ++tile_index) {
    max_fragments_stats_.per_tile_all =
        std::max(max_fragments_stats_.per_tile_all, total_fragments_per_tile[tile_index]);
    uint64_t deep_fragment_count =
        read_stats[per_tile_stats_offset + tile_index].deep_pixel_fragment_count;
    max_fragments_stats_.per_tile_deep =
        std::max(max_fragments_stats_.per_tile_deep, deep_fragment_count);
    fragment_stats_.deep_pixel_fragment_count += deep_fragment_count;
  }

  auto total_time = timer_stop_microseconds(stats_clock_start);
  VLOG(1) << "Max batch fragment count time: " << total_time;
  if constexpr (kPrintPPLLStats || kPrintTimestampStats) {
    print_stat << "\nGenerate fragment stats (batched)" << std::endl;
  }
  if constexpr (kPrintPPLLStats) {
    print_stat("  Num batches", num_primitives_per_batch.size());
  }
  // Read timestamps
  if constexpr (kPrintTimestampStats) {
    auto timestamps = query_pool_->getTimestampResults(0, kNumTimestamps);
    print_stat("  Total time (CPU)", total_time, " us");
    print_stat("    Count fragments (GPU)",
               query_pool_->timestampToMicroseconds(timestamps[kDrawEnd] -
                                                    timestamps[kDrawStart]),
               " us");
    print_stat("    Stats compute (GPU)",
               query_pool_->timestampToMicroseconds(timestamps[kComputeEnd] -
                                                    timestamps[kComputeStart]),
               " us");
  }

  // Check for stats count visualization before 0 fragments to
  // ensure the image is cleared
  if ((color_output_mode_ >= ColorOutputMode::kBeginStatsVis) &&
      (color_output_mode_ < ColorOutputMode::kBeginCompVis)) {
    drawStatsVisualization();
    return false;
  }

  // Check for 0 fragments
  if (fragment_stats_.total_fragment_count == 0u) {
    return false;
  }

  return true;
}

void PPLLRender::prepareStorage(size_t payload_size,
                                const std::vector<uint32_t>& num_primitives_per_batch) {
  VLOG(1) << "Prepare storage";

  // Determine if we're batching
  using_batches_ = enable_batching_ && num_primitives_per_batch.size() > 1;

  // Check if we can use the shallow/deep pixel optimization
  // TODO(scb): various_xform_test_8 and _9 fail if this is enabled with ~40 different
  // pixels (clearly missing fragments), so kAllowShallowPixelOptimization is currently
  // disabled in production
  bool using_shallow_pixel_optimization =
      kAllowShallowPixelOptimization && !payload_size && !using_batches_;

  //
  // Compute fragment buffer sizes (full render vs tiled)
  //
  uint64_t image_fragment_count_to_use, tile_fragment_count_to_use;
  if (using_shallow_pixel_optimization) {
    image_fragment_count_to_use = fragment_stats_.deep_pixel_fragment_count;
    tile_fragment_count_to_use = max_fragments_stats_.per_tile_deep;
  } else if (using_batches_) {
    image_fragment_count_to_use = max_fragments_stats_.per_batch;
    tile_fragment_count_to_use = max_fragments_stats_.per_batch_tiled;
  } else {
    image_fragment_count_to_use = fragment_stats_.total_fragment_count;
    tile_fragment_count_to_use = max_fragments_stats_.per_tile_all;
  }

  uint32_t num_render_pixels = render_width_ * render_height_;
  auto num_big_tile_pixels = global_resources_.getNumPixelsInLargestTile();

  uint64_t full_list_buffer_size =
      (num_render_pixels + image_fragment_count_to_use) * sizeof(uint64_t);
  uint64_t tiled_list_buffer_size =
      (num_big_tile_pixels + tile_fragment_count_to_use) * sizeof(uint64_t);

  uint64_t full_payload_buffer_size = image_fragment_count_to_use * payload_size;
  uint64_t tiled_payload_buffer_size = tile_fragment_count_to_use * payload_size;

  float fragment_to_primitive_ratio = static_cast<float>(image_fragment_count_to_use) /
                                      static_cast<float>(num_primitives_);

  //
  // Apply batching and tiling heuristics
  //
  // Check for tiling using poly count to fragment count ratio and memory size
  using_tiles_ =
      enable_tiling_ && (fragment_to_primitive_ratio > kMaxPPLLFragmentCountToPolyRatio ||
                         full_list_buffer_size > kMaxPPLLFullFragmentBufferSize);

  // Assign list and payload buffer sizes and tile indices to use
  uint64_t list_buffer_size, payload_buffer_size;
  if (using_tiles_) {
    list_buffer_size = tiled_list_buffer_size;
    payload_buffer_size = tiled_payload_buffer_size;
  } else {
    list_buffer_size = full_list_buffer_size;
    payload_buffer_size = full_payload_buffer_size;
  }

  auto log_stats = [&, this](StreamStatFormatter& log_stat) {
    log_stat("  Total fragments", fragment_stats_.total_fragment_count);
    log_stat("    Max per tile", max_fragments_stats_.per_tile_all);
    log_stat("    Max per batch", max_fragments_stats_.per_batch);
    log_stat("    Max per batch (tiled)", max_fragments_stats_.per_batch_tiled);
    log_stat("    Max per pixel", fragment_stats_.max_per_pixel_fragment_count);
    log_stat("    Max per pixel (batch)", max_fragments_stats_.per_pixel_batched);
    log_stat("  Deep-pixel fragments", fragment_stats_.deep_pixel_fragment_count);
    log_stat("    Max per tile", max_fragments_stats_.per_tile_deep);
    log_stat("  Shallow pixel optimization", using_shallow_pixel_optimization);
    log_stat("  Fragment to primitive ratio", fragment_to_primitive_ratio);
    log_stat("  Using tiles", using_tiles_);
    log_stat("  Using batches", using_batches_);
    log_stat("  Using fragment count",
             using_tiles_ ? tile_fragment_count_to_use : image_fragment_count_to_use);
    log_stat << "Fragment storage (bytes)\n";
    if (using_tiles_) {
      log_stat.memory_stat("  List heads", num_big_tile_pixels * sizeof(uint64_t));
      log_stat.memory_stat("  List records",
                           tile_fragment_count_to_use * sizeof(uint64_t));
      if (payload_size) {
        log_stat.memory_stat("  Payloads", tiled_payload_buffer_size);
      }
      if (using_batches_) {
        log_stat.memory_stat("  Total (batched)", list_buffer_size);
      } else {
        log_stat.memory_stat("  Total (no batches)", list_buffer_size);
      }
    } else {
      log_stat.memory_stat("  List heads", num_render_pixels * sizeof(uint64_t));
      log_stat.memory_stat("  List records",
                           image_fragment_count_to_use * sizeof(uint64_t));
      if (payload_size) {
        log_stat.memory_stat("  Payloads", full_payload_buffer_size);
      }
      log_stat.memory_stat("  Total", full_list_buffer_size + full_payload_buffer_size);
    }
  };

  auto log_stats_on_failure = [&] {
    std::stringstream ss;
    StreamStatFormatter string_logger(ss);
    log_stats(string_logger);
    LOG(ERROR) << "PPLL Render Failure: Stats as follows:\n" << ss.str();
  };

  if constexpr (kPrintPPLLStats) {
    log_stats(print_stat);
  }

  //
  // Sanity check buffer sizes (hard errors, no retries)
  //
  auto max_ssbo_size = device_.getLimits().max_shader_storage_buffer_size;
  if (list_buffer_size > max_ssbo_size) {
    log_stats_on_failure();
    THROW_RUNTIME_EX("PPLL Render Failure: List Buffer size (" +
                     std::to_string(list_buffer_size) + ") exceeds GPU driver limit (" +
                     std::to_string(max_ssbo_size) + ")");
  }
  if (payload_buffer_size > max_ssbo_size) {
    log_stats_on_failure();
    THROW_RUNTIME_EX("PPLL Render Failure: Payload Buffer size (" +
                     std::to_string(payload_buffer_size) +
                     ") exceeds GPU driver limit (" + std::to_string(max_ssbo_size) +
                     ")");
  }

  // Check against available memory
  // If insufficient, throw an OutOfGpuMemory error to trigger a renderer clear_gpu
  auto mem_budget = device_.getMemoryBudget();
  if (list_buffer_size > mem_budget.available) {
    log_stats_on_failure();
    throw gfx::OutOfGpuMemoryError("PPLL Render Failure: List Buffer size (" +
                                   std::to_string(list_buffer_size) +
                                   ") exceeds currently available GPU memory (" +
                                   std::to_string(mem_budget.available) + ")");
  }
  if (payload_buffer_size > mem_budget.available) {
    log_stats_on_failure();
    throw gfx::OutOfGpuMemoryError("PPLL Render Failure: Payload Buffer size (" +
                                   std::to_string(payload_buffer_size) +
                                   ") exceeds currently available GPU memory (" +
                                   std::to_string(mem_budget.available) + ")");
  }
  if (list_buffer_size + payload_buffer_size > mem_budget.available) {
    log_stats_on_failure();
    throw gfx::OutOfGpuMemoryError(
        "PPLL Render Failure: Combined PPLL List Buffer size (" +
        std::to_string(list_buffer_size) + ") and Payload Buffer size (" +
        std::to_string(payload_buffer_size) +
        ") exceeds currently available GPU memory (" +
        std::to_string(mem_budget.available) + ")");
  }

  //
  // Allocate / resize the list record buffer and optionally the payload buffer
  //
  std::optional<gfx::LoggingCallback> oom_logging_cb = std::nullopt;
  if (VLOGGING(1)) {  // Always log stats if verbose logging enabled
    std::stringstream ss;
    StreamStatFormatter string_logger(ss);
    log_stats(string_logger);
    VLOG(1) << "PPLL stats\n" << ss.str();
  } else {  // Log stats on OOM
    oom_logging_cb = [&](std::ostream& os) {
      os << "PPLL stats\n\n";
      StreamStatFormatter oom_logger(os);
      log_stats(oom_logger);
    };
  }
  global_resources_.createOrResizeListBuffers(
      list_buffer_size, payload_buffer_size, oom_logging_cb);
}

//
// Capture and Composite
//
void PPLLRender::captureAndComposite(
    const std::vector<uint32_t>& num_primitives_per_batch) {
  VLOG(1) << "Capture and composite fragments";
  if (using_batches_) {
    VLOG(1) << "  num batches: " << num_primitives_per_batch.size();
  }

  updateCaptureAndCompositeDescriptorSets();

  auto& cmd_list = device_.getCommandList();
  auto* list_buffer = global_resources_.getListBuffer();
  auto* payload_buffer = global_resources_.getPayloadBuffer();

  //
  // Update composite fragments material
  //
  pipeline_resources_[PipelineType::kComposite].material->setUniformAttribute(
      "colorOutputMode", static_cast<int>(color_output_mode_));

  //
  // Clear fragment buffer heads
  //
  cmd_list.pushLabel("Clear list heads")
      .fillBuffer(list_buffer->getBuffer(),
                  0u,
                  using_tiles_
                      ? global_resources_.getNumPixelsInLargestTile() * sizeof(uint64_t)
                      : render_width_ * render_height_ * sizeof(uint64_t))
      .bufferMemoryBarrier(list_buffer->getBuffer(),
                           gfx::BufferMemoryBarrierType::kTransferToCompute)
      .popLabel();

  //
  // Capture and composite a batch of draw calls
  //
  PPLLResolveStats resolve_stats;
  static_assert(sizeof(resolve_stats) <= sizeof(PPLLFragmentStats));

  //
  // Batch lambda
  //
  // Clear stats and alloc buffers
  // Set batch offset push constant for DrawID offset
  // Set tile index for capture and comp shaders via push constant
  // Draw primitives capturing fragments
  // Composite fragments with default sort array size
  // if 2-stage composite enabled:
  //   Read back composite stats and check for sort array size overflow
  //   If overflow occurred, run a specialized composite with max fragments unique IDs
  uint32_t max_sort_array_items = 0u;
  uint32_t max_complex_pixel_count = 0u;
  auto& alloc_buffer = global_resources_.getAllocBuffer();
  auto const& stats_buffer = global_resources_.getStatsBuffer();
  auto& counts_texture = *global_resources_.getCountsTexture();
  auto& capture_pipeline = *pipeline_resources_[PipelineType::kCapture].pipeline;
  auto& composite_pipeline = *pipeline_resources_[PipelineType::kComposite].pipeline;
  auto subgroup_size = device_.getLimits().subgroup_size;

  // Capture and composite timestamps per batch
  enum Timestamps {
    kCaptureStart,
    kCaptureEnd,
    kCompositeStart,
    kCompositeEnd,
    kNumTimestampsPerBatch
  };

  auto process_batch = [&](DrawPrimitivesCB draw_callback,
                           uint32_t batch_index,
                           uint32_t batch_size,
                           uint32_t batch_offset,
                           const gfx::Rect2D& tile,
                           uint32_t tile_index) {
    CHECK_NE(batch_size, 0u);
    auto base_timestamp = batch_index * kNumTimestampsPerBatch;
    cmd_list.writeTimestamp(
        *query_pool_, PipelineStageBits::kTopOfPipeBit, base_timestamp);

    // Clear buffers alloc and stats buffers
    cmd_list.fillBuffer(alloc_buffer.getBuffer(), 0u, sizeof(uint32_t))
        .fillBuffer(stats_buffer.getBuffer(), 0u, sizeof(PPLLResolveStats))
        .bufferMemoryBarrier(alloc_buffer.getBuffer(),
                             gfx::BufferMemoryBarrierType::kTransferToFragmentShader)
        .bufferMemoryBarrier(stats_buffer.getBuffer(),
                             gfx::BufferMemoryBarrierType::kTransferToFragmentShader);

    // Set push constants for this command buffer
    if (enable_batching_) {
      cmd_list.setPushConstantUInt32(capture_pipeline,
                                     "batchOffset",
                                     gfx::ShaderStageBits::kVertex,
                                     batch_offset,
                                     0u);
    }
    if (enable_tiling_) {
      cmd_list.setPushConstantUInt32(
          capture_pipeline,
          "tileIndex",
          gfx::ShaderStageBits::kFragment | gfx::ShaderStageBits::kCompute,
          tile_index,
          kTileIndexPushConstantOffset);
    }

    // Draw polygons and capture fragments
    draw_callback(
        cmd_list, "Capture fragments", capture_pipeline, batch_size, batch_offset);

    cmd_list.bufferMemoryBarrier(list_buffer->getBuffer(),
                                 gfx::BufferMemoryBarrierType::kFragmentShaderToCompute);
    if (payload_buffer) {
      cmd_list.bufferMemoryBarrier(
          payload_buffer->getBuffer(),
          gfx::BufferMemoryBarrierType::kFragmentShaderToCompute);
    }

    cmd_list.writeTimestamp(*query_pool_,
                            PipelineStageBits::kFragmentShaderBit,
                            base_timestamp + kCaptureEnd);

    // Composite fragments
    // TODO: replace flush with memory barrier if 2-stage disabled
    cmd_list.pushLabel("Comp fragments 1")
        .writeTimestamp(*query_pool_,
                        PipelineStageBits::kTopOfPipeBit,
                        base_timestamp + kCompositeStart)
        .dispatchCompute(composite_pipeline,
                         kDefaultComposite,
                         (tile.w + subgroup_size - 1) / subgroup_size,
                         tile.h,
                         1u)
        .writeTimestamp(*query_pool_,
                        PipelineStageBits::kComputeShaderBit,
                        base_timestamp + kCompositeEnd)
        .popLabel()
        .flush("Capture and comp", gfx::CommandList::SubmitType::kWaitComplete);

    if constexpr (kPrintCaptureAllocMaxima) {
      uint32_t alloc_value;
      alloc_buffer.getData(&alloc_value, sizeof(uint32_t));
      std::cout << "Capture alloc value: " << alloc_value << std::endl;
    }

    if (enable_2_stage_comp_) {
      // Get stats to check for sort array overflow
      stats_buffer.getData(&resolve_stats, sizeof(PPLLResolveStats));
      if constexpr (kPrintPPLLStats) {
        max_sort_array_items =
            std::max(max_sort_array_items, resolve_stats.max_unique_ids);
        if (resolve_stats.overflow_pixel_count) {
          print_stat("  num overflow pixels", resolve_stats.overflow_pixel_count);
        }
      }

      //
      // Handle sort array overflow
      //
      if (resolve_stats.overflow_pixel_count > 0) {
        // Create a new specialization
        // The only upper bound we have is max fragments per pixel so use that
        // only build a new specialization if the old one is non-existent or was too small
        uint32_t specialization_size =
            std::min(max_fragments_stats_.per_pixel_batched, comp_array_size_limit_);
        const bool is_max_specialization_size =
            specialization_size == comp_array_size_limit_;
        if (current_comp_specialization_size_ < specialization_size) {
          struct {
            uint32_t size;
            uint32_t is_overflow;
          } spec_constants;
          spec_constants.size = specialization_size;
          spec_constants.is_overflow = 1;
          static_cast<ComputePipeline*>(
              pipeline_resources_[PipelineType::kComposite].pipeline)
              ->createSpecialization(
                  kOverflowComposite, &spec_constants, sizeof(spec_constants));
          current_comp_specialization_size_ = specialization_size;
          if constexpr (kPrintPPLLStats) {
            print_stat("  new specialization size", current_comp_specialization_size_);
          }
        }

        if (is_max_specialization_size) {
          // Clear the stats so we can detect pixels that exceed shared memory size
          cmd_list.fillBuffer(stats_buffer.getBuffer(), 0u, sizeof(PPLLResolveStats))
              .bufferMemoryBarrier(stats_buffer.getBuffer(),
                                   gfx::BufferMemoryBarrierType::kTransferToCompute);
        }

        // re-run composite using new pipeline to fill in remaining deeper pixels
        // We only need a memory barrier here since we don't need to read anything
        // back unless printing stats
        cmd_list.pushLabel("Comp fragments 2")
            .dispatchCompute(composite_pipeline,
                             kOverflowComposite,
                             (tile.w + subgroup_size - 1) / subgroup_size,
                             tile.h,
                             1u)
            .popLabel();
        if (is_max_specialization_size || kPrintPPLLStats) {
          // flush commands and wait to retrieve stats
          cmd_list.flush("Comp fragments 2", gfx::CommandList::SubmitType::kWaitComplete);
          // Retrieve the shared memory overflow pixel count
          stats_buffer.getData(&resolve_stats, sizeof(PPLLResolveStats));
          max_sort_array_items =
              std::max(max_sort_array_items, resolve_stats.max_unique_ids);
          max_complex_pixel_count =
              std::max(max_complex_pixel_count, resolve_stats.overflow_pixel_count);
          if constexpr (kPrintPPLLStats) {
            print_stat("  complex pixels: ", resolve_stats.overflow_pixel_count);
          }
        } else {
          cmd_list.imageMemoryBarrier(
              counts_texture, gfx::ImageMemoryBarrierType::kComputeToFragmentShader);
        }
      }
    }
  };

  //
  // Capture and composite batches
  //
  auto num_batches = num_primitives_per_batch.size();
  auto const& tile_rects = global_resources_.getTileRects();
  const std::vector<uint32_t>& render_tile_indices =
      using_tiles_ ? tile_indices : image_tile_index;

  // Timing
  uint64_t total_capture_time{0}, total_comp_time{0};
  std::array<std::pair<uint64_t, uint64_t>, kNumPPLLTiles> per_tile_times;
  auto start_time = timer_start();
  for (auto const tile_index : render_tile_indices) {
    CHECK_LE(tile_index, 4U);
    auto const& tile = tile_rects[tile_index];
    cmd_list.setRenderArea(tile.x, tile.y, tile.w, tile.h);
    // reset timestamps
    cmd_list.resetQueryPool(*query_pool_, 0, num_batches * kNumTimestampsPerBatch);

    if (using_batches_) {
      uint32_t batch_offset = 0u;
      for (uint32_t batch_index = 0; batch_index < num_batches; ++batch_index) {
        auto batch_size = num_primitives_per_batch[batch_index];
        CHECK_NE(batch_size, 0U);
        process_batch(
            draw_callback_, batch_index, batch_size, batch_offset, tile, tile_index);
        batch_offset += batch_size;
      }
    } else {
      process_batch(draw_callback_, 0, num_primitives_, 0u, tile, tile_index);
      cmd_list.flush("Render batches", gfx::CommandList::SubmitType::kImmediateReturn);
    }

    // Read timestamps for this tile
    if constexpr (kPrintTimestampStats) {
      auto timestamps = query_pool_->getTimestampResults(0, num_batches * 4);
      uint64_t tile_capture_time{0}, tile_comp_time{0};
      for (uint32_t batch_index = 0; batch_index < num_batches; ++batch_index) {
        auto base_timestamp = batch_index * kNumTimestampsPerBatch;
        auto capture_time_us = query_pool_->timestampToMicroseconds(
            timestamps[base_timestamp + kCaptureEnd] -
            timestamps[base_timestamp + kCaptureStart]);
        auto comp_time_us = query_pool_->timestampToMicroseconds(
            timestamps[base_timestamp + kCompositeEnd] -
            timestamps[base_timestamp + kCompositeStart]);

        tile_capture_time += capture_time_us;
        tile_comp_time += comp_time_us;
      }

      if (using_tiles_) {
        per_tile_times[tile_index] = {tile_capture_time, tile_comp_time};
      }
      total_capture_time += tile_capture_time;
      total_comp_time += tile_comp_time;
    }
  }
  LOG_IF(WARNING, max_complex_pixel_count > 0u)
      << "Maximum polygon depth complexity exceeded in at least "
      << max_complex_pixel_count << " pixels";

  // restore renderArea to the full image
  cmd_list.setRenderArea(0, 0, render_width_, render_height_);

  //
  // Ensure all commands submitted and wait
  //
  cmd_list.flush("Render batches", gfx::CommandList::SubmitType::kWaitComplete);
  auto total_time = timer_stop_microseconds(start_time);

  if constexpr (kPrintTimestampStats) {
    stats_stream << "Capture and Composite" << std::endl;
    print_stat("  Total time (CPU)", total_time, " us");
    if (using_tiles_) {
      for (uint32_t tile_index = 0; tile_index < kNumPPLLTiles; ++tile_index) {
        auto [capture_time, comp_time] = per_tile_times[tile_index];
        stats_stream << "    Tile " << tile_index << " (GPU):  "
                     << "capture " << std::left << std::setw(7) << capture_time
                     << "  composite " << comp_time << std::endl;
      }
      stats_stream << "    Totals (GPU):  "
                   << "capture " << std::left << std::setw(7) << total_capture_time
                   << "  composite " << total_comp_time << std::endl;
    } else {
      print_stat("    Capture (GPU)", total_capture_time, " us");
      print_stat("    Composite (GPU)", total_comp_time, " us");
    }
  }

  if constexpr (kPrintPPLLStats) {
    stats_stream << "Render and Composite stats:\n";
    print_stat("  using tiles", using_tiles_);
    print_stat("  using batches", using_batches_);
    print_stat("  num batches", using_batches_ ? num_primitives_per_batch.size() : 0u);
    print_stat("  max sort array items", max_sort_array_items);
  }
  VLOG(1) << "PPLL render complete";
}

const PPLLFragmentStats& PPLLRender::getFragmentStats() const {
  return fragment_stats_;
}

}  // namespace gfx
