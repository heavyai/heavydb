/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <bitset>
#include <functional>
#include <memory>
#include <vector>

#include <boost/noncopyable.hpp>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Commands/QueryPool.h"
#include "GfxDriver/Render/PPLLResources.h"
#include "GfxDriver/Render/shaders/PPLL/ppllCommon.h"
#include "GfxDriver/Types.h"

namespace gfx {

template <int size>
struct DirtyBits {
  std::bitset<size> is_dirty;

  bool operator[](int i) const { return is_dirty[i]; }
  typename std::bitset<size>::reference operator[](int i) { return is_dirty[i]; }

  void setAll() { is_dirty.set(); };
  void clearAll() { is_dirty.reset(); };
};

/**
 * PPLLRender class
 *
 * Class to manage C++ side of per-pixel linked list (PPLL) rendering,
 * including descriptor updates, buffer allocation and sizing, and
 * capture / composite, including tiling and batched draw support
 *
 * PPLL rendering flow:
 *   1/ Accumulate fragment counts in each pixel using imageAtomicAdd
 *   2/ Generate PPLLFragmentStats for the image and optionally per image tile
 *   3/ OPTIONAL apply tiling and batching heuristics
 *   4/ Allocate storage for list records (64-bits per fragment record)
 *   5/ OPTIONAL allocate separate payload buffer (custom size per fragment)
 *   6/ Capture all fragments in the list including separate payload if used
 *   7/ Composite all fragments, resolving unique primitives and sorting as needed
 *   8/ Output final framebuffer storage (single or multi-sampled)
 *
 * PipelineResources (Material*, Pipeline*) must be supplied for
 *   Counting (fragment accumulation), Capture, and Compositing (resolve).
 *   Counting and capture are raster passes
 *   Composite is a compute pass
 *
 * The PPLL class handles setting the common PPLL resource descriptors and
 * will call updateDescriptorSets on each Material prior to draw / dispatch.
 * Special descriptors (eg poly / line property SSBOs) must be set by client code
 *
 * An instance of this class is required for each GPU
 *
 * Usage:
 * - Call renderBegin() to initialize all per-render variables
 * - Call countFragmentsAndComputeStats() to generate stats required
 *   for buffer sizing. This function may return false if rendering
 *   should stop (eg debug vis shader is in use via OutputColorMode)
 * - Call prepareStorage() to allocate the list record buffer
 *   Passing a value > 0 for payload size will trigger a secondary payload
 *   buffer to be created/sized (payload size is per fragment!)
 * - Transition output images to kGeneral layout (for output from compute)
 * - Call captureFragmentsAndComposite() to capture and resolve/composite
 *   fragments
 *   If using batches a vector of batch sizes must be supplied.
 *   Application of batches is via the count and index_offset params
 *   passed to the DrawPrimitives callback
 * - Transition output images to next required layout (kAttachment / kPresent)
 * */

class PPLLRender : private boost::noncopyable {
 public:
  enum PipelineType { kCount, kCapture, kComposite };
  static constexpr int kNumPipelines = 3;

  //
  // Color output modes for debugging
  //
  enum class ColorOutputMode {
    // Standard output (no debug)
    kNormal,
    // ppllDebugVisualizer shader controls
    // Does not gather or composite fragments
    kFragmentDensityMono,
    kFragmentDensityColor,
    kFragmentCountMono,
    kFragmentCountColor,
    // Fragment compositor shader controls
    // Requires support in the compositor shader
    kPolyID,
    kDepth,
    kIDA,
    // offsets for setting uniforms
    kBeginStatsVis = kFragmentDensityMono,
    kEndStatsVis = kPolyID,
    kBeginCompVis = kPolyID
  };

  // Pair Pipelines and associated Materials
  struct PipelineResources {
    Material* material = nullptr;
    Pipeline* pipeline = nullptr;
  };

  // Draw callback used to generate fragments
  // When using batches count and index offset are used to limit to current batch
  using DrawPrimitivesCB = std::function<void(CommandList&,
                                              std::string_view,
                                              Pipeline&,
                                              uint32_t count,
                                              uint32_t index_offset)>;

  explicit PPLLRender(const DeviceContext& device,
                      PPLLResources& global_resources,
                      const bool enable_tiling,
                      const bool enable_batching,
                      const bool enable_2_stage_comp,
                      const uint32_t initial_comp_array_size);
  ~PPLLRender();

  // Pass all required Pipeline and Material pairs
  void setPipelineResources(PipelineResources&& count_resources,
                            PipelineResources&& capture_resources,
                            PipelineResources&& composite_resources);

  // renderBegin is called to initialize all per-render state
  // output_rgb_texture is required but currently only used when
  // ColorOutput mode results in call the debug visualization pipeline
  void renderBegin(const uint32_t render_width,
                   const uint32_t render_height,
                   Texture& output_rgb_texture,
                   DrawPrimitivesCB draw_callback,
                   uint32_t num_primitives,
                   ColorOutputMode color_mode = ColorOutputMode::kNormal);
  //
  // Fragment counting and stats
  // methods return true if rendering should continue
  //
  // Full image and tile stats only
  // Requires counts texture
  bool countFragmentsAndComputeStats();

  // Full image, tile, and batch stats
  // Requires counts and batch counts textures
  bool countFragmentsAndComputeStats(
      const std::vector<uint32_t>& num_primitives_per_batch);

  // Prepare list buffer storage
  // if payload_size > 0 then a separate payload buffer will be allocated
  void prepareStorage(size_t payload_size,
                      const std::vector<uint32_t>& num_primitives_per_batch);

  // Capture fragments then run the composite compute pipeline
  // If batching is enabled and active num_primitives_per_batch controls the
  // individual batch sizes
  void captureAndComposite(const std::vector<uint32_t>& num_primitives_per_batch);

  // Get stats for entire rendered frame
  // If tiling is enabled this returns the aggregate stats across all tiles
  const PPLLFragmentStats& getFragmentStats() const;

 private:
  // Composite shader specializations to handle local sort array overflow
  // kDefault specializes array size to initial_comp_array_size
  // kOverflow will specialize to current_comp_specialization_size_
  // current_comp_array_size_ will be updated from stats generated during
  // the first composite pass (that overflows)
  enum CompSpecialization { kDefaultComposite, kOverflowComposite };

  // Push constants for capture and composite consisting of two uint32_t
  // low uint is batch index offset, hi uint is tile index
  static const uint32_t kTileIndexPushConstantOffset{4};
  static std::vector<gfx::PushConstantRange> capture_and_comp_push_constants;

  const DeviceContext& device_;
  PPLLResources& global_resources_;
  DrawPrimitivesCB draw_callback_;
  const bool enable_tiling_;
  const bool enable_batching_;
  const bool enable_2_stage_comp_;
  const uint32_t initial_comp_array_size_;
  const uint32_t comp_array_size_limit_;
  std::unique_ptr<PPLLResources::NotifyChangedCallback> resources_changed_cb_;

  uint32_t render_width_;
  uint32_t render_height_;
  Texture* output_rgb_texture_;
  uint32_t num_primitives_;
  ColorOutputMode color_output_mode_;

  // materials and pipelines
  std::array<PipelineResources, kNumPipelines> pipeline_resources_;

  // descriptor dirty bits to track requirement to bind resources prior
  // to a draw / dispatch step
  struct DescriptorBits : public DirtyBits<3> {
    enum { kCount, kCapture, kComposite };
  };
  DescriptorBits is_descriptor_dirty_;

  // dynamic draw time stuff
  bool using_tiles_;
  bool using_batches_;

  PPLLFragmentStats fragment_stats_;
  struct {
    uint64_t per_tile_all{0};
    uint64_t per_tile_deep{0};
    uint64_t per_batch{0};
    uint64_t per_batch_tiled{0};
    uint32_t per_pixel_batched{0};
    void reset() {
      per_tile_all = 0;
      per_tile_deep = 0;
      per_batch = 0;
      per_batch_tiled = 0;
      per_pixel_batched = 0;
    }
  } max_fragments_stats_;

  uint32_t current_comp_specialization_size_;

  resource_ptr<QueryPool> query_pool_;

  void updateCountDescriptorSets();
  void updateCaptureAndCompositeDescriptorSets();
  void notifyResourceSizeChanged();
  void drawStatsVisualization();
};

}  // namespace gfx
