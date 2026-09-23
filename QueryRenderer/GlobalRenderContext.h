/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <boost/noncopyable.hpp>

#include "GfxDriver/GfxContext.h"
#include "QueryRenderer/Interface/DataMgr_ForwardDeclarations.h"
#include "QueryRenderer/PerGpuData.h"
#include "Shared/DeviceGroup.h"

namespace QueryRenderer {

class SqlPolyQueryCacheMap;
class QueryResultCache;
class Renderer;

// Utility function to convert a requested render size to a render target size including
// any necessary padding
std::pair<uint32_t, uint32_t> get_render_target_size_from_render_size(
    uint32_t render_width,
    uint32_t render_height);

class GlobalRenderContext : private boost::noncopyable {
 public:
  GlobalRenderContext(const gfx::GfxContext& gfx_context,
                      Data_Namespace::DataMgr* const data_mgr,
                      CudaMgr_Namespace::CudaMgr* const cuda_mgr,
                      const size_t render_mem_bytes,
                      const bool use_last_gpu_for_compositor,
                      const gfx::RasterSampleCount raster_sample_count,
                      const bool renderer_enable_slab_allocation);
  ~GlobalRenderContext();

  // Handle lazy initialization
  void init();

  // Clear memory resources
  void clearGpuMemory();
  void clearCpuMemory();

  // GpuId vs gpu_index:
  // Data can be referenced by physical GpuId or logical gpu_index (currently size_t)
  // GpuId is monotonically increasing 0 based index including ALL devices in the system,
  // even those not in use (e.g. start_gpu=1).
  // gpu_index is monotonically increasing index of just devices IN USE. So if num-gpus is
  // set to 1, gpu_index will always = 0 no matter how many devices are available.

  GpuId getStartGpuId() const;  // TODO(scb): replace with DeviceGroup
  GpuId getLastGpuId() const;
  GpuId getLeastSubscribedGpuId() const;

  inline RootPerGpuDataMap& getRootPerGpuData() { return gpu_data_map_; }
  inline const RootPerGpuDataMap& getRootPerGpuData() const { return gpu_data_map_; }

  // GpuData lookup using physical id
  RootPerGpuData& getGpuData(GpuId gpu_id) const;

  // GpuData lookup based on logical index
  GpuId getGpuId(size_t gpu_index) const;
  RootPerGpuData& getGpuDataFromIndex(size_t gpu_index) const;

  // System components
  const gfx::GfxContext& getGfxContext() const;
  const Renderer& getRenderer() const;
  QueryRenderSMAAPass& getSMAAPass() const { return *smaa_pass_; };
  SeparateMultiSamplesPass* getSeparateMultiSamplesPass() const {
    return separate_multisamples_pass_.get();
  }

  // Compositors
  MultiGpuCompositor* getMultiGpuCompositor() const {
    return multi_gpu_compositor_.get();
  }
  GpuId getCompositorGpuId() const;

  inline size_t getRenderMemBytes() const { return render_mem_bytes_; }
  inline gfx::RasterSampleCount getRasterSampleCount() const {
    return raster_sample_count_;
  }
  inline uint32_t getNumSamples() const { return num_samples_; }

  // Render targets
  // Ensure render targets can accomodate a render of width x height
  void prepareRenderTargets(const QueryRendererContext& render_context,
                            const std::set<GpuId>& used_gpus);

  // Set the internal width/height to 0. This will force a render target
  // rebuild on the next render
  void resetRenderTargetSize();

  inline uint32_t getRenderTargetWidth() const { return render_target_width_; }
  inline uint32_t getRenderTargetHeight() const { return render_target_height_; }

  void createCommonRenderPasses(RootPerGpuData& gpu_data);

  inline QueryResultCache& getRenderQueryCacheMap() { return *query_result_cache_; }
  inline const QueryResultCache& getRenderQueryCacheMap() const {
    return *query_result_cache_;
  }

  // Data/CudaMgr
  Data_Namespace::DataMgr* getDataMgr();
  const CudaMgr_Namespace::CudaMgr* getCudaMgr() const;

  // accum renderer
  AccumRenderer& getAccumRenderer() const {
    CHECK(accum_renderer_);
    return *accum_renderer_;
  }

  gfx::DriverType getDriverType() const;

  void logMemorySummary(std::ostream& os) const;

  bool canUseMeshShaders() const;
  bool canUseSlabAddressTable() const;

  void updateSlabAddressTableAndBuffers(const int gpu_id);

 private:
  const gfx::GfxContext& gfx_context_;
  Data_Namespace::DataMgr* const data_mgr_;
  CudaMgr_Namespace::CudaMgr* const cuda_mgr_;
  RootPerGpuDataMap gpu_data_map_;
  std::unique_ptr<Renderer> renderer_;
  QueryRenderSMAAPassUqPtr smaa_pass_;
  SeparateMultiSamplesPassUqPtr separate_multisamples_pass_;
  std::unique_ptr<MultiGpuCompositor> multi_gpu_compositor_;
  AccumRendererUqPtr accum_renderer_;

  std::unique_ptr<QueryResultCache> query_result_cache_;

  const size_t render_mem_bytes_;
  const bool use_last_gpu_for_compositor_;
  const gfx::RasterSampleCount raster_sample_count_;
  const uint32_t num_samples_;
#ifdef HAVE_CUDA
  const bool renderer_enable_slab_allocation_;
#endif

  // current size of framebuffers and dependent textures
  uint32_t render_target_width_;
  uint32_t render_target_height_;
  uint32_t accum_tx_array_depth_;

  void initCachesAndResources(const heavyai::DeviceGroup& device_group);
  bool areCachesAndResourcesComplete(const bool log_incomplete) const;
};

}  // namespace QueryRenderer
