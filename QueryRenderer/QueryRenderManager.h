/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>

#include "GfxDriver/GfxContext.h"
#include "GfxInterop/BufferMemoryDescriptor.h"
#include "QueryRenderer/Cache/HitTestCacheResults.h"
#include "QueryRenderer/Interface/AggDataTypes.h"
#include "QueryRenderer/Interface/DataMgr_ForwardDeclarations.h"
#include "QueryRenderer/Interface/RenderQueryRunnerInterface.h"
#include "QueryRenderer/Interface/RenderRequestInfo.h"
#include "QueryRenderer/Interface/RenderSessionKey.h"
#include "QueryRenderer/Interface/ResultCacheTypes.h"
#include "QueryRenderer/RenderSession.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

class RenderCmdQueue;
class RenderSessionMgr;
class PolyMgr;
class LineMgr;
class RasterMeshMgr;
class GlobalRenderContext;

class QueryRenderManager {
 public:
  explicit QueryRenderManager(gfx::GfxContext* gfx_context,
                              Data_Namespace::DataMgr* data_mgr,
                              const size_t render_mem_bytes,
                              const size_t render_cache_limit,
                              const bool compositor_use_last_gpu,
                              const gfx::RasterSampleCount raster_sample_count,
                              const bool renderer_enable_slab_allocation);

  ~QueryRenderManager();

  Data_Namespace::DataMgr* getDataMgr();
  const CudaMgr_Namespace::CudaMgr* getCudaMgr() const;
  const Renderer& getRenderer() const;

  //
  // RenderSession API
  //
  // Removes all widgets/sessions for a particular session id.
  void removeSessionId(const SessionId& session_id);

  // get (or create, if doesn't exist) render session
  const RenderSession& getOrCreateRenderSession(
      std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
      const WidgetId widget_id,
      std::string&& vega_json);

  bool hasRenderSession(const RenderSessionKey& render_session_key);

  //
  // GPU list
  //
  // stay
  size_t getNumGpus() const;
  std::vector<GpuId> getAllGpuIds() const;
  GpuId getStartGpuId() const;

  // Get Id for gpu with the most free memory
  // Used to select a gpu for non-insitu renders
  GpuId getLeastSubscribedGpuId() const;

  //
  // QueryEngine interop
  //
  // stay
  gfx::BufferMemoryDescriptor getQueryOutputBufferDescriptor(size_t gpu_index);
  void releaseQueryOutputBufferDescriptor(size_t gpu_index,
                                          size_t num_used_bytes,
                                          const QueryDataLayoutShPtr& vert_layout);

  //
  // Cuda-disabled Rendering
  //
  // stay
  size_t getRenderMemorySizeInBytes() const;
  void bufferVboData(int8_t* data,
                     const size_t num_data_bytes,
                     const size_t vbo_buffer_offset,
                     const size_t gpu_index);

  void setRenderBufferDataLayout(const size_t gpu_index,
                                 const size_t offset_bytes,
                                 const size_t num_used_bytes,
                                 const QueryDataLayoutShPtr& vert_layout);

  //
  // Polys
  //
  // stay
  PolyMgr& getPolyMgr() const;

  //
  // Lines
  //
  // stay
  LineMgr& getLineMgr() const;

  //
  // RasterMesh
  //
  RasterMeshMgr& getRasterMeshMgr() const;

  //
  // Render API
  //
  // move
  RenderRequestInfo runRenderRequest(const RenderSession& render_session,
                                     RenderQueryRunnerUqPtr render_query_runner);

  // get the id at a specific pixel
  // move
  std::tuple<int32_t, int64_t, std::string, int16_t> getIdAt(
      const RenderSession& render_session,
      size_t x,
      size_t y,
      size_t pixelRadius = 0);

  // NOTE(scb): removed GeoReturnType since we are always passed WktString and it was
  // forcing inclusion of the massive ResultSet.h just to get the enum declaration! In the
  // future if we need this enum it needs to be handled differently.
  // stay
  HitTestCacheResults getQueryCacheResults(
      const ResultCacheId cache_id,
      const int64_t rowid_to_unpack/*,
      const ResultSet::GeoReturnType geo_return_type*/) const;

  //
  // Clear gpu/cpu memory
  //
  // stay
  void clearGpuMemory();
  void clearCpuMemory();

  // stay
  void setRenderBufferPeakUsage(uint64_t size);
  void logPeakMemoryUsage();

  // stay
  std::string getRendererStatusJSON() const;
  bool validateRendererStatusJSON(const std::string& other_renderer_status_json) const;

  //
  // wrapper for PNG encoding
  //
  static std::string encodePNG(const RenderPixels& pixels, const int compression_level);

  // facade these
  bool canUseMeshShaders() const;
  bool canUseSlabAddressTable() const;

 private:
  using QueryRenderCB = std::function<RenderPixels(QueryRendererContext&)>;

  std::unique_ptr<RenderCmdQueue> command_queue_;
  gfx::GfxContext* gfx_context_;

  std::unique_ptr<GlobalRenderContext> global_context_;
  std::unique_ptr<RenderSessionMgr> render_session_mgr_;
  std::unique_ptr<PolyMgr> poly_cache_mgr_;
  std::unique_ptr<LineMgr> line_cache_mgr_;
  std::unique_ptr<RasterMeshMgr> raster_mesh_cache_mgr_;

  mutable std::mutex render_mutex_, buffer_mutex_;
  size_t render_mem_bytes_;
  uint64_t render_buffer_peak_usage_;
  uint64_t gfx_peak_memory_usage_;

  void setCudaContext(const size_t gpu_index) const;

  void configureRenderInternal(const RenderSession& render_session);
  void executeVegaParse(const RenderSession& render_session,
                        std::shared_ptr<RenderQueryExecuteTimer>& render_timer);

  RenderRequestInfo executeRender(
      const RenderSession& render_session,
      const std::shared_ptr<RenderQueryExecuteTimer>& render_timer,
      QueryRenderCB query_render_func);

  void clearGpuMemoryInternal();
  void validateRenderTargetSize();

  static void atExitHandler();

  struct RendererStatusJSONWriter;
};

}  // namespace QueryRenderer
