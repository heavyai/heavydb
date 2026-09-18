/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/core/noncopyable.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Render/GeoCountResources.h"
#include "GfxDriver/Render/PPLLResources.h"
#include "GfxDriver/Types.h"
#include "QueryRenderer/Interop/QueryBuffer.h"
#include "QueryRenderer/Interop/Types.h"
#include "QueryRenderer/QueryBufferManager.h"
#include "QueryRenderer/Rendering/QueryBufferPool.h"
#include "QueryRenderer/Rendering/Types.h"
#include "QueryRenderer/Types.h"
#include "Shared/EnumBitmaskOps.h"

namespace QueryRenderer {

struct PolyBufferPtrs {
  QueryVertexBufferShPtr verts;
  QueryIndirectVboShPtr line_draw_struct;
  QueryIndirectVboShPtr poly_draw_struct;
  QueryShaderStorageBufferShPtr per_row_data;
  QueryShaderStorageBufferShPtr poly_rowids;
};

struct LineBufferPtrs {
  QueryVertexBufferShPtr verts;
  QueryIndexBufferShPtr indices;
  QueryShaderStorageBufferShPtr per_row_data;
  QueryIndirectVboShPtr indirect_vertex_struct;
  QueryIndirectIboShPtr indirect_index_struct;
};

struct RasterMeshBufferPtrs {
  QueryVertexBufferShPtr vbo;
  QueryIndexBufferShPtr ibo;
};

struct TmpPolyBufferWkPtrs {
  std::weak_ptr<QueryVertexBuffer> verts;
  std::weak_ptr<QueryIndirectVbo> line_draw_struct;
  std::weak_ptr<QueryIndirectVbo> poly_draw_struct;
  std::weak_ptr<QueryShaderStorageBuffer> per_row_data;
  std::weak_ptr<QueryShaderStorageBuffer> poly_rowids;
};

struct TmpLineBufferWkPtrs {
  std::weak_ptr<QueryVertexBuffer> verts;
  std::weak_ptr<QueryIndexBuffer> indices;
  std::weak_ptr<QueryShaderStorageBuffer> per_row_data;
  std::weak_ptr<QueryIndirectVbo> indirect_vertex_struct;
  std::weak_ptr<QueryIndirectIbo> indirect_index_struct;
};

struct TmpRasterMeshBufferWkPtrs {
  std::weak_ptr<QueryVertexBuffer> vbo;
  std::weak_ptr<QueryIndexBuffer> ibo;
};

using TmpPolyBufferWkPtrMap = std::unordered_map<std::string, TmpPolyBufferWkPtrs>;
using TmpLineBufferWkPtrMap = std::unordered_map<std::string, TmpLineBufferWkPtrs>;
using TmpPolyDrawBatchInfoMap = std::unordered_map<std::string, PolyDrawBatchInfoUqPtr>;
using TmpRasterMeshBufferWkPtrMap =
    std::unordered_map<std::string, TmpRasterMeshBufferWkPtrs>;

struct AccumExtentsPipelines {
  gfx::MaterialUqPtr extents_material;
  gfx::MaterialUqPtr std_dev_material;

  gfx::resource_ptr<gfx::ComputePipeline> extents_pipeline;
  gfx::resource_ptr<gfx::ComputePipeline> std_dev_pipeline;

  void destroyPipelines(gfx::ResourceManager& resource_mgr);
};

struct AccumIdPassResources {
  gfx::MaterialUqPtr material;
  gfx::PipelineDescriptorUqPtr pipeline_desc;
  gfx::resource_ptr<gfx::GraphicsPipeline> pipeline;
  gfx::resource_ptr<gfx::RenderPass> renderpass;
  gfx::Framebuffer* framebuffer;

  void destroyResources(gfx::ResourceManager& resource_mgr);
};

struct ClearQueryOutputBufferResources {
  gfx::MaterialUqPtr material;
  gfx::resource_ptr<gfx::ComputePipeline> pipeline;

  void destroyResources(gfx::ResourceManager& resource_mgr);
};

// CommonRenderPassType
// Single subpass except kDepthStencilThenColor which is 2 subpasses:
//  subpass 0 = DepthStencil only
//  subpass 1 = All attachments (Vulkan TODO: special treatment of depth/stencil?)
enum class CommonRenderPassType {
  kAllAttachmentsClear,
  kAllAttachments,
  kDepthStencilThenAll,
  kCount
};

enum class CommonRenderPassTypeBits : uint32_t {
  kAllAttachmentsClear = 1 << 0,
  kAllAttachments = 1 << 1,
  kDepthStencilThenAll = 1 << 2
};

class RootPerGpuData : private boost::noncopyable {
 public:
  RootPerGpuData(const gfx::DeviceContext& device_ctx,
                 Data_Namespace::DataMgr* data_mgr,
                 const bool renderer_enable_slab_allocation);
  ~RootPerGpuData();

  GpuId getGpuId() const;
  const gfx::DeviceContext& getDeviceContext() const;
  gfx::ResourceManager& getResourceManager() const;
  QueryBufferManager& getQueryBufferManager() const;
  gfx::CommandList& getCommandList() const;

  void prepareRenderTargets(uint32_t width, uint32_t height);
  void prepareAccumTextureArray(uint32_t width, uint32_t height, uint32_t depth);
  void destroyAccumTextureArray();
  void updateAccumIDPassResources();

  QueryVertexBuffer* getQueryResultBuffer() const { return query_result_buffer_.get(); }
  QueryVertexBufferShPtr getQueryResultBufferShPtr() const {
    return query_result_buffer_;
  }
  QueryFramebuffer* getRenderFramebuffer() { return ms_framebuffer_.get(); }
  QueryFramebuffer* getAntiAliasingFramebuffer() const { return aa_framebuffer_.get(); }

  gfx::RenderPass& getCommonRenderPass(CommonRenderPassType type,
                                       bool multisampled) const;
  void prepareCommonFramebuffers(CommonRenderPassTypeBits type_bits);
  gfx::Framebuffer& getFramebufferForCommonRenderPass(CommonRenderPassType type,
                                                      bool multisampled) const;

  const gfx::RenderPass& getEmptyRenderPass() const;
  std::pair<gfx::RenderPass&, gfx::Framebuffer&> getEmptyRenderPassAndFramebuffer() const;

  QueryIdMapPboPool* getIdMapPboPool() { return id_pbo_pool_.get(); }
  QueryIdMapPixelBufferWkPtr getInactiveIdMapPbo(uint32_t width, uint32_t height);
  void setIdMapPboInactive(QueryIdMapPixelBufferWkPtr& pbo);

  QueryBufferPool<QueryVertexBuffer>& getVboBufferPool() {
    CHECK(vbo_buffer_pool_);
    return *vbo_buffer_pool_;
  }
  QueryBufferPool<QueryIndexBuffer>& getIboBufferPool() {
    CHECK(ibo_buffer_pool_);
    return *ibo_buffer_pool_;
  }
  QueryBufferPool<QueryShaderStorageBuffer>& getSsboBufferPool() {
    CHECK(ssbo_buffer_pool_);
    return *ssbo_buffer_pool_;
  }
  QueryBufferPool<QueryIndirectVbo>& getIndVboBufferPool() {
    CHECK(indvbo_buffer_pool_);
    return *indvbo_buffer_pool_;
  }
  QueryBufferPool<QueryIndirectIbo>& getIndIboBufferPool() {
    CHECK(indibo_buffer_pool_);
    return *indibo_buffer_pool_;
  }

  gfx::RenderPass& getAccumRenderPass() const;

  const AccumExtentsPipelines& getAccumExtentsPipelines() const {
    CHECK(accum_extents_pipelines_);
    return *accum_extents_pipelines_;
  }
  gfx::Texture* getAccumTextureArray() const {
    CHECK(accum_tx_array_);
    return accum_tx_array_.get();
  }
  gfx::BufferWrapper* getAccumExtentsBuffer() const {
    CHECK(accum_extents_buffer_);
    return accum_extents_buffer_.get();
  }
  const AccumIdPassResources& getAccumIDPassResources() const {
    CHECK(accum_id_pass_resources_);
    return *accum_id_pass_resources_;
  }

  gfx::GeoCountResources& getGeoCountResources() const {
    CHECK(geo_count_resources_);
    return *geo_count_resources_;
  }

  gfx::PPLLResources& getPPLLResources() const {
    CHECK(ppll_resources_);
    return *ppll_resources_;
  }

  TmpLineBufferWkPtrs* getTmpLineBuffersForDataTable(const std::string& data_table_name);
  TmpLineBufferWkPtrs& createTmpLineBuffersForDataTable(
      const std::string& data_table_name);

  TmpPolyBufferWkPtrs* getTmpPolyBuffersForDataTable(const std::string& data_table_name);
  TmpPolyBufferWkPtrs& createTmpPolyBuffersForDataTable(
      const std::string& data_table_name);
  void createPolyDrawBatchInfoForDataTable(const std::string& data_table_name,
                                           PolyDrawBatchInfoUqPtr&& poly_draw_batch_info);

  bool hasPolyDrawBatchInfoForDataTable(const std::string& data_table_name);
  PolyDrawBatchInfoUqPtr extractPolyDrawBatchInfoForDataTable(
      const std::string& data_table_name);

  TmpRasterMeshBufferWkPtrs* getTmpRasterMeshBuffersForDataTable(
      const std::string& data_table_name);
  TmpRasterMeshBufferWkPtrs& createTmpRasterMeshBuffersForDataTable(
      const std::string& data_table_name);

  void releaseLineAndPolyBuffers();

  void clearQueryOutputBuffer();

  const gfx::BufferWrapper& getSlabAddressTableBuffer() const;

  gfx::BufferAllocatorShPtr getBufferAllocator() const;

 private:
  const gfx::DeviceContext& device_ctx_;
  std::unique_ptr<QueryBufferManager> query_buffer_mgr_;

  QueryVertexBufferShPtr query_result_buffer_;

  QueryFramebufferUqPtr ms_framebuffer_;
  QueryFramebufferUqPtr aa_framebuffer_;
  QueryIdMapPboPoolUqPtr id_pbo_pool_;

  std::array<gfx::resource_ptr<gfx::RenderPass>,
             static_cast<int>(CommonRenderPassType::kCount) * 2>
      common_render_passes_;

  gfx::resource_ptr<gfx::RenderPass> empty_renderpass_;
  gfx::resource_ptr<gfx::Framebuffer> empty_framebuffer_;

  gfx::resource_ptr<gfx::RenderPass> accum_render_pass_;

  std::unique_ptr<QueryBufferPool<QueryVertexBuffer>> vbo_buffer_pool_;
  std::unique_ptr<QueryBufferPool<QueryIndexBuffer>> ibo_buffer_pool_;
  std::unique_ptr<QueryBufferPool<QueryShaderStorageBuffer>> ssbo_buffer_pool_;
  std::unique_ptr<QueryBufferPool<QueryIndirectVbo>> indvbo_buffer_pool_;
  std::unique_ptr<QueryBufferPool<QueryIndirectIbo>> indibo_buffer_pool_;

  gfx::resource_ptr<gfx::Texture> accum_tx_array_;
  gfx::BufferWrapperUqPtr accum_extents_buffer_;
  std::unique_ptr<AccumExtentsPipelines> accum_extents_pipelines_;
  std::unique_ptr<AccumIdPassResources> accum_id_pass_resources_;

  std::unique_ptr<gfx::PPLLResources> ppll_resources_;
  std::unique_ptr<gfx::GeoCountResources> geo_count_resources_;

  // Support multi-layer rendering using buffer pool allocations by mapping the
  // data table name to the temporary buffer pointers and other data
  TmpPolyBufferWkPtrMap tmp_poly_buffer_wk_ptr_map_;
  TmpLineBufferWkPtrMap tmp_line_buffer_wk_ptr_map_;
  TmpPolyDrawBatchInfoMap tmp_poly_draw_batch_info_map_;
  TmpRasterMeshBufferWkPtrMap tmp_raster_mesh_buffer_wk_ptr_map_;

  std::unique_ptr<ClearQueryOutputBufferResources> clear_query_output_buffer_resources_;

  void clearResources();
  friend class GlobalRenderContext;
};

struct inorder {};
struct RootPerGpuDataId {
  using result_type = GpuId;
  result_type operator()(const RootPerGpuDataUqPtr& per_gpu_data) const {
    return per_gpu_data->getGpuId();
  }
};

using RootPerGpuDataMap = ::boost::multi_index_container<
    RootPerGpuDataUqPtr,  // NOTE: using a unique ptr here to avoid some of the const
                          // rules associated with multi-index containers
    ::boost::multi_index::indexed_by<
        // hashed on gpuId
        ::boost::multi_index::ordered_unique<RootPerGpuDataId>,
        ::boost::multi_index::random_access<::boost::multi_index::tag<inorder>>>>;

using RootPerGpuDataMap_in_order = RootPerGpuDataMap::index<inorder>::type;

class BasePerGpuData {
 public:
  explicit BasePerGpuData(RootPerGpuData& root_per_gpu_data)
      : root_per_gpu_data_{root_per_gpu_data} {}
  virtual ~BasePerGpuData() = default;

  GpuId getGpuId() const;
  const RootPerGpuData& getRootPerGpuData() const { return root_per_gpu_data_; }
  RootPerGpuData& getRootPerGpuData() { return root_per_gpu_data_; }
  const gfx::DeviceContext& getDeviceContext() const;
  gfx::ResourceManager& getResourceManager() const;

  QueryIdMapPixelBufferWkPtr getInactiveIdMapPbo(uint32_t width, uint32_t height);
  void setIdMapPboInactive(QueryIdMapPixelBufferWkPtr& pbo);

 private:
  RootPerGpuData& root_per_gpu_data_;
};

}  // namespace QueryRenderer

ENABLE_BITMASK_OPS(::QueryRenderer::CommonRenderPassTypeBits);
