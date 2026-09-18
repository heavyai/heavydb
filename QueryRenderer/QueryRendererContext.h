/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <boost/noncopyable.hpp>

#include "GfxDriver/Math/Matrix2d.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/AggregationContext.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Events/EventedProperty.h"
#include "QueryRenderer/Events/RefEvent.h"
#include "QueryRenderer/Events/RefEventCallbacksMap.h"
#include "QueryRenderer/Interface/DataMgr_ForwardDeclarations.h"
#include "QueryRenderer/Interface/RenderQueryRunnerInterface.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Marks/Types.h"
#include "QueryRenderer/Projections/Types.h"
#include "QueryRenderer/Rendering/HitTestBuffers.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"
#include "QueryRenderer/VegaMetaData.h"
#include "Shared/Rendering/InSituFlags.h"

namespace QueryRenderer {

class VegaElements;
class AggregationContext;

// @TODO(se) here for now
// default to premultiplied so that existing Vega does not change behavior
struct ViewRenderOptions {
  bool premultiplied_alpha = true;
};

//
// QueryRendererContext
//
class QueryRendererContext : boost::noncopyable {
 public:
  explicit QueryRendererContext(const RenderSessionKey& render_session_key,
                                GlobalRenderContext& global_context,
                                bool do_hit_test = false);

  ~QueryRendererContext();

  // system components
  Data_Namespace::DataMgr* getDataMgr() const;
  const CudaMgr_Namespace::CudaMgr* getCudaMgr() const;
  const gfx::ShaderManager& getShaderManager() const;

  // Metadata (may be null)
  const VegaMetaData* getMetaData() const;

  // View definition
  uint32_t getWidth() const { return width_; }
  uint32_t getHeight() const { return height_; }
  void setWidthHeight(const uint32_t width, const uint32_t height);

  const gfx::Math::Matrix2d<float>& getViewProjMatrix() const;

  // View Render Options
  const ViewRenderOptions& getViewRenderOptions() const { return view_render_options_; }

  //
  // User Session
  //
  SessionId getUserId() const;
  WidgetId getWidgetId() const;
  const RenderSessionKey& getRenderSessionKey() const;

  //
  // Hit testing
  //
  bool doHitTest() const { return do_hit_test_; }

  HitInfo getIdAt(uint32_t x, uint32_t y, uint32_t pixel_radius);
  std::string getVegaDataNameFromIndex(const int8_t data_id) const;

  //
  // QueryEngine / Executor
  //
  RenderQueryRunnerInterface* getRenderQueryRunner() const {
    return render_query_runner_.get();
  }
  std::shared_ptr<RenderQueryExecuteTimer> getRenderTimer() const {
    return render_timer_;
  }
  int16_t getNodeIdx() const { return node_index_; }

  void setQueryExecutionParams(RenderQueryRunnerUqPtr render_query_runner,
                               std::shared_ptr<RenderQueryExecuteTimer>& render_timer);
  void unsetQueryExecutionParams();

  //
  // JSON parser
  //
  const rapidjson::Value* getJsonCachePtr() const;

  bool isJSONCacheUpToDate(const rapidjson::Pointer& prev_path,
                           const JSONLocation& obj_location) const;
  JSONLocation getJSONObj(const rapidjson::Pointer& obj_path) const;
  bool jsonNeedsUpdating(const JSONLocation& root_json) const;

  //
  // Global context
  //
  inline GlobalRenderContext& getGlobalContext() { return global_context_; }
  inline const GlobalRenderContext& getGlobalContext() const { return global_context_; }

  //
  // Renderable entities
  //
  const VegaElements& getVegaElements() const { return *vega_elements_; }

  // Data table
  bool hasDataTable(const std::string& table_name) const;
  BaseDataTableShPtr getDataTable(const std::string& table_name) const;
  QueryDataTableSQLJSONShPtr getDataTableSQLJSON(const std::string& table_name) const;
  std::optional<int> getDataIndex(const std::string& table_name) const;
  void resetDataTableStates();

  QueryDataTableQueues& getDataTableQueues() const;

  // Query execution
  bool executeQuery(
      BaseQueryDataTableSQLJSON& data_table,
      const JSONLocation* json_loc,
      const OptionalStr& sql_query_override = std::nullopt,
      const heavyai::InSituFlags insitu_flags = heavyai::InSituFlags::kInSitu);

  // Projections
  bool hasProjection(const std::string& projection_name) const;
  ProjectionShPtr getProjection(const std::string& projection_name) const;

  // Scales
  bool hasScale(const std::string& scale_name) const;
  ScaleShPtr getScale(const std::string& scale_name) const;

  // Marks
  const std::vector<BaseMark*> getMarkConfigsForPassIndex(
      const uint16_t pass_index) const;
  size_t getNumTotalMarks() const;
  void buildMarkShaders(BaseMark& mark,
                        const MarkGpuResourceSlot slot,
                        const std::string& resource_tracking_string,
                        gfx::ShaderManager::BuilderUqPtrVector builders) const;
  void clearMarkShaders(BaseMark& mark) const;
  bool drawMark(const uint32_t mark_config_index,
                const gfx::DeviceContext& device_context,
                gfx::Framebuffer& framebuffer,
                const int accumulator_index) const;

  // Entity validation
  bool isReadyForRender(const BaseScale& scale) const;
  bool isReadyForRender(const BaseMark& mark) const;

  // Get the max used accumulator textures. Used to initialize compositor resources
  uint32_t getNumRequiredAccumulatorTextures() const;

  // Per-pixel linked list requirements
  bool usesPerPixelLinkedLists() const;

  std::set<GpuId> getUsedGpus() const;

  //
  // Aggregation
  //
  AggregationContext& getAggregationContext() { return *agg_context_; }
  const AggregationContext& getAggregationContext() const { return *agg_context_; }

  //
  // Vega serialization
  //
  std::string serializeToVega() const;

  //
  // Event callbacks
  //
  RefCallbackId subscribeToRefEvent(const RefEventType event_type,
                                    const RefObjShPtr& event_obj,
                                    RefEventCallback cb);
  void unsubscribeFromRefEvent(const RefEventType event_type,
                               const RefObjShPtr& event_obj,
                               const RefCallbackId callback_id);
  void notifyRefEvent(const RefEventType event_type, const RefObjShPtr& event_obj);
  void notifyRefEvent(const RefEventType event_type, const JSONRefObject* event_obj);

  PropUpdateCallbackId subscribeToPropEvent(
      const std::string& prop_name,
      std::function<void(const uint32_t, const uint32_t)> callback_func);

  PropUpdateCallbackId subscribeToPropEvent(
      const std::string& prop_name,
      std::function<void(const gfx::Math::Matrix2d<float>&,
                         const gfx::Math::Matrix2d<float>&)> callback_func);

  void unsubscribeFromPropEvent(const PropUpdateCallbackId callback_id);

 private:
  //
  // Renderable entities
  //
  std::unique_ptr<VegaElements> vega_elements_;
  uint32_t num_required_accumulator_textures_;
  bool uses_per_pixel_linked_lists_;

  //
  // System
  //
  GlobalRenderContext& global_context_;

  //
  // QueryEngine / Executor
  //
  RenderQueryRunnerUqPtr render_query_runner_;
  std::shared_ptr<RenderQueryExecuteTimer> render_timer_;
  int16_t node_index_;

  const RenderSessionKey& render_session_key_;

  // DataTable queues
  std::unique_ptr<QueryDataTableQueues> data_table_queues_;

  // Hit testing
  bool do_hit_test_;
  HitTestBuffersUqPtr hit_test_buffers_;

  std::unique_ptr<rapidjson::Document> json_cache_;

  // Aggregation
  std::unique_ptr<AggregationContext> agg_context_;

  void clear();

  // View options
  void setViewRenderOptions(const ViewRenderOptions& view_render_options);

  void updateConfigGpuResources();
  void postJSONUpdate();

  void updateMarksAndBuildShaders();

  void releaseLineOrPolyBuffers();

  //
  // Events
  //
  RefEventCallbacksMap ref_event_callbacks_map_;
  PropUpdateCallbackId curr_prop_callback_id_;

  EventedProperty<uint32_t> width_;
  EventedProperty<uint32_t> height_;
  EventedProperty<gfx::Math::Matrix2d<float>> view_proj_matrix_;

  // @TODO(se) should this be an EventedProperty?
  ViewRenderOptions view_render_options_;

  bool force_mark_uniform_update_;

  friend class Renderer;
  // TODO(scb): remove
  friend class AggregationContext;
  friend class VegaParser;
};

}  // namespace QueryRenderer
