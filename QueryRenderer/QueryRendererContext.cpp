/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/QueryRendererContext.h"

#include <png.h>
#include <cstring>
#include <memory>
#include <vector>

#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/AggregationContext.h"
#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/Data/QueryDataTableQueues.h"
#include "QueryRenderer/Data/QueryLineDataTable.h"
#include "QueryRenderer/Data/QueryPolyDataTable.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Events/RefEvent.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Marks/BaseMark.h"
#include "QueryRenderer/Marks/Types.h"
#include "QueryRenderer/Projections/Projection.h"
#include "QueryRenderer/QueryRenderManager.h"
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/Scales/ScaleAccumRenderState.h"
#include "QueryRenderer/Scales/ScaleAccumState.h"
#include "QueryRenderer/VegaElements.h"

#define LOG_MEMORY_SUMMARY false
#define LOG_MARK_MATERIAL_STATS false

#if LOG_MEMORY_SUMMARY || LOG_MARK_MATERIAL_STATS
#include <iostream>
#endif
#if LOG_MARK_MATERIAL_STATS
#include "GfxDriver/Utils/StatsUtils.h"
#include "Shared/measure.h"

static gfx::StatsMap<uint64_t> g_mark_build_material_stats_map;
#endif

namespace QueryRenderer {

QueryRendererContext::QueryRendererContext(const RenderSessionKey& render_session_key,
                                           GlobalRenderContext& global_context,
                                           bool do_hit_test)
    : vega_elements_{std::make_unique<VegaElements>()}
    , num_required_accumulator_textures_{0}
    , uses_per_pixel_linked_lists_{false}
    , global_context_{global_context}
    , render_session_key_{render_session_key}
    , data_table_queues_{std::make_unique<QueryDataTableQueues>(*this)}
    , do_hit_test_{do_hit_test}
    , hit_test_buffers_{std::make_unique<HitTestBuffers>(*this)}
    , json_cache_{nullptr}
    , agg_context_{std::make_unique<AggregationContext>(*this)}
    , curr_prop_callback_id_{0}
    , width_{0}
    , height_{0}
    , force_mark_uniform_update_{false} {}

QueryRendererContext::~QueryRendererContext() {
#if LOG_MARK_MATERIAL_STATS
  if (!g_mark_build_material_stats_map.is_empty()) {
    g_mark_build_material_stats_map.generateReport(std::cout, "Mark material build", 1);
    std::cout << std::endl;
  }
#endif
  clear();
}

void QueryRendererContext::clear() {
  // clear dependent resources
  hit_test_buffers_->releasePbo();
  releaseLineOrPolyBuffers();
  // clear internals
  width_ = 0;
  height_ = 0;
  vega_elements_->clear();
  ref_event_callbacks_map_.clear();
  json_cache_ = nullptr;
  num_required_accumulator_textures_ = 0;
  force_mark_uniform_update_ = false;
}

Data_Namespace::DataMgr* QueryRendererContext::getDataMgr() const {
  return global_context_.getDataMgr();
}

const CudaMgr_Namespace::CudaMgr* QueryRendererContext::getCudaMgr() const {
  return global_context_.getCudaMgr();
}

void QueryRendererContext::updateConfigGpuResources() {
  RENDER_LOG_SCOPE();
  for (auto data_table : vega_elements_->getDataTableMap()) {
    auto base_data_table = std::dynamic_pointer_cast<BaseDataTable>(data_table);
    CHECK(base_data_table);
    if (base_data_table->update()) {
      auto data_table_json =
          std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_table);
      if (data_table_json) {
        data_table_queues_->addToNotifyQueue(data_table_json, RefEventType::kUpdate);
      }
    }
  }
  data_table_queues_->processQueryQueue();
  data_table_queues_->processSourceTableQueue();
  data_table_queues_->processNotifyQueue();

  num_required_accumulator_textures_ = 0;
  for (auto& scale : vega_elements_->getScaleMap()) {
    auto* accum_state = scale->getAccumState();
    auto* accum_render_state = scale->getAccumRenderState();
    if (accum_state && accum_render_state) {
      accum_render_state->initGpuResources(*this, false);
      num_required_accumulator_textures_ =
          std::max(num_required_accumulator_textures_, accum_state->getNumTextures());
    }
  }

  uses_per_pixel_linked_lists_ = false;
  for (auto& mark : vega_elements_->getMarkVector()) {
    uses_per_pixel_linked_lists_ =
        uses_per_pixel_linked_lists_ | mark->usesPerPixelLinkedLists();
    mark->initGpuResources();
  }
}

void QueryRendererContext::postJSONUpdate() {
  RENDER_LOG_SCOPE();
  for (auto data_table : vega_elements_->getDataTableMap()) {
    data_table->postJSONUpdate();
  }

  for (auto& scale : vega_elements_->getScaleMap()) {
    scale->postJSONUpdate();
  }
}

void QueryRendererContext::setWidthHeight(const uint32_t width, const uint32_t height) {
  auto const& limits = global_context_.getGfxContext().getDeviceLimits();

  RUNTIME_EX_ASSERT(
      (width > 0) && (height > 0) && (width <= limits.max_framebuffer_width) &&
          (height <= limits.max_framebuffer_height),
      "Invalid render size: width=" + std::to_string(width) +
          ", height=" + std::to_string(height) + ". Width must be between 1 and " +
          std::to_string(limits.max_framebuffer_width) +
          " inclusive. Height must be between 1 and " +
          std::to_string(limits.max_framebuffer_height) + " inclusive.");

  if ((width != width_) || (height != height_)) {
    width_ = width;
    height_ = height;

    auto fwidth = static_cast<float>(width);
    auto fheight = static_cast<float>(height);
    auto proja = 2.0f / fwidth;
    auto projb = 2.0f / fheight;
    gfx::Math::Matrix2d<float> tmp = {{{proja, 0, 0, projb, -1, -1}}};
    view_proj_matrix_ = tmp;

    if (do_hit_test_) {
      hit_test_buffers_->resize(width, height);
    }

    // Ensure view dependent uniforms update
    force_mark_uniform_update_ = true;
  }
}

void QueryRendererContext::setViewRenderOptions(
    const ViewRenderOptions& view_render_options) {
  view_render_options_ = view_render_options;
}

const gfx::ShaderManager& QueryRendererContext::getShaderManager() const {
  return global_context_.getGfxContext().getShaderManager();
}

SessionId QueryRendererContext::getUserId() const {
  return render_session_key_.getSessionId();
}
WidgetId QueryRendererContext::getWidgetId() const {
  return render_session_key_.getWidgetId();
}
const RenderSessionKey& QueryRendererContext::getRenderSessionKey() const {
  return render_session_key_;
}

const gfx::Math::Matrix2d<float>& QueryRendererContext::getViewProjMatrix() const {
  return view_proj_matrix_.getDataRef();
}

const rapidjson::Value* QueryRendererContext::getJsonCachePtr() const {
  return json_cache_.get();
}

bool QueryRendererContext::hasDataTable(const std::string& table_name) const {
  return vega_elements_->getDataTableMap().contains(table_name);
}

BaseDataTableShPtr QueryRendererContext::getDataTable(
    const std::string& table_name) const {
  RENDER_LOG_SCOPE() << "table_name: " << table_name;
  auto rtn = vega_elements_->getDataTableMap().find(table_name);
  return rtn ? std::dynamic_pointer_cast<BaseDataTable>(*rtn) : nullptr;
}

QueryDataTableSQLJSONShPtr QueryRendererContext::getDataTableSQLJSON(
    const std::string& table_name) const {
  RENDER_LOG_SCOPE() << "table_name: " << table_name;
  auto rtn = vega_elements_->getDataTableMap().find(table_name);
  return rtn ? *rtn : nullptr;
}

std::optional<int> QueryRendererContext::getDataIndex(
    const std::string& table_name) const {
  return vega_elements_->getDataTableMap().get_index(table_name);
}

void QueryRendererContext::resetDataTableStates() {
  RENDER_LOG_SCOPE();
  data_table_queues_->clear();
  for (auto data_table : vega_elements_->getDataTableMap()) {
    data_table->resetStateFlags();
  }
}

bool QueryRendererContext::executeQuery(BaseQueryDataTableSQLJSON& data_table,
                                        const JSONLocation* json_loc,
                                        const OptionalStr& sql_query_override,
                                        const heavyai::InSituFlags insitu_flags) {
  RENDER_LOG_SCOPE();
  auto& query_sql = data_table.getQuerySQL();
  if (query_sql.hasExecutableSql()) {
    auto render_query_runner = getRenderQueryRunner();
    if (render_query_runner) {
      const std::string& sql_query_to_use{
          sql_query_override ? *sql_query_override : query_sql.getSqlQueryStr()};
      auto render_timer = getRenderTimer();
      JSONLocation loc = (json_loc ? *json_loc : getJSONObj(data_table.getJsonPathRef()));
      CHECK(loc.isValid());
      auto render_execute_info =
          render_query_runner->executeQuery(*render_timer,
                                            sql_query_to_use,
                                            &loc,
                                            query_sql.getRenderQueryOptions(),
                                            data_table.getRenderQuerySpecialtyType(),
                                            insitu_flags);

      data_table.setQueryResult(sql_query_to_use, render_execute_info);
      return true;
    }
  } else {
    data_table.clearLayouts();
  }

  return false;
}

QueryDataTableQueues& QueryRendererContext::getDataTableQueues() const {
  return *data_table_queues_;
}

bool QueryRendererContext::hasProjection(const std::string& projection_name) const {
  return vega_elements_->getProjectionMap().contains(projection_name);
}

ProjectionShPtr QueryRendererContext::getProjection(
    const std::string& projection_name) const {
  auto rtn = vega_elements_->getProjectionMap().find(projection_name);
  return rtn ? *rtn : nullptr;
}

bool QueryRendererContext::hasScale(const std::string& scale_name) const {
  return vega_elements_->getScaleMap().contains(scale_name);
}

ScaleShPtr QueryRendererContext::getScale(const std::string& scale_name) const {
  auto rtn = vega_elements_->getScaleMap().find(scale_name);
  return rtn ? *rtn : nullptr;
}

bool QueryRendererContext::isJSONCacheUpToDate(const rapidjson::Pointer& prev_path,
                                               const JSONLocation& obj_location) const {
  if (!json_cache_) {
    return false;
  }

  const rapidjson::Value* cached_val = GetValueByPointer(*json_cache_, prev_path);

  return (cached_val ? (*cached_val == obj_location.getValueRef()) : false);
}

JSONLocation QueryRendererContext::getJSONObj(const rapidjson::Pointer& obj_path) const {
  return JSONLocation(render_session_key_,
                      json_cache_ ? GetValueByPointer(*json_cache_, obj_path) : nullptr,
                      obj_path);
}

bool QueryRendererContext::jsonNeedsUpdating(const JSONLocation& root_json) const {
  if (!json_cache_ || *json_cache_ != root_json.getValueRef() ||
      (vega_elements_->getMetaData() && vega_elements_->getMetaData()->clear_caches)) {
    return true;
  }

  for (auto const& data_table : vega_elements_->getDataTableMap()) {
    if (data_table->jsonNeedsUpdating()) {
      return true;
    }
  }
  return false;
}

const std::vector<BaseMark*> QueryRendererContext::getMarkConfigsForPassIndex(
    const uint16_t pass_index) const {
  std::vector<BaseMark*> rtn;

  auto& mark_vector = vega_elements_->getMarkVector();
  RUNTIME_EX_ASSERT(pass_index <= mark_vector.size(),
                    "Invalid pass index: " + std::to_string(pass_index) +
                        ". There is at most " + std::to_string(mark_vector.size()) +
                        " passes.");

  uint16_t pass_count{0};
  std::string active_accumulator, curr_accumulator;
  for (size_t i = 0; i < mark_vector.size(); ++i) {
    if (mark_vector[i]->hasAccumulator()) {
      curr_accumulator = mark_vector[i]->getAccumulatorScaleName();
      if (active_accumulator != curr_accumulator) {
        pass_count++;
        active_accumulator = curr_accumulator;
      }
    } else {
      active_accumulator = "";
      pass_count++;
    }

    if (pass_count == pass_index + 1) {
      rtn.push_back(mark_vector[i].get());
    } else if (pass_count > pass_index) {
      break;
    }
  }

  return rtn;
}

size_t QueryRendererContext::getNumTotalMarks() const {
  return vega_elements_->getMarkVector().size();
}

void QueryRendererContext::buildMarkShaders(
    BaseMark& mark,
    const MarkGpuResourceSlot slot,
    const std::string& resource_tracking_string,
    gfx::ShaderManager::BuilderUqPtrVector builders) const {
  RENDER_LOG_SCOPE();
  auto* vega_meta_data = vega_elements_->getMetaData();
  if (vega_meta_data &&
      (vega_meta_data->shader_artifacts_to_save != gfx::ShaderArtifactTypeBits::kNone)) {
    for (auto const& builder : builders) {
      getShaderManager().saveArtifacts(
          *builder, vega_meta_data->short_name, vega_meta_data->shader_artifacts_to_save);
    }
  }

  auto caches = getShaderManager().createCacheVector(std::move(builders));
#if LOG_MARK_MATERIAL_STATS
  auto start_time = timer_start();
#endif
  mark.buildMaterials(slot, resource_tracking_string, caches, mark.getUsedGpus());
#if LOG_MARK_MATERIAL_STATS
  g_mark_build_material_stats_map.accumulate(to_string(mark.getType()),
                                             timer_stop_microseconds(start_time));
#endif

  switch (slot) {
    case MarkGpuResourceSlot::kFill:
      mark.fill_shader_caches_.emplace_back(resource_tracking_string, std::move(caches));
      break;
    case MarkGpuResourceSlot::kStroke:
      mark.stroke_shader_caches_.emplace_back(resource_tracking_string,
                                              std::move(caches));
      break;
  }
}

void QueryRendererContext::clearMarkShaders(BaseMark& mark) const {
  for (auto& itr : mark.per_gpu_data_) {
    auto& mark_per_gpu_data = itr.second;
    mark_per_gpu_data.fill_materials.clear();
    mark_per_gpu_data.stroke_materials.clear();
  }
  mark.fill_shader_caches_.clear();
  mark.stroke_shader_caches_.clear();
}

bool QueryRendererContext::drawMark(const uint32_t mark_config_index,
                                    const gfx::DeviceContext& device_context,
                                    gfx::Framebuffer& framebuffer,
                                    const int accumulator_index) const {
  auto& mark_vector = vega_elements_->getMarkVector();
  CHECK(mark_config_index < mark_vector.size());
  auto& mark = *mark_vector[mark_config_index];
  if (mark.is_empty_ || !mark.isVisible()) {
    return false;
  }
  auto itr = mark.per_gpu_data_.find(device_context.getGpuId());
  if (itr == mark.per_gpu_data_.end()) {
    return false;
  }
  VLOG(1) << "Drawing mark (" << to_string(mark.getType()) << ") on gpu "
          << device_context.getGpuId();
  return mark.draw(device_context, itr->second, framebuffer, accumulator_index);
}

HitInfo QueryRendererContext::getIdAt(uint32_t x, uint32_t y, uint32_t pixel_radius) {
  try {
    return hit_test_buffers_->getIdAt(x, y, pixel_radius);
  } catch (...) {
    hit_test_buffers_->releasePbo();
    releaseLineOrPolyBuffers();
    clear();
    std::rethrow_exception(std::current_exception());
  }
}

std::string QueryRendererContext::getVegaDataNameFromIndex(const int8_t data_id) const {
  auto const& data_table_map = vega_elements_->getDataTableMap();
  if (data_id < 0 || static_cast<size_t>(data_id) >= data_table_map.size()) {
    return "";
  }

  return data_table_map[data_id]->getName();
}

void QueryRendererContext::setQueryExecutionParams(
    RenderQueryRunnerUqPtr render_query_runner,
    std::shared_ptr<RenderQueryExecuteTimer>& render_timer) {
  render_query_runner_ = std::move(render_query_runner);
  render_timer_ = render_timer;
  node_index_ = -1;
}

void QueryRendererContext::unsetQueryExecutionParams() {
  render_query_runner_.reset();
  render_timer_.reset();
  node_index_ = -1;

  releaseLineOrPolyBuffers();
}

void QueryRendererContext::releaseLineOrPolyBuffers() {
  // first tell all the data tables to release their shared ptrs to any
  // line/poly buffers, so that they are then ONLY owned by their pools
  for (auto data : vega_elements_->getDataTableMap()) {
    auto line_data = std::dynamic_pointer_cast<SqlQueryLineDataTableJSON>(data);
    if (line_data) {
      line_data->getGpuResources().resetDataPointers();
    } else {
      auto poly_data = std::dynamic_pointer_cast<SqlQueryPolyDataTableJSON>(data);
      if (poly_data) {
        poly_data->getGpuResources().resetDataPointers();
      }
    }
  }

  // then actually inactivate the buffer resources via their pools
  // and forget the buffer pointers completely
  auto& qrm_per_gpu_data = global_context_.getRootPerGpuData();
  for (auto& gpu_data : qrm_per_gpu_data) {
    gpu_data->releaseLineAndPolyBuffers();
  }
}

RefCallbackId QueryRendererContext::subscribeToRefEvent(const RefEventType event_type,
                                                        const RefObjShPtr& event_obj,
                                                        RefEventCallback callback) {
  auto ref_type = event_obj->getRefType();
  const std::string& event_obj_name = event_obj->getNameRef();

  switch (ref_type) {
    case RefType::kData: {
      RUNTIME_EX_ASSERT(hasDataTable(event_obj_name),
                        "QueryRendererContext " + std::string(render_session_key_) +
                            ": Cannot subscribe to event for data table \"" +
                            event_obj_name + "\". The data table does not exist.");
      break;
    }
    case RefType::kScale:
      RUNTIME_EX_ASSERT(hasScale(event_obj_name),
                        "QueryRendererContext " + std::string(render_session_key_) +
                            ": Cannot subscribe to event for scale \"" + event_obj_name +
                            "\". The scale does not exist.");
      break;
    case RefType::kProjection:
      RUNTIME_EX_ASSERT(hasProjection(event_obj_name),
                        "QueryRendererContext " + std::string(render_session_key_) +
                            ": Cannot subscribe to event for projection \"" +
                            event_obj_name + "\". The projection does not exist.");
      break;
    default:
      THROW_RUNTIME_EX("QueryRendererContext " + std::string(render_session_key_) +
                       ": Cannot subscribe to event for an object of type " +
                       to_string(ref_type) + ". " + to_string(ref_type) +
                       " is an unsupported type.");
  }

  return ref_event_callbacks_map_.subscribe(event_type, event_obj, callback);
}

void QueryRendererContext::unsubscribeFromRefEvent(const RefEventType event_type,
                                                   const RefObjShPtr& event_obj,
                                                   const RefCallbackId callback_id) {
  ref_event_callbacks_map_.unsubscribe(event_type, event_obj, callback_id);
}

PropUpdateCallbackId QueryRendererContext::subscribeToPropEvent(
    const std::string& prop_name,
    std::function<void(const uint32_t, const uint32_t)> callback_func) {
  auto callback_id = ++curr_prop_callback_id_;
  if (prop_name == "width") {
    width_.addCallback(callback_id, callback_func);
  } else if (prop_name == "height") {
    height_.addCallback(callback_id, callback_func);
  } else {
    curr_prop_callback_id_--;  // restore callback id
    THROW_RUNTIME_EX("Invalid size_t property: " + prop_name);
  }

  return callback_id;
}

PropUpdateCallbackId QueryRendererContext::subscribeToPropEvent(
    const std::string& prop_name,
    std::function<void(const gfx::Math::Matrix2d<float>&,
                       const gfx::Math::Matrix2d<float>&)> callback_func) {
  auto callback_id = ++curr_prop_callback_id_;
  if (prop_name == "viewProjMatrix") {
    view_proj_matrix_.addCallback(callback_id, callback_func);
  } else {
    curr_prop_callback_id_--;  // restore callback id
    THROW_RUNTIME_EX("Invalid matrix2d property: " + prop_name);
  }

  return callback_id;
}

void QueryRendererContext::unsubscribeFromPropEvent(
    const PropUpdateCallbackId callback_id) {
  if (width_.removeCallback(callback_id) || height_.removeCallback(callback_id) ||
      view_proj_matrix_.removeCallback(callback_id)) {
    return;
  }
  THROW_RUNTIME_EX("Invalid callback id: " + std::to_string(callback_id));
}

std::set<GpuId> QueryRendererContext::getUsedGpus() const {
  std::set<GpuId> rtn;

  for (auto& mark : vega_elements_->getMarkVector()) {
    if (mark->isVisible()) {
      for (auto gpu_id : mark->getUsedGpus()) {
        rtn.insert(gpu_id);
      }
    }
  }

  return rtn;
}

uint32_t QueryRendererContext::getNumRequiredAccumulatorTextures() const {
  return num_required_accumulator_textures_;
}

bool QueryRendererContext::usesPerPixelLinkedLists() const {
  return uses_per_pixel_linked_lists_;
}

bool QueryRendererContext::isReadyForRender(const BaseScale& /*scale*/) const {
  return true;
}

bool QueryRendererContext::isReadyForRender(const BaseMark& /*mark*/) const {
  return true;
}

std::string QueryRendererContext::serializeToVega() const {
  rapidjson::StringBuffer s;
  rapidjson::Writer<rapidjson::StringBuffer> writer(s);

  rapidjson::Document d;
  auto& allocator = d.GetAllocator();

  d.SetObject();
  std::unique_ptr<rapidjson::Value> array_val;
  std::unordered_set<std::string> visited_scales;
  auto const& mark_vector = vega_elements_->getMarkVector();
  std::for_each(mark_vector.begin(),
                mark_vector.end(),
                [&array_val, &allocator, &visited_scales](const BaseMarkUqPtr& mark) {
                  const auto scales = mark->getScaleRefs();
                  for (const auto& scale : scales) {
                    if ((scale->hasDataRef() ||
                         scale->getAccumulatorType() == AccumulatorType::kDensity) &&
                        visited_scales.find(scale->getName()) == visited_scales.end()) {
                      if (!array_val) {
                        array_val =
                            std::make_unique<rapidjson::Value>(rapidjson::kArrayType);
                      }
                      array_val->PushBack(scale->toJSON(&allocator), allocator);
                      visited_scales.insert(scale->getName());
                    }
                  }
                });
  if (array_val) {
    d.AddMember("scales", *array_val, allocator);
  }
  d.Accept(writer);
  return s.GetString();
}

void QueryRendererContext::notifyRefEvent(const RefEventType event_type,
                                          const RefObjShPtr& event_obj) {
  ref_event_callbacks_map_.notify(event_type, event_obj);
}

void QueryRendererContext::notifyRefEvent(const RefEventType event_type,
                                          const JSONRefObject* event_obj) {
  auto const* scale = dynamic_cast<const BaseScale*>(event_obj);
  auto const& scale_map = vega_elements_->getScaleMap();
  auto const& data_table_map = vega_elements_->getDataTableMap();
  if (scale) {
    auto scale = scale_map.find(event_obj->getName());
    CHECK(scale);
    notifyRefEvent(event_type, *scale);
  } else {
    auto const* data_table_json =
        dynamic_cast<const BaseQueryDataTableSQLJSON*>(event_obj);
    RUNTIME_EX_ASSERT(data_table_json,
                      "Cannot find the JSON object \"" + event_obj->getName() +
                          "\" as either a valid scale or data json object. Cannot fire "
                          "reference event type " +
                          to_string(event_type) + ".");
    auto data_table = data_table_map.find(event_obj->getName());
    CHECK(data_table);
    notifyRefEvent(event_type, *data_table);
  }
}

void QueryRendererContext::updateMarksAndBuildShaders() {
  RENDER_LOG_SCOPE();
  for (auto& mark : vega_elements_->getMarkVector()) {
    if (force_mark_uniform_update_) {
      mark->setUniformsDirty();
    }
    mark->update();
  }
  force_mark_uniform_update_ = false;

#if LOG_MEMORY_SUMMARY
  global_context_.logMemorySummary(std::cout);
  std::cout.flush();
#endif
}

}  // namespace QueryRenderer
