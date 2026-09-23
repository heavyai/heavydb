/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/BaseRenderProperty.h"

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Interop/QueryBuffer.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Marks/Utils.h"
#include "QueryRenderer/QueryDataLayout.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Scales/BaseScale.h"

namespace QueryRenderer {

BaseRenderProperty::BaseRenderProperty(const std::string& name,
                                       QueryRendererContext& ctx,
                                       BaseMarkFacade& mark_facade,
                                       const RenderPropertyFlagBits flag_bits)
    : name_{name}
    , get_glsl_func_name_{"get" + name_}
    , mark_facade_{mark_facade}
    , flag_bits_{flag_bits}
    , decimal_exp_scale_{0}
    , vbo_attr_name_{""}
    , real_col_id_{-1}
    , real_table_id_{-1}
    , real_col_type_{kNULLT}
    , real_col_encoding_{EncodingType::kENCODING_NONE}
    , per_gpu_data_{}
    , vbo_init_type_{VboInitType::kUndefined}
    , data_{nullptr}
    , ctx_{ctx}
    , in_type_{nullptr}
    , out_type_{nullptr}
    , scale_config_{nullptr}
    , scale_ref_subscription_id_{0}
    , data_ref_subscription_id_{0} {}

BaseRenderProperty::~BaseRenderProperty() {
  unsubscribeFromDataEvent();
  unsubscribeFromScaleEvent(scale_);
}

void BaseRenderProperty::initializeFromJSONObj(const JSONLocation& obj_loc,
                                               const BaseDataTableShPtr& data) {
  if (obj_loc.isObject()) {
    bool has_data = false;
    bool has_scale = false, update_scale = false;

    auto const scale_loc = obj_loc.getMember(JSONSchema_v1::Marks::kScaleProp);
    if (scale_loc.isValid()) {
      if (!ctx_.isJSONCacheUpToDate(scale_json_path_, scale_loc)) {
        RUNTIME_EX_ASSERT(
            any_bits_set(flag_bits_ & RenderPropertyFlagBits::kUseScale),
            RapidJSONUtils::createJsonParseError(
                scale_loc,
                "Render property \"" + name_ + "\" does not support scale references."));
        update_scale = true;
      } else if (scale_ && ctx_.getScale(scale_->getName()) != scale_) {
        update_scale = true;
      }

      scale_json_path_ = scale_loc.getPathRef();
      has_scale = true;
    } else {
      // need to clear out the scale_json_path_
      scale_json_path_ = rapidjson::Pointer();
    }

    auto const field_loc = obj_loc.getMember(JSONSchema_v1::Marks::kFieldProp);
    if (field_loc.isValid()) {
      // need to clear out the value path
      value_json_path_ = rapidjson::Pointer();

      if (!ctx_.isJSONCacheUpToDate(field_json_path_, field_loc) || data != data_) {
        RUNTIME_EX_ASSERT(
            data != nullptr,
            RapidJSONUtils::createJsonParseError(
                field_loc,
                "A data reference for the mark is not defined. Cannot access \"" +
                    std::string(JSONSchema_v1::Marks::kFieldProp) + "\"."));
        RUNTIME_EX_ASSERT(field_loc.isString(),
                          RapidJSONUtils::createJsonParseError(
                              field_loc, "Property must be a string."));

        internalInitFromData(field_loc.getString(), data, has_scale, update_scale);
      }

      field_json_path_ = field_loc.getPathRef();

      has_data = true;
    } else {
      // need to clear out the field path
      clearFieldPath();
    }

    auto const value_loc = obj_loc.getMember(JSONSchema_v1::Marks::kValueProp);
    if (value_loc.isValid()) {
      if (!ctx_.isJSONCacheUpToDate(value_json_path_, value_loc)) {
        initValueFromJSONObj(value_loc, has_scale, !has_data);
      } else if (!has_data && vbo_init_type_ == VboInitType::kFromDataRef) {
        if (!has_scale) {
          resetTypes();
        } else {
          resetTypes(true, false);
        }
        validateValue(has_scale, value_loc.getPathRef());
      }
      value_json_path_ = value_loc.getPathRef();

      has_data = true;
    } else {
      resetValue();
      value_json_path_ = rapidjson::Pointer();
    }

    if (update_scale) {
      // initialize according to the scale
      clearScalePtr();

      // NOTE: not resetting the in/out types as they will be set in
      // the following initScaleFromJSONobj() call

      initScaleFromJSONObj(scale_loc);
      validateScale();
    }

    if (!has_scale) {
      // need some value source, either by "field" or by "value" if there's no scale
      // reference
      RUNTIME_EX_ASSERT(has_data,
                        RapidJSONUtils::createJsonParseError(
                            obj_loc,
                            "Invalid mark property object. Must contain a data reference "
                            "via a \"field\" property or a \"value\" property."));

      clearScalePtr();
      if (vbo_init_type_ == VboInitType::kFromScaleRef) {
        resetTypes();
        validateValue(
            false, value_loc.isValid() ? value_loc.getPathRef() : obj_loc.getPathRef());
      } else if (vbo_init_type_ == VboInitType::kFromDataRef) {
        if (!out_type_ || !in_type_ || !validateOutType(out_type_)) {
          if (any_bits_set(flag_bits_ & RenderPropertyFlagBits::kFlexibleType) ||
              !in_type_) {
            out_type_ = in_type_;
          } else {
            out_type_ = createDefaultType();
          }
          notifyChanged(ChangeType::kStructure);
        }
      } else {
        resetTypes(false, true);
      }
    }

    initFromJSONObj(obj_loc);

  } else {
    // need to clear out the object paths
    clearFieldPath();
    clearScalePtr();

    value_json_path_ = rapidjson::Pointer();
    scale_json_path_ = rapidjson::Pointer();
    initValueFromJSONObj(obj_loc, false, true);
  }
  json_path_ = obj_loc.getPathRef();
}

bool BaseRenderProperty::initializeFromData(const std::string& attr_name,
                                            const BaseDataTableShPtr& data) {
  return internalInitFromData(attr_name, data, false, false);
}

bool BaseRenderProperty::internalInitFromData(const std::string& attr_name,
                                              const BaseDataTableShPtr& data,
                                              const bool has_scale,
                                              const bool updating_scale) {
  RUNTIME_EX_ASSERT(data != nullptr,
                    std::string(*this) + ": Cannot initialize mark property " + name_ +
                        " from data. A valid data reference hasn't been initialized.");

  bool data_changed = data != data_;
  if (data_changed) {
    unsubscribeFromDataEvent();
  }

  data_ = data;
  vbo_attr_name_ = attr_name;

  RUNTIME_EX_ASSERT(initBuffers(data_->getAttributeDataBuffers(attr_name)),
                    "Cannot initialize mark property " + name_ +
                        " from data. The attr \"" + attr_name +
                        "\" does not exist in the data buffers.");
  vbo_init_type_ = VboInitType::kFromDataRef;

  auto [in_changed, out_changed] = initTypeFromBuffer(has_scale);

  int real_col_id = -1, real_table_id = -1;
  SQLTypes real_col_type = kNULLT;
  EncodingType real_col_encoding = EncodingType::kENCODING_NONE;
  auto layout = getDataLayoutForAttribute(data_, vbo_attr_name_);
  if (layout && layout->isKnownAlias(vbo_attr_name_)) {
    auto const& aliasinfo = layout->getAliasInfo(vbo_attr_name_);
    real_table_id = aliasinfo.table_id;
    real_col_id = aliasinfo.col_id;
    real_col_type = aliasinfo.type_info.get_type();
    real_col_encoding = aliasinfo.type_info.get_compression();
  }
  bool real_col_changed =
      (real_col_id != real_col_id_ || real_table_id != real_table_id_);
  real_col_id_ = real_col_id;
  real_table_id_ = real_table_id;
  bool real_col_geo_type_changed =
      IS_GEO(real_col_type) && (real_col_type != real_col_type_);
  bool real_col_geo_encoding_changed =
      IS_GEO(real_col_type) && (real_col_encoding != real_col_encoding_);
  real_col_type_ = real_col_type;
  real_col_encoding_ = real_col_encoding;

  if (real_col_geo_type_changed) {
    // force shader update as we may need to switch between vert and mesh
    notifyChanged(ChangeType::kStructure);
  } else if (real_col_geo_encoding_changed) {
    // force uniform update to re-capture compression bits
    notifyChanged(ChangeType::kValues);
  }

  if (in_changed || out_changed) {
    if (in_changed && has_scale && !updating_scale && scale_) {
      // need to make sure our scale reference is properly
      // adjusted for a possible new data type
      updateScalePtr(scale_);
    }
    notifyChanged(ChangeType::kData);
  } else if (real_col_changed && has_scale && !updating_scale && scale_ &&
             scale_->getPrimaryDomainDataType() == QueryDataType::STRING) {
    // we've got a scenario where the vbo_attr_name_ hasn't changed between queries, but
    // it is an alias and the column it references has changed and it is a
    // dictionary-encoded string column. The scale in this case should be using strings
    // for its domain, so the scale ref needs to be rebuilt to account for the new column
    // so the dictionary-encoded values for the new column are rebuilt
    updateScalePtr(scale_);
  }

  RENDER_LOG() << "data_changed=" << data_changed;
  if (data_changed) {
    // setup callbacks for data updates
    CHECK_EQ(data_ref_subscription_id_, RefCallbackId(0));
    auto data_json = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data);
    CHECK(data_json);
    data_ref_subscription_id_ =
        ctx_.subscribeToRefEvent(RefEventType::kAll, data_json, [this](auto&&... args) {
          return this->dataRefUpdateCB(std::forward<decltype(args)>(args)...);
        });

    notifyChanged(ChangeType::kData);
  }

  return in_changed || out_changed;
}

int BaseRenderProperty::size(const GpuId& gpu_id) const {
  auto itr = per_gpu_data_.find(gpu_id);
  if (itr != per_gpu_data_.end() && !itr->second.vbo.expired()) {
    auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
    return itr->second.vbo.lock()->numVertices(data ? data->getVboQueryDataLayout()
                                                    : nullptr);
  }
  return 0;
}

bool BaseRenderProperty::hasVboPtr() const {
  for (auto const& itr : per_gpu_data_) {
    if (!itr.second.vbo.expired()) {
      return true;
    }
  }
  return false;
}

bool BaseRenderProperty::hasVboPtr(const GpuId& gpu_id) const {
  auto itr = per_gpu_data_.find(gpu_id);

  return (itr != per_gpu_data_.end() && !itr->second.vbo.expired());
}

QueryVertexBufferShPtr BaseRenderProperty::getVboPtr(const GpuId& gpu_id) const {
  auto itr = per_gpu_data_.find(gpu_id);
  if (itr != per_gpu_data_.end()) {
    return itr->second.vbo.lock();
  }

  return nullptr;
}

QueryVertexBufferShPtr BaseRenderProperty::getVboPtr() const {
  auto itr = per_gpu_data_.begin();
  if (itr != per_gpu_data_.end()) {
    return itr->second.vbo.lock();
  }

  return nullptr;
}

bool BaseRenderProperty::hasSsboPtr() const {
  for (auto const& itr : per_gpu_data_) {
    if (!itr.second.ssbo.expired()) {
      return true;
    }
  }
  return false;
}

bool BaseRenderProperty::hasSsboPtr(const GpuId& gpu_id) const {
  auto itr = per_gpu_data_.find(gpu_id);

  return (itr != per_gpu_data_.end() && !itr->second.ssbo.expired());
}

QueryShaderStorageBufferShPtr BaseRenderProperty::getSsboPtr(const GpuId& gpu_id) const {
  auto itr = per_gpu_data_.find(gpu_id);
  if (itr != per_gpu_data_.end()) {
    return itr->second.ssbo.lock();
  }

  return nullptr;
}

QueryShaderStorageBufferShPtr BaseRenderProperty::getSsboPtr() const {
  auto itr = per_gpu_data_.begin();
  if (itr != per_gpu_data_.end()) {
    return itr->second.ssbo.lock();
  }

  return nullptr;
}

const gfx::TypeGLSLShPtr& BaseRenderProperty::getInTypeGLSL() const {
  RUNTIME_EX_ASSERT(in_type_ != nullptr,
                    std::string(*this) + " getInTypeGLSL(): input type for \"" + name_ +
                        "\" is uninitialized.");

  return in_type_;
}

const gfx::TypeGLSLShPtr& BaseRenderProperty::getOutTypeGLSL() const {
  if (scale_config_) {
    return scale_config_->getRangeTypeGLSL();
  } else if (scale_) {
    return scale_->getRangeTypeGLSL();
  }

  RUNTIME_EX_ASSERT(out_type_ != nullptr,
                    std::string(*this) + " getOutTypeGLSL(): input type for \"" + name_ +
                        "\" is uninitialized.");

  return out_type_;
}

bool BaseRenderProperty::isCoord() const {
  return any_bits_set(RenderPropertyFlagBits::kIsCoord & flag_bits_);
}

bool BaseRenderProperty::usesBDA() const {
  return any_bits_set(RenderPropertyFlagBits::kUseBDA & flag_bits_);
}

void BaseRenderProperty::addToPrimitiveAssemblyAttrInfo(
    const GpuId& gpu_id,
    gfx::PrimitiveAssemblyAttrInfo& attr_info) const {
  auto itr = per_gpu_data_.find(gpu_id);

  CHECK(itr != per_gpu_data_.end()) << std::string(*this);
  RUNTIME_EX_ASSERT(!itr->second.vbo.expired(),
                    std::string(*this) +
                        " addToPrimitiveAssemblyAttrInfo(): A vertex buffer is not "
                        "defined. Cannot add vbo attrs to "
                        "vbo->shader attr map.");

  auto const* vertex_buffer = itr->second.vbo.lock()->unmapForDraw();
  auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);

  auto layout = data ? data->getVboBufferLayout() : nullptr;
  if (attr_info.vbo_and_layout.vertex_buffer) {
    // VBO and layout must be the same for all attrs in this info
    CHECK_EQ(attr_info.vbo_and_layout.vertex_buffer, vertex_buffer);
    CHECK_EQ(attr_info.vbo_and_layout.buffer_layout.get(), layout.get());
  } else {
    attr_info.vbo_and_layout.vertex_buffer = vertex_buffer;
    attr_info.vbo_and_layout.buffer_layout = layout;
  }
  attr_info.attr_pairs.emplace_back(vbo_attr_name_, name_);
}

void BaseRenderProperty::setDecimalScaleUniformAttribute(
    gfx::Material& active_material) const {
  if (decimal_exp_scale_ > 0) {
    active_material.setUniformAttribute<uint64_t>(name_ + "_ExpScale",
                                                  decimal_exp_scale_);
  }
}

bool BaseRenderProperty::initGpuResources(const std::vector<GpuId>& add_gpus,
                                          const std::vector<GpuId>& remove_gpus) {
  bool reset = false;
  auto const& global_gpu_data_map = ctx_.getGlobalContext().getRootPerGpuData();
  for (auto const& gpu_id : add_gpus) {
    if (per_gpu_data_.find(gpu_id) == per_gpu_data_.end()) {
      auto qrm_itr = global_gpu_data_map.find(gpu_id);
      CHECK(qrm_itr != global_gpu_data_map.end());
      auto [per_gpu_itr, emplace_succeeded] =
          per_gpu_data_.try_emplace(gpu_id, **qrm_itr);
      CHECK(emplace_succeeded);
      auto& gpu_data = per_gpu_itr->second;
      if (data_) {
        if (data_->hasAttribute(vbo_attr_name_)) {
          QueryLayoutBufferShPtr buffer =
              data_->getAttributeDataBuffer(gpu_id, vbo_attr_name_).lock();
          switch (buffer->getQueryBufferType()) {
            case QueryBufferType::kVertex:
              CHECK(buffer->getBufferWrapper()->getResourceType() ==
                    gfx::ResourceType::kVertexBuffer);
              gpu_data.vbo = std::dynamic_pointer_cast<QueryVertexBuffer>(buffer);
              CHECK(!gpu_data.vbo.expired());
              break;
            case QueryBufferType::kStorage:
              CHECK(any_bits_set(buffer->getBufferWrapper()->getUsageBits() &
                                 gfx::BufferUsageBits::kStorageBufferBit));
              gpu_data.ssbo = std::dynamic_pointer_cast<QueryShaderStorageBuffer>(buffer);
              CHECK(!gpu_data.ssbo.expired());
              break;
            default:
              CHECK(false) << "Unsupported QueryBuffer type "
                           << buffer->getQueryBufferType() << " for render properties.";
          }
        } else if (!isUndefined()) {
          // could just be empty data, so only reset if it's not empty data
          // TODO(croot): add an API at the data level to query whether it's empty or not
          // For polygons that could mean a cached ibo/vbo exists, but the ubo would
          // be empty. If that is the preferred route, then need to remove
          // BaseMark::_isEmpty and possibly remove BaseRenderProperty::isUndefined()
          reset = true;
          break;
        }
      }
    }
  }

  for (auto const& gpu_id : remove_gpus) {
    per_gpu_data_.erase(gpu_id);
  }

  if (reset) {
    // reaching this code likely means that there was an attempt to initGpuResources
    // before a complete update was performed, so clearing everything out first.
    LOG(WARNING) << "Possible sync issue when updating render property " << printInfo();
    clearFieldPath();
    clearScalePtr();
    resetTypes();
  }

  return !reset;
}

void BaseRenderProperty::notifyChanged(ChangeType change_type) {
  mark_facade_.notifyChanged(change_type);
}

std::string BaseRenderProperty::printInfo() const {
  return "(property name: " + name_ + ", vbo attr name: \"" + vbo_attr_name_ +
         "\", parent mark type: " + to_string(mark_facade_.getType()) + ") " +
         std::string(ctx_.getRenderSessionKey());
}

bool BaseRenderProperty::initBuffers(
    const std::map<GpuId, QueryLayoutBufferWkPtr>& buffer_map) {
  RENDER_LOG_SCOPE();
  std::vector<GpuId> add_gpus;
  for (auto const& itr : buffer_map) {
    if (per_gpu_data_.find(itr.first) == per_gpu_data_.end()) {
      add_gpus.push_back(itr.first);
    }
  }
  std::vector<GpuId> remove_gpus;
  for (auto const& kv : per_gpu_data_) {
    if (buffer_map.find(kv.first) == buffer_map.end()) {
      remove_gpus.push_back(kv.first);
    }
  }

  auto rtn = initGpuResources(add_gpus, remove_gpus);
  if (!rtn) {
    return false;
  }
  CHECK(buffer_map.size() == per_gpu_data_.size());

  if (!buffer_map.size()) {
    return true;
  }

  auto first_item = buffer_map.begin();
  CHECK(!first_item->second.expired());
  auto query_layout_buffer = first_item->second.lock();
  auto query_buffer_type = query_layout_buffer->getQueryBufferType();

  switch (query_buffer_type) {
    case QueryBufferType::kVertex: {
      QueryVertexBufferShPtr qvbo;
      for (auto const& itr : buffer_map) {
        auto my_itr = per_gpu_data_.find(itr.first);
        CHECK(my_itr != per_gpu_data_.end() && !itr.second.expired());
        auto resource = itr.second.lock();
        CHECK(query_buffer_type == resource->getQueryBufferType());
        qvbo = std::dynamic_pointer_cast<QueryVertexBuffer>(resource);
        CHECK(qvbo);
        my_itr->second.vbo = qvbo;
      }
      break;
    }
    case QueryBufferType::kStorage: {
      QueryShaderStorageBufferShPtr qssbo;
      for (auto const& itr : buffer_map) {
        auto my_itr = per_gpu_data_.find(itr.first);
        CHECK(my_itr != per_gpu_data_.end() && !itr.second.expired());
        auto resource = itr.second.lock();
        CHECK(query_buffer_type == resource->getQueryBufferType());
        qssbo = std::dynamic_pointer_cast<QueryShaderStorageBuffer>(resource);
        my_itr->second.ssbo = qssbo;
      }
      break;
    }

    default:
      CHECK(false) << "Unsupported QueryBuffer type " << query_buffer_type
                   << " for render properties.";
  }

  return true;
}

void BaseRenderProperty::dataRefUpdateCB(RefEventType ref_event_type,
                                         const RefObjShPtr& ref_obj) {
  auto data = std::dynamic_pointer_cast<BaseDataTable>(ref_obj);
  CHECK(data);
  switch (ref_event_type) {
    case RefEventType::kUpdate:
      CHECK(data == data_);
    // pass thru to the REPLACE code
    case RefEventType::kReplace: {
      CHECK(vbo_attr_name_.size());
      auto const* sql_data = dynamic_cast<BaseQueryDataTableSQLJSON*>(data.get());
      // TODO(croot): do an improved check here for when we're doing a distributed render?
      // The getResultSet() check here verifies that the query was actually performed
      // on this node, but the chris/vega_xforms branch has some slightly improved ways
      // of determining distributed queries that might be better here.
      if ((!sql_data || sql_data->getResultSet()) && data->hasData() &&
          !data->hasAttribute(vbo_attr_name_)) {
        if (any_bits_set(flag_bits_ & RenderPropertyFlagBits::kResetOnEmptyDataUpdate)) {
          clearReferences();
        } else {
          THROW_RUNTIME_EX(std::string(*this) + " The data table " +
                           RapidJSONUtils::getPointerPath(ref_obj->getJsonPathRef()) +
                           " does not contain the attribute \"" + vbo_attr_name_ +
                           "\". Cannot update render property from a " +
                           to_string(ref_event_type) + " data event.");
        }
      } else {
        if (internalInitFromData(vbo_attr_name_,
                                 data,
                                 scale_config_ != nullptr || scale_ != nullptr,
                                 (scale_ ? scale_->isMarkedForDeletion() : false))) {
          notifyChanged(ChangeType::kStructure);
        }
      }
      break;
    }
    case RefEventType::kRemove:
      THROW_RUNTIME_EX(
          std::string(*this) + ": Error, data table " + ref_obj->getName() +
          " has been removed but is still being referenced by this render property.")
      break;
    default:
      THROW_RUNTIME_EX(std::string(*this) + ": Ref event type: " +
                       std::to_string(static_cast<int>(ref_event_type)) +
                       " isn't currently supported for data reference updates.");
      break;
  }
}

void BaseRenderProperty::clearFieldPath() {
  field_json_path_ = rapidjson::Pointer();
  clearDataPtr();
  vbo_attr_name_ = "";
  decimal_exp_scale_ = 0;
}

void BaseRenderProperty::clearDataPtr() {
  unsubscribeFromDataEvent();
  data_ = nullptr;
}

void BaseRenderProperty::clearScalePtr() {
  if (scale_config_ || scale_) {
    notifyChanged(ChangeType::kStructure);
  }

  unsubscribeFromScaleEvent(scale_);
  clearAccumulatorFromScale(scale_);
  scale_config_ = nullptr;
  scale_ = nullptr;
}

void BaseRenderProperty::clearScalePtrForReplacement(const ScaleShPtr& scale) {
  unsubscribeFromScaleEvent(scale);
  clearAccumulatorFromScale(scale);
  notifyChanged(ChangeType::kStructure);
}

bool BaseRenderProperty::checkAccumulator(const ScaleShPtr& scale) {
  bool scale_accumulation = scale->hasAccumulator();
  RUNTIME_EX_ASSERT(
      any_bits_set(flag_bits_ & RenderPropertyFlagBits::kAllowAccumulator) ||
          !scale_accumulation,
      std::string(*this) + " The scale \"" + scale->getName() +
          "\" is an accumulator scale but " + to_string(mark_facade_.getType()) +
          " mark property \"" + name_ + "\" doesn't accept accumulator scales.");
  return scale_accumulation;
}

void BaseRenderProperty::setAccumulatorFromScale(const ScaleShPtr& scale) {
  if (scale && scale->hasAccumulator()) {
    // mark will ensure it doesn't already have a scale
    mark_facade_.setAccumulatorFromScale(scale, scale_config_);
  }
}

void BaseRenderProperty::clearAccumulatorFromScale(const ScaleShPtr& scale) {
  if (scale) {
    mark_facade_.clearAccumulatorFromScale(scale);
  }
}

void BaseRenderProperty::unsubscribeFromScaleEvent(const ScaleShPtr& scale) {
  if (scale && scale_ref_subscription_id_) {
    ctx_.unsubscribeFromRefEvent(RefEventType::kAll, scale, scale_ref_subscription_id_);
    scale_ref_subscription_id_ = 0;
  }
}

void BaseRenderProperty::unsubscribeFromDataEvent() {
  if (data_ && data_ref_subscription_id_) {
    auto data_json = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
    CHECK(data_json);
    ctx_.unsubscribeFromRefEvent(
        RefEventType::kAll, data_json, data_ref_subscription_id_);
    data_ref_subscription_id_ = 0;
  }
}

void BaseRenderProperty::updateScalePtr(const ScaleShPtr& scale) {
  scale_ = scale;
  setAccumulatorFromScale(scale);
}

}  // namespace QueryRenderer