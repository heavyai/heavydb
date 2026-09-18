/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/BaseMark.h"

#include <algorithm>
#include <optional>
#include <regex>
#include <set>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Render/GeoCountResources.h"
#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/Resources/ShaderBlockLayout.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Cache/RowIdHitTestOffsetData.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Marks/GlslPropWriters.h"
#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderProperty.h"  // For setColorConvertSubroutines
#include "QueryRenderer/Marks/RenderPropertyBufferStateHandler.h"
#include "QueryRenderer/Marks/RenderPropertyContainer.h"
#include "QueryRenderer/Marks/Utils.h"
#include "QueryRenderer/Projections/Projection.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Scales/ScaleAccumState.h"

namespace QueryRenderer {

JSONLocation BaseMark::initPropFromJSONObj(const QueryRendererContext* ctx,
                                           const BaseDataTableShPtr& data,
                                           const bool data_changed,
                                           const JSONLocation& prop_loc,
                                           BaseRenderProperty* prop,
                                           const rapidjson::Pointer& properties_json_path,
                                           ValidateFunc validate_type_func,
                                           JSONParseCBFunc post_update_func,
                                           JSONParseCBFunc post_data_update_func,
                                           JSONParseCBFunc post_up_to_date_func,
                                           JSONParseCBFunc post_empty_func) {
  auto prop_name = prop->getName();
  auto const prop_item_loc = prop_loc.getMember(prop_name);
  if (prop_item_loc.isValid()) {
    auto const prev_path = prop->getJsonPath();
    auto const prop_json_path = prop_item_loc.getPathRef();
    if (!ctx->isJSONCacheUpToDate(prev_path, prop_item_loc)) {
      if (validate_type_func) {
        validate_type_func(prop_name, prop_item_loc);
      }
      prop->initializeFromJSONObj(prop_item_loc, data);
      if (post_update_func) {
        post_update_func(prop_item_loc);
      }
    } else if (data_changed) {
      prop->initializeFromJSONObj(prop_item_loc, data);
      if (post_data_update_func) {
        post_data_update_func(prop_item_loc);
      }
    } else {
      prop->updateJsonPath(prop_json_path);
      if (post_up_to_date_func) {
        post_up_to_date_func(prop_item_loc);
      }
    }
  } else {
    // clear the property of all scale/data references
    prop->clear();
    if (post_empty_func) {
      post_empty_func(prop_loc);
    }
  }

  return prop_item_loc;
}

BaseMark::ValidateFunc BaseMark::validateNumPropFunc(BaseMark& mark) {
  return [](const std::string& prop_name, const JSONLocation& json_loc) {
    CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
    RUNTIME_EX_ASSERT(
        (json_loc.isObject() || json_loc.isNumber()),
        RapidJSONUtils::createJsonParseError(
            json_loc,
            "\"" + prop_name +
                "\" mark property must be a scale/data reference or a number."));
  };
}

BaseMark::ValidateFunc BaseMark::validateColorPropFunc(BaseMark& mark) {
  return [](const std::string& prop_name, const JSONLocation& json_loc) {
    CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
    RUNTIME_EX_ASSERT((json_loc.isObject() || json_loc.isString()) || json_loc.isInt() ||
                          json_loc.isUint(),
                      RapidJSONUtils::createJsonParseError(
                          json_loc,
                          "\"" + prop_name +
                              "\" mark color property must be a scale/data reference, a "
                              "string, or a color packed into a 32-bit int/uint."));
  };
}

BaseMark::ValidateFunc BaseMark::validateEnumPropFunc(BaseMark& mark) {
  return [](const std::string& prop_name, const JSONLocation& json_loc) {
    CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
    RUNTIME_EX_ASSERT(
        json_loc.isInt() || json_loc.isUint64() || json_loc.isString(),
        RapidJSONUtils::createJsonParseError(json_loc,
                                             "\"" + prop_name +
                                                 "\" mark enum property must be an enum "
                                                 "value (i.e. an int or a string)."));
  };
}

BaseMark::ValidateFunc BaseMark::validateBoolPropFunc(BaseMark& mark) {
  return [](const std::string& prop_name, const JSONLocation& json_loc) {
    CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
    RUNTIME_EX_ASSERT(json_loc.isBool() || json_loc.isInt() || json_loc.isUint64() ||
                          json_loc.isString(),
                      RapidJSONUtils::createJsonParseError(
                          json_loc,
                          "\"" + prop_name +
                              "\" mark bool property must be a boolean value (i.e. a "
                              "boolean, int, or string)."));
  };
}

class BaseMarkPropFacadeImpl : public BaseRenderProperty::BaseMarkFacade {
 public:
  explicit BaseMarkPropFacadeImpl(BaseMark& base_mark) : base_mark_{base_mark} {}
  ~BaseMarkPropFacadeImpl() override = default;

  GeomType getType() override { return base_mark_.getType(); }
  void notifyChanged(BaseRenderProperty::ChangeType type) override {
    base_mark_.propChangedCallback(type);
  }
  void setAccumulatorFromScale(const ScaleShPtr scale,
                               const ScaleRefShPtr scale_ref) override {
    base_mark_.setAccumulatorScale(scale, scale_ref);
  }
  void clearAccumulatorFromScale(const ScaleShPtr scale) override {
    base_mark_.clearAccumulatorScale(scale);
  }
  BaseDataTableShPtr getDataPtr() override { return base_mark_.getDataPtr(); }

 private:
  BaseMark& base_mark_;
};

BaseMark::BaseMark(GeomType geom_type,
                   QueryRendererContext& ctx,
                   const JSONLocation& obj_loc,
                   DataOutputFormat data_output_format,
                   bool must_use_data_ref)
    : type_{geom_type}
    , invalid_key_{std::numeric_limits<int64_t>::max()}
    , data_{nullptr}
    , per_gpu_data_{}
    , ctx_{ctx}
    , data_output_format_{data_output_format}
    , is_empty_{true}
    , shader_dirty_{true}
    , props_dirty_{true}
    , pipelines_dirty_{true}
    , uniforms_dirty_{true}
    , prop_mark_facade_{std::make_unique<BaseMarkPropFacadeImpl>(*this)}
    , is_visible_{false}
    , can_render_with_mesh_shader_{ctx.getGlobalContext().canUseSlabAddressTable() &&
                                   ctx.getGlobalContext().canUseMeshShaders()}
    , use_mesh_shader_{false} {
  initFromJSONObj(obj_loc, must_use_data_ref);
}

BaseMark::~BaseMark() {
  ctx_.clearMarkShaders(*this);
  unsubscribeFromProjectionEvent();
  unsubscribeFromDataEvent();
}

void BaseMark::initIds(const bool data_changed) {
  auto [ids_size_changed, id_values_changed] =
      render_props_->initIds(data_changed, data_);
  if (ids_size_changed) {
    setShaderDirty();
  }
  if (id_values_changed) {
    setPropsDirty();
  }
}

const std::unordered_set<BaseDataTableShPtr> BaseMark::getDataRefs() const {
  std::unordered_set<BaseDataTableShPtr> rtn;
  if (data_) {
    rtn.insert(data_);
  }
  auto const used_props = getUsedProps();
  for (auto const& prop : used_props) {
    auto const scale_ref = prop->getScale();
    if (scale_ref) {
      auto const data_refs = scale_ref->getDataRefs();
      for (auto const& data_ref : data_refs) {
        rtn.insert(data_ref);
      }
    }
  }
  return rtn;
}

const std::unordered_set<ScaleShPtr> BaseMark::getScaleRefs() const {
  std::unordered_set<ScaleShPtr> rtn;
  auto const used_props = getUsedProps();
  for (auto const& prop : used_props) {
    auto const scale_ref = prop->getScale();
    if (scale_ref) {
      rtn.insert(scale_ref);
    }
  }
  return rtn;
}

bool BaseMark::hasAccumulator() const {
  return !active_accumulator_.expired();
}

std::string BaseMark::getAccumulatorScaleName() const {
  auto scale = active_accumulator_.lock();
  if (scale) {
    return scale->getName();
  }
  return "";
}

ScaleShPtr BaseMark::getAccumulatorScale() const {
  return active_accumulator_.lock();
}

void BaseMark::setAccumulatorScale(const ScaleShPtr& scale,
                                   const ScaleRefShPtr& scale_ref) {
  CHECK(scale);
  RUNTIME_EX_ASSERT(active_accumulator_.expired() ||
                        active_accumulator_.lock()->getName() == scale->getName(),
                    std::string(*this) + ": An accumulator scale named: \"" +
                        active_accumulator_.lock()->getName() +
                        "\" is already set active on the mark. Only one accumulator "
                        "scale per mark is currently supported.");

  active_accumulator_ = scale;
}

void BaseMark::clearAccumulatorScale(const ScaleShPtr& scale) {
  CHECK(scale);
  // TODO(croot): what if there are two accumulator scales in the
  // same mark properties?
  if (scale->getName() == getAccumulatorScaleName()) {
    active_accumulator_.reset();
  }
}

bool BaseMark::hasProjection() const {
  return (!active_projection_.expired());
}

ProjectionShPtr BaseMark::getProjection() const {
  return active_projection_.lock();
}

void BaseMark::setProjection(const std::string& projection_name,
                             ProjectionShPtr projection) {
  CHECK(projection);
  RUNTIME_EX_ASSERT(active_projection_.expired() ||
                        active_projection_.lock()->getName() == projection->getName(),
                    std::string(*this) + ": A projection named: \"" +
                        active_projection_.lock()->getName() +
                        "\" is already set active on the mark. Only one projection per "
                        "mark is currently supported.");
  active_projection_ = projection;
  subscribeToProjectionEvent(projection);
  setShaderDirty();
}

bool BaseMark::initFromJSONObj(const JSONLocation& obj_loc, bool must_use_data_ref) {
  bool data_ref_changed = false;
  CHECK(obj_loc.isValid()) << RapidJSONUtils::getPointerPath(obj_loc.getPathRef());
  RUNTIME_EX_ASSERT(obj_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        obj_loc, "Definition for marks must be an object."));

  auto const from_loc = obj_loc.getMember(JSONSchema_v1::Marks::kFromProp);
  BaseDataTableShPtr curr_data;
  if (from_loc.isValid()) {
    if (!ctx_.isJSONCacheUpToDate(data_ptr_json_path_, from_loc)) {
      RUNTIME_EX_ASSERT(from_loc.isObject(),
                        RapidJSONUtils::createJsonParseError(
                            from_loc, "Mark data reference must be an object."));

      auto const data_loc = from_loc.getMember(JSONSchema_v1::Marks::kDataProp);
      RUNTIME_EX_ASSERT(
          data_loc.isValid() && data_loc.isString(),
          RapidJSONUtils::createJsonParseError(
              data_loc.isValid() ? data_loc : from_loc,
              "Mark data reference must contain a \"" +
                  std::string(JSONSchema_v1::Marks::kDataProp) + "\" string property."));

      curr_data = ctx_.getDataTable(data_loc.getString());
      RUNTIME_EX_ASSERT(curr_data,
                        RapidJSONUtils::createJsonParseError(
                            data_loc,
                            "Data reference \"" + std::string(data_loc.getString()) +
                                "\" does not exist in the vega."));

      RUNTIME_EX_ASSERT(curr_data->getInputFormat() != DataInputFormat::kSourced,
                        RapidJSONUtils::createJsonParseError(
                            data_loc,
                            "\"" + std::string(data_loc.getString()) + "\" is a " +
                                to_string(curr_data->getInputFormat()) +
                                " data table. Referencing tables of type " +
                                to_string(curr_data->getInputFormat()) +
                                " by marks is not currently supported."));

      RUNTIME_EX_ASSERT(
          curr_data->getOutputFormat() == data_output_format_,
          RapidJSONUtils::createJsonParseError(
              data_loc,
              "Mark data table reference is the wrong type. It is a " +
                  to_string(curr_data->getOutputFormat()) + " table but a " +
                  to_string(data_output_format_) + " table is required."));

      data_ref_changed = (curr_data != data_);
    }
    data_ptr_json_path_ = from_loc.getPathRef();
  } else {
    data_ref_changed = (data_ != nullptr);
    RUNTIME_EX_ASSERT(
        !must_use_data_ref,
        RapidJSONUtils::createJsonParseError(
            obj_loc,
            "A data reference (i.e. \"" + std::string(JSONSchema_v1::Marks::kFromProp) +
                "\") is not defined for mark. It is required."));
  }

  if (data_ref_changed) {
    updateDataPtr(curr_data);
  }

  return data_ref_changed;
}

void BaseMark::initTransformsFromJSONObj(
    const JSONLocation& obj_loc,
    const std::vector<CoordAttrInfo2d>& coord_props) {
  auto const xform_loc = obj_loc.getMember(JSONSchema_v1::Marks::kTransformProp);
  if (xform_loc.isValid()) {
    if (!ctx_.isJSONCacheUpToDate(transform_ptr_json_path_, xform_loc)) {
      RUNTIME_EX_ASSERT(xform_loc.isObject(),
                        RapidJSONUtils::createJsonParseError(
                            xform_loc, "Mark transform reference must be an object."));

      // TODO(adb): should enumerate supported transformations somewhere -- enum or
      // array or somesuch?
      // TODO(adb): this is fine for now. However, once we support more than one type of
      // transform, we will need to ensure that we clear each type of transform
      // independently (if, for example, the user switches from a projection transform
      // to a different type of transform)
      auto const proj_loc = xform_loc.getMember(JSONSchema_v1::Marks::kProjectionProp);
      if (proj_loc.isValid()) {
        RUNTIME_EX_ASSERT(proj_loc.isString(),
                          std::string(*this) + ": The \"" +
                              std::string(JSONSchema_v1::Marks::kProjectionProp) +
                              "\" member of a mark transform must be a string.");
        auto const projection_name = proj_loc.getString();
        auto const projection = ctx_.getProjection(projection_name);
        RUNTIME_EX_ASSERT(
            projection,
            RapidJSONUtils::createJsonParseError(proj_loc,
                                                 "Unable to find projection \"" +
                                                     std::string(projection_name) +
                                                     "\" in projections list for mark."));

        // validate the state of the projected attrs. All projections currently require
        // both x/y to be data driven and in the same buffer
        // TODO(croot): handle a validate function on the projection for this for
        // per-projection customization
        for (auto const& coord_prop : coord_props) {
          RUNTIME_EX_ASSERT(
              coord_prop.getXAttr(),
              RapidJSONUtils::createJsonParseError(
                  obj_loc,
                  "Property \"" + coord_prop.x_prop->getName() + "\" at " +
                      RapidJSONUtils::getPointerPath(coord_prop.x_prop->getJsonPath()) +
                      " must reference a data field to support transform projections"));

          RUNTIME_EX_ASSERT(
              coord_prop.getYAttr(),
              RapidJSONUtils::createJsonParseError(
                  obj_loc,
                  "Property \"" + coord_prop.y_prop->getName() + "\" at " +
                      RapidJSONUtils::getPointerPath(coord_prop.y_prop->getJsonPath()) +
                      " must reference a data field to support transform projections"));
        }
        if (hasProjection()) {
          clearProjection();
        }
        setProjection(projection_name, projection);
      } else {
        THROW_RUNTIME_EX(
            std::string(*this) +
            ": Unable to find any supported transforms in transform object.");
      }
    }

    transform_ptr_json_path_ = xform_loc.getPathRef();
  } else {
    clearTransforms();
    transform_ptr_json_path_ = rapidjson::Pointer();
  }
}

void BaseMark::updateProps(const BaseRenderPropertyConstSet& used_props) {
  // Now update which props are vbo/ssbo/ubo-defined,
  // while auto adding the invalid key property along the way
  RENDER_LOG_SCOPE();

  QueryDataLayoutShPtr vbo_layout, ssbo_layout;
  if (data_) {
    auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
    if (data) {
      vbo_layout = data->getVboQueryDataLayout();
      ssbo_layout = data->getSsboQueryDataLayout();
    }
  }

  PropBufferStateHandler prop_state_handler(
      type_, prop_buf_loc_state_, vbo_layout.get(), ssbo_layout.get());

  // props used from the mark implementations
  for (auto const& prop : used_props) {
    prop_state_handler.handleProp(prop);
  }

  // Ids
  for (auto const& id : render_props_->ids) {
    prop_state_handler.handleProp(id.get());
  }

  // Invalid key property (special-case)
  prop_state_handler.handleKey(*render_props_->key, data_);

  // Finalize and dirty shader if necessary
  prop_state_handler.propHandlingFinished();

  is_empty_ = prop_state_handler.isEmpty();
  if (prop_state_handler.isDirty()) {
    setShaderDirty();
  }
}

bool BaseMark::updateFromJSONObj(const JSONLocation& obj_loc) {
  RENDER_LOG_SCOPE();
  bool rtn = false;
  if (!ctx_.isJSONCacheUpToDate(json_path_, obj_loc)) {
    auto const data_changed = BaseMark::initFromJSONObj(obj_loc, false);
    initPropertiesFromJSONObj(obj_loc, data_changed, false);
    initTransformsFromJSONObj(obj_loc, getCoordPropAttrInfos());
    rtn = true;
  } else if (json_path_ != obj_loc.getPathRef()) {
    // TODO(croot) - Bug! What if the cache is up-to-date, but the path has changed --
    // we need to update the paths for this and all sub-objects that manage paths. NOTE:
    // We should never get in here since marks are stored as an array but if we ever
    // change the storage container in the future (i.e. an unordered_map or the like),
    // we'd reach this
    THROW_RUNTIME_EX(std::string(*this) + ": The cache for mark \"" +
                     RapidJSONUtils::getPointerPath(obj_loc.getPathRef()) +
                     "\" is up-to-date, but the path in the JSON has changed from " +
                     RapidJSONUtils::getPointerPath(json_path_) + " to " +
                     RapidJSONUtils::getPointerPath(obj_loc.getPathRef()) +
                     ", so the path caches need updating. This "
                     "has yet to be implemented.");
  }

  // TODO(croot): if the obj hasn't changed, but the path has, we need
  // to trickle that path change to all subobjects who cache their
  // json data. How should we handle this?
  json_path_ = obj_loc.getPathRef();

  return rtn;
}

void BaseMark::update() {
  std::shared_ptr<SqlQueryRowDataTableJSON> data;
  bool force_fill = false, force_stroke = false;
  RENDER_LOG_SCOPE();
  if (!per_gpu_data_.size() && pending_gpu_data_additions_.empty()) {
    return;
  }

  if (!props_dirty_) {
    // Check if we need to rebuild PrimitiveAssemblies

    // TODO: change the force_fill and force_stroke flags to class level PrimitiveAssembly
    // dirty flags. These can then also be set during initGpuResources when checking
    // removed PerGpuData. Finally, since the flags would be persistent across renders and
    // gpu transitions it should be safe to remove force_fill and force_stroke from the if
    // (isVisible()) conditional below so we never run that code for invisible marks
    data = std::dynamic_pointer_cast<SqlQueryRowDataTableJSON>(data_);
    if (!data || !data->hasLayoutChanged(QDTLayoutChangedFlags::kVboContentsOrOffset)) {
      for (auto const& itr : per_gpu_data_) {
        if (!force_fill) {
          for (auto const& pa : itr.second.fill_primitive_assemblies) {
            if (pa && pa->isDirty()) {
              force_fill = true;
              break;
            }
          }
        }

        if (!force_stroke) {
          for (auto const& pa : itr.second.stroke_primitive_assemblies) {
            if (pa && pa->isDirty()) {
              force_stroke = true;
              break;
            }
          }
        }

        if (force_fill && force_stroke) {
          break;
        }
      }

      if (!force_fill && !force_stroke && !shader_dirty_) {
        RENDER_LOG() << "No structural updates";
        if (uniforms_dirty_) {
          updateUniformProperties();
        }
        return;
      }
    } else {
      force_fill = force_stroke = true;
    }
  }

  RENDER_LOG() << "props_dirty_: " << props_dirty_ << "  force_fill: " << force_fill
               << "  force_stroke: " << force_stroke;
  initIds(props_dirty_ || force_fill || force_stroke);
  updateProps(getUsedProps());
  // force flags somewhat speculative here...
  if (isVisible() || force_fill || force_stroke) {
    if (shader_dirty_) {
      createPendingGpuData();
      // Rebuild everything on all gpus, so clear the pending data vector
      pending_gpu_data_additions_.clear();
      // Build shaders, spirv caches, Materials, PrimitiveAssemblies and Pipelines
      updateShader();
      updatePrimitiveAssemblies(data, force_fill, force_stroke);
      updatePipelines();
    } else if (!pending_gpu_data_additions_.empty()) {
      // Shader spirv is valid
      // Update existing PerGpuData first
      updatePrimitiveAssemblies(data, force_fill, force_stroke);
      updatePipelines();
      // Create dependent resources on the new gpus
      // (Materials, PrimitiveAssemblies, Pipelines)
      createPendingGpuData();
      buildResourcesOnNewGpus();
    } else if (!data || force_fill || force_stroke) {
      // Data layout changed so VAOs are dirty and need to be rebuilt
      updatePrimitiveAssemblies(data, force_fill, force_stroke);
      updatePipelines();
    }
    // All props dependent bits are updated so clear the flag
    props_dirty_ = false;

    // Always update uniforms (tests fail if checking the dirty flag - needs fixing)
    updateUniformProperties();
  }
}

void BaseMark::createPendingGpuData() {
  auto& global_gpu_data_map = ctx_.getGlobalContext().getRootPerGpuData();
  for (auto gpu_id : pending_gpu_data_additions_) {
    auto global_itr = global_gpu_data_map.find(gpu_id);
    CHECK(global_itr != global_gpu_data_map.end());

    auto [per_gpu_itr, emplace_succeeded] =
        per_gpu_data_.try_emplace((*global_itr)->getGpuId(), **global_itr);
    CHECK(emplace_succeeded);
    // Ensure framebuffers are ready for any custom renderpasses
    // TODO(scb): can we get rid of this?
    (*global_itr)->prepareCommonFramebuffers(getRequiredCommonRenderPassTypes());
    if (usesPerPixelLinkedLists()) {
      initPPLLPerGpuData(per_gpu_itr->second);
    }
  }
}

std::vector<CoordAttrInfo2d> BaseMark::getCoordPropAttrInfos() const {
  return {{render_props_->getProperty(RenderPropertyContainer::PropId::kX),
           render_props_->getProperty(RenderPropertyContainer::PropId::kY)}};
}

void BaseMark::insertFragShaderMain(ShaderBuilder& frag_builder) {
  frag_builder.appendTemplate("Outputs/output_ID.glsl");
  if (hasAccumulator()) {
    // insert accumulation main
    auto scale = ctx_.getScale(getAccumulatorScaleName());
    auto* accum_state = scale->getAccumState();
    CHECK(accum_state);
    frag_builder.appendSubBuilder(accum_state->get1stPassFragSubBuilder());
  } else {
    // insert standard raster mode main
    frag_builder.appendTemplate("Marks/mainTemplate_StdRaster.glsl");
  }
}

void BaseMark::setKeyInShaderBuilder(ShaderBuilder& builder) {
  bool use_key = render_props_->key->hasVboPtr();
  builder.replaceFirstTag("useKey", std::to_string(use_key));
}

void BaseMark::setProjectionUniformAttributes(gfx::Material& active_material) {
  if (hasProjection()) {
    // bind projection properties separately from viewport, since we need renderer
    // context to bind viewport uniform
    auto const projection = getProjection();
    RUNTIME_EX_ASSERT(projection,
                      "Mark references a projection but no projection object was found.");
    projection->setUniformAttributes(active_material);
  }
}

void BaseMark::bindIDPropUniformAttributes(gfx::Material& active_material) {
  if (ctx_.doHitTest() && data_) {
    auto* sql_data_table = dynamic_cast<BaseQueryDataTableSQLJSON*>(data_.get());
    if (sql_data_table) {
      auto const& query_sql = sql_data_table->getQuerySQL();
      auto result_cache_id = query_sql.getResultCacheId();
      // Offset +1 to handle case where data index not found (-1), which will CHECK
      auto opt_vega_data_id = ctx_.getDataIndex(sql_data_table->getNameRef());
      CHECK(opt_vega_data_id);
      auto vega_data_id = *opt_vega_data_id;
      CHECK(vega_data_id >= 0 && vega_data_id < 32)
          << "Invalid vega data index: " << vega_data_id;
      auto node_id = static_cast<uint32_t>(ctx_.getNodeIdx() + 1);
      result_cache_id =
          (result_cache_id << 17) | (static_cast<uint32_t>(vega_data_id) << 12) | node_id;
      active_material.setUniformAttribute("uResultCacheId", result_cache_id);
      if (render_props_->ids.size() > 1) {
        auto* rowid_offsets = query_sql.getRowIdOffsetData();
        CHECK(rowid_offsets);
        CHECK_EQ(render_props_->ids.size(), rowid_offsets->rowid_offsets.size());
        // don't include the last bit-shift info. Only N-1 offset bits are required. See:
        // https://github.com/omnisci/omniscidb-internal/blob/master/QueryRenderer/Marks/shaders/polyTemplate.vert#L89
        // So going to copy the offsets and pop the last one off.
        auto rowid_offset_copy = rowid_offsets->rowid_offsets;
        rowid_offset_copy.pop_back();
        active_material.setUniformAttribute("id_offsets", rowid_offset_copy);
      }
    }
  }
}

void BaseMark::bindKeyPropUniformAttributes(gfx::Material& active_material) {
  // TODO(croot): create a static invalidKeyAttrName string on the class
  static const std::string invalid_key_attr_name = "invalidKey";
  if (render_props_->key->hasVboPtr()) {
    if (active_material.hasUniformAttribute(invalid_key_attr_name)) {
      auto vbo = render_props_->key->getVboPtr();
      CHECK(vbo);
      auto* query_result_buffer = dynamic_cast<QueryVertexBuffer*>(vbo.get());
      if (query_result_buffer) {
        auto query_data_layout = query_result_buffer->getQueryDataLayout();
        CHECK(query_data_layout);
        active_material.setUniformAttribute<int64_t>(invalid_key_attr_name,
                                                     query_data_layout->getInvalidKey());
      }
    }
  }
}

void BaseMark::setColorConvertSubroutines(ShaderBuilder& builder,
                                          const BaseRenderProperty* color_prop_base) {
  std::string unpack_name("unpack" + color_prop_base->getName());
  std::string transform_name("transform" + color_prop_base->getName() + "ToRGB");

  // Unpacking is made optional just in case the optimizer has removed it. This should
  // match the original logic
  auto const* color_prop = dynamic_cast<const ColorRenderProperty*>(color_prop_base);
  CHECK(color_prop);
  auto unpack_color = color_prop->isColorPacked();
  switch (color_prop->getColorType()) {
    case gfx::ColorType::RGBA:
      builder.addSubroutineBinding(transform_name, "transformRGBtoRGB", true);
      if (unpack_color) {
        builder.addSubroutineBinding(unpack_name, "unpackRGBAColor", false);
      }
      break;
    case gfx::ColorType::HSL:
      builder.addSubroutineBinding(transform_name, "transformHSLtoRGB", true);
      break;
    case gfx::ColorType::LAB:
      builder.addSubroutineBinding(transform_name, "transformLABtoRGB", true);
      if (unpack_color) {
        builder.addSubroutineBinding(unpack_name, "unpackLABColor", false);
      }
      break;
    case gfx::ColorType::HCL:
      builder.addSubroutineBinding(transform_name, "transformHCLtoRGB", true);
      break;
    default:
      THROW_RUNTIME_EX(std::string(*this) + ": unsupported fill color type " +
                       std::to_string(static_cast<int>(color_prop->getColorType())) +
                       ". Cannot bind fill color uniform");
  }
}

void BaseMark::updateUniformProperties() {
  RENDER_LOG_SCOPE();
  for (auto& itr : per_gpu_data_) {
    auto& mark_gpu_data = itr.second;
    setUniformAttributes(mark_gpu_data);
  }
  uniforms_dirty_ = false;
}

void BaseMark::updatePrimitiveAssemblies(
    const std::shared_ptr<SqlQueryRowDataTableJSON>& data,
    const bool force_fill,
    const bool force_stroke) {
  if (!props_dirty_ && !force_fill && !force_stroke) {
    return;
  }

  // TODO(croot): make thread safe?
  for (auto& itr : per_gpu_data_) {
    auto& mark_gpu_data = itr.second;

    if (mark_gpu_data.fill_materials.empty()) {
      mark_gpu_data.fill_primitive_assemblies.clear();
    } else if (mark_gpu_data.fill_primitive_assemblies.empty() || props_dirty_ || !data ||
               force_fill) {
      CHECK(mark_gpu_data.fill_materials[0]);
      // build vertex array object that binds buffers and maps how vertex buffer
      // attributes will be attached to shader attributes
      buildFillPrimitiveAssemblies(mark_gpu_data);
    }

    if (mark_gpu_data.stroke_materials.empty()) {
      mark_gpu_data.stroke_primitive_assemblies.clear();
    } else if (mark_gpu_data.stroke_primitive_assemblies.empty() || props_dirty_ ||
               !data || force_stroke) {
      CHECK(mark_gpu_data.stroke_materials[0]);
      buildStrokePrimitiveAssemblies(mark_gpu_data);
    }
  }

  // mark pipelines dirty
  setPipelinesDirty();
}

void BaseMark::updatePipelines() {
  if (!pipelines_dirty_) {
    return;
  }

  // shared stuff
  buildPipelineDescriptors();

  // per-GPU stuff
  for (auto& itr : per_gpu_data_) {
    auto& mark_gpu_data = itr.second;
    buildPipelines(mark_gpu_data);
  }

  pipelines_dirty_ = false;
}

void BaseMark::buildMaterials(const MarkGpuResourceSlot slot,
                              const std::string& resource_tracking_string,
                              gfx::ShaderCacheShPtrVector& caches,
                              const std::vector<GpuId>& gpus) {
  RENDER_LOG_SCOPE_P(gpus);

  for (auto gpu_id : gpus) {
    auto gpu_data_itr = per_gpu_data_.find(gpu_id);
    CHECK(gpu_data_itr != per_gpu_data_.end());
    auto& gpu_data = gpu_data_itr->second;
    auto& root_gpu_data = gpu_data.getRootPerGpuData();
    auto material = root_gpu_data.getResourceManager().createMaterial(
        resource_tracking_string, caches);

    switch (slot) {
      case MarkGpuResourceSlot::kFill:
        gpu_data.fill_materials.push_back(std::move(material));
        break;
      case MarkGpuResourceSlot::kStroke:
        gpu_data.stroke_materials.push_back(std::move(material));
        break;
    }
  }
}

void BaseMark::initGpuResources() {
  used_gpus_.clear();
  pending_gpu_data_additions_.clear();
  std::vector<GpuId> removed_gpus;

  if (!data_) {
    auto global_itr = ctx_.getGlobalContext().getRootPerGpuData().begin();
    auto gpu_id = (*global_itr)->getGpuId();  // invariant for a server run
    used_gpus_.push_back(gpu_id);
    auto per_gpu_itr = per_gpu_data_.find(gpu_id);
    if (per_gpu_itr == per_gpu_data_.end()) {
      pending_gpu_data_additions_.push_back(gpu_id);
    }
  } else {
    // Build set of existing gpu data. This ensures we are capturing the actual
    // state of the gpu data. This will be used to populate the removed gpus vector
    std::set<GpuId> unused_gpu_data;
    for (auto const& gpu_data_itr : per_gpu_data_) {
      unused_gpu_data.insert(gpu_data_itr.second.getGpuId());
    }

    // Get the gpus used by the data and populate add/remove vectors
    used_gpus_ = data_->getUsedGpuIds();
    for (auto const& gpu_id : used_gpus_) {
      auto per_gpu_itr = per_gpu_data_.find(gpu_id);
      if (per_gpu_itr == per_gpu_data_.end()) {
        pending_gpu_data_additions_.push_back(gpu_id);
      }
      unused_gpu_data.erase(gpu_id);
    }
    removed_gpus.insert(
        removed_gpus.begin(), unused_gpu_data.begin(), unused_gpu_data.end());
  }

  RENDER_LOG() << "use " << render_logger::format_gpuid_vector(used_gpus_) << "  add "
               << render_logger::format_gpuid_vector(pending_gpu_data_additions_)
               << "  remove " << render_logger::format_gpuid_vector(removed_gpus);

  // If there is no overlap between the old set of gpus and the new set, dirty flags
  // may be lost that were set by data changes during query execution. We need to
  // ensure the dirty state is captured.
  // See also note in update() regarding the force_fill and force_stroke flags
  if (!props_dirty_) {
    for (auto const& gpu_id : removed_gpus) {
      auto itr = per_gpu_data_.find(gpu_id);
      if (itr != per_gpu_data_.end()) {
        for (auto const& pa : itr->second.fill_primitive_assemblies) {
          if (pa->isDirty()) {
            props_dirty_ = true;
            break;
          }
        }
        for (auto const& pa : itr->second.stroke_primitive_assemblies) {
          if (pa->isDirty()) {
            props_dirty_ = true;
            break;
          }
        }
      }
    }
  }

  // Remove PerGpuData for GPUs that are no longer in use
  RENDER_LOG() << "Removing gpu data from "
               << render_logger::format_gpuid_vector(removed_gpus);
  for (auto const& gpu_id : removed_gpus) {
    per_gpu_data_.erase(gpu_id);
  }

  if (used_gpus_.size() && per_gpu_data_.size() == 0) {
    // TODO(croot): make a makeAllDirty() function
    setShaderDirty();
    setPropsDirty();
  }

  // Update all RenderProperty gpu resources
  render_props_->key->initGpuResources(pending_gpu_data_additions_, removed_gpus);

  for (auto& id : render_props_->ids) {
    id->initGpuResources(pending_gpu_data_additions_, removed_gpus);
  }
  updateRenderPropertyGpuResources(pending_gpu_data_additions_, removed_gpus);
}

void BaseMark::buildResourcesOnNewGpus() {
  RENDER_LOG_SCOPE_P(pending_gpu_data_additions_);

  CHECK(!(fill_shader_caches_.empty() && stroke_shader_caches_.empty()));

  // Create materials using shader caches
  for (auto& cache : fill_shader_caches_) {
    buildMaterials(MarkGpuResourceSlot::kFill,
                   cache.tracking_string,
                   cache.caches,
                   pending_gpu_data_additions_);
  }
  for (auto& cache : stroke_shader_caches_) {
    buildMaterials(MarkGpuResourceSlot::kStroke,
                   cache.tracking_string,
                   cache.caches,
                   pending_gpu_data_additions_);
  }

  // Build PrimitiveAssemblies and Pipelines
  for (auto const& gpu_id : pending_gpu_data_additions_) {
    auto itr = per_gpu_data_.find(gpu_id);
    CHECK(itr != per_gpu_data_.end());
    auto& mark_gpu_data = itr->second;

    if (!mark_gpu_data.fill_materials.empty()) {
      buildFillPrimitiveAssemblies(mark_gpu_data);
    }

    if (!mark_gpu_data.stroke_materials.empty()) {
      buildStrokePrimitiveAssemblies(mark_gpu_data);
    }

    buildPipelines(mark_gpu_data);
  }
  pending_gpu_data_additions_.clear();
}

CommonRenderPassTypeBits BaseMark::getRequiredCommonRenderPassTypes() const {
  return CommonRenderPassTypeBits::kAllAttachments;
}

void BaseMark::buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  CHECK(false) << "Unsupported";
  return;
}

void BaseMark::buildStrokePrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  CHECK(false) << "Unsupported";
  return;
}

void BaseMark::unsubscribeFromProjectionEvent() {
  auto projection = active_projection_.lock();
  if (projection && projection_ref_subscription_id_) {
    ctx_.unsubscribeFromRefEvent(
        RefEventType::kAll, projection, projection_ref_subscription_id_);
    projection_ref_subscription_id_ = 0;
  }
}

void BaseMark::subscribeToProjectionEvent(const ProjectionShPtr& projection) {
  CHECK_EQ(projection_ref_subscription_id_, RefCallbackId(0));
  projection_ref_subscription_id_ =
      ctx_.subscribeToRefEvent(RefEventType::kAll, projection, [this](auto&&... args) {
        return this->projectionRefUpdateCB(std::forward<decltype(args)>(args)...);
      });
}

void BaseMark::projectionRefUpdateCB(RefEventType ref_event_type,
                                     const RefObjShPtr& ref_obj) {
  auto projection = std::dynamic_pointer_cast<Projection>(ref_obj);
  CHECK(projection);
  switch (ref_event_type) {
    case RefEventType::kUpdate: {
      // Dirtying the uniforms since all projection properties are applied via UBOs.
      // NOTE(adb): Currently we do not need to mark shaders dirty on mercator projection
      // changes. However, if we add attributes to the projection that require rebuilding
      // the shader, we will need to trigger that action here by dirtying the shader.
      setUniformsDirty();
      break;
    }
    case RefEventType::kReplace: {
      auto current_projection = active_projection_.lock();
      if (current_projection != projection) {
        updateProjectionPtr(projection);
      }
      break;
    }
    case RefEventType::kRemove:
      THROW_RUNTIME_EX(
          std::string(*this) + ": error, projection: " + ref_obj->getName() +
          " has been removed but is still being referenced by this render property.")
      break;
    default:
      THROW_RUNTIME_EX(std::string(*this) + ": Ref event type: " +
                       std::to_string(static_cast<int>(ref_event_type)) +
                       " isn't currently supported for projection reference updates.");
  }
}

std::string BaseMark::buildVertexShaderInputs() const {
  auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
  // TODO: what do we do if the dynamic cast fails?
  CHECK(data) << "Mark must have a valid BaseQueryDataTableSQLJSON data object";
  auto const* layout = data->getVboBufferLayout().get();
  CHECK(layout);
#if 0
  // Dump layout attribute infos
  auto num_layout_attrs = layout->numAttributes();
  std::cout << "Layout from BaseQueryDataTable:" << std::endl;
  for (int i = 0; i < num_layout_attrs; ++i) {
    auto const& attr_info = (*layout)[i];
    std::cout << "layout attr [" << i << "]: \'" << attr_info.name
              << "\' offset: " << attr_info.offset << std::endl;
  }
#endif

  // Build VBO render props string
  // We need to pair the BaseRenderProperty with the gfx::BufferLayout's attr info
  struct PropAttrInfo {
    const BaseRenderProperty* prop;
    const gfx::BufferAttrInfo* attr_info;
    PropAttrInfo(const BaseRenderProperty* prop, const gfx::BufferAttrInfo* attr_info)
        : prop{prop}, attr_info{attr_info} {}
  };

  std::vector<PropAttrInfo> prop_attr_infos;
  for (auto const& prop : prop_buf_loc_state_.vbo_props) {
    auto const& attr_info = layout->getAttributeInfo(prop->getDataColumnName());
    prop_attr_infos.emplace_back(prop, &attr_info);
  }

  // Sort properties in ascending byte offset order
  std::sort(prop_attr_infos.begin(),
            prop_attr_infos.end(),
            [](const PropAttrInfo& a, const PropAttrInfo& b) {
              return a.attr_info->offset < b.attr_info->offset;
            });

  std::stringstream ss;
  int location = 0;

#define LOG_SHADER_INPUTS 0

#if LOG_SHADER_INPUTS
  std::cout << "Vertex shader inputs: " << std::endl;
#endif

  // Loop over the sorted PropAttrInfo structs and generate the vertex shader 'in' string
  for (auto const& info : prop_attr_infos) {
    ss << "layout (location = " << location << ") in "
       << info.prop->getInTypeGLSL()->declString() << " " << info.prop->getName()
       << ";\n";
    location++;

#if LOG_SHADER_INPUTS
    std::cout << "(location = " << location << ") "
              << info.prop->getInTypeGLSL()->declString() << " " << info.prop->getName()
              << "  "
              << "layout attr: \'" << info.attr_info->name
              << "\' offset: " << info.attr_info->offset << std::endl;
#endif
  }

  return ss.str();
}

std::string BaseMark::buildVertexDataStruct() const {
  //
  // For a vertex shader, we require the property names with incrementing locations.
  // The actual mapping from VBO column to vertex attribute is done in the primitive
  // assembly.
  //
  // In the mesh shader case, there is no primitive assembly, so this struct must
  // be of the actual VBO columns directly from the layout.
  //

  auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
  // TODO: what do we do if the dynamic cast fails?
  CHECK(data) << "Mark must have a valid BaseQueryDataTableSQLJSON data object";
  auto const* layout = data->getVboBufferLayout().get();
  CHECK(layout);
  gfx::GlslStructBuilder struct_builder("VertexData");
  struct_builder.addMembersFromLayout(*layout);
  std::stringstream ss;
  ss << "struct " << struct_builder.createStructString() << ";\n";

  return ss.str();
}

std::string BaseMark::buildVertexAttributeFetches() const {
  std::stringstream ss;
  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    ss << prop->getInTypeGLSL()->declString() << " " << prop->getName()
       << " = VERTEX_DATA(vboDeviceAddress).vertex_data[vbo_index]."
       << prop->getDataColumnNameRef() << ";" << std::endl;
  }
  return ss.str();
}

void BaseMark::addCommonRenderPropUniforms(
    gfx::GlslStructBuilder& ubo_struct_builder) const {
  auto check_for_decimal = [&](const BaseRenderProperty* prop) {
    if (prop->isDecimal()) {
      ubo_struct_builder.addMember(prop->getName() + "_ExpScale",
                                   gfx::BufferAttrType::kUint64);
    }
  };

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    ubo_struct_builder.addMember(prop->getName(), *prop->getInTypeGLSL());
    check_for_decimal(prop);
  }

  // Add decimal scales for vertex attributes
  for (auto const& prop : prop_buf_loc_state_.vbo_props) {
    check_for_decimal(prop);
  }

  // Add decimal scales for ssbo attributes
  for (auto const& prop : prop_buf_loc_state_.ssbo_props) {
    check_for_decimal(prop);
  }

  // ID offsets array when using more than one ID
  auto num_ids = render_props_->ids.size();
  if (num_ids > 1) {
    ubo_struct_builder.addMember(
        "id_offsets", gfx::BufferAttrType::kInt, {}, true, num_ids);
  }
}

void BaseMark::streamPropertyTypeInfoDefines(const BaseRenderPropertyConstSet& props,
                                             std::ostream& os) const {
  for (auto const* prop : props) {
    auto const& name = prop->getName();
    auto const& in_type = *prop->getInTypeGLSL();
    auto const& out_type = *prop->getOutTypeGLSL();
    os << "#define inT" << name << " " << in_type.declString() << "\n";
    os << "#define inT" << name << "Enum " << in_type.enumString() << "\n";
    os << "#define outT" << name << " " << out_type.declString() << "\n";
    os << "#define outT" << name << "Enum " << out_type.enumString() << "\n\n";
  }
}

void BaseMark::streamUseUniformDefines(std::ostream& os) const {
  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    os << "#define useU" << prop->getName() << " 1\n";
  }
}

// Write "get<propname>" function definitions to convert the value of a renderProperty
// from inType (the sql result type) to outType (the type the shaders need)
//
// The function takes the general form:
//   outType getpropName(inType value) {
//     return outType(value);
//   }
//
// There are 2 special cases:
//   1/ Properties that are projected (e.g. x, y coords that may be mercator projected).
//      These properties must be converted to double as input to a project() function
//      This requires direct access to the result data, so is currently only available
//      for PolyMark and LineMark which handle this in the thrust pass
//   2/ ColorRenderProperties that must be convertable to vec4. These may be packed
//      colors (RGBA8) which must be unpacked into a vec4
//
// TODO(scb): add streamPropertyGetter(std::ostream&) to RenderProperty and this once
// BDA and mesh shader support are working
void BaseMark::streamPropertyGetters(
    const BaseRenderPropertyConstSet& props,
    std::ostream& os,
    MarkProjectionShaderPolicy* projection_policy) const {
  for (auto const* prop : props) {
    auto const& name = prop->getName();

    // Check if this property goes through a projection and the property supports
    // compressed geo coords
    if (projection_policy && projection_policy->isProjectionProp(prop) &&
        any_bits_set(getSupportedCoordPackingTypes() &
                     CoordPackingTypeBits::kCompressedGeo)) {
      // Property is a packed geo coord which requires the exact form:
      //  double getPropName(value) { return value; }
      // There is no type substitution for inType since inType will be an
      // integer, but project needs a double
      auto const attr_type = prop->getInTypeGLSL()->attrType();
      if (attr_type == gfx::BufferAttrType::kInt ||
          attr_type == gfx::BufferAttrType::kDouble) {
        os << "double get" << name << "(double " << name << ") "
           << "{\n  return " << name;
      }
    } else {
      auto const& in_type = *prop->getInTypeGLSL();
      auto const& out_type = *prop->getOutTypeGLSL();
      auto in_attr_type = in_type.attrType();
      auto out_attr_type = out_type.attrType();
      os << out_type.declString() << " get" << name << "(" << in_type.declString() << " "
         << name << ") {\n  return ";
      auto color_prop = dynamic_cast<const ColorRenderProperty*>(prop);

      if (color_prop) {
        if ((in_attr_type == gfx::BufferAttrType::kInt ||
             in_attr_type == gfx::BufferAttrType::kUint) &&
            out_attr_type == gfx::BufferAttrType::kVec4f) {
          os << "unpack" << name << "(" << name << ")";
        } else if (in_attr_type == gfx::BufferAttrType::kInt64 ||
                   in_attr_type == gfx::BufferAttrType::kUint64) {
          os << "unpack" << name << "(uint(" << name << "))";
        } else {
          os << name;
        }
      } else {
        if (in_attr_type == out_attr_type) {
          os << name;
        } else {
          os << out_type.declString() << "(" << prop->getName() << ")";
        }
      }
    }
    os << ";\n}\n\n";
  }
}

////////////////////////////////////////////////////////////////////////////////////////
//
// This function injects render property-related code into the templatized shader
// strings. The shader templates follow fairly strict syntax rules for this injection
// to work correctly. The general property rules in the templates are as follows:
//
// As an example, we'll discuss the following rules using a render property x, which
// is a positional property that can be 2d projected (i.e. lat/lon to mercator)
//
// 1) The following macros for defining the property type (input type & type enum,
//    output type & type enum) and use case (uniform or buffer attr,
//    input type and type enum, & output type and type enum) defined at the head of
//    the shader. Note that if a property is to always be uniform or buffer defined,
//    the useU<propname> macro is excluded. Also note that the type enum is used for
//    possible type-related macros at other points in the shader:
//       useU<propname>
//       inT<propname>
//       inT<propname>Enum
//       outT<propname>
//       outT<propname>Enum
//
//    So, using the example, the x property would defined like so:
//      #define useUx <useUx>
//      #define inTx <inTxType>
//      #define inTxEnum <inTxEnum>
//      #define outTx <outTxType>
//      #define outTxEnum <outTxEnum>
//
// 2) The property is the defined using structure like the following:
//    #if useUx == 1
//      uniform inTx x;
//    #else
//      in inTx x;
//    #endif
//      uniform uint64_t x_ExpScale;
//
//    The x_ExpScale is an additional scale used by decimal attributes. Decimals are
//    defined as int64_t with a scale to convert that int into a floating-point value.
//    So if x is a decimal, the x_ExpScale would be set to convert the int to a float.
//
// 3) Every property (other than special properties id & key) have dummy getters such
// as
//    this:
//
//    outTx getx(in inTx x) {
//      #if inTxEnum == outTxEnum
//        return x;
//      #else
//        return outTx(x);
//      #endif
//    }
//
//    These dummy getters are swapped out for other transform functions that may be
//    defined by scales. The functions by default act as a pass-thru. If no scales are
//    defined by the prop, these functions are left alone.
//
// 4) If a property is a position property that can be 2D projected, then a project
//    function is also needed:
//
//    inTx projectx(in inTx x) {
//      return x;
//    }
//
//    Again, this is just a dummy function that gets replaced with projection code if
//    a projection is defined. If not, this dummy function is left as is.
//
// 5) Finally, when a property is first used in the shader, it should be retrieved
// with
//    code such as the following:
//
//    getx(projectx(x));
//
////////////////////////////////////////////////////////////////////////////////////////
void BaseMark::insertPropertyCodeInShaderBuilders(
    ShaderBuilderVector& builders,
    const BaseRenderPropertyConstSet& props,
    const MarkProjectionShaderPolicy& projection_policy,
    const std::string* ssbo_name,
    const std::string* ssbo_instance_name,
    const bool auto_inject_main) {
  // Insert type information defines for all RenderProperties in the set
  // Injects 4 defines for all properties:
  //   #define inT<name>
  //   #define inT<name>Enum
  //   #define outT<name>
  //   #define outT<name>Enum
  // Injects 1 additional define for uniform properties. No define is added if the
  // property is a vertex attribute
  //   #define useU<name> 1
  // Example (if x is a double without a scale, and not a uniform):
  //      #define inTx double
  //      #define inTxEnum DOUBLE
  //      #define outTx double
  //      #define outTxEnum DOUBLE
  std::stringstream ss;
  streamPropertyTypeInfoDefines(props, ss);
  streamUseUniformDefines(ss);
  for (auto& builder : builders) {
    builder->replaceFirstTag("RenderPropertyTypeInfos", ss.str());
  }

  // update SSBO use flag..
  // This flag, if active, will define the properties using an ssbo interface block
  // struct, so properties that can be potentially defined via ssbos can have a
  // different input source. See:
  // https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL) for more on interface
  // block structures if curious
  const bool use_ssbo =
      (ssbo_name && ssbo_instance_name && prop_buf_loc_state_.ssbo_props.size() > 0);
  for (auto& builder : builders) {
    builder->replaceFirstTag("useSSBO", std::to_string(use_ssbo));
  }
  if (use_ssbo) {
    // double check the ssbo block layout
    // matches that in the shader
    CHECK(data_);
    CHECK_GT(per_gpu_data_.size(), 0ul);
    auto ssbo = (*prop_buf_loc_state_.ssbo_props.begin())
                    ->getSsboPtr(per_gpu_data_.begin()->first);
    CHECK(ssbo);

    CHECK_EQ(ssbo->getLayoutManager().getNumBufferLayouts(), 1u);
    auto shader_block_layout = std::dynamic_pointer_cast<gfx::ShaderBlockLayout>(
        ssbo->getLayoutManager().getBufferLayoutAtIndex(0));

    // Storage buffers are tagged as readonly
    std::string shader_block_code =
        shader_block_layout->buildShaderBlockCode(*ssbo_name, *ssbo_instance_name);

    for (auto& builder : builders) {
      builder->replaceFirstTag(*ssbo_instance_name, shader_block_code);
    }
  }

  // Now update id properties
  if (ctx_.doHitTest() && render_props_->ids.size()) {
    const bool has_vbo = render_props_->ids[0]->hasVboPtr();
    const bool has_ssbo = render_props_->ids[0]->hasSsboPtr();
    for (auto const& id : render_props_->ids) {
      CHECK_EQ(has_vbo, id->hasVboPtr());
      CHECK_EQ(has_ssbo, id->hasSsboPtr());
      auto prop_writer = std::make_unique<PropArgWriter>(
          id.get(),
          (ssbo_instance_name && prop_buf_loc_state_.ssbo_props.find(id.get()) !=
                                     prop_buf_loc_state_.ssbo_props.end()
               ? ssbo_instance_name
               : nullptr));
      auto const template_prop = prop_writer->getTemplateStr();
      auto const final_prop = prop_writer->getFinalStr();
      if (template_prop != final_prop) {
        for (auto& builder : builders) {
          builder->replaceAll(template_prop, final_prop);
        }
      }
    }
  }

  for (auto& builder : builders) {
    // set the number of separate rowids we need to handle
    builder->replaceFirstTag("numid", std::to_string(render_props_->ids.size()));
  }

  // Check if the query generates a packed pixel coord data column
  // Used by rectbin and hexbin heatmapping
  bool has_packed_pixel_coord_data_column{false};
  std::string packed_pixel_coord_data_column_name;
  if (any_bits_set(getSupportedCoordPackingTypes() &
                   CoordPackingTypeBits::kPackedPixel)) {
    has_packed_pixel_coord_data_column =
        hasPackedPixelCoordDataColumn(packed_pixel_coord_data_column_name);
  }

  // now inject extra function calls on the property. This can be done either via
  // scales, projections, decimal-to-double conversion, and/or casting.
  for (auto const& prop : props) {
    std::unique_ptr<GlslAbstractPropWriter> prop_writer = std::make_unique<PropArgWriter>(
        prop,
        (ssbo_instance_name && prop_buf_loc_state_.ssbo_props.find(prop) !=
                                   prop_buf_loc_state_.ssbo_props.end()
             ? ssbo_instance_name
             : nullptr));

    if (has_packed_pixel_coord_data_column &&
        prop->getDataColumnNameRef() == packed_pixel_coord_data_column_name) {
      //
      // this is mutually exclusive with everything else
      //

      // replace getx(projectx(x)) with unpack_pixel_coord_x(x)
      prop_writer = std::make_unique<PackedPixelCoordWriter>(std::move(prop_writer));
    } else {
      //
      // all other possible prop replacements
      //

      auto const is_decimal = prop->isDecimal();
      if (is_decimal) {
        // adds the decimal conversion function convertDecimalToDouble() around the
        // property. So:
        //   getx(projectx(x)) would now be getx(projectx(convertDecimalToDouble(x,
        //   x_ExpScale)))
        prop_writer = std::make_unique<DecimalWriter>(std::move(prop_writer));
      }

      if (projection_policy.isProjectionProp(prop)) {
        // for geo (line or poly) renders...
        if (any_bits_set(getSupportedCoordPackingTypes() &
                         CoordPackingTypeBits::kCompressedGeo)) {
          // first add a decompression, if needed
          // type must be double or int32_t
          auto const attr_type = prop->getInTypeGLSL()->attrType();
          CHECK(attr_type == gfx::BufferAttrType::kInt ||
                attr_type == gfx::BufferAttrType::kDouble);
          // if int_32_t, we need a decompression
          if (attr_type == gfx::BufferAttrType::kInt) {
            prop_writer =
                std::make_unique<DecompressGeoCoordWriter>(std::move(prop_writer));
          }
        }

        // then add the projection
        prop_writer =
            std::make_unique<ProjectWriter>(std::move(prop_writer), hasProjection());
      }

      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        // Replace the getter function with the appropriate scale function. Note these are
        // generally found in the vertex shader, but they can be found in other shaders,
        // so we need to loop through them all.
        //
        // So: getx(projectx(x)) would now be
        // getQuantitativeScale_x_x(projectx(x)) if a quantitative scale is used, for
        // example.
        //
        // NOTE: Because the domains of scales can be coerced into
        // the render property's type, we need to provide a new
        // set of GLSL code for each scale reference, even tho
        // it is possible to reference the same scale multiple times.

        // TODO(croot): there are ways we can reduce the amount of
        // shader code here. Domains of certain scales can be coerced,
        // but not all scales, so we can find some optimizations there.
        // Also, ranges can not be coerced, so optimizations can be
        // done there as well, but it is likely rare that the same
        // scale be referenced many times at this point (11/9/15), so
        // it's probably not worth the effort to optimize.
        auto const suffix = "_" + prop->getName();
        auto const prop_func_name = prop->getGLSLFunc();
        auto const scale_builder = scale_ref->getShaderSubBuilder(suffix);
        auto const domain_type = scale_ref->getDomainGLSLTypeName(suffix);
        auto const func_name = scale_ref->getScaleGLSLFuncName(suffix);

        if ((!is_decimal && *prop->getInTypeGLSL() != *scale_ref->getDomainTypeGLSL()) ||
            (is_decimal &&
             *prop->getDecimalTypeGLSL() != *scale_ref->getDomainTypeGLSL())) {
          prop_writer = std::make_unique<CastWriter>(std::move(prop_writer), domain_type);
        }

        prop_writer = std::make_unique<ScaleFunctionWriter>(
            std::move(prop_writer), prop_func_name, func_name);

        for (auto& builder : builders) {
          // Replace original function definition with the entirety of the scale code,
          // including supporting functions, UBO, etc.
          builder->replaceFunctionWithSubBuilder(
              prop_func_name, std::move(scale_builder), false);
        }
      }
    }

    // do the actual replacement for this prop
    auto const template_prop = prop_writer->getTemplateStr();
    auto const final_prop = prop_writer->getFinalStr();
    if (template_prop != final_prop) {
      for (auto& builder : builders) {
        builder->replaceAll(template_prop, final_prop);
      }
    }
  }

  // now inject the projection code:
  // replace projectx {} with the true projection function
  if (hasProjection()) {
    auto const projection = getProjection();
    RUNTIME_EX_ASSERT(
        projection,
        "Mark " + std::string(*this) +
            " references a projection but projection object is uninitialized.");
    for (auto& builder : builders) {
      projection->updateShader(*builder, projection_policy);
    }
  }

  // Now inject any accumulation functionality into the shader
  if (auto_inject_main && builders.size() > 1 && builders[1]) {
    insertFragShaderMain(*builders[1]);
  }
}

void BaseMark::updateProjectionPtr(ProjectionShPtr projection) {
  unsubscribeFromProjectionEvent();

  active_projection_ = projection;
  subscribeToProjectionEvent(projection);
  setShaderDirty();
}

void BaseMark::clearProjection() {
  unsubscribeFromProjectionEvent();

  active_projection_.reset();
  setShaderDirty();
}

void BaseMark::clearTransforms() {
  clearProjection();
}

void BaseMark::dataRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj) {
  // noop: let derived classes override if needed
}

void BaseMark::updateDataPtr(BaseDataTableShPtr& data) {
  if (data != data_) {
    unsubscribeFromDataEvent();
    data_ = data;
    if (data) {
      subscribeToDataEvent();
    }
  }
}

void BaseMark::unsubscribeFromDataEvent() {
  if (data_ && data_ref_subscription_id_) {
    auto data_json = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
    CHECK(data_json);
    ctx_.unsubscribeFromRefEvent(
        RefEventType::kAll, data_json, data_ref_subscription_id_);
    data_ref_subscription_id_ = 0;
  }
}

void BaseMark::subscribeToDataEvent() {
  // setup callbacks for data updates
  CHECK_EQ(data_ref_subscription_id_, RefCallbackId(0));
  auto data_json = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
  CHECK(data_json);
  data_ref_subscription_id_ =
      ctx_.subscribeToRefEvent(RefEventType::kAll, data_json, [this](auto&&... args) {
        return this->dataRefUpdateCB(std::forward<decltype(args)>(args)...);
      });
}

bool BaseMark::hasPackedPixelCoordDataColumn(std::string& data_column_name) const {
  auto data_table_ptr = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
  if (data_table_ptr) {
    auto const& sql_str = data_table_ptr->getQuerySQL().getSqlQueryStr();
    // check for any pixel_bin_packed function and capture its "AS" name
    std::regex pixel_bin_packed_regex(
        "(?:rect_|reg_hex_(?:horiz_|vert_))pixel_bin_packed\\(.+?\\)\\s+AS\\s+(\\w+)",
        std::regex::icase);
    std::smatch as_match;
    if (std::regex_search(sql_str, as_match, pixel_bin_packed_regex)) {
      CHECK_GE(as_match.size(), 2U) << "Unexpected smatch (" << as_match.size() << ")";
      data_column_name = as_match[1].str();
      return true;
    }
  }
  return false;
}

bool BaseMark::needsMultisampleEnabled() const {
  return getRasterizationSampleCount() != gfx::RasterSampleCount::k1;
}

gfx::RasterSampleCount BaseMark::getRasterizationSampleCount() const {
  return hasAccumulator() ? gfx::RasterSampleCount::k1
                          : ctx_.getGlobalContext().getRasterSampleCount();
}

void BaseMark::updateVisibility(const bool is_visible) {
  is_visible_ = is_visible;
}

gfx::RenderPass& BaseMark::selectDrawRenderPass(
    const QueryRenderer::RootPerGpuData& root_gpu_data,
    const int accumulator_index) const {
  // select a render pass for draw
  // for accum renders, we render single-sampled
  // for FIRST accum renders, we also clear
  // used by Line, Point, and Symbol
  auto const has_accumulator = hasAccumulator();
  auto const render_pass_type = has_accumulator && accumulator_index == 0
                                    ? CommonRenderPassType::kAllAttachmentsClear
                                    : CommonRenderPassType::kAllAttachments;
  return root_gpu_data.getCommonRenderPass(
      render_pass_type, getRasterizationSampleCount() != gfx::RasterSampleCount::k1);
}

void BaseMark::doManualClear(const QueryRenderer::RootPerGpuData& root_gpu_data,
                             const int accumulator_index,
                             gfx::Framebuffer& framebuffer) const {
  // multi-render-pass operation, so for now we clear manually
  // for FIRST accum render, we clear all (SS)
  // for all other cases, we clear depth only
  // @TODO(se/scb) do clear in first render pass
  auto& cmd_list = root_gpu_data.getCommandList();
  cmd_list.pushLabel("BaseMark doManualClear");
  if (hasAccumulator() && accumulator_index == 0) {
    cmd_list
        .beginRenderPass(root_gpu_data.getCommonRenderPass(
                             CommonRenderPassType::kAllAttachmentsClear, false),
                         framebuffer)
        .endRenderPass();
  } else {
    cmd_list.clearTexture(*framebuffer.getAttachmentManager().getAttachmentTexture(
        gfx::Framebuffer::Attachment::kDepthStencil));
  }
  cmd_list.popLabel();
}

void BaseMark::updateGeoPropInfoAndPropCompressionBits(
    const std::set<SQLTypes> supported_geo_types) {
  //
  // computes a bit pattern for the provided properties
  // each bit will be 1 if the property is a compressed geo
  // bit positions are in order of the provided prop names
  // must match #defines in shader
  //

  // reset
  auto& gpi = geo_prop_info_;
  gpi.prop_compression_bits = 0u;
  gpi.prop_name_to_render.clear();
  gpi.prop_type = kNULLT;
  gpi.prop_encoding = EncodingType::kENCODING_NONE;

  // update bit pattern
  auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
  CHECK(data) << "Mark must have a valid BaseQueryDataTableSQLJSON data object";
  auto layout = data->getVboQueryDataLayout();
  CHECK(layout) << "Mark data object must have a valid layout";
  auto const aai = layout->getAllAttrInfo();
  CHECK(aai.size()) << "Mark data object layout must have some attributes";
  uint32_t bit{1u};
  auto const& coord_props = render_props_->getCoordProperties();
  for (auto const* prop : coord_props) {
    auto const& prop_name = prop->getDataColumnNameRef();
    if (prop_name.length()) {
      for (auto const& ai : aai) {
        if (ai.attr_alias == prop_name) {
          auto const& ti = ai.type_info;
          // is it geo?
          if (IS_GEO(ti.get_type())) {
            // check that we can handle geo types at all
            if (!ctx_.getGlobalContext().canUseSlabAddressTable()) {
              THROW_RUNTIME_EX("Direct geo rendering not supported with CUDA disabled");
            }
            // check it's a supported geo type
            RUNTIME_EX_ASSERT(
                supported_geo_types.find(ti.get_type()) != supported_geo_types.end(),
                "Geo type '" + std::string(ti.get_type_name()) +
                    "' not supported for Mark type '" + to_string(type_) + "'");
            // get compression (can vary across props, although it never will)
            if (ti.get_compression() == kENCODING_GEOINT) {
              CHECK_EQ(ti.get_comp_param(), 32)
                  << "Unsupported geo compression param value!";
              gpi.prop_compression_bits |= bit;
            }
            // check that the prop name is the same for all coord props
            if (gpi.prop_name_to_render.size() && gpi.prop_name_to_render != prop_name) {
              THROW_RUNTIME_EX("Mixed geo/non-geo coordinate properties not supported (" +
                               std::string(prop_name) + "/" + gpi.prop_name_to_render +
                               ")");
            }
            // capture name, type, and encoding
            gpi.prop_name_to_render = prop_name;
            gpi.prop_type = ti.get_type();
            gpi.prop_encoding = ti.get_compression();
          }
          break;
        }
      }
    }
    bit <<= 1;
  }
}

void BaseMark::updateSlabAddressTableAndPropCompressionBitsUniforms(
    const MarkPerGpuData& mark_gpu_data) const {
  auto const& root_gpu_data = mark_gpu_data.getRootPerGpuData();
  for (auto& material : mark_gpu_data.fill_materials) {
    CHECK(material);
    material->bindExternalUniformBufferToBlock("SLAB_ADDRESS_TABLE_UBO",
                                               root_gpu_data.getSlabAddressTableBuffer());
    if (material->hasUniformAttribute("propCompressionBits")) {
      material->setUniformAttribute("propCompressionBits",
                                    geo_prop_info_.prop_compression_bits);
    }
  }
  for (auto& material : mark_gpu_data.stroke_materials) {
    CHECK(material);
    material->bindExternalUniformBufferToBlock("SLAB_ADDRESS_TABLE_UBO",
                                               root_gpu_data.getSlabAddressTableBuffer());
    if (material->hasUniformAttribute("propCompressionBits")) {
      material->setUniformAttribute("propCompressionBits",
                                    geo_prop_info_.prop_compression_bits);
    }
  }
}

void BaseMark::updateUseMeshShader(const std::set<SQLTypes> supported_geo_types) {
  if (supported_geo_types.find(geo_prop_info_.prop_type) != supported_geo_types.end()) {
    RUNTIME_EX_ASSERT(can_render_with_mesh_shader_,
                      "Rendering of " + to_string(geo_prop_info_.prop_type) +
                          " requires GPU Mesh Shader support");
    use_mesh_shader_ = true;
  } else {
    use_mesh_shader_ = false;
  }
}

const bool BaseMark::useMeshShader() const {
  return use_mesh_shader_;
}

const uint32_t BaseMark::startMeshShaderDraw(const MarkPerGpuData& mark_gpu_data) {
  // prepare the QRB VBO for drawing (previously done by the PrimitiveAssembly)
  auto* qrb = mark_gpu_data.getRootPerGpuData().getQueryResultBuffer();
  CHECK(qrb);
  qrb->unmapForDraw();

  // get num vertices to draw
  // @TODO(scb) decide on canonical way to get the layout
  auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
  CHECK(data) << "Mark must have a valid BaseQueryDataTableSQLJSON data object";
  auto layout = data->getVboBufferLayout();
  CHECK(layout);
  auto const num_vertices = qrb->getLayoutManager().numItems(layout);

  // get vertex buffer offset (multi-layer)
  auto const& layout_data = qrb->getLayoutManager().getBufferLayoutDataToUse(layout);
  auto const vertex_buffer_offset_bytes = layout_data.offset_bytes;

  // prepare for geo count pass
  const std::string geo_count_column_attr_name =
      geo_prop_info_.prop_name_to_render + "_num";
  const bool is_compressed =
      geo_prop_info_.prop_encoding == EncodingType::kENCODING_GEOINT;

  // the buffer base and column info
  const uint64_t base_address =
      qrb->getBufferWrapper()->getDeviceAddress() + vertex_buffer_offset_bytes;
  const uint32_t num_columns = layout->getNumBytesPerItem() >> 3;
  const uint32_t column_index =
      layout->getAttributeByteOffset(geo_count_column_attr_name) >> 3;

  // in this context, one "value" is a point (two scalars)
  const uint32_t value_byte_size =
      is_compressed ? sizeof(uint32_t) * 2 : sizeof(double) * 2;

  // create work units
  auto& geo_count_resources = mark_gpu_data.getRootPerGpuData().getGeoCountResources();
  auto& root_gpu_data = mark_gpu_data.getRootPerGpuData();
  auto const num_work_units =
      geo_count_resources.createWorkUnits(root_gpu_data.getCommandList(),
                                          base_address,
                                          num_vertices,
                                          num_columns,
                                          column_index,
                                          value_byte_size);

  // any work units
  if (num_work_units == 0U) {
    return 0U;
  }

  auto const* work_units_buffer = geo_count_resources.getWorkUnitsBuffer();

  // update all materials
  for (auto& material : mark_gpu_data.fill_materials) {
    CHECK(material);
    // set VBO device address with offset
    material->setUniformAttribute(
        "vboDeviceAddress",
        qrb->getBufferWrapper()->getDeviceAddress() + vertex_buffer_offset_bytes);
    // bind work units buffer
    material->bindShaderStorageBufferToBlock("WORK_UNITS_SSBO", *work_units_buffer);
  }
  for (auto& material : mark_gpu_data.stroke_materials) {
    CHECK(material);
    // set VBO device address with offset
    material->setUniformAttribute(
        "vboDeviceAddress",
        qrb->getBufferWrapper()->getDeviceAddress() + vertex_buffer_offset_bytes);
    // bind work units buffer
    material->bindShaderStorageBufferToBlock("WORK_UNITS_SSBO", *work_units_buffer);
  }

  // return this for the actual mesh shader invocation(s)
  return num_work_units;
}

void BaseMark::endMeshShaderDraw(const MarkPerGpuData& mark_gpu_data) {
  auto& geo_count_resources = mark_gpu_data.getRootPerGpuData().getGeoCountResources();
  geo_count_resources.destroyWorkUnitsBuffer();
}

}  // namespace QueryRenderer
