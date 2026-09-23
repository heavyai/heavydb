/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/WindBarbMark.h"

#include <boost/algorithm/string/find.hpp>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderPropertyContainer.h"
#include "QueryRenderer/Marks/Utils.h"
#include "QueryRenderer/Marks/WindBarbDefinitions.h"

namespace QueryRenderer {

using ::gfx::ShaderManager;
using ::gfx::ShaderStage;
using ShaderBuilder = ::gfx::ShaderManager::Builder;

static WindBarbDefinitions g_wind_barb_defs = {};

WindBarbMark::WindBarbMark(const JSONLocation& obj_loc, QueryRendererContext& ctx)
    : BaseMark(GeomType::kWindBarbs, ctx, obj_loc, DataOutputFormat::kRows, false)
    , size_("size", ctx, *prop_mark_facade_)
    , speed_("speed", ctx, *prop_mark_facade_)
    , direction_("direction", ctx, *prop_mark_facade_)
    , anchor_scale_("anchorScale", ctx, *prop_mark_facade_)
    , do_quantize_direction_("quantizeDirection",
                             QueryDataType::INT,
                             ctx,
                             *prop_mark_facade_) {
  using type = RenderPropertyCreateInfo::Type;
  using flag = RenderPropertyFlagBits;
  std::vector<RenderPropertyCreateInfo> render_property_ci = {
      {type::kFillColor, flag::kUseScale, false, gfx::ColorUnion(0.f, 0.f, 0.f, 1.f)},
      {type::kStrokeColor, flag::kUseScale, false, gfx::ColorUnion(0.f, 0.f, 0.f, 1.f)},
      {type::kStrokeWidth, flag::kUnspecified, false, 2.0f}};

  render_props_ = std::make_unique<RenderPropertyContainer>(
      ctx, *prop_mark_facade_, std::move(render_property_ci));

  auto const& coord_props = render_props_->getCoordProperties();
  projection_policy_ = std::make_unique<MarkProjectionShaderPolicy>(
      MarkProjectionShaderPolicy::PropMap{coord_props.begin(), coord_props.end()});

  initPropertiesFromJSONObj(obj_loc, true, true);
  initTransformsFromJSONObj(obj_loc, getCoordPropAttrInfos());
  json_path_ = obj_loc.getPathRef();
  init_wind_barb_defs(g_wind_barb_defs);
}

WindBarbMark::~WindBarbMark() {}

BaseRenderPropertyConstSet WindBarbMark::getUsedProps() const {
  return used_props_const_;
}

void WindBarbMark::initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                             const bool data_changed,
                                             const bool init) {
  RENDER_LOG_SCOPE() << " data_changed: " << data_changed << "  init: " << init;
  auto const prop_loc = obj_loc.getMember(JSONSchema_v1::Marks::kPropertiesProp);
  RUNTIME_EX_ASSERT(prop_loc.isValid(),
                    RapidJSONUtils::createJsonParseError(
                        obj_loc, "Mark objects must have a \"properties\" property."));

  auto const prev_path = properties_json_path_;
  properties_json_path_ = obj_loc.getPathRef();
  if (!ctx_.isJSONCacheUpToDate(prev_path, prop_loc) || data_changed || init) {
    RUNTIME_EX_ASSERT(prop_loc.isObject(),
                      RapidJSONUtils::createJsonParseError(
                          prop_loc, "Property must be a json object."));

    render_props_->initFromJSONObj(obj_loc, data_changed);

    auto const obj_num_check = BaseMark::validateNumPropFunc(*this);
    auto const obj_enum_check = BaseMark::validateEnumPropFunc(*this);
    auto const obj_bool_check = BaseMark::validateBoolPropFunc(*this);
    auto const init_prop_func = [&](const JSONLocation& prop_loc,
                                    BaseRenderProperty* prop,
                                    ValidateFunc validate_type_func = nullptr,
                                    JSONParseCBFunc post_update_func = nullptr,
                                    JSONParseCBFunc post_data_update_func = nullptr,
                                    JSONParseCBFunc post_up_to_date_func = nullptr,
                                    JSONParseCBFunc post_empty_func =
                                        nullptr) -> JSONLocation {
      return BaseMark::initPropFromJSONObj(&ctx_,
                                           data_,
                                           data_changed,
                                           prop_loc,
                                           prop,
                                           properties_json_path_,
                                           validate_type_func,
                                           post_update_func,
                                           post_data_update_func,
                                           post_up_to_date_func,
                                           post_empty_func);
    };

    init_prop_func(prop_loc,
                   &size_,
                   obj_num_check,
                   nullptr,
                   nullptr,
                   nullptr,
                   [this, &prop_loc](const JSONLocation&) {
                     THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
                         prop_loc,
                         "\"" + size_.getName() +
                             "\" mark property must exist for wind barb marks."));
                   });

    init_prop_func(prop_loc,
                   &speed_,
                   obj_num_check,
                   nullptr,
                   nullptr,
                   nullptr,
                   [this, &prop_loc](const JSONLocation&) {
                     THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
                         prop_loc,
                         "\"" + speed_.getName() +
                             "\" mark property must exist for wind barb marks."));
                   });

    init_prop_func(prop_loc,
                   &direction_,
                   obj_num_check,
                   nullptr,
                   nullptr,
                   nullptr,
                   [this, &prop_loc](const JSONLocation&) {
                     THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
                         prop_loc,
                         "\"" + direction_.getName() +
                             "\" mark property must exist for wind barb marks."));
                   });

    init_prop_func(prop_loc,
                   &anchor_scale_,
                   obj_num_check,
                   nullptr,
                   nullptr,
                   nullptr,
                   [this](const JSONLocation&) { anchor_scale_.initializeValue(0.0f); });

    init_prop_func(
        prop_loc,
        &do_quantize_direction_,
        obj_bool_check,
        nullptr,
        nullptr,
        nullptr,
        [this](const JSONLocation&) { do_quantize_direction_.initializeValue(true); });

    initIds(data_changed);

    using PropId = RenderPropertyContainer::PropId;
    auto* opacity = render_props_->getProperty(PropId::kOpacity);
    auto* stroke_color = render_props_->getProperty(PropId::kStrokeColor);
    auto* stroke_width = render_props_->getProperty(PropId::kStrokeWidth);

    auto const& props = render_props_->getProperties();
    BaseRenderPropertySet used_props{props.begin(), props.end()};
    used_props.insert(&size_);
    used_props.insert(&speed_);
    used_props.insert(&direction_);
    used_props.insert(&anchor_scale_);
    used_props.insert(&do_quantize_direction_);

    bool used_props_changed = used_props_ != used_props;
    if (used_props_changed) {
      used_props_ = std::move(used_props);
      used_props_const_.clear();
      used_props_const_.insert(used_props_.cbegin(), used_props_.cend());
    }

    if (init || data_changed || used_props_changed) {
      prop_buf_loc_state_.clear();
      updateProps(getUsedProps());
    }

    bool do_stroke =
        ((stroke_width->isDataDriven() || stroke_width->getUniformValue<float>() > 0) ||
         (anchor_scale_.isDataDriven() || anchor_scale_.getUniformValue<float>() > 0)) &&
        (stroke_color->isDataDriven() ||
         stroke_color->getUniformValue<gfx::ColorUnion>().opacity() > 0) &&
        (opacity->isDataDriven() || opacity->getUniformValue<float>() > 0);

    updateVisibility(render_props_->isFillActive() || do_stroke);
  }
}

void WindBarbMark::updateShader() {
  RENDER_LOG_SCOPE() << "building glsl shaders";

  auto& shader_mgr = ctx_.getShaderManager();

  ShaderManager::BuilderUqPtrVector builders;
  builders = shader_mgr.createBuilderVector({{"Marks/windBarbTemplate.vert"},
                                             {"Marks/windBarbTemplate.frag"},
                                             {"Marks/windBarbTemplate.geom"}});

  std::stringstream get_prop_ss;
  streamPropertyGetters(prop_buf_loc_state_.vbo_props, get_prop_ss);
  streamPropertyGetters(prop_buf_loc_state_.uniform_props, get_prop_ss);

  // Vertex shader inputs
  // Build uniform render props
  gfx::GlslStructBuilder ubo_struct_builder("WIND_BARB_VERT_UBO_TYPE");
  ubo_struct_builder.addMember("invalidKey", gfx::BufferAttrType::kUint64);

  addCommonRenderPropUniforms(ubo_struct_builder);

  bool is_direction_uniform = prop_buf_loc_state_.uniform_props.count(&direction_);

  // Build fragment shader input interface block
  // Also used for vertex output in point mode

  gfx::GlslStructBuilder geometry_inputs("GeometryShaderInputs");
  gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
  generate_wind_barb_interface_blocks(
      geometry_inputs, fragment_inputs, is_direction_uniform);

  builders[0]->replaceFirstTag("GeometryShaderInputs",
                               geometry_inputs.createInterfaceBlockString(true));
  builders[2]->replaceFirstTag(
      "GeometryShaderInputs",
      geometry_inputs.createInterfaceBlockString(true, std::nullopt, true));
  builders[2]->replaceFirstTag("useUdirection", is_direction_uniform ? "1" : "0");

  auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
  builders[1]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
  builders[2]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

  builders[0]->replaceFirstTag("VertexProperties", buildVertexShaderInputs());
  builders[0]->replaceFirstTag("UniformProperties",
                               ubo_struct_builder.createStructString());
  builders[0]->replaceFirstTag("PropertyGetters", get_prop_ss.str());

  BaseMark::setKeyInShaderBuilder(*builders[0]);
  std::string num_barb_types_str = std::to_string(g_wind_barb_defs.pennant_counts.size());
  builders[0]->replaceFirstTag("numBarbTypes", num_barb_types_str);
  builders[1]->replaceFirstTag("numBarbTypes", num_barb_types_str);
  builders[1]->replaceFirstTag("numBarbVerts",
                               std::to_string(g_wind_barb_defs.verts.size()));
  builders[1]->replaceFirstTag("showBillboard", "0");
  builders[2]->replaceFirstTag("numBarbTypes", num_barb_types_str);
  builders[2]->replaceFirstTag("doMirrorY", "0");

  // Set multisampling flag
  // controls use of gl_SamplePosition
  // creates a no-discard zone of 1 pixel from the edge to ensure no contributing samples
  // are also discarded
  builders[1]->replaceFirstTag("isMultiSampling",
                               std::to_string(needsMultisampleEnabled()));

  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      for (auto& builder : builders) {
        scale_ref->buildSubroutineBindings(*builder, "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      for (auto& builder : builders) {
        scale_ref->buildSubroutineBindings(*builder, "_" + prop->getName());
      }
    }
  }

  BaseMark::insertPropertyCodeInShaderBuilders(
      builders, used_props_const_, *projection_policy_);

  // color props in either vertex or geometry shader
  BaseMark::setColorConvertSubroutines(
      *builders[0],
      render_props_->getProperty(RenderPropertyContainer::PropId::kFillColor));

  ctx_.clearMarkShaders(*this);
  ctx_.buildMarkShaders(
      *this, MarkGpuResourceSlot::kFill, "WindBarb", std::move(builders));
  shader_dirty_ = false;

  // set the props dirty to force a rebind with the new shader
  setPropsDirty();
}

void WindBarbMark::buildPipelineDescriptors() {
  if (!pipeline_descriptor_) {
    pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
    CHECK(pipeline_descriptor_);
  }

  pipeline_descriptor_->setRasterSampleCount(getRasterizationSampleCount());
}

void WindBarbMark::buildPipelines(MarkPerGpuData& per_gpu_data) {
  CHECK(per_gpu_data.fill_primitive_assemblies.size());
  CHECK(per_gpu_data.fill_primitive_assemblies[0]);
  CHECK(per_gpu_data.fill_materials.size());
  CHECK(per_gpu_data.fill_materials[0]);

  per_gpu_data.destroyPipelines();
  per_gpu_data.graphics_pipelines.push_back(
      per_gpu_data.getResourceManager().createGraphicsPipeline(
          "WindBarb",
          *per_gpu_data.fill_materials[0],
          *pipeline_descriptor_,
          per_gpu_data.fill_primitive_assemblies[0].get()));
  per_gpu_data.graphics_pipelines[0]->create(
      per_gpu_data.getRootPerGpuData().getCommonRenderPass(
          CommonRenderPassType::kAllAttachments, needsMultisampleEnabled()));
}

void WindBarbMark::buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  auto gpu_id = gpu_data.getGpuId();
  CHECK(!gpu_data.fill_materials.empty());

  gfx::PrimitiveAssemblyAttrInfo attr_info;
  int attr_count = 0;
  int vbo_size = 0;
  int prop_size = 0;
  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    if (!gpu_data.fill_materials[0]->hasVertexAttribute(prop->getName())) {
      continue;
    }
    attr_count++;
    prop_size = prop->size(gpu_id);
    if (attr_count == 1) {
      vbo_size = prop_size;
    } else {
      RUNTIME_EX_ASSERT(prop_size == vbo_size,
                        std::string(*this) +
                            ": Invalid symbol mark. The sizes of the vertex buffer "
                            "attributes do not match for gpuId " +
                            std::to_string(gpu_id) + ". " + std::to_string(vbo_size) +
                            "!=" + std::to_string(prop_size));
    }
    prop->addToPrimitiveAssemblyAttrInfo(gpu_id, attr_info);
  }

  gpu_data.fill_primitive_assemblies.clear();
  gpu_data.fill_primitive_assemblies.push_back(
      gpu_data.getResourceManager().createPrimitiveAssembly(
          "WindBarb",
          gfx::PrimitiveTopology::kPointList,
          *gpu_data.fill_materials[0],
          attr_info));
}

void WindBarbMark::setUniformAttributes(MarkPerGpuData& mark_gpu_data) {
  RENDER_LOG_SCOPE();
  auto& active_material = *mark_gpu_data.fill_materials[0];
  auto viewport_width = ctx_.getWidth();
  auto viewport_height = ctx_.getHeight();

  BaseMark::bindKeyPropUniformAttributes(active_material);

  if (hasProjection()) {
    active_material.setViewportAttributes(0, 0, viewport_width, viewport_height);
  }

  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      RENDER_LOG() << "binding VBO property: " << prop->getName();
      scale_ref->bindUniforms(active_material, "_" + prop->getName());
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      scale_ref->bindUniforms(active_material, "_" + prop->getName());
    }

    prop->setUniformAttribute(active_material, prop->getName());
  }

  for (auto const* prop : prop_buf_loc_state_.decimal_props) {
    prop->setDecimalScaleUniformAttribute(active_material);
  }

  BaseMark::bindIDPropUniformAttributes(active_material);
  BaseMark::setProjectionUniformAttributes(active_material);

  active_material.setUniformAttribute("uBarbCounts", g_wind_barb_defs.barb_counts);
  active_material.setUniformAttribute("uBarbOffsets", g_wind_barb_defs.barb_offsets);
  active_material.setUniformAttribute("uPennantCounts", g_wind_barb_defs.pennant_counts);
  active_material.setUniformAttribute("uPennantOffsets",
                                      g_wind_barb_defs.pennant_offsets);
  active_material.setUniformAttribute("uBarbVerts", g_wind_barb_defs.verts);

  active_material.setUniformAttribute("uVPmatrix",
                                      ctx_.getViewProjMatrix().getDataArrayRef());

  active_material.setUniformAttribute("uInvViewportWidth", 1.0f / (float)viewport_width);
  active_material.setUniformAttribute("uInvViewportHeight",
                                      1.0f / (float)viewport_height);
  if (!direction_.isDataDriven()) {
    float direction = direction_.getUniformValue<float>();
    direction = do_quantize_direction_.getUniformValue<float>()
                    ? ((static_cast<int>(direction) + 5) / 10) * 10.0f
                    : direction;
    direction = -direction * 3.14159265359f / 180.0f;
    active_material.setUniformAttribute("uSinDirection", std::sin(direction));
    active_material.setUniformAttribute("uCosDirection", std::cos(direction));
  }
}

void WindBarbMark::updateRenderPropertyGpuResources(
    const std::vector<GpuId>& add_gpus,
    const std::vector<GpuId>& remove_gpus) {
  for (auto const& prop : used_props_) {
    prop->initGpuResources(add_gpus, remove_gpus);
  }
}

bool WindBarbMark::draw(const gfx::DeviceContext& device_ctx,
                        const MarkPerGpuData& mark_gpu_data,
                        gfx::Framebuffer& framebuffer,
                        const int accumulator_index) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());
  // NOTE: shader should have been updated before calling this

  auto const& root_gpu_data = mark_gpu_data.getRootPerGpuData();
  auto& primitive_assembly = mark_gpu_data.fill_primitive_assemblies[0];
  mark_gpu_data.fill_materials[0]->updateDescriptorSets();

  auto& render_pass = selectDrawRenderPass(root_gpu_data, accumulator_index);

  root_gpu_data.getCommandList()
      .pushLabel("WindBarb draw")
      .beginRenderPass(render_pass, framebuffer)
      .drawVertices(*mark_gpu_data.graphics_pipelines[0],
                    *primitive_assembly->getVertexBuffer(),
                    primitive_assembly->numVertices(),
                    primitive_assembly->getVertexBufferOffsetBytes())
      .endRenderPass()
      .popLabel()
      .flush("WindBarb draw", gfx::CommandList::SubmitType::kImmediateReturn);

  return true;
}

WindBarbMark::operator std::string() const {
  return "WindBarb " + std::string(ctx_.getRenderSessionKey());
}

}  // namespace QueryRenderer
