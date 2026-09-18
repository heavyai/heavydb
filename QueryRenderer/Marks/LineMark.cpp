/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/LineMark.h"

#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Data/QueryLineDataTable.h"
#include "QueryRenderer/Marks/LineUtils.h"
#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderPropertyContainer.h"
#include "QueryRenderer/Marks/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

using ::gfx::ShaderStage;
using ::gfx::ShaderStageBits;
using ShaderBuilder = ::gfx::ShaderManager::Builder;
using ::gfx::ShaderBlockLayout;

namespace {
inline std::string getLineBlockName() {
  return "LineData";
}
}  // namespace

LineMark::LineMark(const JSONLocation& obj_loc, QueryRendererContext& ctx)
    : BaseMark(GeomType::kLines, ctx, obj_loc, DataOutputFormat::kLines, true)
    , fill_below_line_("fillBelowLine", QueryDataType::INT, ctx, *prop_mark_facade_) {
  using type = RenderPropertyCreateInfo::Type;
  using flag = RenderPropertyFlagBits;
  std::vector<RenderPropertyCreateInfo> render_property_ci = {
      {type::kStrokeColor,
       flag::kUnspecified,
       false,
       gfx::ColorUnion(1.f, 1.f, 1.f, 1.f)},
      {type::kStrokeOpacity, flag::kUnspecified, false, 1.0f},
      {type::kStrokeWidth, flag::kUnspecified, false, 0.0f},
      {type::kLineJoinType,
       flag::kFlexibleType,
       false,
       static_cast<int>(LineJoinType::kMiter),
       QueryDataType::LINE_JOIN_ENUM,
       convertStringToLineJoinEnum},
      {type::kMiterLimit, flag::kFlexibleType, false, 10.0f}};  // scb: unused?

  render_props_ = std::make_unique<RenderPropertyContainer>(
      ctx, *prop_mark_facade_, std::move(render_property_ci));

  auto const& props = render_props_->getProperties();
  used_props_.insert(props.begin(), props.end());
  used_props_const_.insert(used_props_.begin(), used_props_.end());

  auto const& coord_props = render_props_->getCoordProperties();
  projection_policy_ = std::make_unique<MarkProjectionShaderPolicy>(
      MarkProjectionShaderPolicy::PropMap{coord_props.begin(), coord_props.end()});

  initPropertiesFromJSONObj(obj_loc, true, true);
  initTransformsFromJSONObj(obj_loc, getCoordPropAttrInfos());
  json_path_ = obj_loc.getPathRef();
}

LineMark::~LineMark() {}

BaseRenderPropertyConstSet LineMark::getUsedProps() const {
  return used_props_const_;
}

void LineMark::initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                         const bool data_changed,
                                         const bool init) {
  RENDER_LOG_SCOPE() << " data_changed: " << data_changed << "  init: " << init;
  auto const prop_loc = obj_loc.getMember(JSONSchema_v1::Marks::kPropertiesProp);
  RUNTIME_EX_ASSERT(
      prop_loc.isValid(),
      RapidJSONUtils::createJsonParseError(
          prop_loc,
          "Line mark objects must have a \"" +
              std::string(JSONSchema_v1::Marks::kPropertiesProp) + "\" property."));

  auto prev_path = properties_json_path_;
  properties_json_path_ = prop_loc.getPathRef();
  if (!ctx_.isJSONCacheUpToDate(prev_path, prop_loc) || data_changed || init) {
    RUNTIME_EX_ASSERT(prop_loc.isObject(),
                      RapidJSONUtils::createJsonParseError(
                          prop_loc, "Property must be a json object."));

    render_props_->initFromJSONObj(obj_loc, data_changed);

    BaseMark::initPropFromJSONObj(&ctx_,
                                  data_,
                                  data_changed,
                                  prop_loc,
                                  &fill_below_line_,
                                  properties_json_path_,
                                  BaseMark::validateBoolPropFunc(*this));

    initIds(data_changed);

    if (init || data_changed) {
      updateProps(getUsedProps());
    }

    updateVisibility(render_props_->isStrokeActive());
  }
}

void LineMark::buildShaders(ShaderBuilderVector& builders,
                            const BaseRenderPropertyConstSet& props,
                            const std::string& ssbo_name,
                            const std::string& ssbo_instance_name) {
  // Build vertex shader uniform render props
  gfx::GlslStructBuilder ubo_struct_builder("LINE_VERT_UBO_TYPE");
  ubo_struct_builder.addMember("uViewProjMatrix", gfx::BufferAttrType::kMat3x2f);

  const bool use_ssbo = (!ssbo_name.empty() && !ssbo_instance_name.empty() &&
                         prop_buf_loc_state_.ssbo_props.size() > 0);

  if (use_ssbo) {
    ubo_struct_builder.addMember("uSSBOIndexBase", gfx::BufferAttrType::kInt);
  }
  ubo_struct_builder.addMember("propCompressionBits", gfx::BufferAttrType::kUint);

  addCommonRenderPropUniforms(ubo_struct_builder);

  // Build stage interface blocks
  gfx::GlslStructBuilder geometry_inputs("GeometryShaderInputs");
  gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
  generate_line_interface_blocks(geometry_inputs, fragment_inputs, hasAccumulator());

  // Write geometry shader inputs to vertex and geometry shaders
  builders[0]->replaceFirstTag("GeometryShaderInputs",
                               geometry_inputs.createInterfaceBlockString(true));
  builders[2]->replaceFirstTag(
      "GeometryShaderInputs",
      geometry_inputs.createInterfaceBlockString(true, std::nullopt, true));

  // Write geometry shader inputs to vertex and geometry shaders
  auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
  builders[1]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
  builders[2]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

  builders[0]->replaceFirstTag("VertexProperties", buildVertexShaderInputs());
  builders[0]->replaceFirstTag("UniformProperties",
                               ubo_struct_builder.createStructString());

  // setup the geom shader for potential accumulation rendering
  builders[2]->replaceFirstTag(
      "doStrokeAccum",
      std::to_string(
          render_props_->getProperty(RenderPropertyContainer::PropId::kStrokeColor)
              ->hasAccumulator()));

  std::stringstream get_prop_ss;
  streamPropertyGetters(used_props_const_, get_prop_ss, projection_policy_.get());
  builders[0]->replaceFirstTag("PropertyGetters", get_prop_ss.str());

  // Inject all the remaining code
  BaseMark::insertPropertyCodeInShaderBuilders(
      builders, props, *projection_policy_, &ssbo_name, &ssbo_instance_name);
}

void LineMark::dataRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj) {
  if (!shader_dirty_ && per_gpu_data_.size() &&
      (ref_event_type == RefEventType::kUpdate ||
       ref_event_type == RefEventType::kReplace)) {
    auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(ref_obj);
    if (data) {
      if (data->hasLayoutChanged(QDTLayoutChangedFlags::kSsboContents)) {
        setShaderDirty();
      }
    }
  }
}

void LineMark::updateShader() {
  RENDER_LOG_SCOPE() << "building glsl shaders";

  updateGeoPropInfoAndPropCompressionBits({kPOINT});

  auto const use_fill_below_geom_template = fill_below_line_.getUniformValue<int>() != 0;

  auto& shader_mgr = ctx_.getShaderManager();
  auto builders = shader_mgr.createBuilderVector(
      {{"Marks/lineTemplate.vert"},
       {"Marks/lineTemplate.frag"},
       {use_fill_below_geom_template ? "Marks/lineTemplate_FillBelow.geom"
                                     : "Marks/lineTemplate.geom"}});

  builders[0]->setExternalUniformBuffers({"SLAB_ADDRESS_TABLE_UBO"});

  buildShaders(builders, used_props_const_, getLineBlockName(), "lineData");

  // Line only has subroutines in the vertex and fragment shaders
  buildSubroutineBindings(*builders[0]);
  buildSubroutineBindings(*builders[1]);

  ctx_.clearMarkShaders(*this);
  ctx_.buildMarkShaders(
      *this, MarkGpuResourceSlot::kStroke, "LineMark Stroke", std::move(builders));

  shader_dirty_ = false;

  // set the props dirty to force a rebind with the new shader
  setPropsDirty();
}

void LineMark::buildPipelineDescriptors() {
  if (!pipeline_descriptor_) {
    pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
    CHECK(pipeline_descriptor_);
  }

  pipeline_descriptor_->setRasterSampleCount(getRasterizationSampleCount());
  pipeline_descriptor_->getPushConstantRanges().clear();
  pipeline_descriptor_->getPushConstantRanges().insert(
      ShaderStageBits::kVertex, 0, sizeof(uint32_t));
}

void LineMark::buildPipelines(MarkPerGpuData& per_gpu_data) {
  CHECK(per_gpu_data.stroke_primitive_assemblies.size());
  CHECK(per_gpu_data.stroke_primitive_assemblies[0]);
  CHECK(per_gpu_data.stroke_materials.size());
  CHECK(per_gpu_data.stroke_materials[0]);

  per_gpu_data.destroyPipelines();

  per_gpu_data.graphics_pipelines.push_back(
      per_gpu_data.getResourceManager().createGraphicsPipeline(
          "LineMark Stroke",
          *per_gpu_data.stroke_materials[0],
          *pipeline_descriptor_,
          per_gpu_data.stroke_primitive_assemblies[0].get()));
  per_gpu_data.graphics_pipelines[0]->create(
      per_gpu_data.getRootPerGpuData().getCommonRenderPass(
          CommonRenderPassType::kAllAttachments, needsMultisampleEnabled()));
}

void LineMark::buildPrimitiveAssemblyData(const GpuId& gpu_id,
                                          const BaseRenderPropertyConstSet& vbo_props,
                                          const BaseDataTableShPtr& data,
                                          gfx::Material& active_material,
                                          gfx::PrimitiveAssemblyAttrInfo& attr_info,
                                          const gfx::IndexBuffer*& ibo) {
  int attr_count = 0;
  int vbo_size = 0;
  int prop_size = 0;
  for (auto const* prop : vbo_props) {
    if (!active_material.hasVertexAttribute(prop->getName())) {
      continue;
    }
    attr_count++;
    prop_size = prop->size(gpu_id);
    if (attr_count == 1) {
      vbo_size = prop_size;
    } else {
      RUNTIME_EX_ASSERT(prop_size == vbo_size,
                        "Invalid line mark. The sizes of the vertex buffer attributes do "
                        "not match for gpuId " +
                            std::to_string(gpu_id) + ". " + std::to_string(vbo_size) +
                            "!=" + std::to_string(prop_size));
    }
    prop->addToPrimitiveAssemblyAttrInfo(gpu_id, attr_info);
  }

  CHECK(data);
  auto line_table = std::dynamic_pointer_cast<BaseLineDataTable>(data);
  CHECK(line_table);
  ibo = line_table->getGpuResources().getIndexBuffer(gpu_id);
}

void LineMark::buildStrokePrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  CHECK(!gpu_data.stroke_materials.empty());
  gfx::PrimitiveAssemblyAttrInfo attr_info;
  const gfx::IndexBuffer* ibo = nullptr;
  buildPrimitiveAssemblyData(gpu_data.getGpuId(),
                             prop_buf_loc_state_.vbo_props,
                             data_,
                             *gpu_data.stroke_materials[0],
                             attr_info,
                             ibo);
  gpu_data.stroke_primitive_assemblies.clear();
  gpu_data.stroke_primitive_assemblies.push_back(
      gpu_data.getResourceManager().createPrimitiveAssembly(
          "LineMark",
          gfx::PrimitiveTopology::kLineStripAdjacency,
          *gpu_data.stroke_materials[0],
          attr_info,
          ibo));
}

void LineMark::buildSubroutineBindings(ShaderBuilder& builder) {
  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    if (used_props_const_.find(prop) != used_props_const_.end()) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->buildSubroutineBindings(builder, "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.ssbo_props) {
    if (used_props_const_.find(prop) != used_props_const_.end()) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->buildSubroutineBindings(builder, "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      scale_ref->buildSubroutineBindings(builder, "_" + prop->getName());
    }
  }
  BaseMark::setColorConvertSubroutines(
      builder, render_props_->getProperty(RenderPropertyContainer::PropId::kStrokeColor));
}

void LineMark::setUniformAttributes(MarkPerGpuData& gpu_data) {
  RENDER_LOG_SCOPE();
  auto& active_material = *gpu_data.stroke_materials[0];
  auto const& props = used_props_const_;

  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    if (props.find(prop) != props.end() &&
        active_material.hasVertexAttribute(prop->getName())) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->bindUniforms(active_material, "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.ssbo_props) {
    if (props.find(prop) != props.end()) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->bindUniforms(active_material, "_" + prop->getName());
      }
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
    if (props.find(prop) != props.end()) {
      prop->setDecimalScaleUniformAttribute(active_material);
    }
  }

  BaseMark::bindIDPropUniformAttributes(active_material);
  BaseMark::setProjectionUniformAttributes(active_material);

  if (prop_buf_loc_state_.ssbo_props.size()) {
    // the same ssbo should be used for all ssbo props,
    // so only need to grab from first one
    auto* ssbo = (*prop_buf_loc_state_.ssbo_props.begin())
                     ->getSsboPtr(gpu_data.getGpuId())
                     ->unmapForDraw();
    CHECK(ssbo);

    active_material.setUniformAttribute("uSSBOIndexBase", static_cast<int32_t>(0));
    active_material.bindShaderStorageBufferToBlock(getLineBlockName(), *ssbo);
  }

  active_material.setUniformAttribute("uViewProjMatrix",
                                      ctx_.getViewProjMatrix().getDataArrayRef());
  // set the viewport in the shader, the viewport is
  // needed to do screen-space line widths
  active_material.setViewportAttributes(0, 0, ctx_.getWidth(), ctx_.getHeight());

  // update prop compression bits again for the case where only the compression changes
  updateGeoPropInfoAndPropCompressionBits({kPOINT});

  updateSlabAddressTableAndPropCompressionBitsUniforms(gpu_data);
}

void LineMark::updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                                const std::vector<GpuId>& remove_gpus) {
  for (auto const& prop : used_props_) {
    prop->initGpuResources(add_gpus, remove_gpus);
  }
}

bool LineMark::draw(const gfx::DeviceContext& device_ctx,
                    const MarkPerGpuData& mark_gpu_data,
                    gfx::Framebuffer& framebuffer,
                    const int accumulator_index) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());
  // NOTE: shader should have been updated before calling this
  CHECK(data_);
  auto line_table = std::dynamic_pointer_cast<BaseLineDataTable>(data_);
  CHECK(line_table);

  // draw lines
  if (mark_gpu_data.stroke_materials.empty()) {
    return false;
  }
  CHECK(!mark_gpu_data.stroke_primitive_assemblies.empty());

  auto const& root_gpu_data = mark_gpu_data.getRootPerGpuData();
  auto& primitive_assembly = mark_gpu_data.stroke_primitive_assemblies[0];
  mark_gpu_data.stroke_materials[0]->updateDescriptorSets();

  auto& cmd_list = root_gpu_data.getCommandList();
  auto const gpu_id = device_ctx.getGpuId();

  auto const* vbo = primitive_assembly->getVertexBuffer();
  CHECK(vbo);

  auto& render_pass = selectDrawRenderPass(root_gpu_data, accumulator_index);

  auto const* indibo = line_table->getGpuResources().getIndirectDrawIndexBuffer(gpu_id);
  if (indibo) {
    auto const* ibo = primitive_assembly->getIndexBuffer();
    CHECK(ibo);
    auto const num_items = indibo->numItems();
    cmd_list.pushLabel("Line draw ibo")
        .setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[0],
                               "totalNumItems",
                               ShaderStageBits::kVertex,
                               std::max(num_items, 1u));
    cmd_list.beginRenderPass(render_pass, framebuffer)
        .drawIndirectIndexed(
            *mark_gpu_data.graphics_pipelines[0], *vbo, *ibo, *indibo, num_items)
        .endRenderPass()
        .popLabel()
        .flush("LineMark IBO draw", gfx::CommandList::SubmitType::kImmediateReturn);
  } else {
    auto const* indvbo =
        line_table->getGpuResources().getIndirectDrawVertexBuffer(gpu_id);
    CHECK(indvbo);
    auto const num_items = indvbo->numItems();
    cmd_list.pushLabel("Line draw vbo")
        .setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[0],
                               "totalNumItems",
                               ShaderStageBits::kVertex,
                               std::max(num_items, 1u));
    cmd_list.beginRenderPass(render_pass, framebuffer)
        .drawIndirect(*mark_gpu_data.graphics_pipelines[0], *vbo, *indvbo, num_items)
        .endRenderPass()
        .popLabel()
        .flush("LineMark VBO draw", gfx::CommandList::SubmitType::kImmediateReturn);
  }

  return true;
}

LineMark::operator std::string() const {
  return "LineMark " + std::string(ctx_.getRenderSessionKey());
}
};  // namespace QueryRenderer
