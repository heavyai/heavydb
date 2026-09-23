/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/PolyMark.h"

#include <algorithm>
#include <vector>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Render/PPLLConstants.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Data/QueryPolyDataTable.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Marks/LineUtils.h"
#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderPropertyContainer.h"
#include "QueryRenderer/Marks/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"

#include "Shared/measure.h"

#define DEFAULT_MAX_UNIQUE_IDS 64u

namespace QueryRenderer {

using ::gfx::ShaderBlockLayout;
using ::gfx::ShaderStage;
using ::gfx::ShaderStageBits;
using ShaderBuilder = ::gfx::ShaderManager::Builder;

namespace {
inline std::string getFillBlockName() {
  return "PolyData";
}

inline std::string getStrokeBlockName() {
  return "LineData";
}
}  // namespace

PolyMark::PolyMark(const JSONLocation& obj_loc, QueryRendererContext& ctx)
    : BaseMark(GeomType::kPolys, ctx, obj_loc, DataOutputFormat::kPolys, true) {
  using type = RenderPropertyCreateInfo::Type;
  using flag = RenderPropertyFlagBits;
  std::vector<RenderPropertyCreateInfo> render_property_ci{
      {type::kFillColor, flag::kUseScale, false, gfx::ColorUnion(0.f, 0.f, 0.f, 1.f)},
      {type::kFillOpacity, flag::kUseScale, false, 1.0f},
      {type::kStrokeColor, flag::kUseScale, false, gfx::ColorUnion(1.f, 1.f, 1.f, 1.f)},
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

  using PropId = RenderPropertyContainer::PropId;
  used_fill_props_ = {render_props_->getProperty(PropId::kX),
                      render_props_->getProperty(PropId::kY),
                      render_props_->getProperty(PropId::kOpacity),
                      render_props_->getProperty(PropId::kFillColor),
                      render_props_->getProperty(PropId::kFillOpacity)};

  used_stroke_props_ = {render_props_->getProperty(PropId::kX),
                        render_props_->getProperty(PropId::kY),
                        render_props_->getProperty(PropId::kOpacity),
                        render_props_->getProperty(PropId::kStrokeColor),
                        render_props_->getProperty(PropId::kStrokeOpacity),
                        render_props_->getProperty(PropId::kStrokeWidth),
                        render_props_->getProperty(PropId::kLineJoinType),
                        render_props_->getProperty(PropId::kMiterLimit)};

  used_fill_props_const_.insert(used_fill_props_.begin(), used_fill_props_.end());
  used_stroke_props_const_.insert(used_stroke_props_.begin(), used_stroke_props_.end());

  auto const& coord_props = render_props_->getCoordProperties();
  used_projection_props_const_ = {coord_props.begin(), coord_props.end()};
  projection_policy_ = std::make_unique<MarkProjectionShaderPolicy>(
      MarkProjectionShaderPolicy::PropMap{coord_props.begin(), coord_props.end()});

  initPropertiesFromJSONObj(obj_loc, true, true);
  initTransformsFromJSONObj(obj_loc, getCoordPropAttrInfos());
  json_path_ = obj_loc.getPathRef();
}

PolyMark::~PolyMark() {}

BaseRenderPropertyConstSet PolyMark::getUsedProps() const {
  BaseRenderPropertyConstSet rtn(used_fill_props_const_);
  rtn.insert(used_stroke_props_const_.begin(), used_stroke_props_const_.end());
  return rtn;
}

void PolyMark::initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                         const bool data_changed,
                                         const bool init) {
  RENDER_LOG_SCOPE() << " data_changed: " << data_changed << "  init: " << init;
  const auto prop_loc = obj_loc.getMember(JSONSchema_v1::Marks::kPropertiesProp);
  RUNTIME_EX_ASSERT(
      prop_loc.isValid(),
      RapidJSONUtils::createJsonParseError(
          obj_loc,
          "Poly mark objects must have a \"" +
              std::string(JSONSchema_v1::Marks::kPropertiesProp) + "\" property."));

  auto prev_path = properties_json_path_;
  properties_json_path_ = obj_loc.getPathRef();
  if (!ctx_.isJSONCacheUpToDate(prev_path, prop_loc) || data_changed || init) {
    RUNTIME_EX_ASSERT(prop_loc.isObject(),
                      RapidJSONUtils::createJsonParseError(
                          prop_loc, "Property must be a json object."));

    render_props_->initFromJSONObj(obj_loc, data_changed);

    initIds(data_changed);

    if (init || data_changed) {
      updateProps(getUsedProps());
    }

    updateVisibility(render_props_->isFillActive() || render_props_->isStrokeActive());
  }
}

void PolyMark::initPPLLPerGpuData(MarkPerGpuData& gpu_data) {
  gpu_data.ppll_render =
      std::make_unique<gfx::PPLLRender>(gpu_data.getDeviceContext(),
                                        gpu_data.getRootPerGpuData().getPPLLResources(),
                                        true,
                                        true,
                                        true,
                                        DEFAULT_MAX_UNIQUE_IDS);
}

void PolyMark::buildShaders(ShaderBuilderVector& builders,
                            const BaseRenderPropertyConstSet& props,
                            const std::string& ssbo_name,
                            const std::string& ssbo_instance_name,
                            const bool auto_inject_main,
                            const std::string& vertex_shader_inputs,
                            const std::string& ubo_block_string) {
  RENDER_LOG_SCOPE();
  builders[0]->replaceFirstTag("VertexProperties", vertex_shader_inputs);
  builders[0]->replaceFirstTag("UniformProperties", std::move(ubo_block_string));

  // Inject remaining code (prop types, scales, projections, casting, etc.)
  auto* ssbo_name_p = ssbo_name.size() ? &ssbo_name : nullptr;
  auto* ssbo_instance_name_p = ssbo_instance_name.size() ? &ssbo_instance_name : nullptr;
  BaseMark::insertPropertyCodeInShaderBuilders(builders,
                                               props,
                                               *projection_policy_,
                                               ssbo_name_p,
                                               ssbo_instance_name_p,
                                               auto_inject_main);
}

void PolyMark::dataRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj) {
  if (!shader_dirty_ && per_gpu_data_.size() &&
      (ref_event_type == RefEventType::kUpdate ||
       ref_event_type == RefEventType::kReplace)) {
    auto data = std::dynamic_pointer_cast<SqlQueryPolyDataTableJSON>(ref_obj);
    if (data) {
      if (data->hasLayoutChanged(QDTLayoutChangedFlags::kSsboContents)) {
        setShaderDirty();
      }
      if (data->hasQueryTypeChanged()) {
        setPropsDirty();
      }
    }
  }
}

void PolyMark::updateShader() {
  RENDER_LOG_SCOPE() << "building glsl shaders";
  ctx_.clearMarkShaders(*this);

  const bool use_ssbo =
      (getFillBlockName().size() && !prop_buf_loc_state_.ssbo_props.empty());

  // Build vertex shader inputs and uniform block strings to use in the various shaders
  gfx::GlslStructBuilder vertex_ubo_builder("POLYMARK_VERTEX_UBO");
  vertex_ubo_builder.addMember("uViewProjMatrix", gfx::BufferAttrType::kMat3x2f);
  addCommonRenderPropUniforms(vertex_ubo_builder);
  if (use_ssbo) {
    vertex_ubo_builder.addMember("uSSBOIndexBase", gfx::BufferAttrType::kInt);
  }
  auto vertex_ubo_string = vertex_ubo_builder.createStructString();

  bool do_fill = render_props_->isFillActive();
  bool do_stroke = render_props_->isStrokeActive();

  std::string vertex_shader_inputs_string;
  if (do_fill || do_stroke) {
    vertex_shader_inputs_string = buildVertexShaderInputs();
  }

  auto& shader_mgr = ctx_.getShaderManager();
  if (do_fill) {
    auto fragment_count_builders = shader_mgr.createBuilderVector(
        {{"Marks/ppllPolyTemplate_count.vert"}, {"PPLL/ppllCountFragments.frag"}});

    auto fragment_capture_builders =
        shader_mgr.createBuilderVector({{"Marks/ppllPolyTemplate_capture.vert"},
                                        {"Marks/ppllPolyTemplate_capture.frag"}});

    fragment_count_builders[0]->replaceFirstTag("VertexProperties",
                                                vertex_shader_inputs_string);
    fragment_capture_builders[0]->replaceFirstTag("VertexProperties",
                                                  vertex_shader_inputs_string);

    fragment_capture_builders[1]->setExternalUniformBuffers({"IMAGE_TILES_UBO"});
    auto num_samples_str = std::to_string(ctx_.getGlobalContext().getNumSamples());
    auto num_tiles_str = std::to_string(gfx::kNumPPLLTiles);
    auto tile_offset_str = std::to_string(kTileIndexPushConstantOffset);

    fragment_capture_builders[1]->replaceFirstTag("numSamples", num_samples_str);
    fragment_capture_builders[1]->replaceFirstTag("numTiles", num_tiles_str);
    fragment_capture_builders[1]->replaceFirstTag("tileIndexOffset", tile_offset_str);

    {
      // count and capture shaders only need x and y
      std::stringstream get_prop_ss;
      streamPropertyGetters(
          used_projection_props_const_, get_prop_ss, projection_policy_.get());
      auto get_prop_string = get_prop_ss.str();
      fragment_count_builders[0]->replaceFirstTag("PropertyGetters", get_prop_string);
      fragment_capture_builders[0]->replaceFirstTag("PropertyGetters",
                                                    std::move(get_prop_string));
    }

    auto fragment_composite_builders =
        shader_mgr.createBuilderVector({{"Marks/ppllPolyTemplate_composite.comp"}});

    fragment_composite_builders[0]->setExternalUniformBuffers(
        {"IMAGE_INFO_UBO", "IMAGE_TILES_UBO"});
    fragment_composite_builders[0]->replaceFirstTag("numSamples", num_samples_str);
    fragment_composite_builders[0]->replaceFirstTag(
        "defaultMaxUniqueIds", std::to_string(DEFAULT_MAX_UNIQUE_IDS));
    fragment_composite_builders[0]->replaceFirstTag("numTiles", num_tiles_str);
    fragment_composite_builders[0]->replaceFirstTag("tileIndexOffset", tile_offset_str);
    fragment_composite_builders[0]->replaceAllTags(
        "workgroupSize",
        std::to_string(
            ctx_.getGlobalContext().getGfxContext().getDeviceLimits().subgroup_size));

    gfx::GlslStructBuilder comp_ubo_builder("POLYMARK_COMPOSITE_UBO");
    comp_ubo_builder.addMember("colorOutputMode", gfx::BufferAttrType::kInt);
    if (use_ssbo) {
      comp_ubo_builder.addMember("ssboMaxIndex", gfx::BufferAttrType::kInt);
    }
    addCommonRenderPropUniforms(comp_ubo_builder);

    std::stringstream get_prop_ss;
    streamPropertyGetters(used_fill_props_const_, get_prop_ss, projection_policy_.get());
    auto get_prop_string = get_prop_ss.str();

    // TODO (scb): Move prop getter stuff to `buildShaders`?
    fragment_composite_builders[0]->replaceFirstTag("PropertyGetters",
                                                    std::move(get_prop_string));

    buildShaders(fragment_count_builders,
                 used_fill_props_const_,
                 getFillBlockName(),
                 "polyData",
                 false,
                 vertex_shader_inputs_string,
                 vertex_ubo_string);
    buildShaders(fragment_capture_builders,
                 used_fill_props_const_,
                 getFillBlockName(),
                 "polyData",
                 false,
                 vertex_shader_inputs_string,
                 vertex_ubo_string);
    buildShaders(fragment_composite_builders,
                 used_fill_props_const_,
                 getFillBlockName(),
                 "polyData",
                 false,
                 vertex_shader_inputs_string,
                 comp_ubo_builder.createStructString());

    buildSubroutineBindings(*fragment_capture_builders[0], used_fill_props_const_);
    buildSubroutineBindings(*fragment_composite_builders[0], used_fill_props_const_);
    BaseMark::setColorConvertSubroutines(
        *fragment_composite_builders[0],
        render_props_->getProperty(RenderPropertyContainer::PropId::kFillColor));

    ctx_.buildMarkShaders(*this,
                          MarkGpuResourceSlot::kFill,
                          "PolyMark Count Fragments",
                          std::move(fragment_count_builders));
    ctx_.buildMarkShaders(*this,
                          MarkGpuResourceSlot::kFill,
                          "PolyMark Capture Fragments",
                          std::move(fragment_capture_builders));
    ctx_.buildMarkShaders(*this,
                          MarkGpuResourceSlot::kFill,
                          "PolyMark Composite Fragments",
                          std::move(fragment_composite_builders));
  }

  if (do_stroke) {
    auto line_builders = shader_mgr.createBuilderVector({{"Marks/lineTemplate.vert"},
                                                         {"Marks/lineTemplate.frag"},
                                                         {"Marks/lineTemplate.geom"}});
    std::stringstream get_prop_ss;
    streamPropertyGetters(
        used_stroke_props_const_, get_prop_ss, projection_policy_.get());
    line_builders[0]->replaceFirstTag("PropertyGetters", get_prop_ss.str());

    gfx::GlslStructBuilder geometry_inputs("GeometryShaderInputs");
    gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
    generate_line_interface_blocks(geometry_inputs, fragment_inputs, hasAccumulator());

    // Write geometry shader inputs to vertex and geometry shaders
    line_builders[0]->replaceFirstTag("GeometryShaderInputs",
                                      geometry_inputs.createInterfaceBlockString(true));
    line_builders[2]->replaceFirstTag(
        "GeometryShaderInputs",
        geometry_inputs.createInterfaceBlockString(true, std::nullopt, true));

    // Write fragment shader inputs to geometry and fragment shaders
    auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
    line_builders[1]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
    line_builders[2]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

    buildShaders(line_builders,
                 used_stroke_props_const_,
                 getStrokeBlockName(),
                 "lineData",
                 true,
                 vertex_shader_inputs_string,
                 vertex_ubo_string);
    line_builders[2]->replaceFirstTag("doStrokeAccum", std::to_string(false));

    buildSubroutineBindings(*line_builders[0], used_stroke_props_const_);
    buildSubroutineBindings(*line_builders[1], used_stroke_props_const_);
    BaseMark::setColorConvertSubroutines(
        *line_builders[0],
        render_props_->getProperty(RenderPropertyContainer::PropId::kStrokeColor));

    ctx_.buildMarkShaders(
        *this, MarkGpuResourceSlot::kStroke, "PolyMark Stroke", std::move(line_builders));
  }

  shader_dirty_ = false;

  // set the props dirty to force a rebind with the new shader
  setPropsDirty();
}

gfx::PushConstantRanges PolyMark::capture_and_comp_push_constants = {
    // batch offset push constant, offset = 0, size = 4
    gfx::PushConstantRange(gfx::ShaderStageBits::kVertex, 0u, sizeof(uint32_t)),
    // tile index push constant, offset = 4, size = 4
    gfx::PushConstantRange(
        gfx::ShaderStageBits::kFragment | gfx::ShaderStageBits::kCompute,
        kTileIndexPushConstantOffset,
        sizeof(uint32_t))};

void PolyMark::buildPipelineDescriptors() {
  if (!pipeline_descriptors_[kCountFragmentsPipeline]) {
    CHECK(!pipeline_descriptors_[kCaptureFragmentsPipeline]);
    CHECK(!pipeline_descriptors_[kOutlinePipeline]);
    pipeline_descriptors_[kCountFragmentsPipeline] =
        std::make_unique<gfx::PipelineDescriptor>();
    pipeline_descriptors_[kCaptureFragmentsPipeline] =
        std::make_unique<gfx::PipelineDescriptor>();
    pipeline_descriptors_[kCaptureFragmentsPipeline]->setPushConstantRanges(
        capture_and_comp_push_constants);
    pipeline_descriptors_[kOutlinePipeline] = std::make_unique<gfx::PipelineDescriptor>();
    CHECK(pipeline_descriptors_[kCountFragmentsPipeline]);
    CHECK(pipeline_descriptors_[kCaptureFragmentsPipeline]);
    CHECK(pipeline_descriptors_[kOutlinePipeline]);
  }

  auto num_samples = getRasterizationSampleCount();
  pipeline_descriptors_[kCountFragmentsPipeline]->setRasterSampleCount(num_samples);

  pipeline_descriptors_[kCaptureFragmentsPipeline]->setRasterSampleCount(num_samples);

  //
  // outline
  //
  pipeline_descriptors_[kOutlinePipeline]->setRasterSampleCount(num_samples);
  pipeline_descriptors_[kOutlinePipeline]->setEnableDepthTest(true);
  pipeline_descriptors_[kOutlinePipeline]->getPushConstantRanges().clear();
  pipeline_descriptors_[kOutlinePipeline]->getPushConstantRanges().insert(
      ShaderStageBits::kVertex, 0, sizeof(uint32_t));
}

void PolyMark::buildPipelines(MarkPerGpuData& per_gpu_data) {
  per_gpu_data.destroyPipelines();
  // reset the cached overflow specialization size
  per_gpu_data.graphics_pipelines.resize(static_cast<int>(kNumGraphicsPipelines));
  per_gpu_data.compute_pipelines.resize(static_cast<int>(kNumComputePipelines));
  auto& resource_mgr = per_gpu_data.getResourceManager();
  auto& root_gpu_data = per_gpu_data.getRootPerGpuData();

  if (per_gpu_data.fill_primitive_assemblies.size() &&
      per_gpu_data.fill_primitive_assemblies[0]) {
    CHECK(per_gpu_data.fill_materials.size() && per_gpu_data.fill_materials[0]);

    CHECK(per_gpu_data.ppll_render);
    auto& empty_render_pass = root_gpu_data.getEmptyRenderPass();

    per_gpu_data.graphics_pipelines[kCountFragmentsPipeline] =
        resource_mgr.createGraphicsPipeline(
            "PolyMark Count Fragments",
            *per_gpu_data.fill_materials[kCountFragmentsMaterial],
            *pipeline_descriptors_[kCountFragmentsPipeline],
            per_gpu_data.fill_primitive_assemblies[kCountFragmentsPipeline].get());
    per_gpu_data.graphics_pipelines[kCountFragmentsPipeline]->create(empty_render_pass);

    per_gpu_data.graphics_pipelines[kCaptureFragmentsPipeline] =
        resource_mgr.createGraphicsPipeline(
            "PolyMark Capture Fragments",
            *per_gpu_data.fill_materials[kCaptureFragmentsMaterial],
            *pipeline_descriptors_[kCaptureFragmentsPipeline],
            per_gpu_data.fill_primitive_assemblies[kCaptureFragmentsPipeline].get());
    per_gpu_data.graphics_pipelines[kCaptureFragmentsPipeline]->create(empty_render_pass);

    static std::vector<gfx::SpecializationMapEntry> spec_map = {
        {0, 0, sizeof(uint32_t)}, {1, sizeof(uint32_t), sizeof(uint32_t)}};
    per_gpu_data.compute_pipelines[kCompositeFragmentsPipeline] =
        resource_mgr.createComputePipeline(
            "PolyMark Composite Fragments",
            *per_gpu_data.fill_materials[kCompositeFragmentsMaterial],
            spec_map,
            capture_and_comp_push_constants);
    // create default specialization (DEFAULT_MAX_UNIQUE_IDS)
    per_gpu_data.compute_pipelines[kCompositeFragmentsPipeline]->create();

    per_gpu_data.ppll_render->setPipelineResources(
        {per_gpu_data.fill_materials[kCountFragmentsMaterial].get(),
         per_gpu_data.graphics_pipelines[kCountFragmentsPipeline].get()},
        {per_gpu_data.fill_materials[kCaptureFragmentsMaterial].get(),
         per_gpu_data.graphics_pipelines[kCaptureFragmentsPipeline].get()},
        {per_gpu_data.fill_materials[kCompositeFragmentsMaterial].get(),
         per_gpu_data.compute_pipelines[kCompositeFragmentsPipeline].get()});
  }

  if (per_gpu_data.stroke_primitive_assemblies.size() &&
      per_gpu_data.stroke_primitive_assemblies[0]) {
    CHECK(per_gpu_data.stroke_materials.size() && per_gpu_data.stroke_materials[0]);

    // outline
    per_gpu_data.graphics_pipelines[kOutlinePipeline] =
        resource_mgr.createGraphicsPipeline(
            "PolyMark Outline",
            *per_gpu_data.stroke_materials[0],
            *pipeline_descriptors_[kOutlinePipeline],
            per_gpu_data.stroke_primitive_assemblies[0].get());
    per_gpu_data.graphics_pipelines[kOutlinePipeline]->create(
        per_gpu_data.getRootPerGpuData().getCommonRenderPass(
            CommonRenderPassType::kAllAttachments, needsMultisampleEnabled()));
  }
}

void PolyMark::buildSubroutineBindings(ShaderBuilder& builder,
                                       const BaseRenderPropertyConstSet& props) {
  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    if (props.find(prop) != props.end()) {
      const auto& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->buildSubroutineBindings(builder, "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.ssbo_props) {
    if (props.find(prop) != props.end()) {
      const auto& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->buildSubroutineBindings(builder, "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    if (props.find(prop) != props.end()) {
      const auto& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->buildSubroutineBindings(builder, "_" + prop->getName());
      }
    }
  }
}

void PolyMark::buildAttrMap(const GpuId& gpu_id,
                            const BaseRenderPropertyConstSet& vbo_props,
                            gfx::Material& active_material,
                            gfx::PrimitiveAssemblyAttrInfo& attr_info) {
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
                        "Invalid poly mark. The sizes of the vertex buffer attributes do "
                        "not match for gpuId " +
                            std::to_string(gpu_id) + ". " + std::to_string(vbo_size) +
                            "!=" + std::to_string(prop_size));
    }
    prop->addToPrimitiveAssemblyAttrInfo(gpu_id, attr_info);
  }
}

void PolyMark::buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  CHECK(!gpu_data.fill_materials.empty());
  gpu_data.fill_primitive_assemblies.clear();
  auto gpu_id = gpu_data.getGpuId();
  auto& resource_mgr = gpu_data.getResourceManager();

  gfx::PrimitiveAssemblyAttrInfo attr_info_count_fragments, attr_info_capture_fragments;
  buildAttrMap(gpu_id,
               prop_buf_loc_state_.vbo_props,
               *gpu_data.fill_materials[kCountFragmentsMaterial],
               attr_info_count_fragments);
  buildAttrMap(gpu_id,
               prop_buf_loc_state_.vbo_props,
               *gpu_data.fill_materials[kCaptureFragmentsMaterial],
               attr_info_capture_fragments);
  gpu_data.fill_primitive_assemblies.push_back(resource_mgr.createPrimitiveAssembly(
      "PolyMark (Fill, Count)",
      gfx::PrimitiveTopology::kTriangleFan,
      *gpu_data.fill_materials[kCountFragmentsMaterial],
      attr_info_count_fragments,
      nullptr));
  gpu_data.fill_primitive_assemblies.push_back(resource_mgr.createPrimitiveAssembly(
      "PolyMark (Fill, Capture)",
      gfx::PrimitiveTopology::kTriangleFan,
      *gpu_data.fill_materials[kCaptureFragmentsMaterial],
      attr_info_capture_fragments,
      nullptr));
}

void PolyMark::buildStrokePrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  CHECK(!gpu_data.stroke_materials.empty());
  gfx::PrimitiveAssemblyAttrInfo attr_info_stroke;
  buildAttrMap(gpu_data.getGpuId(),
               prop_buf_loc_state_.vbo_props,
               *gpu_data.stroke_materials[0],
               attr_info_stroke);
  gpu_data.stroke_primitive_assemblies.clear();
  gpu_data.stroke_primitive_assemblies.push_back(
      gpu_data.getResourceManager().createPrimitiveAssembly(
          "PolyMark (Stroke)",
          gfx::PrimitiveTopology::kLineStripAdjacency,
          *gpu_data.stroke_materials[0],
          attr_info_stroke,
          nullptr));
}

void PolyMark::setUniformAttributes(MarkPerGpuData& gpu_data) {
  RENDER_LOG_SCOPE();
  auto gpu_id = gpu_data.getGpuId();
  CHECK(data_);
  auto poly_table = std::dynamic_pointer_cast<BasePolyDataTable>(data_);
  CHECK(poly_table);
  if (!gpu_data.fill_materials.empty()) {
    // Count fragments material
    auto& count_fragments_material = *gpu_data.fill_materials[kCountFragmentsMaterial];

    setUniformAttributes(count_fragments_material, used_fill_props_const_, true, true);

    // Capture fragments material
    auto& capture_fragments_material =
        *gpu_data.fill_materials[kCaptureFragmentsMaterial];

    setUniformAttributes(capture_fragments_material, used_fill_props_const_, true, true);

    // Composite fragments material
    auto& composite_fragment_material =
        *gpu_data.fill_materials[kCompositeFragmentsMaterial];

    if (!prop_buf_loc_state_.ssbo_props.empty()) {
      auto* ssbo =
          (*prop_buf_loc_state_.ssbo_props.begin())->getSsboPtr(gpu_id)->unmapForDraw();
      CHECK(ssbo);
      composite_fragment_material.bindShaderStorageBufferToBlock(getFillBlockName(),
                                                                 *ssbo);
      composite_fragment_material.setUniformAttribute(
          "ssboMaxIndex", ssbo->getLayoutManager()->numItems() - 1);
    }

    setUniformAttributes(
        composite_fragment_material, used_fill_props_const_, false, false);
  }
  if (!gpu_data.stroke_materials.empty()) {
    auto& stroke_material = *gpu_data.stroke_materials[0];
    setUniformAttributes(stroke_material, used_stroke_props_const_, false, true);

    if (prop_buf_loc_state_.ssbo_props.size()) {
      // the same SSBO should be used for all SSBO
      // props, so only need to grab from first one
      auto* ssbo =
          (*prop_buf_loc_state_.ssbo_props.begin())->getSsboPtr(gpu_id)->unmapForDraw();
      CHECK(ssbo);

      // Validate SSBO item counts
      auto indvbo_lines =
          poly_table->getGpuResources().getIndirectDrawVertexBuffer_lines(gpu_id);
      CHECK(indvbo_lines);
      CHECK(ssbo->getLayoutManager()->numItems() == indvbo_lines->numItems());

      stroke_material.bindShaderStorageBufferToBlock(getStrokeBlockName(), *ssbo);
      stroke_material.setUniformAttribute("uSSBOIndexBase", static_cast<int32_t>(0));
    }
  }
}

void PolyMark::setUniformAttributes(gfx::Material& active_material,
                                    const BaseRenderPropertyConstSet& props,
                                    const bool is_stencil_pass,
                                    const bool requires_projection) {
  RENDER_LOG_SCOPE();

  if (!is_stencil_pass) {
    for (auto const* prop : prop_buf_loc_state_.vbo_props) {
      if (props.find(prop) != props.end() &&
          active_material.hasVertexAttribute(prop->getName())) {
        const auto& scale_ref = prop->getScaleReference();
        if (scale_ref != nullptr) {
          RENDER_LOG() << "bind vbo property: " << prop->getName();
          scale_ref->bindUniforms(active_material, "_" + prop->getName());
        }
      }
    }

    for (auto const* prop : prop_buf_loc_state_.ssbo_props) {
      if (props.find(prop) != props.end()) {
        const auto& scale_ref = prop->getScaleReference();
        if (scale_ref != nullptr) {
          RENDER_LOG() << "bind ssbo property: " << prop->getName();
          scale_ref->bindUniforms(active_material, "_" + prop->getName());
        }
      }
    }

    for (auto const* prop : prop_buf_loc_state_.uniform_props) {
      if (props.find(prop) != props.end()) {
        const auto& scale_ref = prop->getScaleReference();
        if (scale_ref != nullptr) {
          scale_ref->bindUniforms(active_material, "_" + prop->getName());
        }

        prop->setUniformAttribute(active_material, prop->getName());
      }
    }

    for (auto const* prop : prop_buf_loc_state_.decimal_props) {
      if (props.find(prop) != props.end()) {
        prop->setDecimalScaleUniformAttribute(active_material);
      }
    }

    BaseMark::bindIDPropUniformAttributes(active_material);
  }

  if (requires_projection) {
    BaseMark::setProjectionUniformAttributes(active_material);

    active_material.setUniformAttribute("uViewProjMatrix",
                                        ctx_.getViewProjMatrix().getDataArrayRef());
    if (hasProjection()) {
      active_material.setViewportAttributes(0, 0, ctx_.getWidth(), ctx_.getHeight());
    }
  }
}

void PolyMark::updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                                const std::vector<GpuId>& remove_gpus) {
  BaseRenderPropertySet props(used_fill_props_);
  props.insert(used_stroke_props_.begin(), used_stroke_props_.end());
  for (auto const& prop : props) {
    prop->initGpuResources(add_gpus, remove_gpus);
  }
}

CommonRenderPassTypeBits PolyMark::getRequiredCommonRenderPassTypes() const {
  return CommonRenderPassTypeBits::kAllAttachments |
         CommonRenderPassTypeBits::kDepthStencilThenAll;
}

void PolyMark::drawFill(const gfx::DeviceContext& device_ctx,
                        MarkPerGpuData& mark_gpu_data,
                        gfx::Framebuffer& framebuffer,
                        const int accumulator_index) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());

  CHECK(data_);
  auto poly_table = std::dynamic_pointer_cast<BasePolyDataTable>(data_);
  CHECK(poly_table);

  // Get Gpu data
  auto& root_gpu_data = mark_gpu_data.getRootPerGpuData();
  bool do_hit_testing = ctx_.doHitTest() && render_props_->ids.size();

  //
  // Get framebuffer attachments since we write to them directly with imageStore
  //
  auto& attachment_mgr = framebuffer.getAttachmentManager();
  auto& output_rgb_texture =
      *attachment_mgr.getAttachmentTexture(gfx::Framebuffer::Attachment::kColor0);
  auto& composite_material = *mark_gpu_data.fill_materials[kCompositeFragmentsMaterial];
  composite_material.setImageLoadStoreAttribute("outputImageMS", output_rgb_texture);
  if (do_hit_testing) {
    composite_material.setImageLoadStoreAttribute(
        "outputIDA",
        *attachment_mgr.getAttachmentTexture(gfx::Framebuffer::Attachment::kColor1));
    composite_material.setImageLoadStoreAttribute(
        "outputIDB",
        *attachment_mgr.getAttachmentTexture(gfx::Framebuffer::Attachment::kColor2));
    composite_material.setImageLoadStoreAttribute(
        "outputResultCacheId",
        *attachment_mgr.getAttachmentTexture(gfx::Framebuffer::Attachment::kColor3));
  }

  //
  // Draw setup and lambda
  //
  CHECK(!mark_gpu_data.fill_primitive_assemblies.empty());
  auto const gpu_id = device_ctx.getGpuId();
  auto const* vbo = mark_gpu_data.fill_primitive_assemblies[0]->getVertexBuffer();
  CHECK(vbo);
  auto const& data_table_resources = poly_table->getGpuResources();
  auto const* indvbo_polys =
      data_table_resources.getIndirectDrawVertexBuffer_polys(gpu_id);
  CHECK(indvbo_polys);
  auto const num_polys_to_draw = indvbo_polys->numItems();

  auto empty_rp_and_fb = root_gpu_data.getEmptyRenderPassAndFramebuffer();
  auto& empty_render_pass = empty_rp_and_fb.first;
  auto& empty_framebuffer = empty_rp_and_fb.second;

  // Draw polygons lambda
  auto draw_polygons_callback = [&](gfx::CommandList& cmd_list,
                                    std::string_view label,
                                    gfx::Pipeline& draw_pipeline,
                                    uint32_t num_polys,
                                    uint32_t first_poly) {
    cmd_list.pushLabel(label)
        .beginRenderPass(empty_render_pass, empty_framebuffer)
        .drawIndirect(draw_pipeline, *vbo, *indvbo_polys, num_polys, first_poly)
        .endRenderPass()
        .popLabel();
  };

  CHECK(mark_gpu_data.ppll_render);
  auto& ppll_render = mark_gpu_data.ppll_render;

  ppll_render->renderBegin(ctx_.getWidth(),
                           ctx_.getHeight(),
                           output_rgb_texture,
                           draw_polygons_callback,
                           num_polys_to_draw,
                           gfx::PPLLRender::ColorOutputMode::kNormal);

  // Get batch info
  auto const& poly_draw_batch_info = data_table_resources.getPolyDrawBatchInfo(gpu_id);
  auto const& num_polys_per_batch = poly_draw_batch_info.num_polys;

  // Count fragments and return if rendering complete
  // (fragment count == 0 or debug visualization ran)
  if (!ppll_render->countFragmentsAndComputeStats(num_polys_per_batch)) {
    return;
  }

  // Allocate list storage
  ppll_render->prepareStorage(0ul, num_polys_per_batch);

  //
  // Update capture material
  //
  auto& capture_material = *mark_gpu_data.fill_materials[kCaptureFragmentsMaterial];

  auto const* ssbo_poly_rowids =
      data_table_resources.getShaderStorageBuffer_poly_rowids(gpu_id);
  CHECK(ssbo_poly_rowids);
  capture_material.bindShaderStorageBufferToBlock("POLY_CAPTURE_POLYGON_ID_SSBO",
                                                  *ssbo_poly_rowids);

  //
  // Transition framebuffer attachment(s) to kGeneral layout for storage image
  //
  auto& cmd_list = device_ctx.getCommandList();

  cmd_list.pushLabel("Image layout");
  if (do_hit_testing) {
    cmd_list.transitionFramebufferLayout(framebuffer, gfx::ImageLayout::kGeneral);
  } else {
    cmd_list
        .imageMemoryBarrier(output_rgb_texture,
                            gfx::ImageMemoryBarrierType::kImageLayout,
                            gfx::ImageLayout::kGeneral)
        .popLabel();
  }

  //
  // Capture and composite batches
  //

  ppll_render->captureAndComposite(num_polys_per_batch);

  //
  // Restore framebuffer layout before returning
  //

  cmd_list.pushLabel("Image layout");
  if (do_hit_testing) {
    cmd_list.transitionFramebufferLayout(framebuffer, gfx::ImageLayout::kAttachment);
  } else {
    cmd_list.imageMemoryBarrier(output_rgb_texture,
                                gfx::ImageMemoryBarrierType::kImageLayout,
                                gfx::ImageLayout::kAttachment);
  }
  cmd_list.popLabel().flush("Framebuffer layout",
                            gfx::CommandList::SubmitType::kImmediateReturn);
}

#define PROFILE_FILL false
bool PolyMark::draw(const gfx::DeviceContext& device_ctx,
                    const MarkPerGpuData& mark_gpu_data,
                    gfx::Framebuffer& framebuffer,
                    const int accumulator_index) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());
  // NOTE: shader should have been updated before calling this
  CHECK(data_);
  auto const& root_gpu_data = mark_gpu_data.getRootPerGpuData();

  doManualClear(root_gpu_data, accumulator_index, framebuffer);

#if PROFILE_FILL
  auto clock_begin = timer_start();
#endif

  if (!mark_gpu_data.fill_materials.empty()) {
    VLOG(1) << "Drawing poly fill";
    drawFill(device_ctx,
             const_cast<MarkPerGpuData&>(mark_gpu_data),
             framebuffer,
             accumulator_index);
  }

#if PROFILE_FILL
  device_ctx.getCommandExecutor().waitForCompletion(true);
  auto const wall_time = timer_stop_microseconds(clock_begin);
  std::cout << "total time for fill: " << wall_time << "(us)" << std::endl;
#endif

  // now draw outlines
  if (!mark_gpu_data.stroke_materials.empty()) {
    VLOG(1) << "Drawing poly stroke";
    for (auto& sm : mark_gpu_data.stroke_materials) {
      sm->updateDescriptorSets();
    }

    // reset the ubo to null and re-grab if there are stroke-related props in it
    CHECK(!mark_gpu_data.stroke_primitive_assemblies.empty());

    auto const gpu_id = device_ctx.getGpuId();
    auto* vbo = mark_gpu_data.stroke_primitive_assemblies[0]->getVertexBuffer();
    CHECK(vbo);
    auto poly_table = std::dynamic_pointer_cast<BasePolyDataTable>(data_);
    CHECK(poly_table);
    auto* indvbo_lines =
        poly_table->getGpuResources().getIndirectDrawVertexBuffer_lines(gpu_id);
    CHECK(indvbo_lines);

    auto const num_items = indvbo_lines->numItems();
    auto& cmd_list = root_gpu_data.getCommandList();
    cmd_list.pushLabel("Poly stroke")
        .setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[kOutlinePipeline],
                               "totalNumItems",
                               ShaderStageBits::kVertex,
                               std::max(num_items, 1u));

    auto& render_pass = root_gpu_data.getCommonRenderPass(
        CommonRenderPassType::kAllAttachments, needsMultisampleEnabled());

    cmd_list.beginRenderPass(render_pass, framebuffer)
        .drawIndirect(*mark_gpu_data.graphics_pipelines[kOutlinePipeline],
                      *vbo,
                      *indvbo_lines,
                      num_items)
        .endRenderPass()
        .popLabel()
        .flush("PolyMark stroke", gfx::CommandList::SubmitType::kImmediateReturn);
  }

  return true;
}

PolyMark::operator std::string() const {
  return "PolyMark " + std::string(ctx_.getRenderSessionKey());
}

}  // namespace QueryRenderer
