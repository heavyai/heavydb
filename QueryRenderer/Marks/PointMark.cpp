/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/PointMark.h"

#include <boost/algorithm/string/regex.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Marks/RenderPropertyContainer.h"
#include "QueryRenderer/Marks/Utils.h"
#include "Shared/scope.h"

#define PROFILE_MESH_DRAW 0
#if PROFILE_MESH_DRAW
#include "Shared/measure.h"
#endif

// NOTE: Validation layers must be disabled or set to non-fatal to test device lost
#define DO_DEVICE_LOST_TEST 0

namespace QueryRenderer {

using ::gfx::ShaderStage;
using ShaderBuilder = ::gfx::ShaderManager::Builder;

PointMark::PointMark(const JSONLocation& obj_loc, QueryRendererContext& ctx)
    : BaseMark(GeomType::kPoints, ctx, obj_loc, DataOutputFormat::kRows, false)
    , size_{"size", ctx, *prop_mark_facade_} {
  using type = RenderPropertyCreateInfo::Type;
  using flag = RenderPropertyFlagBits;
  std::vector<RenderPropertyCreateInfo> render_property_ci{
      {type::kFillColor, flag::kUnspecified},
      {type::kFillOpacity, flag::kUnspecified, false, {1.0f}}};

  render_props_ = std::make_unique<RenderPropertyContainer>(
      ctx, *prop_mark_facade_, std::move(render_property_ci));

  auto const& props = render_props_->getProperties();
  used_props_.insert(props.begin(), props.end());
  used_props_.insert(&size_);
  used_props_const_.insert(used_props_.begin(), used_props_.end());

  auto const& coord_props = render_props_->getCoordProperties();
  projection_policy_ = std::make_unique<MarkProjectionShaderPolicy>(
      MarkProjectionShaderPolicy::PropMap{coord_props.begin(), coord_props.end()});

  initPropertiesFromJSONObj(obj_loc, true, true);
  initTransformsFromJSONObj(obj_loc, getCoordPropAttrInfos());
  json_path_ = obj_loc.getPathRef();
}

PointMark::~PointMark() {}

BaseRenderPropertyConstSet PointMark::getUsedProps() const {
  return used_props_const_;
}

void PointMark::initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                          const bool data_changed,
                                          const bool init) {
  RENDER_LOG_SCOPE() << " data_changed: " << data_changed << "  init: " << init;
  auto const prop_loc = obj_loc.getMember(JSONSchema_v1::Marks::kPropertiesProp);
  RUNTIME_EX_ASSERT(prop_loc.isValid() && prop_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        prop_loc.isValid() ? prop_loc : obj_loc,
                        "Mark objects must have a \"" +
                            std::string(JSONSchema_v1::Marks::kPropertiesProp) +
                            "\" property and it must be a JSON object."));

  auto prev_path = properties_json_path_;
  properties_json_path_ = prop_loc.getPathRef();
  if (!ctx_.isJSONCacheUpToDate(prev_path, prop_loc) || data_changed || init) {
    RUNTIME_EX_ASSERT(prop_loc.isObject(),
                      RapidJSONUtils::createJsonParseError(
                          prop_loc, "Property must be a json object."));

    render_props_->initFromJSONObj(obj_loc, data_changed);

    initPropFromJSONObj(
        &ctx_,
        data_,
        data_changed,
        prop_loc,
        &size_,
        properties_json_path_,
        BaseMark::validateNumPropFunc(*this),
        nullptr,
        nullptr,
        nullptr,
        [this, &prop_loc](const JSONLocation&) {
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              prop_loc,
              "\"" + size_.getName() + "\" mark property must exist for point marks."))
        });

    initIds(data_changed);

    if (init || data_changed) {
      updateProps(getUsedProps());
    }

    updateVisibility(render_props_->isFillActive());
  }
}

void PointMark::updateShader() {
  RENDER_LOG_SCOPE() << "building glsl shaders";

  auto& shader_mgr = ctx_.getShaderManager();

  // we use the mesh shader path for MULTIPOINT, and the vert shader path for POINT
  updateGeoPropInfoAndPropCompressionBits({kPOINT, kMULTIPOINT});
  updateUseMeshShader({kMULTIPOINT});

  // Create builders and replace tags
  auto builders = shader_mgr.createBuilderVector(
      {{useMeshShader() ? "Marks/pointTemplate.mesh" : "Marks/pointTemplate.vert"},
       {"Marks/pointTemplate.frag"}});

  std::stringstream get_prop_ss;
  streamPropertyGetters(prop_buf_loc_state_.vbo_props, get_prop_ss);
  streamPropertyGetters(prop_buf_loc_state_.uniform_props, get_prop_ss);

  // Build vertex/mesh shader uniform render props
  gfx::GlslStructBuilder ubo_struct_builder("POINT_VERT_UBO_TYPE");
  ubo_struct_builder.addMember("uViewProjMatrix", gfx::BufferAttrType::kMat3x2f);
  ubo_struct_builder.addMember("invalidKey", gfx::BufferAttrType::kUint64);
  ubo_struct_builder.addMember("propCompressionBits", gfx::BufferAttrType::kUint);
  if (useMeshShader()) {
    ubo_struct_builder.addMember("vboDeviceAddress", gfx::BufferAttrType::kUint64);
  }

  addCommonRenderPropUniforms(ubo_struct_builder);

  // Build vertex->fragment interface block
  gfx::GlslStructBuilder block_builder("FragmentShaderInputs");
  std::vector<gfx::GlslStructBuilder::Qualifier> quals = {
      gfx::GlslStructBuilder::Qualifier::kFlat};
  block_builder.addMember("fRowId", gfx::BufferAttrType::kUint64, quals);
  block_builder.addMember("fColor", gfx::BufferAttrType::kVec4f, quals);
  block_builder.addMember("fPointSize", gfx::BufferAttrType::kFloat, quals);
  if (hasAccumulator()) {
    block_builder.addMember("accumIdx", gfx::BufferAttrType::kInt, quals);
  }

  auto block_str = block_builder.createInterfaceBlockString(true);

  // vertex/mesh -> fragment interface block
  builders[0]->replaceFirstTag("FragmentShaderInputs", block_str);
  builders[1]->replaceFirstTag("FragmentShaderInputs", block_str);

  builders[0]->setExternalUniformBuffers({"SLAB_ADDRESS_TABLE_UBO"});

  BaseMark::setKeyInShaderBuilder(*builders[0]);

  // Vertex/Mesh shader inputs
  builders[0]->replaceFirstTag(
      "VertexProperties",
      useMeshShader() ? buildVertexDataStruct() : buildVertexShaderInputs());
  builders[0]->replaceFirstTag("UniformProperties",
                               ubo_struct_builder.createStructString());
  builders[0]->replaceFirstTag("PropertyGetters", get_prop_ss.str());

  // Set multisampling flag
  // controls use of gl_SamplePosition
  builders[0]->replaceFirstTag("isMultiSampling",
                               std::to_string(needsMultisampleEnabled()));
  builders[1]->replaceFirstTag("isMultiSampling",
                               std::to_string(needsMultisampleEnabled()));

  builders[1]->replaceFirstTag("doDeviceLostTestLoop",
                               std::to_string(DO_DEVICE_LOST_TEST));

  if (useMeshShader()) {
    // subgroup size
    auto const subgroup_size =
        ctx_.getGlobalContext().getGfxContext().getDeviceLimits().subgroup_size;
    auto const subgroup_size_bits = uint32_t(log2(subgroup_size));
    builders[0]->replaceFirstTag("workgroupSize", std::to_string(subgroup_size));
    builders[0]->replaceFirstTag("workgroupSizeBits", std::to_string(subgroup_size_bits));

    // vertex attribute fetches
    // WIP
    auto attr_fetch_str = buildVertexAttributeFetches();
    builders[0]->replaceFirstTag("VertexAttributeFetches", attr_fetch_str);
  }

  auto build_subroutine_bindings = [&](BaseRenderPropertyConstSet& props) {
    for (auto const* prop : props) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        for (auto& builder : builders) {
          scale_ref->buildSubroutineBindings(*builder, "_" + prop->getName());
        }
      }
    }
  };

  build_subroutine_bindings(prop_buf_loc_state_.vbo_props);
  build_subroutine_bindings(prop_buf_loc_state_.uniform_props);

  // Inject scale code as needed and handle projections
  BaseMark::insertPropertyCodeInShaderBuilders(
      builders, used_props_const_, *projection_policy_);

  BaseMark::setColorConvertSubroutines(
      *builders[0],
      render_props_->getProperty(RenderPropertyContainer::PropId::kFillColor));

  ctx_.clearMarkShaders(*this);
  ctx_.buildMarkShaders(
      *this, MarkGpuResourceSlot::kFill, "PointMark Fill", std::move(builders));
  shader_dirty_ = false;

  // set the props and pipeline dirty to force a rebind with the new shader
  setPropsDirty();
  setPipelinesDirty();
}

void PointMark::buildPipelineDescriptors() {
  if (!pipeline_descriptor_) {
    pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
    CHECK(pipeline_descriptor_);
  }

  pipeline_descriptor_->setRasterSampleCount(getRasterizationSampleCount());

  pipeline_descriptor_->getPushConstantRanges().clear();
  pipeline_descriptor_->getPushConstantRanges().insert(
      gfx::ShaderStageBits::kMesh, 0, sizeof(uint32_t));
}

void PointMark::buildPipelines(MarkPerGpuData& per_gpu_data) {
  const gfx::PrimitiveAssembly* primitive_assembly_to_use{};
  if (!useMeshShader()) {
    CHECK(per_gpu_data.fill_primitive_assemblies.size());
    CHECK(per_gpu_data.fill_primitive_assemblies[0]);
    primitive_assembly_to_use = per_gpu_data.fill_primitive_assemblies[0].get();
  }

  CHECK(per_gpu_data.fill_materials.size());
  CHECK(per_gpu_data.fill_materials[0]);

  per_gpu_data.destroyPipelines();
  per_gpu_data.graphics_pipelines.push_back(
      per_gpu_data.getResourceManager().createGraphicsPipeline(
          "PointMark Fill",
          *per_gpu_data.fill_materials[0],
          *pipeline_descriptor_,
          primitive_assembly_to_use));
  CHECK(per_gpu_data.graphics_pipelines[0]);

  per_gpu_data.graphics_pipelines[0]->create(
      per_gpu_data.getRootPerGpuData().getCommonRenderPass(
          CommonRenderPassType::kAllAttachments, needsMultisampleEnabled()));
}

void PointMark::buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  if (!useMeshShader()) {
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
      auto gpu_id = gpu_data.getGpuId();
      prop_size = prop->size(gpu_id);
      if (attr_count == 1) {
        vbo_size = prop_size;
      } else {
        RUNTIME_EX_ASSERT(prop_size == vbo_size,
                          std::string(*this) +
                              ": Invalid point mark. The sizes of the vertex buffer "
                              "attributes do not match for gpuId " +
                              std::to_string(gpu_id) + ". " + std::to_string(vbo_size) +
                              "!=" + std::to_string(prop_size));
      }
      prop->addToPrimitiveAssemblyAttrInfo(gpu_id, attr_info);
    }

    gpu_data.fill_primitive_assemblies.clear();
    gpu_data.fill_primitive_assemblies.push_back(
        gpu_data.getResourceManager().createPrimitiveAssembly(
            "PointMark",
            gfx::PrimitiveTopology::kPointList,
            *gpu_data.fill_materials[0],
            attr_info));
  }
}

void PointMark::setUniformAttributes(MarkPerGpuData& gpu_data) {
  RENDER_LOG_SCOPE();
  auto& material = *gpu_data.fill_materials[0];

  BaseMark::bindKeyPropUniformAttributes(material);

  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      scale_ref->bindUniforms(material, "_" + prop->getName());
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      scale_ref->bindUniforms(material, "_" + prop->getName());
    }

    prop->setUniformAttribute(material, prop->getName());
  }

  for (auto const* prop : prop_buf_loc_state_.decimal_props) {
    prop->setDecimalScaleUniformAttribute(material);
  }

  BaseMark::bindIDPropUniformAttributes(material);
  BaseMark::setProjectionUniformAttributes(material);

  material.setUniformAttribute("uViewProjMatrix",
                               ctx_.getViewProjMatrix().getDataArrayRef());
  if (hasProjection()) {
    material.setViewportAttributes(0, 0, ctx_.getWidth(), ctx_.getHeight());
  }

  // update prop compression bits again for the case where only the compression changes
  updateGeoPropInfoAndPropCompressionBits({kPOINT, kMULTIPOINT});

  updateSlabAddressTableAndPropCompressionBitsUniforms(gpu_data);

#if DO_DEVICE_LOST_TEST
  static bool do_device_lost_loop = true;
  if (do_device_lost_loop) {
    do_device_lost_loop = false;
    material.setUniformAttribute("uTestLoopEnd", 0);
  } else {
    material.setUniformAttribute("uTestLoopEnd", 1);
  }
#endif
}

void PointMark::updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                                 const std::vector<GpuId>& remove_gpus) {
  for (auto const& prop : used_props_) {
    prop->initGpuResources(add_gpus, remove_gpus);
  }
}

bool PointMark::draw(const gfx::DeviceContext& device_ctx,
                     const MarkPerGpuData& mark_gpu_data,
                     gfx::Framebuffer& framebuffer,
                     const int accumulator_index) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());

  // NOTE: shader should have been updated before calling this
  CHECK(!mark_gpu_data.fill_materials.empty());

  auto& render_pass =
      selectDrawRenderPass(mark_gpu_data.getRootPerGpuData(), accumulator_index);

  if (useMeshShader()) {
    drawWithMeshShader(device_ctx, mark_gpu_data, framebuffer, render_pass);
  } else {
    drawWithVertexShader(device_ctx, mark_gpu_data, framebuffer, render_pass);
  }

  return true;
}

void PointMark::drawWithMeshShader(const gfx::DeviceContext& device_ctx,
                                   const MarkPerGpuData& mark_gpu_data,
                                   gfx::Framebuffer& framebuffer,
                                   gfx::RenderPass& render_pass) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());

#if PROFILE_MESH_DRAW
  LOG(INFO) << "DEBUG: starting mesh shader draw";
  auto total_timer = timer_start();
  auto start_mesh_draw_timer = timer_start();
#endif

  // start
  // this does the count pass on the recognized geo prop, builds the work
  // units buffer, and returns the number of work units (mesh shader warps)

  auto const num_work_units = startMeshShaderDraw(mark_gpu_data);
  if (num_work_units == 0U) {
    // nothing to draw
    return;
  }

  ScopeGuard end_mesh_shader_draw = [&]() { endMeshShaderDraw(mark_gpu_data); };

#if PROFILE_MESH_DRAW
  LOG(INFO) << "DEBUG: start mesh draw took " << timer_stop(start_mesh_draw_timer)
            << "ms, num_work_units = " << num_work_units;
#endif

  // finalize
  mark_gpu_data.fill_materials[0]->updateDescriptorSets();

  // draw in batches of max workgroup count / 4
  // this is an arbitrary safety factor, considering the ease of triggering
  // a DL during testing when approaching the full workgroup count
  auto const batch_workgroup_count =
      device_ctx.getLimits().max_mesh_workgroup_count[0] / 4;
  uint32_t first_work_unit = 0u;
  auto& command_list = device_ctx.getCommandList();
  command_list.pushLabel("Point Draw (Mesh Shader)")
      .beginRenderPass(render_pass, framebuffer);
  while (first_work_unit < num_work_units) {
    auto const group_count_x =
        std::min(batch_workgroup_count, num_work_units - first_work_unit);
    command_list
        .setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[0],
                               "firstWorkUnit",
                               gfx::ShaderStageBits::kMesh,
                               first_work_unit)
        .drawMeshTasks(*mark_gpu_data.graphics_pipelines[0], group_count_x, 1u, 1u);
    first_work_unit += group_count_x;
  }
  command_list.endRenderPass().popLabel().flush("Point Draw (Mesh Shader)");

  // can do more draw passes in here with the same work units buffer

#if PROFILE_MESH_DRAW
  LOG(INFO) << "DEBUG: ending mesh shader draw, took " << timer_stop(total_timer) << "ms";
#endif
}

void PointMark::drawWithVertexShader(const gfx::DeviceContext& device_ctx,
                                     const MarkPerGpuData& mark_gpu_data,
                                     gfx::Framebuffer& framebuffer,
                                     gfx::RenderPass& render_pass) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());

  // get VBO info
  CHECK(!mark_gpu_data.fill_primitive_assemblies.empty());
  auto& primitive_assembly = mark_gpu_data.fill_primitive_assemblies[0];
  auto const* vertex_buffer = primitive_assembly->getVertexBuffer();
  auto const num_vertices = primitive_assembly->numVertices();
  auto const vertex_buffer_offset_bytes =
      primitive_assembly->getVertexBufferOffsetBytes();

  // finalize
  mark_gpu_data.fill_materials[0]->updateDescriptorSets();

  // draw
  device_ctx.getCommandList()
      .pushLabel("Point draw")
      .beginRenderPass(render_pass, framebuffer)
      .drawVertices(*mark_gpu_data.graphics_pipelines[0],
                    *vertex_buffer,
                    num_vertices,
                    vertex_buffer_offset_bytes)
      .endRenderPass()
      .popLabel()
      .flush("PointMark draw", gfx::CommandList::SubmitType::kImmediateReturn);
}

PointMark::operator std::string() const {
  return "PointMark " + std::string(ctx_.getRenderSessionKey());
}

}  // namespace QueryRenderer
