/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/Mesh2dMark.h"

#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Data/QueryMeshDataTable.h"
#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderPropertyContainer.h"
#include "QueryRenderer/Marks/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

Mesh2dMark::Mesh2dMark(const JSONLocation& obj_loc, QueryRendererContext& ctx)
    : BaseMark(GeomType::kMesh2d, ctx, obj_loc, DataOutputFormat::kMesh2d, false) {
  using type = RenderPropertyCreateInfo::Type;
  using flag = RenderPropertyFlagBits;
  std::vector<RenderPropertyCreateInfo> render_property_ci = {
      {type::kFillColor, flag::kUseScale, false, gfx::ColorUnion(0.f, 0.f, 0.f, 1.f)},
      {type::kFillOpacity, flag::kUnspecified, false, 1.0f}};

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

Mesh2dMark::~Mesh2dMark() {}

Mesh2dMark::operator std::string() const {
  return "Mesh2d " + std::string(ctx_.getRenderSessionKey());
}

BaseRenderPropertyConstSet Mesh2dMark::getUsedProps() const {
  return used_props_const_;
}

void Mesh2dMark::initPropertiesFromJSONObj(const JSONLocation& obj_loc,
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

    initIds(data_changed);

    if (init || data_changed) {
      updateProps(getUsedProps());
    }

    updateVisibility(render_props_->isFillActive());
  }
}

void Mesh2dMark::updateShader() {
  RENDER_LOG_SCOPE() << "building glsl shaders";

  auto& shader_mgr = ctx_.getShaderManager();

  gfx::ShaderManager::BuilderUqPtrVector builders;
  builders = shader_mgr.createBuilderVector(
      {{"Marks/mesh2dTemplate.vert"}, {"Marks/mesh2dTemplate.frag"}});

  std::stringstream get_prop_ss;
  streamPropertyGetters(prop_buf_loc_state_.vbo_props, get_prop_ss);
  streamPropertyGetters(prop_buf_loc_state_.uniform_props, get_prop_ss);

  // Build vertex shader uniform render props
  gfx::GlslStructBuilder ubo_struct_builder("MESH2D_VERT_UBO_TYPE");
  ubo_struct_builder.addMember("uViewProjMatrix", gfx::BufferAttrType::kMat3x2f);

  addCommonRenderPropUniforms(ubo_struct_builder);

  // Vertex shader inputs
  builders[0]->replaceFirstTag("VertexProperties", buildVertexShaderInputs());
  builders[0]->replaceFirstTag("UniformProperties",
                               ubo_struct_builder.createStructString());
  builders[0]->replaceFirstTag("PropertyGetters", get_prop_ss.str());

  // NOTE: there is no handling of invalid key here as all mesh queries should be run
  // non-insitu, hence no invalid key

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

  // color props in vertex shader
  BaseMark::setColorConvertSubroutines(
      *builders[0],
      render_props_->getProperty(RenderPropertyContainer::PropId::kFillColor));

  ctx_.clearMarkShaders(*this);
  ctx_.buildMarkShaders(*this, MarkGpuResourceSlot::kFill, "Mesh2d", std::move(builders));
  shader_dirty_ = false;

  // set the props dirty to force a rebind with the new shader
  setPropsDirty();
}

void Mesh2dMark::setUniformAttributes(MarkPerGpuData& per_gpu_data) {
  RENDER_LOG_SCOPE_P(per_gpu_data.getGpuId());
  auto& active_material = *per_gpu_data.fill_materials[0];
  auto viewport_width = ctx_.getWidth();
  auto viewport_height = ctx_.getHeight();

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

  active_material.setUniformAttribute("uViewProjMatrix",
                                      ctx_.getViewProjMatrix().getDataArrayRef());
}

void Mesh2dMark::buildPipelineDescriptors() {
  if (!pipeline_descriptor_) {
    pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
    CHECK(pipeline_descriptor_);
  }

  pipeline_descriptor_->setRasterSampleCount(getRasterizationSampleCount());
}

void Mesh2dMark::buildPipelines(MarkPerGpuData& per_gpu_data) {
  RENDER_LOG_SCOPE_P(per_gpu_data.getGpuId());
  CHECK(per_gpu_data.fill_primitive_assemblies.size());
  CHECK(per_gpu_data.fill_primitive_assemblies[0]);
  CHECK(per_gpu_data.fill_materials.size());
  CHECK(per_gpu_data.fill_materials[0]);

  per_gpu_data.destroyPipelines();
  per_gpu_data.graphics_pipelines.push_back(
      per_gpu_data.getResourceManager().createGraphicsPipeline(
          "Mesh2d",
          *per_gpu_data.fill_materials[0],
          *pipeline_descriptor_,
          per_gpu_data.fill_primitive_assemblies[0].get()));
  per_gpu_data.graphics_pipelines[0]->create(
      per_gpu_data.getRootPerGpuData().getCommonRenderPass(
          CommonRenderPassType::kAllAttachments, needsMultisampleEnabled()));
}

void Mesh2dMark::buildFillPrimitiveAssemblies(MarkPerGpuData& per_gpu_data) {
  RENDER_LOG_SCOPE_P(per_gpu_data.getGpuId());
  auto gpu_id = per_gpu_data.getGpuId();
  CHECK(!per_gpu_data.fill_materials.empty());

  auto mesh_table = std::dynamic_pointer_cast<SqlQueryMeshDataTableJSON>(data_);
  CHECK(mesh_table);
  auto const* ibo = mesh_table->getIndexBuffer(gpu_id);
  CHECK(ibo);

  gfx::PrimitiveAssemblyAttrInfo attr_info;
  int attr_count = 0;
  int vbo_size = 0;
  int prop_size = 0;
  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    if (!per_gpu_data.fill_materials[0]->hasVertexAttribute(prop->getName())) {
      continue;
    }
    attr_count++;
    prop_size = prop->size(gpu_id);
    if (attr_count == 1) {
      vbo_size = prop_size;
    } else {
      RUNTIME_EX_ASSERT(prop_size == vbo_size,
                        std::string(*this) +
                            ": Invalid mesh2d mark. The sizes of the vertex buffer "
                            "attributes do not match for gpuId " +
                            std::to_string(gpu_id) + ". " + std::to_string(vbo_size) +
                            "!=" + std::to_string(prop_size));
    }
    prop->addToPrimitiveAssemblyAttrInfo(gpu_id, attr_info);
  }

  per_gpu_data.fill_primitive_assemblies.clear();
  per_gpu_data.fill_primitive_assemblies.push_back(
      per_gpu_data.getResourceManager().createPrimitiveAssembly(
          "Mesh2d",
          gfx::PrimitiveTopology::kTriangleList,
          *per_gpu_data.fill_materials[0],
          attr_info,
          ibo));
}

void Mesh2dMark::updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                                  const std::vector<GpuId>& remove_gpus) {
  for (auto const& prop : used_props_) {
    prop->initGpuResources(add_gpus, remove_gpus);
  }
}

bool Mesh2dMark::draw(const gfx::DeviceContext& device_ctx,
                      const MarkPerGpuData& per_gpu_data,
                      gfx::Framebuffer& framebuffer,
                      const int accumulator_index) {
  RENDER_LOG_SCOPE_P(per_gpu_data.getGpuId());
  // NOTE: shader should have been updated before calling this

  auto const& root_gpu_data = per_gpu_data.getRootPerGpuData();
  auto& primitive_assembly = per_gpu_data.fill_primitive_assemblies[0];
  per_gpu_data.fill_materials[0]->updateDescriptorSets();

  auto& render_pass = selectDrawRenderPass(root_gpu_data, accumulator_index);

  root_gpu_data.getCommandList()
      .pushLabel("Mesh2d draw")
      .beginRenderPass(render_pass, framebuffer)
      .drawIndexed(*per_gpu_data.graphics_pipelines[0],
                   *primitive_assembly->getVertexBuffer(),
                   *primitive_assembly->getIndexBuffer(),
                   primitive_assembly->numIndices())
      .endRenderPass()
      .popLabel()
      .flush("Mesh2d draw", gfx::CommandList::SubmitType::kImmediateReturn);

  return true;
}

}  // namespace QueryRenderer
