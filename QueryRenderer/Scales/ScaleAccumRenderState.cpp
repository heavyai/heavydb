/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/ScaleAccumRenderState.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Scales/ScaleAccumState.h"

namespace QueryRenderer {

using ::gfx::DeviceContext;

// TODO(scb): max_textures seems quite arbitrary, and needs to handled at a higher level
// in conjunction with device introspection.
const uint32_t ScaleAccumRenderState::max_textures = 30;

namespace {
template <typename T>
void bind_percent_uniforms(gfx::Material& material,
                           const std::string& cat_uniform_name,
                           const std::string& margin_uniform_name,
                           const AnyDataType* pct_cat_val,
                           const AnyDataType* pct_margin_val) {
  material.setUniformAttribute<T>(cat_uniform_name, pct_cat_val->getVal<T>());
  material.setUniformAttribute<T>(margin_uniform_name,
                                  pct_margin_val ? pct_margin_val->getVal<T>() : (T)0);
}
}  // end namespace

void ScaleAccumRenderState::bindPercentUniforms(gfx::Material& material,
                                                const std::string& extra_suffix,
                                                const AnyDataType* pct_cat_val,
                                                const AnyDataType* pct_margin_val) {
  CHECK(pct_cat_val &&
        (!pct_margin_val || pct_cat_val->getType() == pct_margin_val->getType()));
  auto category_name = scale_accum_state_.getPercentCategoryUniformName() + extra_suffix;
  auto margin_name = scale_accum_state_.getPercentMarginUniformName() + extra_suffix;

  switch (pct_cat_val->getType()) {
    case QueryDataType::UINT:
      bind_percent_uniforms<QueryDataTypeSelector<QueryDataType::UINT>::type>(
          material, category_name, margin_name, pct_cat_val, pct_margin_val);
      break;
    case QueryDataType::INT:
      bind_percent_uniforms<QueryDataTypeSelector<QueryDataType::INT>::type>(
          material, category_name, margin_name, pct_cat_val, pct_margin_val);
      break;
    case QueryDataType::FLOAT:
      bind_percent_uniforms<QueryDataTypeSelector<QueryDataType::FLOAT>::type>(
          material, category_name, margin_name, pct_cat_val, pct_margin_val);
      break;
    case QueryDataType::DOUBLE:
      bind_percent_uniforms<QueryDataTypeSelector<QueryDataType::DOUBLE>::type>(
          material, category_name, margin_name, pct_cat_val, pct_margin_val);
      break;
    case QueryDataType::UINT64:
      bind_percent_uniforms<QueryDataTypeSelector<QueryDataType::UINT64>::type>(
          material, category_name, margin_name, pct_cat_val, pct_margin_val);
      break;
    case QueryDataType::INT64:
      bind_percent_uniforms<QueryDataTypeSelector<QueryDataType::INT64>::type>(
          material, category_name, margin_name, pct_cat_val, pct_margin_val);
      break;
    default:
      THROW_RUNTIME_EX("A percent accumulation scale with a category value of type " +
                       to_string(pct_cat_val->getType()) +
                       " is not supported. Cannot bind attributes.");
  }
}

class ScaleAccumRenderState::PerGpuData : public BasePerGpuData {
 public:
  gfx::MaterialUqPtr accum_2nd_pass_material;
  gfx::resource_ptr<gfx::GraphicsPipeline> pipeline;

  explicit PerGpuData(RootPerGpuData& root_data) : BasePerGpuData(root_data) {}
};

ScaleAccumRenderState::ScaleAccumRenderState(ScaleAccumState& scale_accum_state)
    : scale_accum_state_{scale_accum_state} {}

ScaleAccumRenderState::~ScaleAccumRenderState() {
  clearResources();
  destroyPipelines();
  per_gpu_data_map_.clear();
}

ScaleAccumRenderState::PerGpuData& ScaleAccumRenderState::getGpuData(GpuId gpu_id) {
  auto itr = per_gpu_data_map_.find(gpu_id);
  CHECK(itr != per_gpu_data_map_.end()) << "GpuId: " << gpu_id;
  return itr->second;
}

//
// Uniforms
//
void ScaleAccumRenderState::bindUniforms(gfx::Material& material,
                                         const std::string& extra_suffix,
                                         const AnyDataType* pct_cat_val,
                                         const AnyDataType* pct_margin_val) {
  RENDER_LOG_SCOPE();
  auto& gpu_data = getGpuData(material.getDeviceContext().getGpuId());
  auto const* accum_texture_array = gpu_data.getRootPerGpuData().getAccumTextureArray();
  CHECK(accum_texture_array);
  material.setImageLoadStoreAttribute("inTxArrayPixelCounter", *accum_texture_array);

  // TODO(croot): Do we need to be concerned about the possibility of
  // the pct data being overridden?

  // FIXME(scb): Find a better way to handle these Percent specializations, which are
  // scattered everywhere. It *might* be worth subclassing ScaleAccumStateer, but I
  // don't think that captures all the dependencies across the renderer.
  if (scale_accum_state_.getType() == AccumulatorType::kPct) {
    bindPercentUniforms(
        material,
        extra_suffix,
        pct_cat_val != nullptr ? pct_cat_val : scale_accum_state_.pct_cat_val_.get(),
        pct_margin_val != nullptr ? pct_margin_val
                                  : scale_accum_state_.pct_margin_val_.get());
  }
}

//
// Rendering
//

gfx::Material* ScaleAccumRenderState::get2ndPassMaterial(const GpuId gpu_id) {
  return getGpuData(gpu_id).accum_2nd_pass_material.get();
}

gfx::Pipeline* ScaleAccumRenderState::getPipeline(const GpuId gpu_id) {
  return getGpuData(gpu_id).pipeline.get();
}

void ScaleAccumRenderState::initGpuResources(QueryRendererContext& render_context,
                                             bool initializing) {
  RENDER_LOG_SCOPE() << " initializing: " << initializing;
  // Get the number of texture to use per-gpu
  uint32_t texture_array_size = scale_accum_state_.getNumTextures();

  // There's currently no clear way to determine if a query is going to result in
  // single or multi-gpu results from here, so we build two versions of the shaders
  // in advance, and select the correct one during the render call.
  // If we switch to always using texture arrays, or we are able to cleanly determine
  // single vs multi-gpu earlier in the setup process we can change this, but this is
  // low overhead enough to not stress about it.

  if (per_gpu_data_map_.empty()) {
    if (!initializing) {
      // doing a lazy init, so only initialize when properly instructed to do so
      return;
    }
    auto& root_gpu_data_map = render_context.getGlobalContext().getRootPerGpuData();
    for (auto& root_gpu_data : root_gpu_data_map) {
      per_gpu_data_map_.try_emplace(root_gpu_data->getGpuId(), *root_gpu_data);
    }
    scale_accum_state_.is_shader_dirty_ = true;
  }

  if (scale_accum_state_.is_shader_dirty_) {
    RENDER_LOG() << "Building ScaleAccumRenderState gpu resources";
    auto builders = scale_accum_state_.get2ndPassShaderBuilders(texture_array_size);
    auto caches =
        render_context.getShaderManager().createCacheVector(std::move(builders));

    destroyPipelines();
    pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();

    // Multi-sampling impacts the color subpass only. Extents and stddev subpasses
    // do not write to the framebuffer and their pipelines single sample
    pipeline_descriptor_->setRasterSampleCount(
        render_context.getGlobalContext().getRasterSampleCount());

    for (auto& gpu_data_itr : per_gpu_data_map_) {
      auto& gpu_data = gpu_data_itr.second;
      gpu_data.accum_2nd_pass_material =
          gpu_data.getResourceManager().createMaterial("Accum 2nd Pass", caches);
      gpu_data.pipeline = gpu_data.getResourceManager().createGraphicsPipeline(
          "ScaleAccum Accum", *gpu_data.accum_2nd_pass_material, *pipeline_descriptor_);

      // Create the pipeline using the RenderPass that matches the subpass count
      // do_clear can be true or false as this doesn't impact pipeline compatibility.
      gpu_data.pipeline->create(gpu_data.getRootPerGpuData().getAccumRenderPass());
    }
    scale_accum_state_.is_shader_dirty_ = false;
  }
}

void ScaleAccumRenderState::clearResources() {
  for (auto& gpu_data_itr : per_gpu_data_map_) {
    auto& gpu_data = gpu_data_itr.second;
    gpu_data.accum_2nd_pass_material = nullptr;
  }
}

void ScaleAccumRenderState::destroyPipelines() {
  for (auto& gpu_data_itr : per_gpu_data_map_) {
    auto& gpu_data = gpu_data_itr.second;
    if (gpu_data.pipeline) {
      gpu_data.getResourceManager().destroyPipeline(std::move(gpu_data.pipeline));
    }
  }
  pipeline_descriptor_ = nullptr;
}

AccumulatorType ScaleAccumRenderState::getAccumulatorType() const {
  return scale_accum_state_.accumulator_type_;
}
uint32_t ScaleAccumRenderState::getNumTextures() const {
  return scale_accum_state_.getNumTextures();
}
bool ScaleAccumRenderState::getDoFindMinDensity() const {
  return scale_accum_state_.do_find_min_density_;
}
bool ScaleAccumRenderState::getDoFindMaxDensity() const {
  return scale_accum_state_.do_find_max_density_;
}
bool ScaleAccumRenderState::getDoFindStdDev() const {
  return scale_accum_state_.do_find_std_dev_;
}
bool ScaleAccumRenderState::getDoFindExtents() const {
  return scale_accum_state_.getDoFindExtents();
}
uint32_t ScaleAccumRenderState::getMinDensity() const {
  return scale_accum_state_.min_density_;
}
uint32_t ScaleAccumRenderState::getMaxDensity() const {
  return scale_accum_state_.max_density_;
}
uint8_t ScaleAccumRenderState::getNumMinStdDev() const {
  return scale_accum_state_.num_min_std_dev_;
}
uint8_t ScaleAccumRenderState::getNumMaxStdDev() const {
  return scale_accum_state_.num_max_std_dev_;
}
void ScaleAccumRenderState::setAccumStats(RenderAccumStatsUqPtr&& accum_stats) const {
  scale_accum_state_.setAccumStats(std::move(accum_stats));
}
BaseScale& ScaleAccumRenderState::getParentScale() const {
  return scale_accum_state_.parent_scale_;
}

}  // namespace QueryRenderer
