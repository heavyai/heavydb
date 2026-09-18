/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "GfxDriver/Types.h"
#include "QueryRenderer/Data/QueryRowDataTable.h"
#include "QueryRenderer/Marks/BaseRenderProperty.h"
#include "QueryRenderer/Marks/CoordAttrInfo2d.h"
#include "QueryRenderer/Marks/Enums.h"
#include "QueryRenderer/Marks/MarkPerGpuData.h"
#include "QueryRenderer/Marks/RenderPropertyBufferState.h"
#include "QueryRenderer/Types.h"
#include "Shared/EnumBitmaskOps.h"

namespace QueryRenderer {

class MarkProjectionShaderPolicy;
class RenderPropertyContainer;

class BaseMark {
 public:
  BaseMark(GeomType geom_type,
           QueryRendererContext& ctx,
           const JSONLocation& obj_loc,
           DataOutputFormat data_output_format,
           bool must_use_data_ref);
  virtual ~BaseMark();

  GeomType getType() { return type_; }

  // NOTE: MarkPerGpuData is built lazily and is not guaranteed to exist for all gpus in
  // the vector
  std::vector<GpuId> getUsedGpus() const { return used_gpus_; }

  void propChangedCallback(BaseRenderProperty::ChangeType change_type) {
    using ct = BaseRenderProperty::ChangeType;
    switch (change_type) {
      case ct::kStructure:
        shader_dirty_ = true;
        break;
      case ct::kData:
        props_dirty_ = true;
        break;
      case ct::kValues:
        uniforms_dirty_ = true;
        break;
    }
  };

  void setShaderDirty() { shader_dirty_ = true; }
  void setPropsDirty() { props_dirty_ = true; }
  void setPipelinesDirty() { pipelines_dirty_ = true; }
  void setUniformsDirty() { uniforms_dirty_ = true; }

  bool isVisible() const { return is_visible_; }

  const BaseDataTableShPtr getDataPtr() const { return data_; }
  const std::unordered_set<BaseDataTableShPtr> getDataRefs() const;
  const std::unordered_set<ScaleShPtr> getScaleRefs() const;

  bool hasProjection() const;
  ProjectionShPtr getProjection() const;
  void setProjection(const std::string& projection_name, ProjectionShPtr projection);

  bool hasAccumulator() const;
  std::string getAccumulatorScaleName() const;
  ScaleShPtr getAccumulatorScale() const;

  virtual bool usesPerPixelLinkedLists() const { return false; }

  // set/clear accumulator should only be called by render property
  // TODO(croot): make these functions private and make render properties friends?
  void setAccumulatorScale(const ScaleShPtr& scale, const ScaleRefShPtr& scale_ref);
  void clearAccumulatorScale(const ScaleShPtr& scale);

  virtual bool draw(const gfx::DeviceContext& device_ctx,
                    const MarkPerGpuData& mark_gpu_data,
                    gfx::Framebuffer& framebuffer,
                    const int accumulator_index) = 0;

  void setInvalidKey(const int64_t invalid_key) { invalid_key_ = invalid_key; }

  bool updateFromJSONObj(const JSONLocation& obj_loc);
  void update();

  virtual std::vector<CoordAttrInfo2d> getCoordPropAttrInfos() const;

  virtual operator std::string() const = 0;

 protected:
  enum class CoordPackingTypeBits {
    kNone = 0,
    kCompressedGeo = 1 << 0,
    kPackedPixel = 1 << 1
  };

  using ValidateFunc = std::function<void(const std::string&, const JSONLocation&)>;
  using JSONParseCBFunc = std::function<void(const JSONLocation&)>;
  static JSONLocation initPropFromJSONObj(const QueryRendererContext* ctx,
                                          const BaseDataTableShPtr& data,
                                          const bool data_changed,
                                          const JSONLocation& prop_loc,
                                          BaseRenderProperty* prop,
                                          const rapidjson::Pointer& properties_json_path,
                                          ValidateFunc validate_type_func = nullptr,
                                          JSONParseCBFunc post_update_func = nullptr,
                                          JSONParseCBFunc post_data_update_func = nullptr,
                                          JSONParseCBFunc post_up_to_date_func = nullptr,
                                          JSONParseCBFunc post_empty_func = nullptr);

  static ValidateFunc validateNumPropFunc(BaseMark& mark);
  static ValidateFunc validateColorPropFunc(BaseMark& mark);
  static ValidateFunc validateEnumPropFunc(BaseMark& mark);
  static ValidateFunc validateBoolPropFunc(BaseMark& mark);

  GeomType type_;

  int64_t invalid_key_;

  BaseDataTableShPtr data_;

  struct ShaderCacheInfo {
    std::string tracking_string;
    gfx::ShaderCacheShPtrVector caches;
    ShaderCacheInfo(std::string tracking_string, gfx::ShaderCacheShPtrVector&& caches)
        : tracking_string(std::move(tracking_string)), caches(std::move(caches)) {}
    ShaderCacheInfo(ShaderCacheInfo&& other)
        : tracking_string(std::move(other.tracking_string))
        , caches(std::move(other.caches)) {}
    ShaderCacheInfo& operator=(const ShaderCacheInfo& other) = default;
  };

  std::vector<ShaderCacheInfo> fill_shader_caches_;
  std::vector<ShaderCacheInfo> stroke_shader_caches_;

  using PerGpuDataMap = std::map<GpuId, MarkPerGpuData>;
  using ShaderBuilder = ::gfx::ShaderManager::Builder;
  using ShaderBuilderVector = ::gfx::ShaderManager::BuilderUqPtrVector;

  PerGpuDataMap per_gpu_data_;
  QueryRendererContext& ctx_;
  DataOutputFormat data_output_format_;

  rapidjson::Pointer data_ptr_json_path_;
  rapidjson::Pointer properties_json_path_;
  rapidjson::Pointer transform_ptr_json_path_;
  rapidjson::Pointer json_path_;

  // Gpus used by input data
  // Will not have valid MarkPerGpuData until the Mark is visible and has been updated
  std::vector<GpuId> used_gpus_;

  // Pending MarkPerGpuData additions
  std::vector<GpuId> pending_gpu_data_additions_;

  bool is_empty_;
  bool shader_dirty_;
  bool props_dirty_;
  bool pipelines_dirty_;
  bool uniforms_dirty_;

  RenderPropertyBufferState prop_buf_loc_state_;

  // common mark properties
  std::unique_ptr<BaseRenderProperty::BaseMarkFacade> prop_mark_facade_;
  std::unique_ptr<RenderPropertyContainer> render_props_;
  void initIds(const bool data_changed);

  bool initFromJSONObj(const JSONLocation& obj_loc, bool must_use_data_ref);

  void initTransformsFromJSONObj(const JSONLocation& obj_loc,
                                 const std::vector<CoordAttrInfo2d>& coord_props);

  void initGpuResources();
  virtual CommonRenderPassTypeBits getRequiredCommonRenderPassTypes() const;

  virtual BaseRenderPropertyConstSet getUsedProps() const = 0;

  void updateProps(const BaseRenderPropertyConstSet& used_props);
  void insertFragShaderMain(ShaderBuilder& frag_builder);
  void setKeyInShaderBuilder(ShaderBuilder& builder);
  void setColorConvertSubroutines(ShaderBuilder& builder,
                                  const BaseRenderProperty* color_prop);
  // Insert type decl defines and type enum defines for RenderProperty in/out types
  void streamPropertyTypeInfoDefines(const BaseRenderPropertyConstSet& props,
                                     std::ostream& os) const;
  void streamUseUniformDefines(std::ostream& os) const;
  void streamPropertyGetters(
      const BaseRenderPropertyConstSet& props,
      std::ostream& os,
      MarkProjectionShaderPolicy* projection_policy = nullptr) const;

  void setProjectionUniformAttributes(gfx::Material& active_material);
  void bindIDPropUniformAttributes(gfx::Material& active_material);
  void bindKeyPropUniformAttributes(gfx::Material& active_material);

  std::string buildVertexShaderInputs() const;
  std::string buildVertexDataStruct() const;
  std::string buildVertexAttributeFetches() const;
  void addCommonRenderPropUniforms(gfx::GlslStructBuilder& ubo_struct_builder) const;

  void insertPropertyCodeInShaderBuilders(
      ShaderBuilderVector& builders,
      const BaseRenderPropertyConstSet& props,
      const MarkProjectionShaderPolicy& projection_policy,
      const std::string* ssbo_name = nullptr,
      const std::string* ssbo_instance_name = nullptr,
      const bool auto_inject_main = true);

  virtual void dataRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj);

  bool needsMultisampleEnabled() const;
  gfx::RasterSampleCount getRasterizationSampleCount() const;
  void updateVisibility(const bool is_visible);

  // centralize these for now
  // @TODO(se/scb) improve this
  gfx::RenderPass& selectDrawRenderPass(
      const QueryRenderer::RootPerGpuData& root_gpu_data,
      const int accumulator_index) const;
  void doManualClear(const QueryRenderer::RootPerGpuData& root_gpu_data,
                     const int accumulator_index,
                     gfx::Framebuffer& framebuffer) const;

  void updateGeoPropInfoAndPropCompressionBits(
      const std::set<SQLTypes> supported_geo_types);
  void updateSlabAddressTableAndPropCompressionBitsUniforms(
      const MarkPerGpuData& mark_gpu_data) const;

  void updateUseMeshShader(const std::set<SQLTypes> supported_geo_types);
  const bool useMeshShader() const;

  const uint32_t startMeshShaderDraw(const MarkPerGpuData& mark_gpu_data);
  void endMeshShaderDraw(const MarkPerGpuData& mark_gpu_data);

 private:
  // Create new MarkPerGpuData struct for gpus in pending_gpu_data_additions_
  void createPendingGpuData();
  virtual void initPPLLPerGpuData(MarkPerGpuData& gpu_data) {}

  // Build Material, PrimitiveAssemblies, and Pipelines for gpus in
  // pending_gpu_data_additions_ using ShaderCaches from ShaderCacheInfo
  // Used to populate new gpus when the GLSL shader is still valid
  void buildResourcesOnNewGpus();

  void updatePrimitiveAssemblies(const std::shared_ptr<SqlQueryRowDataTableJSON>& data,
                                 const bool force_fill,
                                 const bool force_stroke);
  void updateUniformProperties();

  virtual void initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                         const bool data_changed,
                                         const bool init) = 0;

  virtual CoordPackingTypeBits getSupportedCoordPackingTypes() const {
    return CoordPackingTypeBits::kNone;
  }

  // Build GLSL shaders, generate ShaderCaches and Materials, and dirty props
  // so that dependent resources are built / updated
  virtual void updateShader() = 0;

  // Build Material resources in the fill or stroke slot
  void buildMaterials(const MarkGpuResourceSlot slot,
                      const std::string& resource_tracking_string,
                      gfx::ShaderCacheShPtrVector& caches,
                      const std::vector<GpuId>& gpus);

  virtual void setUniformAttributes(MarkPerGpuData& gpu_data) = 0;

  virtual void updatePipelines();

  virtual void buildPipelineDescriptors() = 0;
  virtual void buildPipelines(MarkPerGpuData& gpu_data) = 0;
  virtual void buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data);
  virtual void buildStrokePrimitiveAssemblies(MarkPerGpuData& gpu_data);

  virtual void updateRenderPropertyGpuResources(
      const std::vector<GpuId>& add_gpus,
      const std::vector<GpuId>& remove_gpus) = 0;

  void unsubscribeFromProjectionEvent();
  void subscribeToProjectionEvent(const ProjectionShPtr& projection);
  void projectionRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj);
  void updateProjectionPtr(ProjectionShPtr projection);

  void clearProjection();
  void clearTransforms();

  void updateDataPtr(BaseDataTableShPtr& data);
  void unsubscribeFromDataEvent();
  void subscribeToDataEvent();

  bool hasPackedPixelCoordDataColumn(std::string& name) const;

  bool is_visible_;
  ScaleWkPtr active_accumulator_;
  ProjectionWkPtr active_projection_;
  RefCallbackId projection_ref_subscription_id_ = 0;
  RefCallbackId data_ref_subscription_id_ = 0;

  struct GeoPropInfo {
    uint32_t prop_compression_bits = 0u;
    std::string prop_name_to_render;
    SQLTypes prop_type = kNULLT;
    EncodingType prop_encoding = EncodingType::kENCODING_NONE;
  };

  GeoPropInfo geo_prop_info_;

  const bool can_render_with_mesh_shader_;
  bool use_mesh_shader_;

  friend class QueryRendererContext;
};

}  // namespace QueryRenderer

ENABLE_BITMASK_OPS(QueryRenderer::BaseMark::CoordPackingTypeBits);
