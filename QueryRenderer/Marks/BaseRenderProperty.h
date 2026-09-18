/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>
#include <variant>

#include <rapidjson/pointer.h>

#include "GfxDriver/Colors/ColorUnion.h"
#include "GfxDriver/Pipeline/Types.h"
#include "GfxDriver/TypeGLSL.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Events/RefEvent.h"
#include "QueryRenderer/Events/Types.h"
#include "QueryRenderer/Interop/Types.h"
#include "QueryRenderer/JSONRefObject.h"  // for RefCallbackId
#include "QueryRenderer/Marks/Enums.h"
#include "QueryRenderer/Marks/Types.h"
#include "QueryRenderer/PerGpuData.h"
#include "QueryRenderer/Scales/BaseScaleRef.h"
#include "QueryRenderer/Scales/Types.h"
#include "Shared/EnumBitmaskOps.h"

namespace QueryRenderer {
enum class RenderPropertyFlagBits : uint32_t {
  kNone = 0,
  kUseScale = 1 << 0,
  kFlexibleType = 1 << 1,
  kAllowAccumulator = 1 << 2,
  kAllowNonColorStrings = 1 << 3,
  kResetOnEmptyDataUpdate = 1 << 4,
  kIsCoord = 1 << 5,
  kUseBDA = 1 << 6,
  kUnspecified = 0xffffffff
};

using RenderPropertyValue = std::
    variant<float, double, int32_t, uint32_t, int64_t, uint64_t, bool, gfx::ColorUnion>;

class BaseRenderProperty {
 public:
  enum class ChangeType {
    kStructure,  // Scale changed or anything requiring a new shader build
    kData,       // Query or some other significant change that *may* require a shader or
                 // primitive assembly rebuild
    kValues      // Internal changes that may result in uniform updates
  };

  // API facade to parent BaseMark
  class BaseMarkFacade {
   public:
    virtual ~BaseMarkFacade() = default;

    // Used for error message construction
    virtual GeomType getType() = 0;

    // DataTable pointer required for JSON parsing
    virtual BaseDataTableShPtr getDataPtr() = 0;

    // General change notification for resource dirtying (shaders, uniforms, etc)
    virtual void notifyChanged(ChangeType type) = 0;

    // Notify Mark of accumulator changes
    // Marks are restricted to one accumulator so must be notified on changes
    virtual void setAccumulatorFromScale(const ScaleShPtr scale,
                                         const ScaleRefShPtr scale_ref) = 0;
    virtual void clearAccumulatorFromScale(const ScaleShPtr scale) = 0;
  };

  BaseRenderProperty(const std::string& name,
                     QueryRendererContext& ctx,
                     BaseMarkFacade& mark_facade,
                     const RenderPropertyFlagBits flag_bits);
  virtual ~BaseRenderProperty();

  void initializeFromJSONObj(const JSONLocation& obj_loc, const BaseDataTableShPtr& data);

  bool initializeFromData(const std::string& attr_name, const BaseDataTableShPtr& data);

  virtual void initializeValue(const RenderPropertyValue& value) = 0;

  inline rapidjson::Pointer getJsonPath() const { return json_path_; }
  inline void updateJsonPath(const rapidjson::Pointer& path) { json_path_ = path; }

  int size(const GpuId& gpu_id) const;

  const std::string& getName() const { return name_; }

  const std::string& getGLSLFunc() const { return get_glsl_func_name_; }

  const gfx::TypeGLSLShPtr& getInTypeGLSL() const;
  const gfx::TypeGLSLShPtr& getOutTypeGLSL() const;

  bool hasVboPtr() const;
  bool hasVboPtr(const GpuId& gpu_id) const;

  bool hasSsboPtr() const;
  bool hasSsboPtr(const GpuId& gpu_id) const;

  bool isDecimal() const { return decimal_exp_scale_ != 0; }
  bool isCoord() const;
  bool usesBDA() const;

  gfx::TypeGLSLShPtr getDecimalTypeGLSL() const {
    return std::make_shared<gfx::TypeGLSL<double, 1>>();
  }

  bool isUndefined() const { return !in_type_ && !out_type_; }

  bool isDataDriven() const { return hasVboPtr() || hasSsboPtr() || usesScaleConfig(); }

  void clear() {
    clearReferences();
    json_path_ = rapidjson::Pointer();
  }

  QueryVertexBufferShPtr getVboPtr(const GpuId& gpu_id) const;
  QueryVertexBufferShPtr getVboPtr() const;

  QueryShaderStorageBufferShPtr getSsboPtr(const GpuId& gpu_id) const;
  QueryShaderStorageBufferShPtr getSsboPtr() const;

  bool usesScaleConfig() const { return (scale_config_ != nullptr || scale_ != nullptr); }
  ScaleShPtr getScale() const {
    return (scale_ ? scale_ : (scale_config_ ? scale_config_->getScalePtr() : nullptr));
  }

  const ScaleRefShPtr& getScaleReference() const { return scale_config_; }
  bool hasAccumulatorScale() const {
    return (scale_config_ ? scale_config_->hasAccumulator() : false);
  }

  void addToPrimitiveAssemblyAttrInfo(const GpuId& gpu_id,
                                      gfx::PrimitiveAssemblyAttrInfo& attr_info) const;

  virtual void setUniformAttribute(gfx::Material& active_material,
                                   const std::string& uniform_attr_name) const = 0;

  template <typename T>
  T getUniformValue() const {
    auto val = getUniformValueAsRenderPropertyValue();
    CHECK(std::holds_alternative<T>(val));
    return std::get<T>(val);
  }

  void setDecimalScaleUniformAttribute(gfx::Material& active_material) const;

  std::string getDataColumnName() const { return vbo_attr_name_; }
  const std::string& getDataColumnNameRef() const { return vbo_attr_name_; }
  const BaseDataTableShPtr& getDataTablePtr() const { return data_; }

  bool initGpuResources(const std::vector<GpuId>& add_gpus,
                        const std::vector<GpuId>& remove_gpus);

  virtual QueryDataType getDataType() const = 0;
  virtual bool hasAccumulator() const { return false; }
  virtual operator std::string() const = 0;

 protected:
  enum class VboInitType { kFromValue = 0, kFromDataRef, kFromScaleRef, kUndefined };

  std::string name_;
  std::string get_glsl_func_name_;
  BaseMarkFacade& mark_facade_;
  rapidjson::Pointer json_path_;
  const RenderPropertyFlagBits flag_bits_;
  uint64_t decimal_exp_scale_;

  std::string vbo_attr_name_;
  int real_col_id_;
  int real_table_id_;
  SQLTypes real_col_type_;
  EncodingType real_col_encoding_;

  class PerGpuData : public BasePerGpuData {
   public:
    QueryVertexBufferWkPtr vbo;
    QueryShaderStorageBufferWkPtr ssbo;

    explicit PerGpuData(RootPerGpuData& root_data,
                        const QueryVertexBufferShPtr& vbo = nullptr,
                        const QueryShaderStorageBufferShPtr& ssbo = nullptr)
        : BasePerGpuData{root_data}, vbo{vbo}, ssbo{ssbo} {}

    ~PerGpuData() override = default;
  };
  using PerGpuDataMap = std::map<GpuId, PerGpuData>;

  PerGpuDataMap per_gpu_data_;

  VboInitType vbo_init_type_;
  BaseDataTableShPtr data_;

  QueryRendererContext& ctx_;

  gfx::TypeGLSLShPtr in_type_;
  gfx::TypeGLSLShPtr out_type_;

  ScaleRefShPtr scale_config_;
  ScaleShPtr scale_;

  rapidjson::Pointer field_json_path_;
  rapidjson::Pointer value_json_path_;
  rapidjson::Pointer scale_json_path_;

  RefCallbackId scale_ref_subscription_id_;
  RefCallbackId data_ref_subscription_id_;

  virtual void initScaleFromJSONObj(const JSONLocation& obj_loc) = 0;
  virtual void initFromJSONObj(const JSONLocation& obj) {}
  virtual void initValueFromJSONObj(const JSONLocation& obj_loc,
                                    const bool has_scale,
                                    const bool reset_types = false) = 0;
  virtual void validateValue(const bool has_scale,
                             const rapidjson::Pointer& value_path) = 0;
  virtual void resetValue() = 0;
  virtual RenderPropertyValue getUniformValueAsRenderPropertyValue() const = 0;
  virtual bool resetTypes(const bool reset_in_type = true,
                          const bool reset_out_type = true) = 0;
  virtual std::pair<bool, bool> initTypeFromBuffer(const bool has_scale = false) = 0;
  virtual void validateScale() = 0;
  virtual void scaleRefUpdateCB(RefEventType ref_event_type,
                                const RefObjShPtr& ref_obj) = 0;
  virtual void updateScalePtr(const ScaleShPtr& scale);

  virtual gfx::TypeGLSLShPtr createDefaultType() const = 0;
  virtual bool validateInType(const gfx::TypeGLSLShPtr& type) = 0;
  virtual bool validateOutType(const gfx::TypeGLSLShPtr& type) = 0;

  virtual void dataRefUpdateCB(RefEventType ref_event_type, const RefObjShPtr& ref_obj);
  void clearFieldPath();
  void clearDataPtr();
  void clearScalePtr();
  void clearScalePtrForReplacement(const ScaleShPtr& scale);

  bool checkAccumulator(const ScaleShPtr& scale);
  void setAccumulatorFromScale(const ScaleShPtr& scale);
  void clearAccumulatorFromScale(const ScaleShPtr& scale);
  void unsubscribeFromScaleEvent(const ScaleShPtr& scale);
  void unsubscribeFromDataEvent();
  void notifyChanged(ChangeType type);

  std::string printInfo() const;

 private:
  bool initBuffers(const std::map<GpuId, QueryLayoutBufferWkPtr>& buffer_map);
  bool internalInitFromData(const std::string& attr_name,
                            const BaseDataTableShPtr& data,
                            const bool has_scale,
                            const bool updating_scale);

  virtual void clearReferences() = 0;

  friend class BaseScaleRef;
};

using BaseRenderPropertyConstSet = std::unordered_set<const BaseRenderProperty*>;
using BaseRenderPropertySet = std::unordered_set<BaseRenderProperty*>;

}  // namespace QueryRenderer

ENABLE_BITMASK_OPS(::QueryRenderer::RenderPropertyFlagBits);
