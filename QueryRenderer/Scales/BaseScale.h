/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>

#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/Scales/Types.h"

namespace QueryRenderer {

class ScaleImplBase;
using ScaleImplUqPtr = std::unique_ptr<ScaleImplBase>;

class BaseScale : public JSONRefObject {
 public:
  enum class ScaleShaderType : uint8_t { kQuantitative, kOrdinal, kQuantize, kThreshold };

  struct BindOptions {
    bool use_domain;
    bool use_range;
    bool use_accum;
    bool use_null;
  };

  BaseScale(const JSONLocation& json_loc,
            QueryRendererContext& ctx,
            const std::string& name,
            const ScaleType type);

  ~BaseScale() override;

  //
  // Factory settings
  //
  void setScaleImpl(ScaleImplUqPtr impl);
  void setDomainData(ScaleDomainRangeDataUqPtr domain_data);
  void setRangeData(ScaleDomainRangeDataUqPtr range_data);

  ScaleType getType() { return type_; }

  // returns true if domain or range data, null value, accumulator, or any
  // other special properties changed (eg clamp, which impacts the shader).
  bool updateFromJSONObj(const JSONLocation& json_loc);
  // cleanup changed flags
  void postJSONUpdate();
  void markForDeletion();

  //
  // Accumulation rendering
  //
  // Support flags. Kept direct for now. Will move to bitflags if
  // necessary in the future.
  AccumulatorType getValidAccumTypeMask() const;
  bool supportsAccumNulls() const;

  AccumulatorType getAccumulatorType() const;
  bool hasAccumulator() const;

  uint32_t getNumValuesForAccumulation() const;
  void bindAccumulatorColors(gfx::Material& material, const std::string& attr_name);

  // these objects exist in sync (both or neither of these accessors may return null)
  ScaleAccumState* getAccumState() const;
  ScaleAccumRenderState* getAccumRenderState() const;

  //
  // Domain / Range data input
  //
  QueryDataType getDomainDataType() const;
  QueryDataType getRangeDataType() const;

  gfx::ColorType getRangeColorType() const;

  // This is ultimately checking for STRING data types and is only
  // called by RenderProperty.
  QueryDataType getPrimaryDomainDataType() const;

  // When get_orig == false, returns the TypeGLSL to use. In cases with an override type
  // (e.g. uint for string data), it returns the override type.
  // This flag exists due to external callers that expected the non-override return,
  // particularly in RenderProperty.
  // FIXME(scb): continue cleaning up external callers so get_orig is not required.
  const gfx::TypeGLSLShPtr& getDomainTypeGLSL(bool get_orig = true) const;
  const gfx::TypeGLSLShPtr& getRangeTypeGLSL(bool get_orig = true) const;

  // DomainRangeData accessors
  const BaseScaleDomainRangeData* getDomainData(const bool get_orig = false) const;
  const BaseScaleDomainRangeData* getRangeData(const bool get_orig = false) const;

  BaseScaleDomainRangeData* getDomainData(const bool get_orig = false) {
    return const_cast<BaseScaleDomainRangeData*>(
        static_cast<const BaseScale&>(*this).getDomainData(get_orig));
  }
  BaseScaleDomainRangeData* getRangeData(const bool get_orig = false) {
    return const_cast<BaseScaleDomainRangeData*>(
        static_cast<const BaseScale&>(*this).getRangeData(get_orig));
  }

  // DataRef accessor
  bool hasDataRef() const;
  const std::unordered_set<BaseDataTableShPtr> getDataRefs() const;

  //
  // Shader code
  //
  std::string getScaleGLSLFuncName(const std::string& extra_suffix, bool use_accumulator);

  // Called by ScaleRef (from Marks) and ScaleAccumState
  gfx::ShaderManager::BuilderShPtr getShaderSubBuilder(const BaseScaleRef* ref,
                                                       const std::string& extra_suffix,
                                                       bool use_accum) const;

  std::string getDomainGLSLTypeName(const std::string& extra_suffix = "") {
    return "domainType_" + name_ + extra_suffix;
  }
  std::string getRangeGLSLTypeName(const std::string& extra_suffix = "") {
    return "rangeType_" + name_ + extra_suffix;
  }

  std::string getDomainGLSLUniformName() { return "uDomains_" + name_; }
  std::string getRangeGLSLUniformName() { return "uRanges_" + name_; }

  void bindUniforms(gfx::Material& material,
                    const std::string& extra_suffix,
                    bool use_domain,
                    bool use_range,
                    bool use_accum,
                    const SQLTypeInfo* sql_type_info);

  void buildSubroutineBindings(gfx::ShaderManager::Builder& builder,
                               const std::string& extra_suffix,
                               bool is_accum_final_pass);
  //
  // Change notification
  //
  bool hasDomainDataChanged() const {
    return (dr_changed_flags_ & ScaleDRChangedFlags::kDomain) !=
           ScaleDRChangedFlags::kNone;
  }
  bool hasRangeDataChanged() const {
    return (dr_changed_flags_ & ScaleDRChangedFlags::kRange) !=
           ScaleDRChangedFlags::kNone;
  }

  bool isMarkedForDeletion() const { return marked_for_deletion_; }
  bool isShaderDirty() const;

  //
  // Domain / Range uniform access
  //
  using DomainTypeUniforms =
      std::pair<QueryDataType, std::unordered_map<std::string, std::any>>;
  DomainTypeUniforms getDomainTypeUniforms(const std::string& extra_suffix,
                                           const SQLTypeInfo* sql_type_info) const;

  using RangeTypeUniforms =
      std::pair<QueryDataType, std::unordered_map<std::string, std::any>>;
  RangeTypeUniforms getRangeTypeUniforms(const std::string& extra_suffix) const;

 private:
  ScaleType type_;
  ScaleImplUqPtr impl_;
  ScaleAccumStateUqPtr accum_state_;
  ScaleAccumRenderStateUqPtr accum_render_state_;

  ScaleDomainRangeDataUqPtr domain_data_;
  ScaleDomainRangeDataUqPtr range_data_;

  // TODO(croot): somehow consolidate all the types and use typeid() or the like
  // to handle type-ness.
  gfx::TypeGLSLShPtr domain_type_glsl_;
  gfx::TypeGLSLShPtr range_type_glsl_;

  ScaleDRChangedFlags dr_changed_flags_;
  bool accumulator_changed_;
  bool marked_for_deletion_;

  struct OverrideData {
    ScaleDomainRangeDataShPtr data;
    const QueryDataTableSQL* data_table{nullptr};

    void reset() {
      data = nullptr;
      data_table = nullptr;
    }
  };

  OverrideData domain_override_data_;
  ScaleDomainRangeDataShPtr range_override_data_;

  bool updateAccumulatorFromJSONObj(const JSONLocation& json_loc);

  void setJSONPathRef(const rapidjson::Pointer& obj_path) { json_path_ = obj_path; }

  void toJSONInternal(rapidjson::Value& obj,
                      rapidjson::Document::AllocatorType& allocator) const override;

  inline std::string getNullGLSLAttrName(const std::string& extra_suffix) const {
    return "nullDomainVal_" + this->name_ + extra_suffix;
  }

  // Domain / Range overrides
  // These are set by ScaleRef when coercing types
  void setDomainOverride(const ScaleDomainRangeDataShPtr& domain_override,
                         const QueryDataTableSQL& domain_override_table);
  void setRangeOverride(const ScaleDomainRangeDataShPtr& range_override);
  bool hasDomainOverride() const;
  bool hasRangeOverride() const;
  std::string getDomainOverrideTableName() const;

  void setDRChangedFlags(ScaleDRChangedFlags flags);

  std::string printInfo() const;
  // FIXME(scb) - verify this still outputs adequate info.
  // Make operator explicit or convert to a toString(). The latter
  // is more typical for classes that don't need to convert to string.
  operator std::string() const { return printInfo(); }

  // calls numerous ScaleDomainRangeData override related functions
  friend class BaseScaleRef;

  // call methods to invalidate size or values of data:
  //   _setDRChangedFlags()
  //   _validateDomainRangeSizes()
  friend class BaseScaleDomainRangeData;

  template <typename DomainType, typename RangeType>
  friend class Scale;
};

// Interface class for scale implementations that will be called by BaseScale.
// This class is exposed for now but do not call into directly (call through BaseScale).
// Do not add common code here. Optional stubs are OK for private methods, but
// anything with a non-trivial default implementation should be protected here
// and the default implementation goes in Scale
class ScaleImplBase {
 public:
  virtual ~ScaleImplBase() {}

 protected:
  // Accumulation
  virtual AccumulatorType getValidAccumTypeMask() const = 0;
  virtual bool supportsAccumNulls() const = 0;

  virtual uint32_t getNumValuesForAccumulation() const = 0;
  virtual void bindAccumulatorColors(gfx::Material& material,
                                     const std::string& attr_name) = 0;
  // Properties and nulls
  virtual void initNullValueFromJSONObj(const JSONLocation& json_loc) = 0;
  virtual bool hasNullValue() const = 0;
  virtual bool havePropertiesChanged() const = 0;
  virtual bool haveUniformPropertiesChanged() const = 0;

  // Range binding
  virtual gfx::ColorType getRangeColorType() const = 0;
  virtual BaseScale::RangeTypeUniforms getRangeTypeUniforms(
      const std::string& extra_suffix) const = 0;

  // Only used by Threshold
  virtual void validateDomainRangeSizes(const JSONLocation& json_loc) = 0;

  // Serialization
  virtual void toJSONInternal(rapidjson::Value& obj,
                              rapidjson::Document::AllocatorType& allocator) const = 0;

 private:
  //
  // JSON parsing
  //
  // Primary parsing. Called prior to domain range updating
  virtual void updateFromJSONObj(const JSONLocation& json_loc) = 0;
  // Let the connected DomainRangeData objects update
  virtual ScaleDRChangedFlags updateDRDataFromJSONObj(const JSONLocation& json_loc) = 0;
  // Final (optional) update opportunity after the JSON cache has been checked and the
  // DomainRangeData inputs have been updated.
  // Only called if the json cache was updated or a DomainRangeData updated
  virtual void postDRDataJSONUpdate(const JSONLocation& json_loc) {}

  //
  // Uniform and Subroutine handling
  //
  virtual void modifyBindOptions(BaseScale::BindOptions& opt) = 0;
  virtual BaseScale::ScaleShaderType getShaderType() = 0;
  virtual void modifyShaderTemplate(gfx::ShaderManager::Builder& builder) const {}

  virtual void bindUniforms(gfx::Material& material,
                            const std::string& extra_suffix,
                            const BaseScale::BindOptions& bind_opt,
                            const SQLTypeInfo* sql_type_info) = 0;

  virtual void getDomainTypeUniformsInternal(BaseScale::DomainTypeUniforms& uniform_data,
                                             const std::string& extra_suffix,
                                             const SQLTypeInfo* sql_type_info) const = 0;

  // Set subroutines specifc to the implementation (optional)
  virtual void bindImplementationSubroutines(gfx::ShaderManager::Builder& builder,
                                             const std::string& extra_suffix,
                                             bool is_accum_final_pass) {}

  friend class BaseScale;
};

}  // namespace QueryRenderer
