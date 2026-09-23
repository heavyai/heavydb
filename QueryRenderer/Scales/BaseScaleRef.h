/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>

#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "GfxDriver/Types.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Marks/Types.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/AnyDataType.h"

namespace QueryRenderer {

class AnyDataType;

class BaseScaleRef {
 public:
  BaseScaleRef(QueryRendererContext& ctx,
               const ScaleShPtr& scale,
               BaseRenderProperty* rndr_prop);

  virtual ~BaseScaleRef() {}

  std::string getName() const;
  const std::string& getNameRef();

  ScaleShPtr getScalePtr() { return scale_; }

  virtual QueryDataType getDomainDataType() const = 0;
  virtual QueryDataType getRangeDataType() const = 0;

  virtual const gfx::TypeGLSLShPtr& getDomainTypeGLSL();
  virtual const gfx::TypeGLSLShPtr& getRangeTypeGLSL();

  // GLSL support
  // used by BaseMark to handle writing property types and mangled
  // scale functions.
  // TODO(scb): possibly move the ScaleRef handling code in BaseMark
  // into either the Scale or ScaleRef (Scale would actually be the more
  // consistent location. Most of these functions could then be made
  // private
  std::string getDomainGLSLTypeName(const std::string& extra_suffix);
  std::string getRangeGLSLTypeName(const std::string& extra_suffix);
  std::string getScaleGLSLFuncName(const std::string& extra_suffix);

  virtual const gfx::TypeGLSLShPtr& getDomainTypeGLSL() const = 0;
  virtual const gfx::TypeGLSLShPtr& getRangeTypeGLSL() const = 0;

  virtual gfx::ShaderManager::BuilderShPtr getShaderSubBuilder(
      const std::string& extra_suffix) const = 0;

  virtual void bindUniforms(gfx::Material& material, const std::string& extra_suffix) = 0;

  void buildSubroutineBindings(gfx::ShaderManager::Builder& builder,
                               const std::string& extra_suffix);

  // Get the active Domain and Range data objects. When type coercion is
  // in effect this will be the converted data, otherwise returns the
  // original data objects from BaseScale.
  virtual BaseScaleDomainRangeData* getDomainData() = 0;
  virtual BaseScaleDomainRangeData* getRangeData() = 0;

  // Get the Pct accumulation category types. Currently returns nullptr if
  // no override is in effect (FIXME(scb))
  const AnyDataType* getPctCatValPtr() const;
  const AnyDataType* getPctMarginValPtr() const;

  virtual void updateScaleRef(const ScaleShPtr& scale) = 0;

  const std::unordered_set<BaseDataTableShPtr> getDataRefs() const;

  AccumulatorType getAccumulatorType() const;
  bool hasAccumulator() const;

  // logging
  virtual operator std::string() const = 0;

 protected:
  void verifyScalePointer() const;

  std::string printInfo() const;

  const BaseDataTableShPtr& getDataTablePtr();
  std::string getDataColumnName();
  std::string getRenderPropertyName();

  void initScalePtr(const ScaleDomainRangeDataShPtr& domain_data,
                    const ScaleDomainRangeDataShPtr& range_data);
  void deleteScalePtr();

  QueryRendererContext& ctx_;
  ScaleShPtr scale_;

  std::unique_ptr<AnyDataType> pct_cat_val_;
  std::unique_ptr<AnyDataType> pct_margin_val_;

 private:
  BaseRenderProperty* rndr_prop_;
};

}  // namespace QueryRenderer
