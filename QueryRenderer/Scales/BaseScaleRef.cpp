/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/BaseScaleRef.h"

#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/Data/QueryDataTableSQL.h"
#include "QueryRenderer/Marks/BaseRenderProperty.h"  // for getPrimaryTableName
#include "QueryRenderer/QueryRendererContext.h"      // for getRenderSessionKey
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/Scales/BaseScaleDomainRangeData.h"
#include "QueryRenderer/Scales/ScaleAccumRenderState.h"
#include "QueryRenderer/Scales/ScaleAccumState.h"

namespace QueryRenderer {

using ::gfx::ColorHCL;
using ::gfx::ColorHSL;
using ::gfx::ColorLAB;
using ::gfx::ColorRGBA;

BaseScaleRef::BaseScaleRef(QueryRendererContext& ctx,
                           const ScaleShPtr& scale,
                           BaseRenderProperty* rndr_prop)
    : ctx_(ctx), scale_(scale), rndr_prop_(rndr_prop) {}

std::string BaseScaleRef::getName() const {
  verifyScalePointer();
  return scale_->getName();
}

const std::string& BaseScaleRef::getNameRef() {
  verifyScalePointer();
  return scale_->getNameRef();
}

const gfx::TypeGLSLShPtr& BaseScaleRef::getDomainTypeGLSL() {
  verifyScalePointer();
  return scale_->getDomainTypeGLSL(false);
}

const gfx::TypeGLSLShPtr& BaseScaleRef::getRangeTypeGLSL() {
  verifyScalePointer();
  return scale_->getRangeTypeGLSL(false);
}

std::string BaseScaleRef::getDomainGLSLTypeName(const std::string& extra_suffix) {
  verifyScalePointer();
  return scale_->getDomainGLSLTypeName(extra_suffix);
}

std::string BaseScaleRef::getRangeGLSLTypeName(const std::string& extra_suffix) {
  verifyScalePointer();
  return scale_->getRangeGLSLTypeName(extra_suffix);
}

std::string BaseScaleRef::getScaleGLSLFuncName(const std::string& extra_suffix) {
  verifyScalePointer();
  return scale_->getScaleGLSLFuncName(extra_suffix, true);
}

void BaseScaleRef::buildSubroutineBindings(gfx::ShaderManager::Builder& builder,
                                           const std::string& extra_suffix) {
  scale_->buildSubroutineBindings(builder, extra_suffix, false);
  auto* accum_state = scale_->getAccumState();
  if (accum_state) {
    accum_state->bindScaleSubroutines(builder);
  }
}

const AnyDataType* BaseScaleRef::getPctCatValPtr() const {
  return pct_cat_val_.get();
}

const AnyDataType* BaseScaleRef::getPctMarginValPtr() const {
  return pct_margin_val_.get();
}

void BaseScaleRef::verifyScalePointer() const {
  RUNTIME_EX_ASSERT(
      scale_ != nullptr,
      std::string(*this) + ": The scale reference object is uninitialized.");
}

const BaseDataTableShPtr& BaseScaleRef::getDataTablePtr() {
  return rndr_prop_->getDataTablePtr();
}

std::string BaseScaleRef::getDataColumnName() {
  return rndr_prop_->getDataColumnName();
}

std::string BaseScaleRef::getRenderPropertyName() {
  return rndr_prop_->getName();
}

// FIXME(scb): What if 2 ScaleRefs have different domain overrides?
// Scales should pull the override from the ScaleRef similar to coerced
// types. This should allow making the ScalePtr const (which it needs to
// be if we want to allow multiple refs to the same scale to always work)
void BaseScaleRef::initScalePtr(const ScaleDomainRangeDataShPtr& domainDataPtr,
                                const ScaleDomainRangeDataShPtr& rangeDataPtr) {
  // doing this here because we only need to initialize
  // the gpu resources for a scale when it is being used
  // by a mark and it is guaranteed to be used by a mark
  // when a scale ref is created/updated, so we can
  // initialize the gpu resources here.
  auto accumulates = scale_->hasAccumulator();
  if (accumulates) {
    auto scaleDomainDataPtr = scale_->getDomainData(true);
    CHECK(scaleDomainDataPtr);
    if (scaleDomainDataPtr->getType() == QueryDataType::STRING) {
      // we've got a scale that is accumulating, but it has
      // strings for its domain. That means the strings are
      // dictionary encoded and have been converted to
      // their encoded values here in this scale ref. We need
      // to explicitly set the parent scale's domain to those
      // encoded values so the accumulation shader can get
      // set appropriately. We're going to override the
      // domain values on the scales

      // TODO(croot): is there ever a case where range values
      // are dictionary encoded? If so, we need to support that
      // here too. Or, if there if a day ever comes where non
      // dictionary-encoded strings are supported, then that
      // needs handling too. This logic could possibly be
      // removed once data references are supported in scales.
      // If that's the case, then the dictionary-encoding
      // conversion can take place on the parent scale and
      // therefore this wouldn't be needed.

      auto data_table =
          std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(getDataTablePtr());
      CHECK(data_table);
      auto const& data_table_sql = data_table->getQuerySQL();

      bool has_override = scale_->hasDomainOverride();
      RUNTIME_EX_ASSERT(!has_override || scale_->getDomainOverrideTableName() ==
                                             data_table_sql.getPrimaryTableName(),
                        std::string(*this) +
                            "Error trying to initialize a scale reference object. The "
                            "accumulator scale \"" +
                            scale_->getName() +
                            "\" is being referenced, but its domain has "
                            "dictionary-encoded strings and it's already "
                            "tied to table: " +
                            scale_->getDomainOverrideTableName() +
                            " whereas this reference is tied to table: " +
                            data_table_sql.getPrimaryTableName() +
                            ". Any references to accumulation scales that have a "
                            "dictionary-encoded string domain "
                            "must use the same data table.");

      if (!has_override) {
        scale_->setDomainOverride(domainDataPtr, data_table_sql);
      }

      scale_->setRangeOverride(rangeDataPtr);
    }
  }

  if (scale_->getAccumRenderState()) {
    scale_->getAccumRenderState()->initGpuResources(ctx_, true);
  }

  if (accumulates) {
    rndr_prop_->setAccumulatorFromScale(scale_);
  } else {
    rndr_prop_->clearAccumulatorFromScale(scale_);
  }
}

void BaseScaleRef::deleteScalePtr() {
  rndr_prop_->clearScalePtrForReplacement(scale_);
  scale_ = nullptr;
}

std::string BaseScaleRef::printInfo() const {
  std::string rtn = std::string(ctx_.getRenderSessionKey());
  if (scale_) {
    rtn += ", scale reference: " + std::string(*scale_);
  }
  if (rndr_prop_) {
    rtn += ", render property: " + std::string(*rndr_prop_);
  }

  return rtn;
}

AccumulatorType BaseScaleRef::getAccumulatorType() const {
  verifyScalePointer();
  return scale_->getAccumulatorType();
}

bool BaseScaleRef::hasAccumulator() const {
  verifyScalePointer();
  return scale_->hasAccumulator();
}

const std::unordered_set<BaseDataTableShPtr> BaseScaleRef::getDataRefs() const {
  return (scale_ ? scale_->getDataRefs() : std::unordered_set<BaseDataTableShPtr>());
}

}  // namespace QueryRenderer
