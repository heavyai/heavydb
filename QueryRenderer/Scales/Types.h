/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Shared/EnumBitmaskOps.h"

#include <functional>
#include <memory>
#include <vector>

namespace QueryRenderer {

enum class ScaleType {
  kLinear,
  kLog,
  kPow,
  kSqrt,
  kOrdinal,
  kQuantize,
  kThreshold,
  kUndefined
};
enum class ScaleInterpType { kRgb, kHsl, kHslLong, kLab, kHcl, kHclLong, kUndefined };

/**
 * AccumlatorType is used as both an explicit identifier, as well
 * as a type support mask returned by Scales
 * */
enum class AccumulatorType : uint8_t {
  kUndefined = 0x00,
  kMin = 0x01,
  kMax = 0x02,
  kBlend = 0x04,
  kPct = 0x08,
  kDensity = 0x10,
  kAll = 0xFF
};

// Domain / range validity flags
enum class ScaleDRChangedFlags : uint8_t {
  kNone = 0x00,
  kDomainSize = 0x01,
  kDomainVals = 0x02,
  kDomain = 0x03,  // DomainSize | DomainVals
  kRangeSize = 0x04,
  kRangeVals = 0x08,
  kRange = 0x0C  // RangeSize | RangeVals
};

enum class ScaleShaderUpdateFlags : uint8_t {
  kNone = 0x00,
  kTemplate = 0x01,
  kDomain = 0x02,
  kRange = 0x04,
  kNumDomains = 0x08,
  kNumRanges = 0x10,
  kAll = 0xFF
};

class BaseScaleDomainRangeData;
using ScaleDomainRangeDataUqPtr = std::unique_ptr<BaseScaleDomainRangeData>;
using ScaleDomainRangeDataShPtr = std::shared_ptr<BaseScaleDomainRangeData>;

class BaseScale;
using ScaleShPtr = std::shared_ptr<BaseScale>;
using ScaleWkPtr = std::weak_ptr<BaseScale>;

class BaseScaleRef;
using ScaleRefShPtr = std::shared_ptr<BaseScaleRef>;
using ScaleRefWkPtr = std::weak_ptr<BaseScaleRef>;

class ScaleAccumState;
using ScaleAccumStateUqPtr = std::unique_ptr<ScaleAccumState>;
class ScaleAccumRenderState;
using ScaleAccumRenderStateUqPtr = std::unique_ptr<ScaleAccumRenderState>;

std::string to_string(const ScaleType scale_type);
std::string to_string(const AccumulatorType accum_type);
std::string to_string(const ScaleInterpType interp_type);
std::vector<std::string> getScaleInterpTypes(
    const std::vector<ScaleInterpType>& interps = {});

bool isQuantitativeScale(const ScaleType type);
bool isContinuousDomainScale(const ScaleType type);

class JSONLocation;

template <typename T>
using ValidateFuncT = std::function<void(const JSONLocation&, const T&)>;

template <typename T>
using ConvertFuncT = std::function<T(const T&)>;

}  // namespace QueryRenderer

ENABLE_BITMASK_OPS(QueryRenderer::AccumulatorType)
ENABLE_BITMASK_OPS(QueryRenderer::ScaleDRChangedFlags)
ENABLE_BITMASK_OPS(QueryRenderer::ScaleShaderUpdateFlags)
