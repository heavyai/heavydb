/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace QueryRenderer {

enum class XformType {
  kAggregate = 0,
  kFormula,
  kMaxXformType  // Always keep kMaxXformType as last item
};

enum class OpType {
  kCount = 0,
  kCountValid,
  kCountMissing,
  kMin,
  kMax,
  kSum,
  kAvg,
  kAverage = kAvg,
  kMean = kAvg,
  kVariance,
  kVarianceP,
  kStdDev,
  kStdDevP,
  kFormula,
  kDistinct,
  kCountDistinct,
  kMedian,
  kQuantile,
  kNonClientFacingSeparator,  // All items prior to this are client facing (and can be
                              // referenced in vega), all items after are only operators
                              // available internally, and can't be referenced in vega
  kSQDiffSum,
  kDistinctHistogram,  // TODO(croot): look for a way to generalize this into group-by,
                       // because ultimately a discrete histogram is a group-by with a
                       // count aggregate
  kTopK,      // Not exposing topk/bottomk yet. Ultimately they're group-by functions. So,
              // keeping
  kBottomK,   // them around for as they may have some properties useful for exposing
              // group-by in the future
  kMaxOpType  // Always keep an MAX_OP_TYPE as last item
};

std::string to_string(const XformType value);
int convert_string_to_xform_type_enum(const std::string& val);
std::string get_xform_types_as_string();

std::string to_string(const OpType value);
int convert_string_to_op_type_enum(const std::string& val);

bool is_agg_op_type(const OpType op_type);
bool is_client_facing_op_type(const OpType op_type);
std::string get_agg_ops_as_string();

std::ostream& operator<<(std::ostream& os, const XformType value);
std::ostream& operator<<(std::ostream& os, const OpType value);

}  // namespace QueryRenderer
