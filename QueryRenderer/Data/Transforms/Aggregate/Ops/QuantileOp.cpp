/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Transforms/Aggregate/Ops/QuantileOp.h"

#ifdef HAVE_CUDA
#include "GfxInterop/Utils/CudaErrorCheck.h"
#endif  // HAVE_CUDA

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/DataMgr.h"
#include "QueryRenderer/Data/Transforms/Aggregate/AggError.h"
#include "QueryRenderer/Data/Transforms/Aggregate/Ops/DistinctHistogramOp.h"
#include "QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h"
#include "QueryRenderer/Utils/TypeUtils.h"

namespace QueryRenderer {

namespace {

inline std::string get_default_dep_name() {
  return "distincthistogramval";
}

std::string serialize_deps_from_props(const QuantileOp::Props& props) {
  std::stringstream ss;
  DistinctHistogramOp::serializeFromProps(ss, props);
  return ss.str();
}

template <typename T>
std::vector<T> get_quantiles_from_distinct_histogram(
    const AnyDataType& distinct_vals,
    const std::vector<uint32_t>& distinct_cnts,
    const uint64_t population_size,
    const QuantileProps& props) {
  if (props.num_quantiles == 1) {
    return {distinct_vals.getValAtIndex<T>(0)};
  }

  std::vector<T> quantiles;
  if (props.include_extrema) {
    quantiles.push_back(distinct_vals.getValAtIndex<T>(0));
  }
  if (population_size == 1) {
    quantiles.resize(quantiles.size() + props.num_quantiles - 1,
                     distinct_vals.getValAtIndex<T>(0));
  } else {
    double curr_idx{0};
    auto population_size_d = static_cast<double>(population_size);
    double quantile_diff =
        (population_size_d - 1) / std::max(static_cast<double>(props.num_quantiles), 1.0);
    double curr_quantile = quantile_diff;
    double quantile_idx = std::floor(curr_quantile);
    bool finished = false;
    for (size_t i = 0; i < distinct_cnts.size() && !finished; ++i) {
      curr_idx += static_cast<double>(distinct_cnts[i]);
      while (quantile_idx < curr_idx) {
        auto value0 = distinct_vals.getValAtIndex<T>(i);
        if (quantile_idx + 1 == curr_idx) {
          auto value1 = distinct_vals.getValAtIndex<T>(i + 1);
          quantiles.push_back(value0 +
                              (value1 - value0) * (curr_quantile - quantile_idx));
        } else {
          quantiles.push_back(value0);
        }

        curr_quantile += quantile_diff;
        if (std::abs(curr_quantile - population_size_d + 1) <= 0.00001) {
          finished = true;
          break;
        }
        quantile_idx = std::floor(curr_quantile);
      }
    }
  }
  if (props.include_extrema) {
    quantiles.push_back(distinct_vals.getValAtIndex<T>(distinct_cnts.size() - 1));
    CHECK_EQ(quantiles.size(), static_cast<size_t>(props.num_quantiles + 1));
  } else {
    CHECK_EQ(quantiles.size(), static_cast<size_t>(props.num_quantiles - 1));
  }
  return quantiles;
}

std::vector<std::shared_ptr<AnyDataType>>
build_quantile_result_from_distinct_histogram_results(
    const std::vector<std::shared_ptr<AnyDataType>>& distinct_result,
    const QuantileProps& props) {
  const std::vector<uint32_t>* cnt_vec;
  if (distinct_result[1]->getType() == QueryDataType::UINT) {
    cnt_vec = &distinct_result[1]->getVectorRef<uint32_t>();
  } else {
    // TODO(croot): we should be assured that all values in the int case (which is
    // created due to sql_types not having an unsigned int version, so can't convert to
    // unsigned int during distributed) are > 0. But should we check? And should we
    // check that they're all < min<int>?
    cnt_vec = reinterpret_cast<const std::vector<uint32_t>*>(
        &distinct_result[1]->getVectorRef<int32_t>());
  }
  uint64_t sum{0};
  for (auto const& item_cnt : *cnt_vec) {
    CHECK_GT(item_cnt, 0u) << item_cnt;
    CHECK_LE(item_cnt, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        << item_cnt;
    sum += item_cnt;
  }

  if (!sum) {
    // TODO(croot): how should we appropriately handle the case where we're trying to
    // calculate the median/quantile on empty data
    return {std::make_shared<AnyDataType>(std::vector<int>())};
  } else {
    switch (distinct_result[0]->getType()) {
      case QueryDataType::INT:
      case QueryDataType::UINT:
      case QueryDataType::FLOAT:
        return {
            std::make_shared<AnyDataType>(get_quantiles_from_distinct_histogram<float>(
                *distinct_result[0], *cnt_vec, sum, props))};
        break;
      case QueryDataType::INT64:
      case QueryDataType::UINT64:
      case QueryDataType::DOUBLE:
        return {
            std::make_shared<AnyDataType>(get_quantiles_from_distinct_histogram<double>(
                *distinct_result[0], *cnt_vec, sum, props))};
        break;
      default:
        throw std::runtime_error("Data of type " +
                                 to_string(distinct_result[0]->getType()) +
                                 ". Only single-value types are currently supported "
                                 "for calculating median.");
    }
  }
  CHECK(false);
  return {};
}

AggDataList execute_quantile_op(const QuantileOp* parent_xform_op,
                                Data_Namespace::DataMgr& data_mgr,
                                const XformOp::DependencyOpResultsMap& dependency_results,
                                const std::string& distinct_histogram_attr,
                                const QuantileProps& props) {
  auto distinct_itr = dependency_results.find(distinct_histogram_attr);
  CHECK(distinct_itr != dependency_results.end());

  // NOTE: not checking that varresult are defined
  // because the getResult() call does that
  auto const& distinct_result = distinct_itr->second;
  CHECK_EQ(distinct_result.size(), 2u);
  CHECK(distinct_result[0]);
  CHECK(distinct_result[1]);
  CHECK(distinct_result[0]->isVector());
  CHECK(distinct_result[1]->isVector());
  CHECK_EQ(distinct_result[0]->size(), distinct_result[1]->size());
  CHECK(distinct_result[1]->getType() == QueryDataType::UINT ||
        distinct_result[1]->getType() == QueryDataType::INT)
      << distinct_result[1]->getType();

  const size_t min_size_for_thrust =
      50000;  // 50K seems like a good middleground # to determine when to
              // use thrust. On my machine, thrust started winning with about 200K
              // values, but my machine has Intel Xeon E5-1650 w/ 3.5GHz and 6 cores.
              // 50K seemed like a good middle-ground value to handle slower processors.
              // TODO(croot): make this dynamic or system-dependent?
  if (distinct_result[0]->size() > min_size_for_thrust) {
    // NOTE: we're not running this through the executeThrustDependencyOp()
    // call because that evaluator is intended to be run for merge-able operators.
    // Quantiles are not merge-able as they require all
    // the merged histogram results up front, so no merging is necessary. That means,
    // when we get to this point, all the data required should be available.

#ifdef HAVE_CUDA
    auto& cuda_ctx_vector = data_mgr.getCudaMgr()->getDeviceContexts();
    // TODO(croot): have a better metric for which gpu to use here
    // Use the one with less gpu memory?
    // Defaulting to the first for now.
    CHECK_RENDER_CUDA_ERRORS(cuCtxSetCurrent(cuda_ctx_vector[0]), 0);
#endif  // HAVE_CUDA
    auto thrust_context = DataMgrThrustContext(ThrustAllocator(&data_mgr, 0));
    auto thrust_op_executor = ThrustOpExecutor(thrust_context);
    return thrust_op_executor.executeQuantileOp(distinct_result, props);
  } else {
    return build_quantile_result_from_distinct_histogram_results(distinct_result, props);
  }
}

}  // namespace

QuantileOp::QuantileOp(const XformShPtr& parent_xform,
                       const LayoutAttrInfo& input_info,
                       const VisitedInputsSetShPtr& visited_inputs,
                       const uint16_t num_quantiles,
                       const bool include_extrema,
                       const bool approximate,
                       const size_t num_bins)
    : AggDepOp(parent_xform, input_info, visited_inputs, true)
    , is_median_{false}
    , props_{num_quantiles, include_extrema, approximate, num_bins}
    , dep_outputs_{{{serialize_deps_from_props(props_), get_default_dep_name()}}} {
  validateInputs();
  validateInputBuffers();
}

QuantileOp::QuantileOp(const XformShPtr& parent_xform,
                       const LayoutAttrInfo& input_info,
                       const VisitedInputsSetShPtr& visited_inputs,
                       const bool approximate,
                       const size_t num_bins)
    : AggDepOp(parent_xform, input_info, visited_inputs, true)
    , is_median_{true}
    , props_{2, false, approximate, num_bins}
    , dep_outputs_{{{serialize_deps_from_props(props_), get_default_dep_name()}}} {
  validateInputs();
  validateInputBuffers();
}

QuantileOp::QuantileOp(const XformShPtr& parent_xform,
                       const LayoutAttrInfo& input_info,
                       const VisitedInputsSetShPtr& visited_inputs,
                       const JSONLocation& json_loc)
    : QuantileOp(parent_xform,
                 input_info,
                 visited_inputs,
                 Props::getNumQuantilesFromJSONObj(json_loc),
                 Props::getIncludeExtremaFromJSONObj(json_loc),
                 DistinctHistogramOp::Props::getApproximateFromJSONObj(json_loc),
                 DistinctHistogramOp::Props::getNumBinsFromJSONObj(json_loc)) {}

SQLTypeInfo QuantileOp::getOutputType() const {
  // NOTE: median is a floating-pt value to properly handle the case
  // where there is an even number of values to compute the median
  return get_float_equivalent_type(AggOp::getOutputType());
}

const XformOp::OpResult QuantileOp::executeOp(
    const std::string& evaluator_name,
    InteropBufferMgr* mapped_buffers,
    const DependencyOpResultsMap& dependency_results) {
  try {
    CHECK_EQ(dep_outputs_.size(), 1u);
    auto const& distinct_histogram_attr = dep_outputs_.begin()->second;
    return {true,
            execute_quantile_op(
                this, getDataMgr(), dependency_results, distinct_histogram_attr, props_)};
  } catch (...) {
    LOG_AGG_INFO_AND_THROW(*this);
  }
  return {true, {}};
}

void QuantileOp::serializePropsFromJSONObj(std::stringstream& ss,
                                           const JSONLocation& json_loc) {
  Props props(json_loc);
  props.serialize(ss);
}

/******************** dependency info ******************/
XformOp::DependencyOpTypeMap QuantileOp::getRequiredDependencyInfo() const {
  return generateInputDependencyInfo(this, dep_outputs_);
}

void QuantileOp::setDependency(const XformOpShPtr& op) {
  setInputDependency(this, dep_outputs_, dependent_ops_, op);
}

const XformOp::DependencyOpMap* QuantileOp::getDependencyOps(
    const std::string* op_type) const {
  auto op = OpType::kMaxOpType;
  std::vector<AnyDataType> args;
  if (op_type) {
    std::tie(op, args) = XformOp::deserializeOperatorAndProps(*op_type);
    auto const dep_outputs = dep_outputs_;
    CHECK(dep_outputs.find(*op_type) != dep_outputs.end());
    CHECK_EQ(args.size(), 2u);
    CHECK_EQ(props_.approximate, args[0].getVal<bool>());
    CHECK_EQ(props_.num_bins, args[1].getVal<size_t>());
  }
  if (!op_type || op == OpType::kDistinctHistogram) {
    auto const output = get_default_dep_name();
    if (dependent_ops_.find(output) == dependent_ops_.end()) {
      CHECK(dependent_ops_
                .try_emplace(output,
                             std::make_shared<DistinctHistogramOp>(parent_xform_.lock(),
                                                                   getInputInfo(),
                                                                   visited_inputs_,
                                                                   props_.approximate,
                                                                   props_.num_bins))
                .second);
    }
  }
  return &dependent_ops_;
}

}  // namespace QueryRenderer
