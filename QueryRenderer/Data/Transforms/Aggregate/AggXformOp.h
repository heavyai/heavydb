/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Transforms/XformOp.h"

namespace QueryRenderer {

class AggOp : public XformOp {
 public:
  using VisitedInputsSetShPtr = std::shared_ptr<std::unordered_set<std::string>>;

  ~AggOp() override = default;

  SQLTypeInfo getOutputType() const override;
  gfx::BufferAttrType getOutputBufferAttrType() const override;

  // extracts the input attr name for aggregator ops.
  // The should all only have 1 input.
  // Primarily used as an accessor for
  struct AggInputNameExtractor {
    using result_type = std::string;

    result_type operator()(const XformOpShPtr& xform_op) {
      CHECK(xform_op);
      auto const& inputs = xform_op->getInputs();
      CHECK_EQ(inputs.size(), 1u);
      return inputs.begin()->attr_name;
    }
  };

  LayoutAttrInfo getInputInfo() const {
    CHECK_EQ(inputs_.size(), 1u);
    return *(inputs_.begin());
  }

  std::string getInputAttrName() const {
    CHECK_EQ(inputs_.size(), 1u);
    return inputs_.begin()->attr_name;
  }

 protected:
  AggOp(const XformShPtr& parent_xform,
        const LayoutAttrInfo& input_info,
        const VisitedInputsSetShPtr& visited_inputs,
        const bool is_vector_op)
      : XformOp(parent_xform, input_info, is_vector_op)
      , visited_inputs_{visited_inputs} {}

  void validateInputBuffers();

  mutable VisitedInputsSetShPtr visited_inputs_;
};

/**
 * A CRTP middle-layer class that provides an interface for handling thrust-generated
 * results. The CRTP pattern is used so that static functions can be used like virtual
 * functions and allow for override ability by derived classes. AggOp-derived classes that
 * make use of the OpExecuteUtils::executeThrustOp() utility method should also
 * inherit this class to conform to the interface that the
 * OpExecuteUtils::executeThrustOp() method expects.
 */
template <class DerivedOp>
class AggOpResultsInterface {
 public:
  /**
   * Creates empty data associated with an operator in the event that there are no results
   * returned from a query to operate on
   */
  static AggDataList createEmptyData(const QueryDataType data_type) {
    CHECK(false) << "Needs to be overridden";
    return {};
  }

  /**
   * Creates null data associated with an operator in the event that there is no results
   * returned from a query on a leaf in distributed. This way merging at the aggregator
   * can be more gracefully handled.
   */
  static AggDataList createNullData(const QueryDataType data_type) {
    CHECK(false) << "Needs to be overridden";
    return {};
  }

  /**
   * Merges/combines/flattens results from different sources and returns the flattened
   * result. This is called in two cases, to flatten results from across all the leaves on
   * the aggregator in distributed, and to flatten results from multiple gpus in a single
   * node instance. A vector of results is taken as an r-value (std::moved) as returned
   * data will be complete merge of all the data, and therefore the original vector should
   * no longer be necessary. The vector will ultimately be destroyed.
   */
  static AggDataList flattenResults(std::vector<AggDataList>&& results_to_merge) {
    // results vector will be destroyed on exit. All results should have been merged into
    // the returned value, so the initial vector of results would be no longer needed
    auto local_results = std::move(results_to_merge);
    CHECK_GT(local_results.size(), 0u);
    auto head_results = local_results.begin();
    auto results_itr = head_results;

    while (++results_itr != local_results.end()) {
      DerivedOp::mergeResults(*head_results, *results_itr);
    }
    return *head_results;
  }

 private:
  static void mergeResults(AggDataList& merged_result,
                           const AggDataList& result_to_merge) {
    CHECK(false) << "Need to be overridden";
  }
};

class AggDepOp : public AggOp {
 public:
  ~AggDepOp() override = default;
  bool isDependentOp() const final { return true; }

 protected:
  AggDepOp(const XformShPtr& parent_xform,
           const LayoutAttrInfo& input_info,
           const VisitedInputsSetShPtr& visited_inputs,
           const bool is_vector_op)
      : AggOp(parent_xform, input_info, visited_inputs, is_vector_op) {}

  static XformOp::DependencyOpTypeMap generateInputDependencyInfo(
      const XformOp* op,
      const XformOp::DependencyOutputsMap& dep_outputs);

  static void setInputDependency(XformOp* this_op,
                                 const XformOp::DependencyOutputsMap& dep_outputs,
                                 XformOp::DependencyOpMap& dependent_ops_map,
                                 const XformOpShPtr& dep_op);

  mutable DependencyOpMap dependent_ops_;
};

}  // namespace QueryRenderer
