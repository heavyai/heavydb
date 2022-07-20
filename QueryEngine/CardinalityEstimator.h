/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    CardinalityEstimator.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Estimators to be used when precise cardinality isn't useful.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_CARDINALITYESTIMATOR_H
#define QUERYENGINE_CARDINALITYESTIMATOR_H

#include "RelAlgExecutionUnit.h"

#include "../Analyzer/Analyzer.h"
#include "IR/Expr.h"
#include "Logger/Logger.h"

class CardinalityEstimationRequired : public std::runtime_error {
 public:
  CardinalityEstimationRequired(const int64_t range)
      : std::runtime_error("CardinalityEstimationRequired"), range_(range) {}

  int64_t range() const { return range_; }

 private:
  const int64_t range_;
};

namespace Analyzer {

/*
 * @type  Estimator
 * @brief Infrastructure to define estimators which take an expression tuple, are called
 * for every row and need a buffer to track state.
 */
class Estimator : public hdk::ir::Expr {
 public:
  Estimator() : Expr(SQLTypeInfo(kINT, true)){};

  // The tuple argument received by the estimator for every row.
  virtual const hdk::ir::ExprPtrList& getArgument() const = 0;

  // The size of the working buffer used by the estimator.
  virtual size_t getBufferSize() const = 0;

  // The name for the estimator runtime function which is called for every row.
  // The runtime function will receive four arguments:
  //   uint8_t* the pointer to the beginning of the estimator buffer
  //   uint32_t the size of the estimator buffer, in bytes
  //   uint8_t* the concatenated bytes for the argument tuple
  //   uint32_t the size of the argument tuple, in bytes
  virtual std::string getRuntimeFunctionName() const = 0;

  hdk::ir::ExprPtr deep_copy() const override {
    CHECK(false);
    return nullptr;
  }

  bool operator==(const Expr& rhs) const override {
    CHECK(false);
    return false;
  }

  std::string toString() const override {
    CHECK(false);
    return "";
  }
};

/*
 * @type  NDVEstimator
 * @brief Provides an estimate for the number of distinct tuples. Not a real
 *        Analyzer expression, it's only used in RelAlgExecutionUnit synthesized
 *        for the cardinality estimation before running an user-provided query.
 */
class NDVEstimator : public Analyzer::Estimator {
 public:
  NDVEstimator(const hdk::ir::ExprPtrList& expr_tuple)
      : expr_tuple_(expr_tuple) {}

  const hdk::ir::ExprPtrList& getArgument() const override {
    return expr_tuple_;
  }

  size_t getBufferSize() const override { return 1024 * 1024; }

  std::string getRuntimeFunctionName() const override {
    return "linear_probabilistic_count";
  }

 private:
  const hdk::ir::ExprPtrList expr_tuple_;
};

class LargeNDVEstimator : public NDVEstimator {
 public:
  LargeNDVEstimator(const hdk::ir::ExprPtrList& expr_tuple)
      : NDVEstimator(expr_tuple) {}

  size_t getBufferSize() const final;
};

}  // namespace Analyzer

RelAlgExecutionUnit create_ndv_execution_unit(const RelAlgExecutionUnit& ra_exe_unit,
                                              const int64_t range);

RelAlgExecutionUnit create_count_all_execution_unit(
    const RelAlgExecutionUnit& ra_exe_unit,
    hdk::ir::ExprPtr replacement_target,
    bool strip_join_covered_quals);

ResultSetPtr reduce_estimator_results(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& results_per_device);

#endif  // QUERYENGINE_CARDINALITYESTIMATOR_H
