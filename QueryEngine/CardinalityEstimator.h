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

#include <glog/logging.h>

class CardinalityEstimationRequired : public std::runtime_error {
 public:
  CardinalityEstimationRequired() : std::runtime_error("CardinalityEstimationRequired") {}
};

namespace Analyzer {

/*
 * @type  NDVEstimator
 * @brief Provides an estimate for the number of distinct tuples. Not a real
 *        Analyzer expression, it's only used in RelAlgExecutionUnit synthesized
 *        for the cardinality estimation before running an user-provided query.
 */
class NDVEstimator : public Analyzer::Expr {
 public:
  NDVEstimator(const std::list<std::shared_ptr<Analyzer::Expr>>& expr_tuple)
      : Expr(SQLTypeInfo(kINT, true)), expr_tuple_(expr_tuple) {}

  const std::list<std::shared_ptr<Analyzer::Expr>>& getArgument() const { return expr_tuple_; }

  std::shared_ptr<Analyzer::Expr> deep_copy() const override {
    CHECK(false);
    return nullptr;
  }

  bool operator==(const Expr& rhs) const override {
    CHECK(false);
    return false;
  }

  void print() const override { CHECK(false); }

  size_t getEstimatorBufferSize() const { return 512 * 1024; }

 private:
  const std::list<std::shared_ptr<Analyzer::Expr>> expr_tuple_;
};

}  // Analyzer

inline RelAlgExecutionUnit create_ndv_execution_unit(const RelAlgExecutionUnit& ra_exe_unit) {
  return {ra_exe_unit.input_descs,
          ra_exe_unit.extra_input_descs,
          ra_exe_unit.input_col_descs,
          ra_exe_unit.simple_quals,
          ra_exe_unit.quals,
          ra_exe_unit.join_type,
          ra_exe_unit.join_dimensions,
          ra_exe_unit.inner_join_quals,
          ra_exe_unit.outer_join_quals,
          {},
          {},
          {},
          std::make_shared<const Analyzer::NDVEstimator>(ra_exe_unit.groupby_exprs),
          SortInfo{{}, SortAlgorithm::Default, 0, 0},
          0};
}

#endif  // QUERYENGINE_CARDINALITYESTIMATOR_H
