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

  std::shared_ptr<Analyzer::Expr> deep_copy() const override {
    CHECK(false);
    return nullptr;
  }

  bool operator==(const Expr& rhs) const override {
    CHECK(false);
    return false;
  }

  void print() const override { CHECK(false); }

 private:
  const std::list<std::shared_ptr<Analyzer::Expr>> expr_tuple_;
};

}  // Analyzer

RelAlgExecutionUnit create_ndv_execution_unit(const RelAlgExecutionUnit&);

#endif  // QUERYENGINE_CARDINALITYESTIMATOR_H
