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
 * @file    RelAlgExecutionUnit.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Execution unit for relational algebra. It's a low-level description
 *          of any relational algebra operation in a format understood by our VM.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_RELALGEXECUTIONUNIT_H
#define QUERYENGINE_RELALGEXECUTIONUNIT_H

#include "InputDescriptors.h"
#include "../Shared/sqldefs.h"

#include <list>
#include <memory>
#include <vector>

enum class SortAlgorithm { Default, SpeculativeTopN, StreamingTopN };

namespace Analyzer {

class Expr;
class NDVEstimator;
struct OrderEntry;

}  // Analyzer

struct SortInfo {
  const std::list<Analyzer::OrderEntry> order_entries;
  const SortAlgorithm algorithm;
  const size_t limit;
  const size_t offset;
};

struct JoinCondition {
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  JoinType type;
};

typedef std::vector<JoinCondition> JoinQualsPerNestingLevel;

struct RelAlgExecutionUnit {
  const std::vector<InputDescriptor> input_descs;
  const std::vector<InputDescriptor> extra_input_descs;
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
  std::list<std::shared_ptr<Analyzer::Expr>> quals;
  const JoinType join_type;
  const JoinQualsPerNestingLevel inner_joins;
  const std::vector<std::pair<int, size_t>> join_dimensions;
  const std::list<std::shared_ptr<Analyzer::Expr>> inner_join_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> outer_join_quals;
  const std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  std::vector<Analyzer::Expr*> target_exprs;
  const std::vector<Analyzer::Expr*> orig_target_exprs;
  const std::shared_ptr<Analyzer::NDVEstimator> estimator;
  const SortInfo sort_info;
  size_t scan_limit;
};

#endif  // QUERYENGINE_RELALGEXECUTIONUNIT_H
