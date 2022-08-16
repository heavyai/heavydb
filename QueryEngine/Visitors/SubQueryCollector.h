/*
 * Copyright 2020 OmniSci, Inc.
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

/*
 * @description SubQueryCollector is a visitor class that collects all
 * ScalarSubquery and InSubquery target nodes.
 */

#pragma once

#include "QueryEngine/ExprDagVisitor.h"

#include <unordered_set>

class SubQueryCollector final : public ExprDagVisitor {
 public:
  static std::unordered_set<const RelAlgNode*> getLiveSubQueries(RelAlgNode const*);

 private:
  void* visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override;
  void* visitInSubquery(const hdk::ir::InSubquery* in_subquery) const override;

  mutable std::unordered_set<const RelAlgNode*> subqueries_;
};
