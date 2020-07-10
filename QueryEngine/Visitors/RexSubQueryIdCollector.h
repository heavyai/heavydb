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
 * @description RexSubQueryIdCollector is a visitor class that collects all
 * RexSubQuery::getId() values for all RexSubQuery nodes. This uses sorted arrays of
 * (hash_code, handler) pairs for tree navigation.
 */

#pragma once

#include "RelRexDagVisitor.h"

#include <unordered_set>

class RexSubQueryIdCollector final : public RelRexDagVisitor {
 public:
  using RelRexDagVisitor::visit;

  using Ids = std::unordered_set<unsigned>;
  static Ids getLiveRexSubQueryIds(RelAlgNode const*);

 private:
  void visit(RexSubQuery const*) override;

  Ids ids_;
};
