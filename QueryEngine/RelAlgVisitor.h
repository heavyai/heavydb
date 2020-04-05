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

#ifndef QUERYENGINE_RELALGVISITOR_H
#define QUERYENGINE_RELALGVISITOR_H

#include "RelAlgDagBuilder.h"

template <class T>
class RelAlgVisitor {
 public:
  T visit(const RelAlgNode* rel_alg) const {
    auto result = defaultResult();
    for (size_t i = 0; i < rel_alg->inputCount(); ++i) {
      result = aggregateResult(result, visit(rel_alg->getInput(i)));
    }
    const auto aggregate = dynamic_cast<const RelAggregate*>(rel_alg);
    if (aggregate) {
      return aggregateResult(result, visitAggregate(aggregate));
    }
    const auto compound = dynamic_cast<const RelCompound*>(rel_alg);
    if (compound) {
      return aggregateResult(result, visitCompound(compound));
    }
    const auto filter = dynamic_cast<const RelFilter*>(rel_alg);
    if (filter) {
      return aggregateResult(result, visitFilter(filter));
    }
    const auto join = dynamic_cast<const RelJoin*>(rel_alg);
    if (join) {
      return aggregateResult(result, visitJoin(join));
    }
    const auto left_deep_inner_join = dynamic_cast<const RelLeftDeepInnerJoin*>(rel_alg);
    if (left_deep_inner_join) {
      return aggregateResult(result, visitLeftDeepInnerJoin(left_deep_inner_join));
    }
    const auto project = dynamic_cast<const RelProject*>(rel_alg);
    if (project) {
      return aggregateResult(result, visitProject(project));
    }
    const auto scan = dynamic_cast<const RelScan*>(rel_alg);
    if (scan) {
      return aggregateResult(result, visitScan(scan));
    }
    const auto sort = dynamic_cast<const RelSort*>(rel_alg);
    if (sort) {
      return aggregateResult(result, visitSort(sort));
    }
    const auto logical_values = dynamic_cast<const RelLogicalValues*>(rel_alg);
    if (logical_values) {
      return aggregateResult(result, visitLogicalValues(logical_values));
    }
    const auto modify = dynamic_cast<const RelModify*>(rel_alg);
    if (modify) {
      return aggregateResult(result, visitModify(modify));
    }
    const auto table_func = dynamic_cast<const RelTableFunction*>(rel_alg);
    if (table_func) {
      return aggregateResult(result, visitTableFunction(table_func));
    }
    const auto logical_union = dynamic_cast<const RelLogicalUnion*>(rel_alg);
    if (logical_union) {
      return aggregateResult(result, visitLogicalUnion(logical_union));
    }
    LOG(FATAL) << "Unhandled rel_alg type: " << rel_alg->toString();
    return {};
  }

  virtual T visitAggregate(const RelAggregate*) const { return defaultResult(); }

  virtual T visitCompound(const RelCompound*) const { return defaultResult(); }

  virtual T visitFilter(const RelFilter*) const { return defaultResult(); }

  virtual T visitJoin(const RelJoin*) const { return defaultResult(); }

  virtual T visitLeftDeepInnerJoin(const RelLeftDeepInnerJoin*) const {
    return defaultResult();
  }

  virtual T visitProject(const RelProject*) const { return defaultResult(); }

  virtual T visitScan(const RelScan*) const { return defaultResult(); }

  virtual T visitSort(const RelSort*) const { return defaultResult(); }

  virtual T visitLogicalValues(const RelLogicalValues*) const { return defaultResult(); }

  virtual T visitModify(const RelModify*) const { return defaultResult(); }

  virtual T visitTableFunction(const RelTableFunction*) const { return defaultResult(); }

  virtual T visitLogicalUnion(const RelLogicalUnion*) const { return defaultResult(); }

 protected:
  virtual T aggregateResult(const T& aggregate, const T& next_result) const {
    return next_result;
  }

  virtual T defaultResult() const { return T{}; }
};

#endif  // QUERYENGINE_RELALGVISITOR_H
