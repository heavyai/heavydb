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

#include "QueryPhysicalInputsCollector.h"

#include "RelAlgDagBuilder.h"
#include "RelAlgVisitor.h"
#include "RexVisitor.h"

namespace {

using PhysicalInputSet = std::unordered_set<PhysicalInput>;

class RelAlgPhysicalInputsVisitor : public RelAlgVisitor<PhysicalInputSet> {
 public:
  PhysicalInputSet visitCompound(const RelCompound* compound) const override;
  PhysicalInputSet visitFilter(const RelFilter* filter) const override;
  PhysicalInputSet visitJoin(const RelJoin* join) const override;
  PhysicalInputSet visitLeftDeepInnerJoin(const RelLeftDeepInnerJoin*) const override;
  PhysicalInputSet visitProject(const RelProject* project) const override;
  PhysicalInputSet visitSort(const RelSort* sort) const override;

 protected:
  PhysicalInputSet aggregateResult(const PhysicalInputSet& aggregate,
                                   const PhysicalInputSet& next_result) const override;
};

class RexPhysicalInputsVisitor : public RexVisitor<PhysicalInputSet> {
 public:
  PhysicalInputSet visitInput(const RexInput* input) const override {
    const auto source_ra = input->getSourceNode();
    const auto scan_ra = dynamic_cast<const RelScan*>(source_ra);
    if (!scan_ra) {
      const auto join_ra = dynamic_cast<const RelJoin*>(source_ra);
      if (join_ra) {
        const auto node_inputs = get_node_output(join_ra);
        CHECK_LT(input->getIndex(), node_inputs.size());
        return visitInput(&node_inputs[input->getIndex()]);
      }
      return PhysicalInputSet{};
    }
    const auto scan_td = scan_ra->getTableDescriptor();
    CHECK(scan_td);
    const int col_id = input->getIndex() + 1;
    const int table_id = scan_td->tableId;
    CHECK_GT(table_id, 0);
    return {{col_id, table_id}};
  }

  PhysicalInputSet visitSubQuery(const RexSubQuery* subquery) const override {
    const auto ra = subquery->getRelAlg();
    CHECK(ra);
    RelAlgPhysicalInputsVisitor visitor;
    return visitor.visit(ra);
  }

  PhysicalInputSet visitOperator(const RexOperator* oper) const override {
    PhysicalInputSet result;
    if (auto window_oper = dynamic_cast<const RexWindowFunctionOperator*>(oper)) {
      for (const auto& partition_key : window_oper->getPartitionKeys()) {
        if (auto input = dynamic_cast<const RexInput*>(partition_key.get())) {
          const auto source_node = input->getSourceNode();
          if (auto filter_node = dynamic_cast<const RelFilter*>(source_node)) {
            // Partitions utilize string dictionary translation in the hash join framework
            // if the partition key is a dictionary encoded string. Ensure we reach the
            // source for all partition keys, so we can access string dictionaries for the
            // partition keys while we build the partition (hash) table
            CHECK_EQ(filter_node->inputCount(), size_t(1));
            const auto parent_node = filter_node->getInput(0);
            const auto node_inputs = get_node_output(parent_node);
            CHECK_LT(input->getIndex(), node_inputs.size());
            result = aggregateResult(result, visitInput(&node_inputs[input->getIndex()]));
          }
          result = aggregateResult(result, visit(input));
        }
      }
    }
    for (size_t i = 0; i < oper->size(); i++) {
      result = aggregateResult(result, visit(oper->getOperand(i)));
    }
    return result;
  }

 protected:
  PhysicalInputSet aggregateResult(const PhysicalInputSet& aggregate,
                                   const PhysicalInputSet& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitCompound(
    const RelCompound* compound) const {
  PhysicalInputSet result;
  for (size_t i = 0; i < compound->getScalarSourcesSize(); ++i) {
    const auto rex = compound->getScalarSource(i);
    CHECK(rex);
    RexPhysicalInputsVisitor visitor;
    const auto rex_phys_inputs = visitor.visit(rex);
    result.insert(rex_phys_inputs.begin(), rex_phys_inputs.end());
  }
  const auto filter = compound->getFilterExpr();
  if (filter) {
    RexPhysicalInputsVisitor visitor;
    const auto filter_phys_inputs = visitor.visit(filter);
    result.insert(filter_phys_inputs.begin(), filter_phys_inputs.end());
  }
  return result;
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitFilter(const RelFilter* filter) const {
  const auto condition = filter->getCondition();
  CHECK(condition);
  RexPhysicalInputsVisitor visitor;
  return visitor.visit(condition);
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitJoin(const RelJoin* join) const {
  const auto condition = join->getCondition();
  if (!condition) {
    return PhysicalInputSet{};
  }
  RexPhysicalInputsVisitor visitor;
  return visitor.visit(condition);
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitLeftDeepInnerJoin(
    const RelLeftDeepInnerJoin* left_deep_inner_join) const {
  PhysicalInputSet result;
  const auto condition = left_deep_inner_join->getInnerCondition();
  RexPhysicalInputsVisitor visitor;
  if (condition) {
    result = visitor.visit(condition);
  }
  CHECK_GE(left_deep_inner_join->inputCount(), size_t(2));
  for (size_t nesting_level = 1; nesting_level <= left_deep_inner_join->inputCount() - 1;
       ++nesting_level) {
    const auto outer_condition = left_deep_inner_join->getOuterCondition(nesting_level);
    if (outer_condition) {
      const auto outer_result = visitor.visit(outer_condition);
      result.insert(outer_result.begin(), outer_result.end());
    }
  }
  return result;
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitProject(
    const RelProject* project) const {
  PhysicalInputSet result;
  for (size_t i = 0; i < project->size(); ++i) {
    const auto rex = project->getProjectAt(i);
    CHECK(rex);
    RexPhysicalInputsVisitor visitor;
    const auto rex_phys_inputs = visitor.visit(rex);
    result.insert(rex_phys_inputs.begin(), rex_phys_inputs.end());
  }
  return result;
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitSort(const RelSort* sort) const {
  CHECK_EQ(sort->inputCount(), size_t(1));
  return visit(sort->getInput(0));
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::aggregateResult(
    const PhysicalInputSet& aggregate,
    const PhysicalInputSet& next_result) const {
  auto result = aggregate;
  result.insert(next_result.begin(), next_result.end());
  return result;
}

class RelAlgPhysicalTableInputsVisitor : public RelAlgVisitor<std::unordered_set<int>> {
 public:
  std::unordered_set<int> visitScan(const RelScan* scan) const override {
    return {scan->getTableDescriptor()->tableId};
  }

 protected:
  std::unordered_set<int> aggregateResult(
      const std::unordered_set<int>& aggregate,
      const std::unordered_set<int>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

}  // namespace

std::unordered_set<PhysicalInput> get_physical_inputs(const RelAlgNode* ra) {
  RelAlgPhysicalInputsVisitor phys_inputs_visitor;
  return phys_inputs_visitor.visit(ra);
}

std::unordered_set<int> get_physical_table_inputs(const RelAlgNode* ra) {
  RelAlgPhysicalTableInputsVisitor phys_table_inputs_visitor;
  return phys_table_inputs_visitor.visit(ra);
}

std::ostream& operator<<(std::ostream& os, PhysicalInput const& physical_input) {
  return os << '(' << physical_input.col_id << ',' << physical_input.table_id << ')';
}
