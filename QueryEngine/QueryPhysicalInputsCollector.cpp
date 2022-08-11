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

#include "ExprDagVisitor.h"
#include "RelAlgDagBuilder.h"
#include "RelAlgVisitor.h"
#include "RexVisitor.h"
#include "Visitors/RelRexDagVisitor.h"

#include "SchemaMgr/ColumnInfo.h"

namespace {

using InputColDescriptorSet = std::unordered_set<InputColDescriptor>;

template <typename RexVisitor, typename ResultType>
class RelAlgPhysicalInputsVisitor : public RelAlgVisitor<ResultType> {
 public:
  RelAlgPhysicalInputsVisitor() {}

  ResultType visitCompound(const RelCompound* compound) const override {
    ResultType result;
    for (size_t i = 0; i < compound->getScalarSourcesSize(); ++i) {
      const auto rex = compound->getScalarSource(i);
      CHECK(rex);
      RexVisitor visitor;
      const auto rex_phys_inputs = visitor.visit(rex);
      result.insert(rex_phys_inputs.begin(), rex_phys_inputs.end());
    }
    const auto filter = compound->getFilterExpr();
    if (filter) {
      RexVisitor visitor;
      const auto filter_phys_inputs = visitor.visit(filter);
      result.insert(filter_phys_inputs.begin(), filter_phys_inputs.end());
    }
    return result;
  }

  ResultType visitFilter(const RelFilter* filter) const override {
    const auto condition = filter->getCondition();
    CHECK(condition);
    RexVisitor visitor;
    return visitor.visit(condition);
  }

  ResultType visitJoin(const RelJoin* join) const override {
    const auto condition = join->getCondition();
    if (!condition) {
      return ResultType{};
    }
    RexVisitor visitor;
    return visitor.visit(condition);
  }

  ResultType visitLeftDeepInnerJoin(
      const RelLeftDeepInnerJoin* left_deep_inner_join) const override {
    ResultType result;
    const auto condition = left_deep_inner_join->getInnerCondition();
    RexVisitor visitor;
    if (condition) {
      result = visitor.visit(condition);
    }
    CHECK_GE(left_deep_inner_join->inputCount(), size_t(2));
    for (size_t nesting_level = 1;
         nesting_level <= left_deep_inner_join->inputCount() - 1;
         ++nesting_level) {
      const auto outer_condition = left_deep_inner_join->getOuterCondition(nesting_level);
      if (outer_condition) {
        const auto outer_result = visitor.visit(outer_condition);
        result.insert(outer_result.begin(), outer_result.end());
      }
    }
    return result;
  }

  ResultType visitProject(const RelProject* project) const override {
    ResultType result;
    for (size_t i = 0; i < project->size(); ++i) {
      const auto rex = project->getProjectAt(i);
      CHECK(rex);
      RexVisitor visitor;
      const auto rex_phys_inputs = visitor.visit(rex);
      result.insert(rex_phys_inputs.begin(), rex_phys_inputs.end());
    }
    return result;
  }

  ResultType visitSort(const RelSort* sort) const override {
    CHECK_EQ(sort->inputCount(), size_t(1));
    return this->visit(sort->getInput(0));
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

template <typename ExprVisitor, typename ResultType>
class PhysicalInputsNodeVisitor : public RelAlgVisitor<ResultType> {
 public:
  PhysicalInputsNodeVisitor() {}

  using RelAlgVisitor<ResultType>::visit;

  ResultType visitCompound(const RelCompound* compound) const override {
    ResultType result;
    ExprVisitor visitor;
    for (auto& expr : compound->getGroupByExprs()) {
      auto inputs = visitor.visit(expr.get());
      result.insert(inputs.begin(), inputs.end());
    }
    for (auto& expr : compound->getExprs()) {
      auto inputs = visitor.visit(expr.get());
      result.insert(inputs.begin(), inputs.end());
    }
    auto filter = compound->getFilter();
    if (filter) {
      auto inputs = visitor.visit(filter.get());
      result.insert(inputs.begin(), inputs.end());
    }
    return result;
  }

  ResultType visitFilter(const RelFilter* filter) const override {
    ExprVisitor visitor;
    return visitor.visit(filter->getConditionExpr());
  }

  ResultType visitJoin(const RelJoin* join) const override {
    auto condition = join->getConditionExpr();
    if (!condition) {
      return ResultType{};
    }
    ExprVisitor visitor;
    return visitor.visit(condition);
  }

  ResultType visitLeftDeepInnerJoin(
      const RelLeftDeepInnerJoin* left_deep_inner_join) const override {
    ResultType result;
    auto condition = left_deep_inner_join->getInnerConditionExpr();
    ExprVisitor visitor;
    if (condition) {
      result = visitor.visit(condition);
    }
    for (size_t nesting_level = 1; nesting_level < left_deep_inner_join->inputCount();
         ++nesting_level) {
      auto outer_condition = left_deep_inner_join->getOuterConditionExpr(nesting_level);
      if (outer_condition) {
        auto outer_result = visitor.visit(outer_condition);
        result.insert(outer_result.begin(), outer_result.end());
      }
    }
    return result;
  }

  ResultType visitProject(const RelProject* project) const override {
    ResultType result;
    ExprVisitor visitor;
    for (auto& expr : project->getExprs()) {
      const auto inputs = visitor.visit(expr.get());
      result.insert(inputs.begin(), inputs.end());
    }
    return result;
  }

  ResultType visitSort(const RelSort* sort) const override {
    return this->visit(sort->getInput(0));
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

template <typename Derived, typename ResultType>
class RexInputVisitorBase : public RexVisitor<ResultType> {
 public:
  RexInputVisitorBase() {}

  ResultType visitSubQuery(const RexSubQuery* subquery) const override {
    const auto ra = subquery->getRelAlg();
    CHECK(ra);
    RelAlgPhysicalInputsVisitor<Derived, ResultType> visitor;
    return visitor.visit(ra);
  }

  ResultType visitOperator(const RexOperator* oper) const override {
    ResultType result;
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
            result = aggregateResult(result,
                                     this->visitInput(&node_inputs[input->getIndex()]));
          }
          result = aggregateResult(result, this->visit(input));
        }
      }
    }
    for (size_t i = 0; i < oper->size(); i++) {
      result = aggregateResult(result, this->visit(oper->getOperand(i)));
    }
    return result;
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

template <typename Derived, typename ResultType>
class InputVisitorBase : public ScalarExprVisitor<ResultType> {
 public:
  InputVisitorBase() {}

  ResultType visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override {
    PhysicalInputsNodeVisitor<Derived, ResultType> visitor;
    return visitor.visit(subquery->getNode());
  }

  ResultType visitInSubquery(const hdk::ir::InSubquery* in_subquery) const override {
    PhysicalInputsNodeVisitor<Derived, ResultType> visitor;
    auto node_res = visitor.visit(in_subquery->getNode());
    auto arg_res = ScalarExprVisitor<ResultType>::visit(in_subquery->getArg().get());
    return aggregateResult(node_res, arg_res);
  }

  ResultType visitWindowFunction(
      const hdk::ir::WindowFunction* window_func) const override {
    ResultType result;
    for (auto& part_key : window_func->getPartitionKeys()) {
      if (auto col_ref = dynamic_cast<const hdk::ir::ColumnRef*>(part_key.get())) {
        if (auto filter = dynamic_cast<const RelFilter*>(col_ref->getNode())) {
          // Partitions utilize string dictionary translation in the hash join framework
          // if the partition key is a dictionary encoded string. Ensure we reach the
          // source for all partition keys, so we can access string dictionaries for the
          // partition keys while we build the partition (hash) table
          auto parent_node = filter->getInput(0);
          auto input = getNodeColumnRef(parent_node, col_ref->getIndex());
          result = aggregateResult(result, this->visit(input.get()));
        }
        result = aggregateResult(result, this->visitColumnRef(col_ref));
      }
    }
    for (const auto& arg : window_func->getArgs()) {
      result = aggregateResult(result, this->visit(arg.get()));
    }
    // TODO: order keys are ignored here to get the same result as for Rex.
    // Should it be fixed to collect proper inputs?
    return result;
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

class RexPhysicalInputsVisitor
    : public RexInputVisitorBase<RexPhysicalInputsVisitor, InputColDescriptorSet> {
 public:
  RexPhysicalInputsVisitor() {}

  InputColDescriptorSet visitInput(const RexInput* input) const override {
    const auto source_ra = input->getSourceNode();
    const auto scan_ra = dynamic_cast<const RelScan*>(source_ra);
    if (!scan_ra) {
      const auto join_ra = dynamic_cast<const RelJoin*>(source_ra);
      if (join_ra) {
        const auto node_inputs = get_node_output(join_ra);
        CHECK_LT(input->getIndex(), node_inputs.size());
        return visitInput(&node_inputs[input->getIndex()]);
      }
      return InputColDescriptorSet{};
    }

    auto col_info = scan_ra->getColumnInfoBySpi(input->getIndex() + 1);
    CHECK_GT(col_info->table_id, 0);
    return {{col_info, 0}};
  }
};

class PhysicalInputsVisitor
    : public InputVisitorBase<PhysicalInputsVisitor, InputColDescriptorSet> {
 public:
  PhysicalInputsVisitor() {}

  InputColDescriptorSet visitColumnRef(const hdk::ir::ColumnRef* col_ref) const override {
    const auto source = col_ref->getNode();
    const auto scan = dynamic_cast<const RelScan*>(source);
    if (!scan) {
      const auto join = dynamic_cast<const RelJoin*>(source);
      if (join) {
        auto input = getNodeColumnRef(join, col_ref->getIndex());
        return visit(input.get());
      }
      return InputColDescriptorSet{};
    }

    auto col_info = scan->getColumnInfoBySpi(col_ref->getIndex() + 1);
    CHECK_GT(col_info->table_id, 0);
    return {{col_info, 0}};
  }
};

template <typename RelAlgVisitor, typename ResultType>
class RexSubqueryVisitorBase : public RexVisitor<ResultType> {
 public:
  RexSubqueryVisitorBase() {}

  ResultType visitSubQuery(const RexSubQuery* subquery) const override {
    const auto ra = subquery->getRelAlg();
    CHECK(ra);
    RelAlgVisitor visitor;
    return visitor.visit(ra);
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

template <typename RelAlgVisitor, typename ResultType>
class SubqueryVisitorBase : public ScalarExprVisitor<ResultType> {
 public:
  SubqueryVisitorBase() {}

  ResultType visitScalarSubquery(const hdk::ir::ScalarSubquery* subquery) const override {
    RelAlgVisitor visitor;
    return visitor.visit(subquery->getNode());
  }

  ResultType visitInSubquery(const hdk::ir::InSubquery* in_subquery) const override {
    RelAlgVisitor visitor;
    return visitor.visit(in_subquery->getNode());
  }

 protected:
  ResultType aggregateResult(const ResultType& aggregate,
                             const ResultType& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

class RelAlgPhysicalColumnInfosVisitor
    : public RelAlgPhysicalInputsVisitor<
          RexSubqueryVisitorBase<RelAlgPhysicalColumnInfosVisitor, ColumnInfoMap>,
          ColumnInfoMap> {
 public:
  RelAlgPhysicalColumnInfosVisitor() {}

  ColumnInfoMap visitScan(const RelScan* scan) const override {
    ColumnInfoMap res;

    for (size_t col_idx = 0; col_idx < scan->size(); ++col_idx) {
      auto col_info = scan->getColumnInfoBySpi(col_idx + 1);
      res.insert({*col_info, col_info});
    }

    return res;
  }
};

class PhysicalColumnInfosNodeVisitor
    : public PhysicalInputsNodeVisitor<
          SubqueryVisitorBase<PhysicalColumnInfosNodeVisitor, ColumnInfoMap>,
          ColumnInfoMap> {
 public:
  PhysicalColumnInfosNodeVisitor() {}

  ColumnInfoMap visitScan(const RelScan* scan) const override {
    ColumnInfoMap res;

    for (size_t col_idx = 0; col_idx < scan->size(); ++col_idx) {
      auto col_info = scan->getColumnInfoBySpi(col_idx + 1);
      res.insert({*col_info, col_info});
    }

    return res;
  }
};

class RelAlgPhysicalTableInputsVisitor : public RelRexDagVisitor {
 public:
  RelAlgPhysicalTableInputsVisitor() {}

  using RelRexDagVisitor::visit;
  using TableIds = std::unordered_set<std::pair<int, int>>;

  static TableIds getTableIds(RelAlgNode const* node) {
    RelAlgPhysicalTableInputsVisitor visitor;
    visitor.visit(node);
    return std::move(visitor.table_ids_);
  }

 private:
  TableIds table_ids_;

  void visit(RelScan const* scan) override {
    table_ids_.insert({scan->getDatabaseId(), scan->getTableId()});
  }
};

class PhysicalTableInputsNodeVisitor : public ExprDagVisitor {
 public:
  PhysicalTableInputsNodeVisitor() {}

  using ExprDagVisitor::visit;
  using TableIds = std::unordered_set<std::pair<int, int>>;

  static TableIds getTableIds(RelAlgNode const* node) {
    PhysicalTableInputsNodeVisitor visitor;
    visitor.visit(node);
    return std::move(visitor.table_ids_);
  }

 private:
  TableIds table_ids_;

  void visitScan(const RelScan* scan) override {
    table_ids_.insert({scan->getDatabaseId(), scan->getTableId()});
  }
};

class RelAlgPhysicalTableInfosVisitor
    : public RelAlgPhysicalInputsVisitor<
          RexSubqueryVisitorBase<RelAlgPhysicalTableInfosVisitor, TableInfoMap>,
          TableInfoMap> {
 public:
  RelAlgPhysicalTableInfosVisitor() {}

  TableInfoMap visitScan(const RelScan* scan) const override {
    TableInfoMap res;
    auto info = scan->getTableInfo();
    res.insert(std::make_pair(TableRef(info->db_id, info->table_id), info));
    return res;
  }
};

class PhysicalTableInfosNodeVisitor
    : public PhysicalInputsNodeVisitor<
          SubqueryVisitorBase<PhysicalTableInfosNodeVisitor, TableInfoMap>,
          TableInfoMap> {
 public:
  PhysicalTableInfosNodeVisitor() {}

  TableInfoMap visitScan(const RelScan* scan) const override {
    TableInfoMap res;
    auto info = scan->getTableInfo();
    res.insert(std::make_pair(TableRef(info->db_id, info->table_id), info));
    return res;
  }
};

}  // namespace

std::unordered_set<InputColDescriptor> get_physical_inputs(const RelAlgNode* ra) {
  RelAlgPhysicalInputsVisitor<RexPhysicalInputsVisitor, InputColDescriptorSet>
      orig_phys_inputs_visitor;
  auto orig_res = orig_phys_inputs_visitor.visit(ra);

  PhysicalInputsNodeVisitor<PhysicalInputsVisitor, InputColDescriptorSet>
      phys_inputs_visitor;
  auto res = phys_inputs_visitor.visit(ra);

  CHECK_EQ(res.size(), orig_res.size()) << "Result mismatch:" << std::endl
                                        << "orig=" << ::toString(orig_res) << std::endl
                                        << "new=" << ::toString(res);
  for (auto& input : orig_res) {
    CHECK(res.count(input)) << "Result mismatch:" << std::endl
                            << "orig=" << ::toString(orig_res) << std::endl
                            << "new=" << ::toString(res);
  }

  return res;
}

std::unordered_set<std::pair<int, int>> get_physical_table_inputs(const RelAlgNode* ra) {
  auto orig_res = RelAlgPhysicalTableInputsVisitor::getTableIds(ra);
  auto res = PhysicalTableInputsNodeVisitor::getTableIds(ra);

  CHECK_EQ(res.size(), orig_res.size()) << "Result mismatch:" << std::endl
                                        << "orig=" << ::toString(orig_res) << std::endl
                                        << "new=" << ::toString(res);
  for (auto& input : orig_res) {
    CHECK(res.count(input)) << "Result mismatch:" << std::endl
                            << "orig=" << ::toString(orig_res) << std::endl
                            << "new=" << ::toString(res);
  }

  return res;
}

ColumnInfoMap get_physical_column_infos(const RelAlgNode* ra) {
  RelAlgPhysicalColumnInfosVisitor orig_visitor;
  auto orig_res = orig_visitor.visit(ra);

  PhysicalColumnInfosNodeVisitor visitor;
  auto res = visitor.visit(ra);

  CHECK_EQ(res.size(), orig_res.size()) << "Result mismatch:" << std::endl
                                        << "orig=" << ::toString(orig_res) << std::endl
                                        << "new=" << ::toString(res);
  for (auto& pr : orig_res) {
    CHECK(res.count(pr.first)) << "Result mismatch:" << std::endl
                               << "orig=" << ::toString(orig_res) << std::endl
                               << "new=" << ::toString(res);
  }

  return res;
}

TableInfoMap get_physical_table_infos(const RelAlgNode* ra) {
  RelAlgPhysicalTableInfosVisitor orig_visitor;
  auto orig_res = orig_visitor.visit(ra);

  PhysicalTableInfosNodeVisitor visitor;
  auto res = visitor.visit(ra);

  CHECK_EQ(res.size(), orig_res.size()) << "Result mismatch:" << std::endl
                                        << "orig=" << ::toString(orig_res) << std::endl
                                        << "new=" << ::toString(res);
  for (auto& pr : orig_res) {
    CHECK(res.count(pr.first)) << "Result mismatch:" << std::endl
                               << "orig=" << ::toString(orig_res) << std::endl
                               << "new=" << ::toString(res);
  }

  return res;
}

std::ostream& operator<<(std::ostream& os, PhysicalInput const& physical_input) {
  return os << '(' << physical_input.col_id << ',' << physical_input.table_id << ')';
}
