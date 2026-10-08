/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RelLeftDeepInnerJoin.h"
#include "Logger/Logger.h"
#include "RelAlgDag.h"
#include "RexVisitor.h"

#include <unordered_map>
#include <unordered_set>

// todo (yoonmin): Can we remove this artificial left-deep join tree node from our query
// plan? because it causes too much complexity when dealing w/ various edge cases of query
// plans
RelLeftDeepInnerJoin::RelLeftDeepInnerJoin(
    const std::shared_ptr<RelFilter>& filter,
    std::vector<std::shared_ptr<const RelAlgNode>> inputs,
    std::vector<std::shared_ptr<const RelJoin>>& original_joins)
    : original_filter_(filter), original_joins_(original_joins) {
  std::vector<std::unique_ptr<const RexScalar>> operands;
  bool is_notnull = true;
  RexDeepCopyVisitor condition_copier;
  if (filter && filter->getCondition()) {
    condition_ = condition_copier.visit(filter->getCondition());
  }
  // Accumulate join conditions from the (explicit) joins themselves and
  // from the filter node at the root of the left-deep tree pattern.
  outer_conditions_per_level_.resize(original_joins.size());
  for (size_t nesting_level = 0; nesting_level < original_joins.size(); ++nesting_level) {
    const auto& original_join = original_joins[nesting_level];
    const auto condition_true =
        dynamic_cast<const RexLiteral*>(original_join->getCondition());
    if (!condition_true || !condition_true->getVal<bool>()) {
      if (dynamic_cast<const RexOperator*>(original_join->getCondition())) {
        is_notnull =
            is_notnull && dynamic_cast<const RexOperator*>(original_join->getCondition())
                              ->getType()
                              .get_notnull();
      }
      switch (original_join->getJoinType()) {
        case JoinType::INNER:
        case JoinType::SEMI:
        case JoinType::ANTI: {
          if (original_join->getCondition()) {
            operands.emplace_back(condition_copier.visit(original_join->getCondition()));
          }
          break;
        }
        case JoinType::LEFT: {
          if (original_join->getCondition()) {
            outer_conditions_per_level_[nesting_level] =
                condition_copier.visit(original_join->getCondition());
          }
          break;
        }
        default:
          CHECK(false);
      }
    }
  }
  if (!operands.empty()) {
    if (condition_) {
      CHECK(dynamic_cast<const RexOperator*>(condition_.get()));
      is_notnull =
          is_notnull &&
          static_cast<const RexOperator*>(condition_.get())->getType().get_notnull();
      operands.emplace_back(std::move(condition_));
    }
    if (operands.size() > 1) {
      condition_.reset(
          new RexOperator(kAND, operands, SQLTypeInfo(kBOOLEAN, is_notnull)));
    } else {
      condition_ = std::move(operands.front());
    }
  }
  if (!condition_) {
    condition_.reset(new RexLiteral(true, kBOOLEAN, kBOOLEAN, 0, 0, 0, 0));
  }
  for (const auto& input : inputs) {
    addManagedInput(input);
  }
}

RelLeftDeepInnerJoin::RelLeftDeepInnerJoin(
    RelAlgInputs inputs,
    std::vector<std::shared_ptr<const RelJoin>> original_joins,
    std::unique_ptr<const RexScalar> condition,
    std::vector<std::unique_ptr<const RexScalar>> outer_conditions)
    : condition_(std::move(condition))
    , outer_conditions_per_level_(std::move(outer_conditions))
    , original_joins_(std::move(original_joins)) {
  CHECK_EQ(outer_conditions_per_level_.size(), original_joins_.size());
  if (!condition_) {
    condition_.reset(new RexLiteral(true, kBOOLEAN, kBOOLEAN, 0, 0, 0, 0));
  }
  for (const auto& input : inputs) {
    addManagedInput(input);
  }
}

const RexScalar* RelLeftDeepInnerJoin::getInnerCondition() const {
  return condition_.get();
}

const RexScalar* RelLeftDeepInnerJoin::getOuterCondition(
    const size_t nesting_level) const {
  CHECK_GE(nesting_level, size_t(1));
  CHECK_LE(nesting_level, outer_conditions_per_level_.size());
  // Outer join conditions are collected depth-first while the returned condition
  // must be consistent with the order of the loops (which is reverse depth-first).
  return outer_conditions_per_level_[outer_conditions_per_level_.size() - nesting_level]
      .get();
}

const JoinType RelLeftDeepInnerJoin::getJoinType(const size_t nesting_level) const {
  CHECK_LE(nesting_level, original_joins_.size());
  return original_joins_[original_joins_.size() - nesting_level]->getJoinType();
}

std::string RelLeftDeepInnerJoin::toString(RelRexToStringConfig config) const {
  if (!config.attributes_only) {
    std::string ret = ::typeName(this) + "(";
    ret += condition_->toString(config);
    if (!config.skip_input_nodes) {
      for (const auto& input : inputs_) {
        ret += " " + input->toString(config);
      }
    } else {
      ret += ", input node id={";
      for (auto& input : inputs_) {
        ret += std::to_string(input->getId()) + " ";
      }
      ret += "}";
    }
    ret += ")";
    return ret;
  } else {
    return ::typeName(this) + "()";
  }
}

size_t RelLeftDeepInnerJoin::size() const {
  CHECK(!inputs_.empty());
  size_t total_size = inputs_[0]->size();
  for (size_t nesting_level = 1; nesting_level <= original_joins_.size();
       ++nesting_level) {
    switch (getJoinType(nesting_level)) {
      case JoinType::SEMI:
      case JoinType::ANTI:
        break;
      default:
        CHECK_LT(nesting_level, inputs_.size());
        total_size += inputs_[nesting_level]->size();
        break;
    }
  }
  return total_size;
}

size_t RelLeftDeepInnerJoin::getOuterConditionsSize() const {
  return outer_conditions_per_level_.size();
}

std::shared_ptr<RelAlgNode> RelLeftDeepInnerJoin::deepCopy() const {
  CHECK(false);
  return nullptr;
}

bool RelLeftDeepInnerJoin::coversOriginalNode(const RelAlgNode* node) const {
  if (node == original_filter_.get()) {
    return true;
  }
  for (const auto& original_join : original_joins_) {
    if (original_join.get() == node) {
      return true;
    }
  }
  return false;
}

const RelFilter* RelLeftDeepInnerJoin::getOriginalFilter() const {
  return original_filter_.get();
}

std::vector<std::shared_ptr<const RelJoin>> RelLeftDeepInnerJoin::getOriginalJoins()
    const {
  std::vector<std::shared_ptr<const RelJoin>> original_joins;
  original_joins.assign(original_joins_.begin(), original_joins_.end());
  return original_joins;
}

namespace {

void collect_left_deep_join_inputs(
    std::deque<std::shared_ptr<const RelAlgNode>>& inputs,
    std::vector<std::shared_ptr<const RelJoin>>& original_joins,
    const std::shared_ptr<const RelJoin>& join) {
  original_joins.push_back(join);
  CHECK_EQ(size_t(2), join->inputCount());
  const auto left_input_join =
      std::dynamic_pointer_cast<const RelJoin>(join->getAndOwnInput(0));
  if (left_input_join) {
    inputs.push_front(join->getAndOwnInput(1));
    collect_left_deep_join_inputs(inputs, original_joins, left_input_join);
  } else {
    inputs.push_front(join->getAndOwnInput(1));
    inputs.push_front(join->getAndOwnInput(0));
  }
}

std::pair<std::shared_ptr<RelLeftDeepInnerJoin>, std::shared_ptr<const RelAlgNode>>
create_left_deep_join(const std::shared_ptr<RelAlgNode>& left_deep_join_root) {
  const auto old_root = get_left_deep_join_root(left_deep_join_root);
  if (!old_root) {
    return {nullptr, nullptr};
  }
  std::deque<std::shared_ptr<const RelAlgNode>> inputs_deque;
  const auto left_deep_join_filter =
      std::dynamic_pointer_cast<RelFilter>(left_deep_join_root);
  const auto join =
      std::dynamic_pointer_cast<const RelJoin>(left_deep_join_root->getAndOwnInput(0));
  CHECK(join);
  std::vector<std::shared_ptr<const RelJoin>> original_joins;
  collect_left_deep_join_inputs(inputs_deque, original_joins, join);
  std::vector<std::shared_ptr<const RelAlgNode>> inputs(inputs_deque.begin(),
                                                        inputs_deque.end());
  return {std::make_shared<RelLeftDeepInnerJoin>(
              left_deep_join_filter, inputs, original_joins),
          old_root};
}

std::optional<RegisteredQueryHint> extract_query_hint(
    std::unordered_map<const RelAlgNode*,
                       std::unordered_map<unsigned, RegisteredQueryHint>>& query_hints,
    const RelAlgNode* node) {
  auto hint_map_it = query_hints.find(node);
  if (hint_map_it == query_hints.end()) {
    return std::nullopt;
  }
  auto& node_hints = hint_map_it->second;
  auto hint_it = node_hints.find(node->getId());
  if (hint_it == node_hints.end()) {
    return std::nullopt;
  }
  const auto hint = hint_it->second;
  if (node_hints.size() == 1) {
    query_hints.erase(hint_map_it);
  } else {
    node_hints.erase(hint_it);
  }
  return hint;
}

void propagate_query_hints_to_left_deep_join(
    std::unordered_map<const RelAlgNode*,
                       std::unordered_map<unsigned, RegisteredQueryHint>>& query_hints,
    const std::shared_ptr<RelLeftDeepInnerJoin>& left_deep_join,
    const std::shared_ptr<RelAlgNode>& left_deep_join_candidate) {
  RegisteredQueryHint combined_hint;
  bool has_hint = false;

  auto merge_hint = [&combined_hint, &has_hint](const RegisteredQueryHint& hint) {
    combined_hint = has_hint ? (combined_hint || hint) : hint;
    has_hint = true;
  };

  if (auto hint = extract_query_hint(query_hints, left_deep_join_candidate.get())) {
    merge_hint(*hint);
  }
  for (const auto& original_join : left_deep_join->getOriginalJoins()) {
    if (auto hint = extract_query_hint(query_hints, original_join.get())) {
      merge_hint(*hint);
    }
  }

  if (!has_hint) {
    return;
  }
  std::unordered_map<unsigned, RegisteredQueryHint> hint_map;
  hint_map.emplace(left_deep_join->getId(), combined_hint);
  query_hints.emplace(left_deep_join.get(), std::move(hint_map));
}

void transfer_left_deep_join_query_hint(
    std::unordered_map<const RelAlgNode*,
                       std::unordered_map<unsigned, RegisteredQueryHint>>& query_hints,
    const RelLeftDeepInnerJoin* old_join,
    const std::shared_ptr<RelLeftDeepInnerJoin>& new_join) {
  if (!old_join || !new_join) {
    return;
  }
  auto old_hint_map_it = query_hints.find(old_join);
  if (old_hint_map_it == query_hints.end()) {
    return;
  }
  auto& old_node_hints = old_hint_map_it->second;
  auto old_hint_it = old_node_hints.find(old_join->getId());
  if (old_hint_it == old_node_hints.end()) {
    return;
  }

  const auto query_hint = old_hint_it->second;
  if (old_node_hints.size() == 1) {
    query_hints.erase(old_hint_map_it);
  } else {
    old_node_hints.erase(old_hint_it);
  }

  auto& new_node_hints = query_hints[new_join.get()];
  new_node_hints.emplace(new_join->getId(), query_hint);
}

class RebindRexInputsFromLeftDeepJoin : public RexVisitor<void*> {
 public:
  RebindRexInputsFromLeftDeepJoin(const RelLeftDeepInnerJoin* left_deep_join)
      : left_deep_join_(left_deep_join) {
    CHECK_GT(left_deep_join->inputCount(), size_t(1));
    for (size_t input_idx = 0; input_idx < left_deep_join->inputCount(); ++input_idx) {
      addInput(input_idx, all_input_size_prefix_sums_, all_input_indices_);
    }
    addOutputInput(0);
    for (size_t nesting_level = 1; nesting_level <= left_deep_join->inputCount() - 1;
         ++nesting_level) {
      switch (left_deep_join->getJoinType(nesting_level)) {
        case JoinType::SEMI:
        case JoinType::ANTI:
          break;
        default:
          addOutputInput(nesting_level);
          break;
      }
    }
  }

  void* visitInput(const RexInput* rex_input) const override {
    const auto source_node = rex_input->getSourceNode();
    if (left_deep_join_->coversOriginalNode(source_node)) {
      if (!rebindInput(rex_input, input_size_prefix_sums_, output_input_indices_)) {
        CHECK(rebindInput(rex_input, all_input_size_prefix_sums_, all_input_indices_));
      }
    }
    return nullptr;
  };

 private:
  void addOutputInput(const size_t input_idx) {
    addInput(input_idx, input_size_prefix_sums_, output_input_indices_);
  }

  void addInput(const size_t input_idx,
                std::vector<size_t>& input_size_prefix_sums,
                std::vector<size_t>& input_indices) {
    CHECK_LT(input_idx, left_deep_join_->inputCount());
    input_indices.push_back(input_idx);
    const auto previous_size =
        input_size_prefix_sums.empty() ? size_t(0) : input_size_prefix_sums.back();
    input_size_prefix_sums.push_back(previous_size +
                                     left_deep_join_->getInput(input_idx)->size());
  }

  bool rebindInput(const RexInput* rex_input,
                   const std::vector<size_t>& input_size_prefix_sums,
                   const std::vector<size_t>& input_indices) const {
    const auto it = std::upper_bound(input_size_prefix_sums.begin(),
                                     input_size_prefix_sums.end(),
                                     rex_input->getIndex());
    if (it == input_size_prefix_sums.end()) {
      return false;
    }
    const auto output_input_idx = std::distance(input_size_prefix_sums.begin(), it);
    CHECK_LT(static_cast<size_t>(output_input_idx), input_indices.size());
    const auto input_node = left_deep_join_->getInput(input_indices[output_input_idx]);
    if (it != input_size_prefix_sums.begin()) {
      const auto prev_input_count = *(it - 1);
      CHECK_LE(prev_input_count, rex_input->getIndex());
      const auto input_index = rex_input->getIndex() - prev_input_count;
      rex_input->setIndex(input_index);
    }
    rex_input->setSourceNode(input_node);
    return true;
  }

  std::vector<size_t> input_size_prefix_sums_;
  std::vector<size_t> output_input_indices_;
  std::vector<size_t> all_input_size_prefix_sums_;
  std::vector<size_t> all_input_indices_;
  const RelLeftDeepInnerJoin* left_deep_join_;
};

class RebindRexInputsThroughSimpleProject : public RexVisitor<void*> {
 public:
  explicit RebindRexInputsThroughSimpleProject(const RelProject* project)
      : project_(project) {
    CHECK(project_);
    CHECK(project_->isSimple());
  }

  void* visitInput(const RexInput* rex_input) const override {
    if (rex_input->getSourceNode() != project_) {
      return nullptr;
    }
    const auto project_expr = project_->getProjectAt(rex_input->getIndex());
    const auto project_input = dynamic_cast<const RexInput*>(project_expr);
    CHECK(project_input);
    rex_input->setSourceNode(project_input->getSourceNode());
    rex_input->setIndex(project_input->getIndex());
    return nullptr;
  }

 private:
  const RelProject* project_;
};

class CopyAndRebindRexInputsThroughCompound : public RexDeepCopyVisitor {
 public:
  explicit CopyAndRebindRexInputsThroughCompound(const RelCompound* compound)
      : compound_(compound) {
    CHECK(compound_);
    CHECK(!compound_->isAggregate());
  }

  RetType visitInput(const RexInput* rex_input) const override {
    if (rex_input->getSourceNode() != compound_) {
      return rex_input->deepCopy();
    }
    CHECK_LT(static_cast<size_t>(rex_input->getIndex()),
             compound_->getScalarSourcesSize());
    return visit(compound_->getScalarSource(rex_input->getIndex()));
  }

 private:
  const RelCompound* compound_;
};

void rebind_inputs_through_simple_project(const RexScalar* rex,
                                          const RelProject* project) {
  if (!rex || !project) {
    return;
  }
  RebindRexInputsThroughSimpleProject rebind(project);
  rebind.visit(rex);
}

std::unique_ptr<const RexScalar> copy_and_rebind_inputs_through_compound(
    const RexScalar* rex,
    const RelCompound* compound) {
  if (!rex) {
    return nullptr;
  }
  CopyAndRebindRexInputsThroughCompound rebind(compound);
  return rebind.visit(rex);
}

bool is_inner_only_left_deep_join(const RelLeftDeepInnerJoin& join) {
  for (size_t nesting_level = 1; nesting_level <= join.inputCount() - 1;
       ++nesting_level) {
    if (join.getJoinType(nesting_level) != JoinType::INNER ||
        join.getOuterCondition(nesting_level)) {
      return false;
    }
  }
  return true;
}

size_t reachable_direct_input_use_count(const RelAlgNode* root,
                                        const RelAlgNode* needle) {
  if (!root || !needle) {
    return 0;
  }

  size_t count{0};
  std::unordered_set<const RelAlgNode*> visited;
  std::vector<const RelAlgNode*> stack{root};
  while (!stack.empty()) {
    const auto node = stack.back();
    stack.pop_back();
    if (!node || !visited.emplace(node).second) {
      continue;
    }
    for (size_t input_idx = 0; input_idx < node->inputCount(); ++input_idx) {
      const auto input = node->getInput(input_idx);
      if (input == needle) {
        ++count;
      }
      stack.push_back(input);
    }
  }
  return count;
}

bool is_inlineable_scalar_compound(const RelCompound* compound) {
  if (!compound || compound->isAggregate() || compound->inputCount() != size_t(1) ||
      compound->isUpdateViaSelect() || compound->isDeleteViaSelect()) {
    return false;
  }
  for (size_t scalar_idx = 0; scalar_idx < compound->getScalarSourcesSize();
       ++scalar_idx) {
    if (!compound->getScalarSource(scalar_idx) ||
        dynamic_cast<const RexSubQuery*>(compound->getScalarSource(scalar_idx))) {
      return false;
    }
  }
  return true;
}

class RexLiteralDetector : public RexVisitor<bool> {
 public:
  bool visitLiteral(const RexLiteral*) const override { return true; }

 protected:
  bool aggregateResult(const bool& aggregate, const bool& next_result) const override {
    return aggregate || next_result;
  }
};

bool contains_literal(const RexScalar* rex) {
  if (!rex) {
    return false;
  }
  RexLiteralDetector detector;
  return detector.visit(rex);
}

bool is_fixed_width_filter_literal(const RexLiteral* literal) {
  if (!literal) {
    return false;
  }
  const auto type = literal->getType();
  return type == kBOOLEAN || type == kTINYINT || type == kSMALLINT || type == kINT ||
         type == kBIGINT || type == kNUMERIC || type == kDECIMAL ||
         type == kINTERVAL_DAY_TIME || type == kINTERVAL_YEAR_MONTH || is_datetime(type);
}

bool is_fixed_width_input_literal_comparison(const RexOperator* oper) {
  if (!oper || !IS_COMPARISON(oper->getOperator()) || oper->size() != size_t(2)) {
    return false;
  }
  const auto lhs_input = dynamic_cast<const RexInput*>(oper->getOperand(0));
  const auto rhs_input = dynamic_cast<const RexInput*>(oper->getOperand(1));
  const auto lhs_literal = dynamic_cast<const RexLiteral*>(oper->getOperand(0));
  const auto rhs_literal = dynamic_cast<const RexLiteral*>(oper->getOperand(1));
  return ((lhs_input && is_fixed_width_filter_literal(rhs_literal)) ||
          (rhs_input && is_fixed_width_filter_literal(lhs_literal)));
}

bool is_text_filter_literal(const RexLiteral* literal) {
  return literal && IS_STRING(literal->getType());
}

bool is_text_equality_input_literal_comparison(const RexOperator* oper) {
  if (!oper || oper->getOperator() != kEQ || oper->size() != size_t(2)) {
    return false;
  }
  const auto lhs_input = dynamic_cast<const RexInput*>(oper->getOperand(0));
  const auto rhs_input = dynamic_cast<const RexInput*>(oper->getOperand(1));
  const auto lhs_literal = dynamic_cast<const RexLiteral*>(oper->getOperand(0));
  const auto rhs_literal = dynamic_cast<const RexLiteral*>(oper->getOperand(1));
  return ((lhs_input && is_text_filter_literal(rhs_literal)) ||
          (rhs_input && is_text_filter_literal(lhs_literal)));
}

bool is_fixed_width_literal_filter(const RexScalar* rex) {
  if (!rex) {
    return true;
  }
  const auto oper = dynamic_cast<const RexOperator*>(rex);
  if (!oper) {
    return !dynamic_cast<const RexLiteral*>(rex);
  }
  if (oper->getOperator() == kAND) {
    for (size_t i = 0; i < oper->size(); ++i) {
      if (!is_fixed_width_literal_filter(oper->getOperand(i))) {
        return false;
      }
    }
    return true;
  }
  return is_fixed_width_input_literal_comparison(oper);
}

bool is_text_equality_literal_filter(const RexScalar* rex) {
  if (!rex) {
    return true;
  }
  const auto oper = dynamic_cast<const RexOperator*>(rex);
  if (!oper) {
    return !dynamic_cast<const RexLiteral*>(rex);
  }
  if (oper->getOperator() == kAND) {
    for (size_t i = 0; i < oper->size(); ++i) {
      if (!is_text_equality_literal_filter(oper->getOperand(i))) {
        return false;
      }
    }
    return true;
  }
  return is_text_equality_input_literal_comparison(oper);
}

size_t scan_row_count(const RelScan* scan) {
  if (!scan || !scan->getTableDescriptor() || !scan->getTableDescriptor()->fragmenter) {
    return 0;
  }
  return scan->getTableDescriptor()->fragmenter->getNumRows();
}

const RelScan* get_direct_scan_input(const RelAlgNode* node) {
  if (const auto scan = dynamic_cast<const RelScan*>(node)) {
    return scan;
  }
  if (const auto compound = dynamic_cast<const RelCompound*>(node);
      compound && compound->inputCount() == size_t(1)) {
    return dynamic_cast<const RelScan*>(compound->getInput(0));
  }
  return nullptr;
}

size_t max_direct_scan_row_count(const RelLeftDeepInnerJoin* join) {
  size_t max_rows{0};
  for (size_t input_idx = 0; join && input_idx < join->inputCount(); ++input_idx) {
    max_rows = std::max(max_rows,
                        scan_row_count(get_direct_scan_input(join->getInput(input_idx))));
  }
  return max_rows;
}

bool can_inline_build_side_filtered_compound(const size_t input_idx,
                                             const RelCompound* compound,
                                             const size_t max_input_scan_rows) {
  if (input_idx == 0 || !compound || !compound->getFilterExpr()) {
    return true;
  }
  if (!contains_literal(compound->getFilterExpr())) {
    return true;
  }
  const auto scan = dynamic_cast<const RelScan*>(compound->getInput(0));
  if (!scan) {
    return false;
  }
  constexpr size_t kSmallTextFilterScanRows = 4096;
  const auto rows = scan_row_count(scan);
  const bool is_known_tiny_scan = rows > 0 && rows <= kSmallTextFilterScanRows;
  const bool is_largest_known_scan =
      max_input_scan_rows > 0 && rows >= max_input_scan_rows;
  // Fixed-width literal filters are cheap to move into the join condition. Text literal
  // filters are only inlined when the scan is provably tiny, or when it is the largest
  // direct scan feeding the join and keeping the filter local would materialize the
  // fact-side payload before the join tree can reduce it.
  return is_fixed_width_literal_filter(compound->getFilterExpr()) ||
         ((is_known_tiny_scan || is_largest_known_scan) &&
          is_text_equality_literal_filter(compound->getFilterExpr()));
}

std::unique_ptr<const RexScalar> copy_rebound_condition(const RexScalar* condition,
                                                        const RelProject* project) {
  RexDeepCopyVisitor copier;
  auto copied_condition = copier.visit(condition);
  rebind_inputs_through_simple_project(copied_condition.get(), project);
  return copied_condition;
}

std::unique_ptr<const RexScalar> copy_rebound_condition(
    const RexScalar* condition,
    const std::vector<std::shared_ptr<const RelCompound>>& compounds) {
  RexDeepCopyVisitor copier;
  auto copied_condition = copier.visit(condition);
  for (const auto& compound : compounds) {
    copied_condition =
        copy_and_rebind_inputs_through_compound(copied_condition.get(), compound.get());
  }
  return copied_condition;
}

std::unique_ptr<const RexScalar> make_conjunction(
    std::vector<std::unique_ptr<const RexScalar>> operands) {
  CHECK(!operands.empty());
  if (operands.size() == size_t(1)) {
    return std::move(operands.front());
  }

  bool is_notnull{true};
  for (const auto& operand : operands) {
    if (const auto oper = dynamic_cast<const RexOperator*>(operand.get())) {
      is_notnull = is_notnull && oper->getType().get_notnull();
    }
  }
  return std::make_unique<RexOperator>(kAND, operands, SQLTypeInfo(kBOOLEAN, is_notnull));
}

void rebind_project_references(std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                               const RelProject* project) {
  for (const auto& node : nodes) {
    if (!node) {
      continue;
    }
    if (const auto compound = dynamic_cast<const RelCompound*>(node.get())) {
      for (size_t source_idx = 0; source_idx < compound->getScalarSourcesSize();
           ++source_idx) {
        rebind_inputs_through_simple_project(compound->getScalarSource(source_idx),
                                             project);
      }
      rebind_inputs_through_simple_project(compound->getFilterExpr(), project);
      continue;
    }
    if (const auto project_node = dynamic_cast<const RelProject*>(node.get())) {
      for (size_t project_idx = 0; project_idx < project_node->size(); ++project_idx) {
        rebind_inputs_through_simple_project(project_node->getProjectAt(project_idx),
                                             project);
      }
      continue;
    }
    if (const auto filter = dynamic_cast<const RelFilter*>(node.get())) {
      rebind_inputs_through_simple_project(filter->getCondition(), project);
      continue;
    }
    if (const auto join = dynamic_cast<const RelJoin*>(node.get())) {
      rebind_inputs_through_simple_project(join->getCondition(), project);
      continue;
    }
    if (const auto left_deep_join =
            dynamic_cast<const RelLeftDeepInnerJoin*>(node.get())) {
      rebind_inputs_through_simple_project(left_deep_join->getInnerCondition(), project);
      for (size_t nesting_level = 1;
           nesting_level <= left_deep_join->getOuterConditionsSize();
           ++nesting_level) {
        rebind_inputs_through_simple_project(
            left_deep_join->getOuterCondition(nesting_level), project);
      }
      continue;
    }
  }
}

void rebind_compound_references(std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                                const RelCompound* compound) {
  for (const auto& node : nodes) {
    if (!node) {
      continue;
    }
    if (auto compound_node = dynamic_cast<RelCompound*>(node.get())) {
      if (compound_node == compound) {
        continue;
      }
      std::vector<std::unique_ptr<const RexScalar>> rebound_sources;
      rebound_sources.reserve(compound_node->getScalarSourcesSize());
      for (size_t source_idx = 0; source_idx < compound_node->getScalarSourcesSize();
           ++source_idx) {
        rebound_sources.push_back(copy_and_rebind_inputs_through_compound(
            compound_node->getScalarSource(source_idx), compound));
      }
      compound_node->setScalarSources(rebound_sources);
      if (compound_node->getFilterExpr()) {
        auto rebound_filter = copy_and_rebind_inputs_through_compound(
            compound_node->getFilterExpr(), compound);
        compound_node->setFilterExpr(rebound_filter);
      }
      continue;
    }
    if (const auto project_node = dynamic_cast<const RelProject*>(node.get())) {
      std::vector<std::unique_ptr<const RexScalar>> rebound_exprs;
      rebound_exprs.reserve(project_node->size());
      for (size_t project_idx = 0; project_idx < project_node->size(); ++project_idx) {
        rebound_exprs.push_back(copy_and_rebind_inputs_through_compound(
            project_node->getProjectAt(project_idx), compound));
      }
      project_node->setExpressions(rebound_exprs);
      continue;
    }
    if (auto filter = dynamic_cast<RelFilter*>(node.get())) {
      if (filter->getCondition()) {
        auto rebound_condition =
            copy_and_rebind_inputs_through_compound(filter->getCondition(), compound);
        filter->setCondition(rebound_condition);
      }
      continue;
    }
    if (auto join = dynamic_cast<RelJoin*>(node.get())) {
      if (join->getCondition()) {
        auto rebound_condition =
            copy_and_rebind_inputs_through_compound(join->getCondition(), compound);
        join->setCondition(rebound_condition);
      }
      continue;
    }
    if (const auto left_deep_join =
            dynamic_cast<const RelLeftDeepInnerJoin*>(node.get())) {
      // A flattened replacement join receives a fully rebound condition when it is
      // constructed. Other left-deep joins should not reference this single-use input.
      (void)left_deep_join;
      continue;
    }
  }
}

using OutputIndexMap = std::unordered_map<size_t, size_t>;

size_t project_output_index_after_flatten(const RelProject* project,
                                          const size_t project_output_idx) {
  CHECK(project);
  CHECK_LT(project_output_idx, project->size());
  const auto project_input =
      dynamic_cast<const RexInput*>(project->getProjectAt(project_output_idx));
  CHECK(project_input);
  CHECK_EQ(project_input->getSourceNode(), project->getInput(0));
  return project_input->getIndex();
}

OutputIndexMap make_output_map_after_simple_project_flatten(
    const RelLeftDeepInnerJoin* join,
    const std::vector<std::shared_ptr<const RelProject>>& inlined_projects) {
  CHECK(join);
  std::unordered_set<const RelProject*> inlined_project_set;
  for (const auto& project : inlined_projects) {
    CHECK(project);
    inlined_project_set.insert(project.get());
  }

  OutputIndexMap old_to_new_index_map;
  size_t old_base = 0;
  size_t new_base = 0;
  for (size_t input_idx = 0; input_idx < join->inputCount(); ++input_idx) {
    const auto input = join->getInput(input_idx);
    const auto project = dynamic_cast<const RelProject*>(input);
    const bool flattened_project =
        project && inlined_project_set.count(project) > size_t(0);
    const auto new_input_size =
        flattened_project ? project->getInput(0)->size() : input->size();
    for (size_t old_local_idx = 0; old_local_idx < input->size(); ++old_local_idx) {
      const auto new_local_idx =
          flattened_project ? project_output_index_after_flatten(project, old_local_idx)
                            : old_local_idx;
      old_to_new_index_map.emplace(old_base + old_local_idx, new_base + new_local_idx);
    }
    old_base += input->size();
    new_base += new_input_size;
  }
  return old_to_new_index_map;
}

bool aggregate_input_can_be_remapped(const RelAggregate* aggregate,
                                     const OutputIndexMap& old_to_new_index_map) {
  CHECK(aggregate);
  for (size_t group_idx = 0; group_idx < aggregate->getGroupByCount(); ++group_idx) {
    const auto group_it = old_to_new_index_map.find(group_idx);
    if (group_it == old_to_new_index_map.end() || group_it->second != group_idx) {
      return false;
    }
  }
  for (const auto& agg_expr : aggregate->getAggExprs()) {
    for (size_t operand_idx = 0; operand_idx < agg_expr->size(); ++operand_idx) {
      if (!old_to_new_index_map.count(agg_expr->getOperand(operand_idx))) {
        return false;
      }
    }
  }
  return true;
}

bool aggregate_parents_can_be_remapped(
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    const RelLeftDeepInnerJoin* old_join,
    const OutputIndexMap& old_to_new_index_map) {
  for (const auto& node : nodes) {
    if (!node || !node->hasInput(old_join)) {
      continue;
    }
    const auto aggregate = dynamic_cast<const RelAggregate*>(node.get());
    if (aggregate && !aggregate_input_can_be_remapped(aggregate, old_to_new_index_map)) {
      return false;
    }
  }
  return true;
}

void remap_aggregate_input(RelAggregate* aggregate,
                           const OutputIndexMap& old_to_new_index_map) {
  CHECK(aggregate_input_can_be_remapped(aggregate, old_to_new_index_map));
  auto old_exprs = aggregate->getAggExprsAndRelease();
  std::vector<std::unique_ptr<const RexAgg>> new_exprs;
  new_exprs.reserve(old_exprs.size());
  for (auto& agg_expr : old_exprs) {
    std::vector<size_t> operands;
    operands.reserve(agg_expr->size());
    for (size_t operand_idx = 0; operand_idx < agg_expr->size(); ++operand_idx) {
      const auto operand_it =
          old_to_new_index_map.find(agg_expr->getOperand(operand_idx));
      CHECK(operand_it != old_to_new_index_map.end());
      operands.push_back(operand_it->second);
    }
    new_exprs.push_back(std::make_unique<RexAgg>(
        agg_expr->getKind(), agg_expr->isDistinct(), agg_expr->getType(), operands));
  }
  aggregate->setAggExprs(new_exprs);
}

std::shared_ptr<RelLeftDeepInnerJoin> flatten_leftmost_simple_project_input(
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    const std::shared_ptr<RelLeftDeepInnerJoin>& outer_join) {
  if (!outer_join || outer_join->inputCount() < size_t(2) ||
      !is_inner_only_left_deep_join(*outer_join)) {
    return nullptr;
  }

  const auto project =
      std::dynamic_pointer_cast<const RelProject>(outer_join->getAndOwnInput(0));
  const auto reachable_project_use_count =
      nodes.empty() ? size_t(0)
                    : reachable_direct_input_use_count(nodes.back().get(), project.get());
  if (!project || !project->isSimple() || project->hasWindowFunctionExpr() ||
      project->isUpdateViaSelect() || project->isDeleteViaSelect() ||
      reachable_project_use_count != size_t(1)) {
    return nullptr;
  }

  const auto inner_join =
      std::dynamic_pointer_cast<const RelLeftDeepInnerJoin>(project->getAndOwnInput(0));
  if (!inner_join || inner_join->inputCount() < size_t(2) ||
      !is_inner_only_left_deep_join(*inner_join)) {
    return nullptr;
  }

  RelAlgInputs flattened_inputs;
  flattened_inputs.reserve(inner_join->inputCount() + outer_join->inputCount() - 1);
  for (size_t input_idx = 0; input_idx < inner_join->inputCount(); ++input_idx) {
    flattened_inputs.push_back(inner_join->getAndOwnInput(input_idx));
  }
  for (size_t input_idx = 1; input_idx < outer_join->inputCount(); ++input_idx) {
    flattened_inputs.push_back(outer_join->getAndOwnInput(input_idx));
  }

  auto original_joins = outer_join->getOriginalJoins();
  auto inner_original_joins = inner_join->getOriginalJoins();
  original_joins.insert(
      original_joins.end(), inner_original_joins.begin(), inner_original_joins.end());

  RexDeepCopyVisitor copier;
  std::vector<std::unique_ptr<const RexScalar>> condition_operands;
  condition_operands.push_back(copier.visit(inner_join->getInnerCondition()));
  condition_operands.push_back(
      copy_rebound_condition(outer_join->getInnerCondition(), project.get()));

  std::vector<std::unique_ptr<const RexScalar>> outer_conditions(original_joins.size());
  auto flattened_join = std::make_shared<RelLeftDeepInnerJoin>(
      std::move(flattened_inputs),
      std::move(original_joins),
      make_conjunction(std::move(condition_operands)),
      std::move(outer_conditions));
  return flattened_join;
}

std::pair<std::shared_ptr<RelLeftDeepInnerJoin>,
          std::vector<std::shared_ptr<const RelCompound>>>
flatten_simple_compound_inputs(const std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                               const std::shared_ptr<RelLeftDeepInnerJoin>& join) {
  if (!join || join->inputCount() < size_t(2) || !is_inner_only_left_deep_join(*join)) {
    return {nullptr, {}};
  }

  RelAlgInputs flattened_inputs;
  flattened_inputs.reserve(join->inputCount());
  std::vector<std::shared_ptr<const RelCompound>> inlined_compounds;
  bool changed{false};
  const auto max_input_scan_rows = max_direct_scan_row_count(join.get());
  for (size_t input_idx = 0; input_idx < join->inputCount(); ++input_idx) {
    const auto input = join->getAndOwnInput(input_idx);
    const auto compound = std::dynamic_pointer_cast<const RelCompound>(input);
    const auto reachable_compound_use_count =
        nodes.empty()
            ? size_t(0)
            : reachable_direct_input_use_count(nodes.back().get(), compound.get());
    const bool blocked_filtered_build_side_compound =
        !can_inline_build_side_filtered_compound(
            input_idx, compound.get(), max_input_scan_rows);
    if (is_inlineable_scalar_compound(compound.get()) &&
        reachable_compound_use_count == size_t(1) &&
        !blocked_filtered_build_side_compound) {
      flattened_inputs.push_back(compound->getAndOwnInput(0));
      inlined_compounds.push_back(compound);
      changed = true;
      continue;
    }
    flattened_inputs.push_back(input);
  }
  if (!changed) {
    return {nullptr, {}};
  }

  auto original_joins = join->getOriginalJoins();
  std::vector<std::unique_ptr<const RexScalar>> condition_operands;
  condition_operands.push_back(
      copy_rebound_condition(join->getInnerCondition(), inlined_compounds));
  RexDeepCopyVisitor copier;
  for (const auto& compound : inlined_compounds) {
    if (!compound->getFilterExpr()) {
      continue;
    }
    auto filter = copy_and_rebind_inputs_through_compound(compound->getFilterExpr(),
                                                          compound.get());
    condition_operands.push_back(std::move(filter));
  }

  std::vector<std::unique_ptr<const RexScalar>> outer_conditions(original_joins.size());
  auto flattened_join = std::make_shared<RelLeftDeepInnerJoin>(
      std::move(flattened_inputs),
      std::move(original_joins),
      make_conjunction(std::move(condition_operands)),
      std::move(outer_conditions));
  return {flattened_join, std::move(inlined_compounds)};
}

std::pair<std::shared_ptr<RelLeftDeepInnerJoin>,
          std::vector<std::shared_ptr<const RelProject>>>
flatten_simple_project_inputs(const std::vector<std::shared_ptr<RelAlgNode>>& nodes,
                              const std::shared_ptr<RelLeftDeepInnerJoin>& join) {
  if (!join || join->inputCount() < size_t(2) || !is_inner_only_left_deep_join(*join)) {
    return {nullptr, {}};
  }

  RelAlgInputs flattened_inputs;
  flattened_inputs.reserve(join->inputCount());
  std::vector<std::shared_ptr<const RelProject>> inlined_projects;
  bool changed{false};
  for (size_t input_idx = 0; input_idx < join->inputCount(); ++input_idx) {
    const auto input = join->getAndOwnInput(input_idx);
    const auto project = std::dynamic_pointer_cast<const RelProject>(input);
    const auto reachable_project_use_count =
        nodes.empty()
            ? size_t(0)
            : reachable_direct_input_use_count(nodes.back().get(), project.get());
    if (project && dynamic_cast<const RelLeftDeepInnerJoin*>(project->getInput(0))) {
      flattened_inputs.push_back(input);
      continue;
    }
    if (project && project->isSimple() && !project->hasWindowFunctionExpr() &&
        !project->isUpdateViaSelect() && !project->isDeleteViaSelect() &&
        reachable_project_use_count == size_t(1)) {
      flattened_inputs.push_back(project->getAndOwnInput(0));
      inlined_projects.push_back(project);
      changed = true;
      continue;
    }
    flattened_inputs.push_back(input);
  }
  if (!changed) {
    return {nullptr, {}};
  }

  auto original_joins = join->getOriginalJoins();
  RexDeepCopyVisitor copier;
  auto condition = copier.visit(join->getInnerCondition());
  for (const auto& project : inlined_projects) {
    rebind_inputs_through_simple_project(condition.get(), project.get());
  }

  std::vector<std::unique_ptr<const RexScalar>> outer_conditions(original_joins.size());
  auto flattened_join =
      std::make_shared<RelLeftDeepInnerJoin>(std::move(flattened_inputs),
                                             std::move(original_joins),
                                             std::move(condition),
                                             std::move(outer_conditions));
  return {flattened_join, std::move(inlined_projects)};
}

void flatten_left_deep_join_simple_project_inputs_impl(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    std::unordered_map<const RelAlgNode*,
                       std::unordered_map<unsigned, RegisteredQueryHint>>& query_hints) {
  bool changed{true};
  while (changed) {
    changed = false;
    for (auto& node : nodes) {
      auto outer_join = std::dynamic_pointer_cast<RelLeftDeepInnerJoin>(node);
      auto flattened_join = flatten_leftmost_simple_project_input(nodes, outer_join);
      bool references_rebound{false};
      bool remap_aggregate_parents{false};
      OutputIndexMap output_index_map;
      if (!flattened_join) {
        std::vector<std::shared_ptr<const RelProject>> inlined_projects;
        std::tie(flattened_join, inlined_projects) =
            flatten_simple_project_inputs(nodes, outer_join);
        if (flattened_join) {
          output_index_map = make_output_map_after_simple_project_flatten(
              outer_join.get(), inlined_projects);
          if (!aggregate_parents_can_be_remapped(
                  nodes, outer_join.get(), output_index_map)) {
            flattened_join.reset();
          } else {
            remap_aggregate_parents = true;
          }
        }
        if (flattened_join) {
          for (const auto& project : inlined_projects) {
            rebind_project_references(nodes, project.get());
          }
          references_rebound = true;
        }
      }
      if (!flattened_join) {
        std::vector<std::shared_ptr<const RelCompound>> inlined_compounds;
        std::tie(flattened_join, inlined_compounds) =
            flatten_simple_compound_inputs(nodes, outer_join);
        if (!flattened_join) {
          continue;
        }
        for (const auto& compound : inlined_compounds) {
          rebind_compound_references(nodes, compound.get());
        }
        references_rebound = true;
      }
      if (!references_rebound) {
        const auto project =
            std::dynamic_pointer_cast<const RelProject>(outer_join->getAndOwnInput(0));
        CHECK(project);
        rebind_project_references(nodes, project.get());
      }
      for (auto& maybe_parent : nodes) {
        if (maybe_parent && maybe_parent->hasInput(outer_join.get())) {
          if (remap_aggregate_parents) {
            if (auto aggregate = dynamic_cast<RelAggregate*>(maybe_parent.get())) {
              remap_aggregate_input(aggregate, output_index_map);
            }
          }
          maybe_parent->replaceInput(outer_join, flattened_join);
        }
      }
      transfer_left_deep_join_query_hint(query_hints, outer_join.get(), flattened_join);
      node = flattened_join;
      changed = true;
      break;
    }
  }
}

}  // namespace

// Recognize the left-deep join tree pattern with an optional filter as root
// with `node` as the parent of the join sub-tree. On match, return the root
// of the recognized tree (either the filter node or the outermost join).
std::shared_ptr<const RelAlgNode> get_left_deep_join_root(
    const std::shared_ptr<RelAlgNode>& node) {
  const auto left_deep_join_filter = dynamic_cast<const RelFilter*>(node.get());
  if (left_deep_join_filter) {
    const auto join = dynamic_cast<const RelJoin*>(left_deep_join_filter->getInput(0));
    if (!join) {
      return nullptr;
    }
    if (join->getJoinType() == JoinType::INNER || join->getJoinType() == JoinType::SEMI ||
        join->getJoinType() == JoinType::ANTI) {
      return node;
    }
  }
  if (!node || node->inputCount() != 1) {
    return nullptr;
  }
  const auto join = dynamic_cast<const RelJoin*>(node->getInput(0));
  if (!join) {
    return nullptr;
  }
  return node->getAndOwnInput(0);
}

void rebind_inputs_from_left_deep_join(const RexScalar* rex,
                                       const RelLeftDeepInnerJoin* left_deep_join) {
  RebindRexInputsFromLeftDeepJoin rebind_rex_inputs_from_left_deep_join(left_deep_join);
  rebind_rex_inputs_from_left_deep_join.visit(rex);
}

void create_left_deep_join(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    std::unordered_map<const RelAlgNode*,
                       std::unordered_map<unsigned, RegisteredQueryHint>>& query_hints) {
  std::list<std::shared_ptr<RelAlgNode>> new_nodes;
  for (auto& left_deep_join_candidate : nodes) {
    std::shared_ptr<RelLeftDeepInnerJoin> left_deep_join;
    std::shared_ptr<const RelAlgNode> old_root;
    std::tie(left_deep_join, old_root) = create_left_deep_join(left_deep_join_candidate);
    if (!left_deep_join) {
      continue;
    }
    CHECK_GE(left_deep_join->inputCount(), size_t(2));
    for (size_t nesting_level = 1; nesting_level <= left_deep_join->inputCount() - 1;
         ++nesting_level) {
      const auto outer_condition = left_deep_join->getOuterCondition(nesting_level);
      if (outer_condition) {
        rebind_inputs_from_left_deep_join(outer_condition, left_deep_join.get());
      }
    }
    rebind_inputs_from_left_deep_join(left_deep_join->getInnerCondition(),
                                      left_deep_join.get());
    propagate_query_hints_to_left_deep_join(
        query_hints, left_deep_join, left_deep_join_candidate);
    for (auto& node : nodes) {
      if (node && node->hasInput(old_root.get())) {
        node->replaceInput(left_deep_join_candidate, left_deep_join);
        std::shared_ptr<const RelJoin> old_join;
        if (std::dynamic_pointer_cast<const RelJoin>(left_deep_join_candidate)) {
          old_join = std::static_pointer_cast<const RelJoin>(left_deep_join_candidate);
        } else {
          CHECK_EQ(size_t(1), left_deep_join_candidate->inputCount());
          old_join = std::dynamic_pointer_cast<const RelJoin>(
              left_deep_join_candidate->getAndOwnInput(0));
        }
        while (old_join) {
          node->replaceInput(old_join, left_deep_join);
          old_join =
              std::dynamic_pointer_cast<const RelJoin>(old_join->getAndOwnInput(0));
        }
      }
    }

    new_nodes.emplace_back(std::move(left_deep_join));
  }

  // insert the new left join nodes to the front of the owned RelAlgNode list.
  // This is done to ensure all created RelAlgNodes exist in this list for later
  // visitation, such as RelAlgDag::resetQueryExecutionState.
  nodes.insert(nodes.begin(), new_nodes.begin(), new_nodes.end());
  flatten_left_deep_join_simple_project_inputs_impl(nodes, query_hints);
}

void flatten_left_deep_join_simple_project_inputs(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes,
    std::unordered_map<const RelAlgNode*,
                       std::unordered_map<unsigned, RegisteredQueryHint>>& query_hints) {
  flatten_left_deep_join_simple_project_inputs_impl(nodes, query_hints);
}
