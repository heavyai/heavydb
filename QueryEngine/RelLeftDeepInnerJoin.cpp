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

#include "RelLeftDeepInnerJoin.h"
#include "RelAlgAbstractInterpreter.h"
#include "RexVisitor.h"

#include <glog/logging.h>

#include <numeric>

RelLeftDeepInnerJoin::RelLeftDeepInnerJoin(
    const std::shared_ptr<RelFilter>& filter,
    std::vector<std::shared_ptr<const RelAlgNode>> inputs,
    std::vector<std::shared_ptr<const RelJoin>>& original_joins)
    : condition_(filter ? filter->getAndReleaseCondition() : nullptr)
    , original_filter_(filter)
    , original_joins_(original_joins) {
  std::vector<std::unique_ptr<const RexScalar>> operands;
  bool is_notnull = true;
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
        case JoinType::INNER: {
          if (original_join->getCondition()) {
            operands.emplace_back(original_join->getAndReleaseCondition());
          }
          break;
        }
        case JoinType::LEFT: {
          if (original_join->getCondition()) {
            outer_conditions_per_level_[nesting_level].reset(
                original_join->getAndReleaseCondition());
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

std::string RelLeftDeepInnerJoin::toString() const {
  std::string result =
      "(RelLeftDeepInnerJoin<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
  result += condition_->toString();
  for (const auto& input : inputs_) {
    result += " " + input->toString();
  }
  result += ")";
  return result;
}

size_t RelLeftDeepInnerJoin::size() const {
  size_t total_size = 0;
  for (const auto& input : inputs_) {
    total_size += input->size();
  }
  return total_size;
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

class RebindRexInputsFromLeftDeepJoin : public RexVisitor<void*> {
 public:
  RebindRexInputsFromLeftDeepJoin(const RelLeftDeepInnerJoin* left_deep_join)
      : left_deep_join_(left_deep_join) {
    std::vector<size_t> input_sizes;
    CHECK_GT(left_deep_join->inputCount(), size_t(1));
    for (size_t i = 0; i < left_deep_join->inputCount(); ++i) {
      input_sizes.push_back(left_deep_join->getInput(i)->size());
    }
    input_size_prefix_sums_.resize(input_sizes.size());
    std::partial_sum(
        input_sizes.begin(), input_sizes.end(), input_size_prefix_sums_.begin());
  }

  void* visitInput(const RexInput* rex_input) const override {
    const auto source_node = rex_input->getSourceNode();
    if (left_deep_join_->coversOriginalNode(source_node)) {
      const auto it = std::lower_bound(input_size_prefix_sums_.begin(),
                                       input_size_prefix_sums_.end(),
                                       rex_input->getIndex(),
                                       std::less_equal<size_t>());
      CHECK(it != input_size_prefix_sums_.end());
      const auto input_node =
          left_deep_join_->getInput(std::distance(input_size_prefix_sums_.begin(), it));
      if (it != input_size_prefix_sums_.begin()) {
        const auto prev_input_count = *(it - 1);
        CHECK_LE(prev_input_count, rex_input->getIndex());
        const auto input_index = rex_input->getIndex() - prev_input_count;
        rex_input->setIndex(input_index);
      }
      rex_input->setSourceNode(input_node);
    }
    return nullptr;
  };

 private:
  std::vector<size_t> input_size_prefix_sums_;
  const RelLeftDeepInnerJoin* left_deep_join_;
};

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
    if (join->getJoinType() == JoinType::INNER) {
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

void create_left_deep_join(std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  for (const auto& left_deep_join_candidate : nodes) {
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
  }
}
