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

RelLeftDeepInnerJoin::RelLeftDeepInnerJoin(const std::shared_ptr<RelFilter>& filter,
                                           std::vector<std::shared_ptr<const RelAlgNode>> inputs,
                                           std::vector<std::shared_ptr<const RelJoin>>& original_joins)
    : condition_(static_cast<const RexOperator*>(filter->getAndReleaseCondition())),
      original_filter_(filter),
      original_joins_(original_joins) {
  CHECK(condition_);
  for (const auto& input : inputs) {
    addManagedInput(input);
  }
}

const RexOperator* RelLeftDeepInnerJoin::getCondition() const {
  return condition_.get();
}

std::string RelLeftDeepInnerJoin::toString() const {
  std::string result = "(RelLeftDeepInnerJoin<" + std::to_string(reinterpret_cast<uint64_t>(this)) + ">(";
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

bool is_left_deep_join_helper(const RelJoin* join) {
  CHECK(join);
  // Only inner joins for now.
  if (join->getJoinType() != JoinType::INNER) {
    return false;
  }
  const auto condition_true = dynamic_cast<const RexLiteral*>(join->getCondition());
  if (!condition_true || !condition_true->getVal<bool>()) {
    return false;
  }
  CHECK_EQ(kBOOLEAN, condition_true->getType());
  CHECK_EQ(size_t(2), join->inputCount());
  if (dynamic_cast<const RelJoin*>(join->getInput(1))) {
    // Not left-deep.
    return false;
  }
  const auto left_input_join = dynamic_cast<const RelJoin*>(join->getInput(0));
  if (left_input_join) {
    return is_left_deep_join_helper(left_input_join);
  }
  // It's disabled for now.
  return false /* true */;
}

void collect_left_deep_join_inputs(std::deque<std::shared_ptr<const RelAlgNode>>& inputs,
                                   std::vector<std::shared_ptr<const RelJoin>>& original_joins,
                                   const std::shared_ptr<const RelJoin>& join) {
  original_joins.push_back(join);
  CHECK_EQ(size_t(2), join->inputCount());
  const auto left_input_join = std::dynamic_pointer_cast<const RelJoin>(join->getAndOwnInput(0));
  if (left_input_join) {
    inputs.push_front(join->getAndOwnInput(1));
    collect_left_deep_join_inputs(inputs, original_joins, left_input_join);
  } else {
    inputs.push_front(join->getAndOwnInput(1));
    inputs.push_front(join->getAndOwnInput(0));
  }
}

std::shared_ptr<RelLeftDeepInnerJoin> create_left_deep_join(const std::shared_ptr<RelAlgNode>& left_deep_join_root) {
  if (!is_left_deep_join(left_deep_join_root.get())) {
    return nullptr;
  }
  std::deque<std::shared_ptr<const RelAlgNode>> inputs_deque;
  const auto left_deep_join_filter = std::dynamic_pointer_cast<RelFilter>(left_deep_join_root);
  CHECK(left_deep_join_filter);
  const auto join = std::dynamic_pointer_cast<const RelJoin>(left_deep_join_filter->getAndOwnInput(0));
  CHECK(join);
  std::vector<std::shared_ptr<const RelJoin>> original_joins;
  collect_left_deep_join_inputs(inputs_deque, original_joins, join);
  std::vector<std::shared_ptr<const RelAlgNode>> inputs(inputs_deque.begin(), inputs_deque.end());
  return std::make_shared<RelLeftDeepInnerJoin>(left_deep_join_filter, inputs, original_joins);
}

class RebindRexInputsFromLeftDeepJoin : public RexVisitor<void*> {
 public:
  RebindRexInputsFromLeftDeepJoin(const RelLeftDeepInnerJoin* left_deep_join) : left_deep_join_(left_deep_join) {
    std::vector<size_t> input_sizes;
    CHECK_GT(left_deep_join->inputCount(), size_t(1));
    for (size_t i = 0; i < left_deep_join->inputCount(); ++i) {
      input_sizes.push_back(left_deep_join->getInput(i)->size());
    }
    input_size_prefix_sums_.resize(input_sizes.size());
    std::partial_sum(input_sizes.begin(), input_sizes.end(), input_size_prefix_sums_.begin());
  }

  void* visitInput(const RexInput* rex_input) const override {
    const auto source_node = rex_input->getSourceNode();
    if (left_deep_join_->coversOriginalNode(source_node)) {
      const auto it = std::lower_bound(input_size_prefix_sums_.begin(),
                                       input_size_prefix_sums_.end(),
                                       rex_input->getIndex(),
                                       std::less_equal<size_t>());
      CHECK(it != input_size_prefix_sums_.end());
      const auto input_node = left_deep_join_->getInput(std::distance(input_size_prefix_sums_.begin(), it));
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

bool is_left_deep_join(const RelAlgNode* left_deep_join_root) {
  const auto left_deep_join_filter = dynamic_cast<const RelFilter*>(left_deep_join_root);
  if (!left_deep_join_filter) {
    return false;
  }
  CHECK_EQ(size_t(1), left_deep_join_filter->inputCount());
  const auto join = dynamic_cast<const RelJoin*>(left_deep_join_filter->getInput(0));
  if (!join) {
    return false;
  }
  return is_left_deep_join_helper(join);
}

void rebind_inputs_from_left_deep_join(const RexScalar* rex, const RelLeftDeepInnerJoin* left_deep_join) {
  RebindRexInputsFromLeftDeepJoin rebind_rex_inputs_from_left_deep_join(left_deep_join);
  rebind_rex_inputs_from_left_deep_join.visit(rex);
}

void create_left_deep_join(std::vector<std::shared_ptr<RelAlgNode>>& nodes) {
  for (const auto& left_deep_join_candidate : nodes) {
    const auto left_deep_join = create_left_deep_join(left_deep_join_candidate);
    if (!left_deep_join) {
      continue;
    }
    rebind_inputs_from_left_deep_join(left_deep_join->getCondition(), left_deep_join.get());
    for (auto& node : nodes) {
      if (node && node->hasInput(left_deep_join_candidate.get())) {
        node->replaceInput(left_deep_join_candidate, left_deep_join);
        CHECK_EQ(size_t(1), left_deep_join_candidate->inputCount());
        auto old_join = std::dynamic_pointer_cast<const RelJoin>(left_deep_join_candidate->getAndOwnInput(0));
        while (old_join) {
          node->replaceInput(old_join, left_deep_join);
          old_join = std::dynamic_pointer_cast<const RelJoin>(old_join->getAndOwnInput(0));
        }
      }
    }
  }
}
