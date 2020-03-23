/*
 * Copyright 2019 OmniSci, Inc.
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

#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"

#include <boost/graph/topological_sort.hpp>

#include "QueryEngine/GroupByAndAggregate.h"
#include "QueryEngine/RelAlgDagBuilder.h"

ExecutionResult::ExecutionResult(const std::shared_ptr<ResultSet>& rows,
                                 const std::vector<TargetMetaInfo>& targets_meta)
    : result_(rows), targets_meta_(targets_meta), filter_push_down_enabled_(false) {}

ExecutionResult::ExecutionResult(ResultSetPtr&& result,
                                 const std::vector<TargetMetaInfo>& targets_meta)
    : targets_meta_(targets_meta), filter_push_down_enabled_(false) {
  result_ = std::move(result);
}

ExecutionResult::ExecutionResult(const ExecutionResult& that)
    : targets_meta_(that.targets_meta_)
    , pushed_down_filter_info_(that.pushed_down_filter_info_)
    , filter_push_down_enabled_(that.filter_push_down_enabled_) {
  if (!pushed_down_filter_info_.empty() ||
      (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
    return;
  }
  result_ = that.result_;
}

ExecutionResult::ExecutionResult(ExecutionResult&& that)
    : targets_meta_(std::move(that.targets_meta_))
    , pushed_down_filter_info_(std::move(that.pushed_down_filter_info_))
    , filter_push_down_enabled_(std::move(that.filter_push_down_enabled_)) {
  if (!pushed_down_filter_info_.empty() ||
      (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
    return;
  }
  result_ = std::move(that.result_);
}

ExecutionResult::ExecutionResult(
    const std::vector<PushedDownFilterInfo>& pushed_down_filter_info,
    bool filter_push_down_enabled)
    : pushed_down_filter_info_(pushed_down_filter_info)
    , filter_push_down_enabled_(filter_push_down_enabled) {}

ExecutionResult& ExecutionResult::operator=(const ExecutionResult& that) {
  if (!that.pushed_down_filter_info_.empty() ||
      (that.filter_push_down_enabled_ && that.pushed_down_filter_info_.empty())) {
    pushed_down_filter_info_ = that.pushed_down_filter_info_;
    filter_push_down_enabled_ = that.filter_push_down_enabled_;
    return *this;
  }
  result_ = that.result_;
  targets_meta_ = that.targets_meta_;
  return *this;
}

const std::vector<PushedDownFilterInfo>& ExecutionResult::getPushedDownFilterInfo()
    const {
  return pushed_down_filter_info_;
}

void RaExecutionDesc::setResult(const ExecutionResult& result) {
  result_ = result;
  body_->setContextData(this);
}

const RelAlgNode* RaExecutionDesc::getBody() const {
  return body_;
}

namespace {

std::vector<Vertex> merge_sort_with_input(const std::vector<Vertex>& vertices,
                                          const DAG& graph) {
  DAG::in_edge_iterator ie_iter, ie_end;
  std::unordered_set<Vertex> inputs;
  for (const auto vert : vertices) {
    if (const auto sort = dynamic_cast<const RelSort*>(graph[vert])) {
      boost::tie(ie_iter, ie_end) = boost::in_edges(vert, graph);
      CHECK(size_t(1) == sort->inputCount() && boost::next(ie_iter) == ie_end);
      const auto in_vert = boost::source(*ie_iter, graph);
      const auto input = graph[in_vert];
      if (dynamic_cast<const RelScan*>(input)) {
        throw std::runtime_error("Standalone sort not supported yet");
      }
      if (boost::out_degree(in_vert, graph) > 1) {
        throw std::runtime_error("Sort's input node used by others not supported yet");
      }
      inputs.insert(in_vert);
    }
  }

  std::vector<Vertex> new_vertices;
  for (const auto vert : vertices) {
    if (inputs.count(vert)) {
      continue;
    }
    new_vertices.push_back(vert);
  }
  return new_vertices;
}

DAG build_dag(const RelAlgNode* sink) {
  DAG graph(1);
  graph[0] = sink;
  std::unordered_map<const RelAlgNode*, int> node_ptr_to_vert_idx{
      std::make_pair(sink, 0)};
  std::vector<const RelAlgNode*> stack(1, sink);
  while (!stack.empty()) {
    const auto node = stack.back();
    stack.pop_back();
    if (dynamic_cast<const RelScan*>(node)) {
      continue;
    }

    const auto input_num = node->inputCount();
    CHECK(input_num == 1 ||
          (dynamic_cast<const RelLogicalValues*>(node) && input_num == 0) ||
          (dynamic_cast<const RelModify*>(node) && input_num == 1) ||
          (input_num == 2 && (dynamic_cast<const RelJoin*>(node) ||
                              dynamic_cast<const RelLeftDeepInnerJoin*>(node))) ||
          (input_num > 2 && (dynamic_cast<const RelLeftDeepInnerJoin*>(node))));
    for (size_t i = 0; i < input_num; ++i) {
      const auto input = node->getInput(i);
      CHECK(input);
      const bool visited = node_ptr_to_vert_idx.count(input) > 0;
      if (!visited) {
        node_ptr_to_vert_idx.insert(std::make_pair(input, node_ptr_to_vert_idx.size()));
      }
      boost::add_edge(node_ptr_to_vert_idx[input], node_ptr_to_vert_idx[node], graph);
      if (!visited) {
        graph[node_ptr_to_vert_idx[input]] = input;
        stack.push_back(input);
      }
    }
  }
  return graph;
}

std::unordered_set<Vertex> get_join_vertices(const std::vector<Vertex>& vertices,
                                             const DAG& graph) {
  std::unordered_set<Vertex> joins;
  for (const auto vert : vertices) {
    if (dynamic_cast<const RelLeftDeepInnerJoin*>(graph[vert])) {
      joins.insert(vert);
      continue;
    }
    if (!dynamic_cast<const RelJoin*>(graph[vert])) {
      continue;
    }
    if (boost::out_degree(vert, graph) > 1) {
      throw std::runtime_error("Join used more than once not supported yet");
    }
    auto [oe_iter, oe_end] = boost::out_edges(vert, graph);
    CHECK(std::next(oe_iter) == oe_end);
    const auto out_vert = boost::target(*oe_iter, graph);
    if (!dynamic_cast<const RelJoin*>(graph[out_vert])) {
      joins.insert(vert);
    }
  }
  return joins;
}

}  // namespace

RaExecutionSequence::RaExecutionSequence(const RelAlgNode* sink,
                                         const bool build_sequence) {
  CHECK(sink);
  if (dynamic_cast<const RelScan*>(sink) || dynamic_cast<const RelJoin*>(sink)) {
    throw std::runtime_error("Query not supported yet");
  }

  graph_ = build_dag(sink);

  boost::topological_sort(graph_, std::back_inserter(ordering_));
  std::reverse(ordering_.begin(), ordering_.end());

  ordering_ = merge_sort_with_input(ordering_, graph_);
  joins_ = get_join_vertices(ordering_, graph_);

  if (build_sequence) {
    while (next()) {
      // noop
    }
  }
}

RaExecutionSequence::RaExecutionSequence(std::unique_ptr<RaExecutionDesc> exec_desc) {
  descs_.emplace_back(std::move(exec_desc));
}

RaExecutionDesc* RaExecutionSequence::next() {
  while (current_vertex_ < ordering_.size()) {
    auto vert = ordering_[current_vertex_++];
    if (joins_.count(vert)) {
      continue;
    }
    auto& node = graph_[vert];
    CHECK(node);
    if (dynamic_cast<const RelScan*>(node)) {
      scan_count_++;
      continue;
    }
    descs_.emplace_back(std::make_unique<RaExecutionDesc>(node));
    return descs_.back().get();
  }
  return nullptr;
}

ssize_t RaExecutionSequence::nextStepId(const bool after_reduction) const {
  if (after_reduction) {
    if (current_vertex_ == ordering_.size()) {
      return -1;
    }
    const auto steps_to_next_reduction = static_cast<ssize_t>(stepsToNextReduction());
    return static_cast<ssize_t>(descs_.size()) + steps_to_next_reduction;
  } else {
    return current_vertex_ == ordering_.size() ? -1 : descs_.size();
  }
}

bool RaExecutionSequence::executionFinished() const {
  if (current_vertex_ == ordering_.size()) {
    // All descriptors visited, execution finished
    return true;
  } else {
    const auto next_step_id = nextStepId(true);
    if (next_step_id < 0 ||
        (static_cast<size_t>(next_step_id) == totalDescriptorsCount())) {
      // One step remains (the current vertex), or all remaining steps can be executed
      // without another reduction
      return true;
    }
  }
  return false;
}

size_t RaExecutionSequence::totalDescriptorsCount() const {
  size_t num_descriptors = 0;
  size_t crt_vertex = 0;
  while (crt_vertex < ordering_.size()) {
    auto vert = ordering_[crt_vertex++];
    if (joins_.count(vert)) {
      continue;
    }
    auto& node = graph_[vert];
    CHECK(node);
    if (dynamic_cast<const RelScan*>(node)) {
      continue;
    }
    ++num_descriptors;
  }
  return num_descriptors;
}

size_t RaExecutionSequence::stepsToNextReduction() const {
  size_t steps_to_next_reduction = 0;
  auto crt_vertex = current_vertex_;
  while (crt_vertex < ordering_.size()) {
    auto vert = ordering_[crt_vertex++];
    auto node = graph_[vert];
    CHECK(node);
    if (joins_.count(vert)) {
      auto join_node = dynamic_cast<const RelLeftDeepInnerJoin*>(node);
      CHECK(join_node);
      if (crt_vertex < ordering_.size() - 1) {
        // Force the parent node of the RelLeftDeepInnerJoin to run on the aggregator.
        // Note that crt_vertex has already been incremented once above for the join node
        // -- increment it again to account for the parent node of the join
        ++steps_to_next_reduction;
        ++crt_vertex;
        continue;
      } else {
        CHECK_EQ(crt_vertex, ordering_.size() - 1);
        // If the join node parent is the last node in the tree, run all remaining steps
        // on the aggregator
        return ++steps_to_next_reduction;
      }
    }
    if (auto sort = dynamic_cast<const RelSort*>(node)) {
      CHECK_EQ(sort->inputCount(), size_t(1));
      node = sort->getInput(0);
    }
    if (dynamic_cast<const RelScan*>(node)) {
      return steps_to_next_reduction;
    }
    if (dynamic_cast<const RelModify*>(node)) {
      // Modify runs on the leaf automatically, run the same node as a noop on the
      // aggregator
      ++steps_to_next_reduction;
      continue;
    }
    if (auto project = dynamic_cast<const RelProject*>(node)) {
      if (project->hasWindowFunctionExpr()) {
        ++steps_to_next_reduction;
        continue;
      }
    }
    for (size_t input_idx = 0; input_idx < node->inputCount(); input_idx++) {
      if (dynamic_cast<const RelScan*>(node->getInput(input_idx))) {
        return steps_to_next_reduction;
      }
    }
    ++steps_to_next_reduction;
  }
  return steps_to_next_reduction;
}
