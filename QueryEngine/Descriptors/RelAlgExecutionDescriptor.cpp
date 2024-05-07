/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include "QueryEngine/GroupByAndAggregate.h"
#include "QueryEngine/QueryPlanDagExtractor.h"
#include "QueryEngine/RelAlgDag.h"

#include <boost/graph/topological_sort.hpp>

#include <algorithm>

ExecutionResult::ExecutionResult()
    : filter_push_down_enabled_(false)
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {}

ExecutionResult::ExecutionResult(const std::shared_ptr<ResultSet>& rows,
                                 const std::vector<TargetMetaInfo>& targets_meta)
    : result_(rows)
    , targets_meta_(targets_meta)
    , filter_push_down_enabled_(false)
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {}

ExecutionResult::ExecutionResult(ResultSetPtr&& result,
                                 const std::vector<TargetMetaInfo>& targets_meta)
    : result_(std::move(result))
    , targets_meta_(targets_meta)
    , filter_push_down_enabled_(false)
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {}

ExecutionResult::ExecutionResult(const ExecutionResult& that)
    : targets_meta_(that.targets_meta_)
    , pushed_down_filter_info_(that.pushed_down_filter_info_)
    , filter_push_down_enabled_(that.filter_push_down_enabled_)
    , success_(that.success_)
    , execution_time_ms_(that.execution_time_ms_)
    , type_(that.type_) {
  if (!pushed_down_filter_info_.empty() ||
      (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
    return;
  }
  result_ = that.result_;
}

ExecutionResult::ExecutionResult(ExecutionResult&& that)
    : targets_meta_(std::move(that.targets_meta_))
    , pushed_down_filter_info_(std::move(that.pushed_down_filter_info_))
    , filter_push_down_enabled_(std::move(that.filter_push_down_enabled_))
    , success_(that.success_)
    , execution_time_ms_(that.execution_time_ms_)
    , type_(that.type_) {
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
    , filter_push_down_enabled_(filter_push_down_enabled)
    , success_(true)
    , execution_time_ms_(0)
    , type_(QueryResult) {}

ExecutionResult& ExecutionResult::operator=(const ExecutionResult& that) {
  if (!that.pushed_down_filter_info_.empty() ||
      (that.filter_push_down_enabled_ && that.pushed_down_filter_info_.empty())) {
    pushed_down_filter_info_ = that.pushed_down_filter_info_;
    filter_push_down_enabled_ = that.filter_push_down_enabled_;
    return *this;
  }
  result_ = that.result_;
  targets_meta_ = that.targets_meta_;
  success_ = that.success_;
  execution_time_ms_ = that.execution_time_ms_;
  type_ = that.type_;
  return *this;
}

const std::vector<PushedDownFilterInfo>& ExecutionResult::getPushedDownFilterInfo()
    const {
  return pushed_down_filter_info_;
}

void ExecutionResult::updateResultSet(const std::string& query,
                                      RType type,
                                      bool success) {
  targets_meta_.clear();
  pushed_down_filter_info_.clear();
  success_ = success;
  type_ = type;
  result_ = std::make_shared<ResultSet>(query);
}

std::string ExecutionResult::getExplanation() {
  if (!empty()) {
    return getRows()->getExplanation();
  }
  return {};
}

void RaExecutionDesc::setResult(const ExecutionResult& result) {
  result_ = result;
  body_->setContextData(this);
}

const RelAlgNode* RaExecutionDesc::getBody() const {
  return body_;
}

namespace {

DAG build_dag(const RelAlgNode* sink,
              std::unordered_map<const RelAlgNode*, int>& node_ptr_to_vert_idx) {
  DAG graph(1);
  graph[0] = sink;
  node_ptr_to_vert_idx.emplace(std::make_pair(sink, 0));
  std::vector<const RelAlgNode*> stack(1, sink);
  while (!stack.empty()) {
    const auto node = stack.back();
    stack.pop_back();
    if (dynamic_cast<const RelScan*>(node)) {
      continue;
    }

    const auto input_num = node->inputCount();
    switch (input_num) {
      case 0:
        CHECK(dynamic_cast<const RelLogicalValues*>(node) ||
              dynamic_cast<const RelTableFunction*>(node));
      case 1:
        break;
      case 2:
        CHECK(dynamic_cast<const RelJoin*>(node) ||
              dynamic_cast<const RelLeftDeepInnerJoin*>(node) ||
              dynamic_cast<const RelLogicalUnion*>(node) ||
              dynamic_cast<const RelTableFunction*>(node));
        break;
      default:
        CHECK(dynamic_cast<const RelLeftDeepInnerJoin*>(node) ||
              dynamic_cast<const RelLogicalUnion*>(node) ||
              dynamic_cast<const RelTableFunction*>(node));
    }
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
                                         Executor* executor,
                                         const bool just_validation,
                                         const bool build_sequence)
    : executor_(executor), just_validation_(just_validation) {
  CHECK(sink);
  CHECK(executor_);
  if (dynamic_cast<const RelScan*>(sink) || dynamic_cast<const RelJoin*>(sink)) {
    throw std::runtime_error("Query not supported yet");
  }
  graph_ = build_dag(sink, node_ptr_to_vert_idx_);

  boost::topological_sort(graph_, std::back_inserter(ordering_));
  std::reverse(ordering_.begin(), ordering_.end());

  ordering_ = mergeSortWithInput(ordering_, graph_);
  joins_ = get_join_vertices(ordering_, graph_);

  if (build_sequence) {
    while (next()) {
      // noop
    }
    if (g_enable_data_recycler && g_use_query_resultset_cache &&
        g_allow_query_step_skipping && !has_step_for_union_ && !g_cluster &&
        !just_validation_) {
      extractQueryStepSkippingInfo();
      skipQuerySteps();
    }
  }
}

RaExecutionSequence::RaExecutionSequence(std::unique_ptr<RaExecutionDesc> exec_desc) {
  descs_.emplace_back(std::move(exec_desc));
}

RaExecutionDesc* RaExecutionSequence::next() {
  auto checkQueryStepSkippable =
      [&](RelAlgNode const* node, QueryPlanHash key, size_t step_id) {
        CHECK_GE(step_id, 0);
        if (executor_->getResultSetRecyclerHolder().hasCachedQueryResultSet(key)) {
          cached_query_steps_.insert(step_id);
          auto const node_id = -int(node->getId());
          cached_resultset_keys_.emplace(node_id, key);
          VLOG(1) << "Skip query step if possible, step_id: " << step_id
                  << ", node_id: " << node_id << ", query plan dag hash: " << key;
          auto const output_meta_info =
              executor_->getResultSetRecyclerHolder().getOutputMetaInfo(key);
          CHECK(output_meta_info);
          node->setOutputMetainfo(*output_meta_info);
        }
      };
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
    auto logical_union = dynamic_cast<const RelLogicalUnion*>(node);
    if (logical_union) {
      has_step_for_union_ = true;
    }
    auto extracted_query_plan_dag =
        QueryPlanDagExtractor::extractQueryPlanDag(node, executor_);
    if (g_allow_query_step_skipping && !just_validation_ &&
        !boost::iequals(extracted_query_plan_dag.extracted_dag, EMPTY_QUERY_PLAN)) {
      SortInfo sort_info;
      if (auto sort_node = dynamic_cast<const RelSort*>(node)) {
        sort_info = SortInfo::createFromSortNode(sort_node);
      }
      auto cache_key = QueryPlanDagExtractor::applyLimitClauseToCacheKey(
          node->getQueryPlanDagHash(), sort_info);
      checkQueryStepSkippable(node, cache_key, descs_.size() - 1);
    }
    return descs_.back().get();
  }
  return nullptr;
}

std::vector<Vertex> RaExecutionSequence::mergeSortWithInput(
    const std::vector<Vertex>& vertices,
    const DAG& graph) {
  DAG::in_edge_iterator ie_iter, ie_end;
  std::unordered_set<Vertex> inputs;
  for (const auto vert : vertices) {
    if (const auto sort = dynamic_cast<const RelSort*>(graph[vert])) {
      boost::tie(ie_iter, ie_end) = boost::in_edges(vert, graph);
      CHECK(size_t(1) == sort->inputCount() && boost::next(ie_iter) == ie_end);
      if (sort->isLimitDelivered()) {
        has_limit_clause_ = true;
      }
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

RaExecutionDesc* RaExecutionSequence::prev() {
  if (descs_.empty()) {
    return nullptr;
  }
  if (descs_.size() == 1) {
    return nullptr;
  }
  CHECK_GE(descs_.size(), size_t(2));
  auto itr = descs_.rbegin();
  return (++itr)->get();
}

void RaExecutionSequence::extractQueryStepSkippingInfo() {
  const auto pushChildNodes = [&](auto& stack, const auto node_id) {
    auto [in_edge_iter, in_edge_end] = boost::in_edges(node_id, graph_);
    while (in_edge_iter != in_edge_end) {
      const auto child_node_id = boost::source(*in_edge_iter, graph_);
      stack.push(graph_[child_node_id]);
      std::advance(in_edge_iter, 1);
    }
  };
  auto top_node_id = descs_.size() - 1;
  auto top_res = skippable_steps_.emplace(top_node_id, std::unordered_set<int>{});
  CHECK(top_res.second);
  for (auto it = descs_.begin(); it != std::prev(descs_.end()); ++it) {
    const auto step = it->get();
    const auto body = step->getBody();
    const auto step_id = static_cast<int>(std::distance(descs_.begin(), it));
    auto res = skippable_steps_.emplace(step_id, std::unordered_set<int>{});
    CHECK(res.second);
    skippable_steps_[top_node_id].insert(step_id);  // top-desc can skip all child descs
    std::stack<const RelAlgNode*> stack;
    pushChildNodes(stack, node_ptr_to_vert_idx_[body]);
    while (!stack.empty()) {
      auto child_node = stack.top();
      stack.pop();
      // descs_ is based on the topologically sorted (flattened) DAG, so we can limit the
      // search range for child descs
      auto is_desc_body = std::find_if(
          descs_.begin(), it, [&child_node](std::unique_ptr<RaExecutionDesc>& ptr) {
            return ptr->getBody() == child_node;
          });
      if (is_desc_body != it) {
        // due to the topological sorting of query plan DAG, we can avoid visiting "all"
        // child nodes
        const auto child_step_id =
            static_cast<int>(std::distance(descs_.begin(), is_desc_body));
        skippable_steps_[step_id].insert(child_step_id);
        skippable_steps_[step_id].insert(skippable_steps_[child_step_id].begin(),
                                         skippable_steps_[child_step_id].end());
      } else {
        pushChildNodes(stack, node_ptr_to_vert_idx_[child_node]);
      }
    }
  }
}

void RaExecutionSequence::skipQuerySteps() {
  CHECK_LE(cached_query_steps_.size(), descs_.size());

  // extract skippable query steps`
  std::unordered_set<int> skippable_query_steps;
  for (const auto cached_step_id : cached_query_steps_) {
    auto it = skippable_steps_.find(cached_step_id);
    CHECK(it != skippable_steps_.end());
    const auto& child_steps = it->second;
    std::for_each(
        child_steps.begin(), child_steps.end(), [&](const auto& skippable_step_id) {
          if (skippable_step_id != cached_step_id) {
            skippable_query_steps.insert(skippable_step_id);
          }
        });
  }

  // modify query step sequence based on the skippable query steps
  if (!skippable_query_steps.empty()) {
    std::vector<std::unique_ptr<RaExecutionDesc>> new_descs;
    for (size_t step_id = 0; step_id < descs_.size(); ++step_id) {
      const auto body = descs_[step_id]->getBody();
      if (!skippable_query_steps.count(step_id)) {
        new_descs.push_back(std::make_unique<RaExecutionDesc>(body));
      } else {
        VLOG(1) << "Skip query step-" << step_id << " (node_id: " << body->getId() << ")";
      }
    }
    const auto prev_num_steps = descs_.size();
    if (!new_descs.empty()) {
      std::swap(descs_, new_descs);
    }
    VLOG(1) << "Skip " << prev_num_steps - descs_.size() << " query steps from "
            << prev_num_steps << " steps";
  }

  for (const auto& desc : descs_) {
    // remove cached resultset info for each desc since
    // it is not skipped
    auto body = desc->getBody();
    auto it = cached_resultset_keys_.find(-body->getId());
    if (it != cached_resultset_keys_.end()) {
      cached_resultset_keys_.erase(it);
    }
  }
}

std::optional<size_t> RaExecutionSequence::nextStepId(const bool after_broadcast) const {
  if (after_broadcast) {
    if (current_vertex_ == ordering_.size()) {
      return std::nullopt;
    }
    return descs_.size() + stepsToNextBroadcast();
  } else if (current_vertex_ == ordering_.size()) {
    return std::nullopt;
  } else {
    return descs_.size();
  }
}

bool RaExecutionSequence::executionFinished() const {
  if (current_vertex_ == ordering_.size()) {
    // All descriptors visited, execution finished
    return true;
  } else {
    const auto next_step_id = nextStepId(true);
    if (!next_step_id || (*next_step_id == totalDescriptorsCount())) {
      // One step remains (the current vertex), or all remaining steps can be executed
      // without another broadcast (i.e. on the aggregator)
      return true;
    }
  }
  return false;
}

namespace {
struct MatchBody {
  unsigned const body_id_;
  bool operator()(std::unique_ptr<RaExecutionDesc> const& desc) const {
    return desc->getBody()->getId() == body_id_;
  }
};
}  // namespace

// Search for RaExecutionDesc* by body, starting at start_idx and decrementing to 0.
RaExecutionDesc* RaExecutionSequence::getDescriptorByBodyId(
    unsigned const body_id,
    size_t const start_idx) const {
  CHECK_LT(start_idx, descs_.size());
  auto const from_end = descs_.size() - (start_idx + 1);
  MatchBody const match_body{body_id};
  auto const itr = std::find_if(descs_.rbegin() + from_end, descs_.rend(), match_body);
  return itr == descs_.rend() ? nullptr : itr->get();
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

size_t RaExecutionSequence::stepsToNextBroadcast() const {
  size_t steps_to_next_broadcast = 0;
  auto crt_vertex = current_vertex_;
  while (crt_vertex < ordering_.size()) {
    auto vert = ordering_[crt_vertex++];
    auto node = graph_[vert];
    CHECK(node);
    if (joins_.count(vert)) {
      auto join_node = dynamic_cast<const RelLeftDeepInnerJoin*>(node);
      CHECK(join_node);
      for (size_t i = 0; i < join_node->inputCount(); i++) {
        const auto input = join_node->getInput(i);
        if (dynamic_cast<const RelScan*>(input)) {
          return steps_to_next_broadcast;
        }
      }
      if (crt_vertex < ordering_.size() - 1) {
        // Force the parent node of the RelLeftDeepInnerJoin to run on the aggregator.
        // Note that crt_vertex has already been incremented once above for the join node
        // -- increment it again to account for the parent node of the join
        ++steps_to_next_broadcast;
        ++crt_vertex;
        continue;
      } else {
        CHECK_EQ(crt_vertex, ordering_.size() - 1);
        // If the join node parent is the last node in the tree, run all remaining steps
        // on the aggregator
        return ++steps_to_next_broadcast;
      }
    }
    if (auto sort = dynamic_cast<const RelSort*>(node)) {
      CHECK_EQ(sort->inputCount(), size_t(1));
      node = sort->getInput(0);
    }
    if (dynamic_cast<const RelScan*>(node)) {
      return steps_to_next_broadcast;
    }
    if (dynamic_cast<const RelModify*>(node)) {
      // Modify runs on the leaf automatically, run the same node as a noop on the
      // aggregator
      ++steps_to_next_broadcast;
      continue;
    }
    if (auto project = dynamic_cast<const RelProject*>(node)) {
      if (project->hasWindowFunctionExpr()) {
        ++steps_to_next_broadcast;
        continue;
      }
    }
    for (size_t input_idx = 0; input_idx < node->inputCount(); input_idx++) {
      if (dynamic_cast<const RelScan*>(node->getInput(input_idx))) {
        return steps_to_next_broadcast;
      }
    }
    ++steps_to_next_broadcast;
  }
  return steps_to_next_broadcast;
}
