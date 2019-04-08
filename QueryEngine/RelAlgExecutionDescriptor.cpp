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

#include "RelAlgExecutionDescriptor.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>

namespace {

using DAG = boost::
    adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS, const RelAlgNode*>;
using Vertex = DAG::vertex_descriptor;

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

std::vector<Vertex> merge_join_with_non_join(const std::vector<Vertex>& vertices,
                                             const DAG& graph) {
  DAG::out_edge_iterator oe_iter, oe_end;
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
    boost::tie(oe_iter, oe_end) = boost::out_edges(vert, graph);
    CHECK(boost::next(oe_iter) == oe_end);
    const auto out_vert = boost::target(*oe_iter, graph);
    if (!dynamic_cast<const RelJoin*>(graph[out_vert])) {
      joins.insert(vert);
    }
  }

  std::vector<Vertex> new_vertices;
  for (const auto vert : vertices) {
    if (joins.count(vert)) {
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
      const auto rel_join = dynamic_cast<const RelJoin*>(node);
      const auto rel_left_deep_join = dynamic_cast<const RelLeftDeepInnerJoin*>(node);
      if ((rel_join || rel_left_deep_join)) {
        // If the input to the join node is something other than RelJoin or RelScan we
        // don't schedule it to run now as we form a subquery out of the sequence and
        // schedule it separately. See create_implicit_subquery_node of
        // RelAlgAbstractInterpreter.cpp. We do it becasue we would like to consturct a
        // subquery of all the imputs of a join which are not RelScan or RelJoin. Because
        // we might need to broadcast those intermediate results to other leaves.
        if (!(dynamic_cast<const RelScan*>(input) ||
              dynamic_cast<const RelJoin*>(input))) {
          continue;
        }
      }
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

std::vector<const RelAlgNode*> schedule_ra_dag(const RelAlgNode* sink) {
  CHECK(sink);
  auto graph = build_dag(sink);
  std::vector<Vertex> ordering;
  boost::topological_sort(graph, std::back_inserter(ordering));
  std::reverse(ordering.begin(), ordering.end());

  std::vector<const RelAlgNode*> nodes;
  ordering = merge_sort_with_input(ordering, graph);
  for (auto vert : merge_join_with_non_join(ordering, graph)) {
    nodes.push_back(graph[vert]);
  }

  return nodes;
}

}  // namespace

std::vector<RaExecutionDesc> get_execution_descriptors(const RelAlgNode* ra_node) {
  CHECK(ra_node);
  if (dynamic_cast<const RelScan*>(ra_node) || dynamic_cast<const RelJoin*>(ra_node)) {
    throw std::runtime_error("Query not supported yet");
  }

  std::vector<RaExecutionDesc> descs;
  for (const auto node : schedule_ra_dag(ra_node)) {
    if (dynamic_cast<const RelScan*>(node)) {
      continue;
    }
    descs.emplace_back(node);
  }

  return descs;
}

std::vector<RaExecutionDesc> get_execution_descriptors(
    const std::vector<const RelAlgNode*>& ra_nodes) {
  CHECK(!ra_nodes.empty());

  std::vector<RaExecutionDesc> descs;
  for (const auto node : ra_nodes) {
    if (dynamic_cast<const RelScan*>(node)) {
      continue;
    }
    CHECK_GT(node->inputCount(), size_t(0));
    descs.emplace_back(node);
  }

  return descs;
}
