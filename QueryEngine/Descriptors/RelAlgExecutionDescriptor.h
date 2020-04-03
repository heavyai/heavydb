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

#pragma once

#include <boost/graph/adjacency_list.hpp>

#include "QueryEngine/Descriptors/QueryMemoryDescriptor.h"
#include "QueryEngine/JoinFilterPushDown.h"
#include "QueryEngine/ResultSet.h"
#include "Shared/TargetInfo.h"

class ResultSet;

class ExecutionResult {
 public:
  ExecutionResult(const ResultSetPtr& rows,
                  const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(ResultSetPtr&& result, const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(const TemporaryTable& results,
                  const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(TemporaryTable&& results,
                  const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(const ExecutionResult& that);

  ExecutionResult(ExecutionResult&& that);

  ExecutionResult(const std::vector<PushedDownFilterInfo>& pushed_down_filter_info,
                  bool filter_push_down_enabled);

  ExecutionResult& operator=(const ExecutionResult& that);

  const ResultSetPtr& getRows() const {
    CHECK_EQ(results_.getFragCount(), 1);
    return results_[0];
  }

  bool empty() const { return results_.empty(); }

  const ResultSetPtr& getDataPtr() const {
    CHECK_EQ(results_.getFragCount(), 1);
    return results_[0];
  }

  const TemporaryTable& getTable() const { return results_; }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

  const std::vector<PushedDownFilterInfo>& getPushedDownFilterInfo() const;

  const bool isFilterPushDownEnabled() const { return filter_push_down_enabled_; }

  void setQueueTime(const int64_t queue_time_ms) {
    CHECK(!results_.empty());
    results_[0]->setQueueTime(queue_time_ms);
  }

 private:
  TemporaryTable results_;
  std::vector<TargetMetaInfo> targets_meta_;
  // filters chosen to be pushed down
  std::vector<PushedDownFilterInfo> pushed_down_filter_info_;
  // whether or not it was allowed to look for filters to push down
  bool filter_push_down_enabled_;
};

class RelAlgNode;

class RaExecutionDesc {
 public:
  RaExecutionDesc(const RelAlgNode* body)
      : body_(body)
      , result_(std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                            ExecutorDeviceType::CPU,
                                            QueryMemoryDescriptor(),
                                            nullptr,
                                            nullptr),
                {}) {}

  const ExecutionResult& getResult() const { return result_; }

  void setResult(const ExecutionResult& result);

  const RelAlgNode* getBody() const;

 private:
  const RelAlgNode* body_;
  ExecutionResult result_;
};

using DAG = boost::
    adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS, const RelAlgNode*>;
using Vertex = DAG::vertex_descriptor;

/**
 * @brief A container for relational algebra descriptors defining the execution order
 * for a relational algebra query.
 * Holds the relational algebra descriptors for executing a relational algebra query. Each
 * descriptor holds both a top-level relational algebra node and a ResultSet ptr holding
 * the results from the execution of the accompany node(s). The sequence can be generated
 * on initialization or lazily with calls to the next() operator.
 */
class RaExecutionSequence {
 public:
  RaExecutionSequence(const RelAlgNode*, const bool build_sequence = true);
  RaExecutionSequence(std::unique_ptr<RaExecutionDesc> exec_desc);

  /**
   * Return the next execution descriptor in the sequence. If no more execution
   * descriptors exist, returns nullptr.
   */
  RaExecutionDesc* next();

  /**
   * Returns the index of the next execution descriptor in the graph. If after_broadcast
   * is true, returns the index of the first execution descriptor after the next global
   * broadcast. Returns -1 if no execution descriptors remain in the graph.
   */
  ssize_t nextStepId(const bool after_broadcast) const;

  bool executionFinished() const;

  RaExecutionDesc* getDescriptor(size_t idx) const {
    CHECK_LT(idx, descs_.size());
    return descs_[idx].get();
  }

  size_t size() const { return descs_.size(); }
  bool empty() const { return descs_.empty(); }

  size_t totalDescriptorsCount() const;

  bool hasTableFunctions() const { return table_functions_ > 0; }

 private:
  DAG graph_;

  std::unordered_set<Vertex> joins_;
  std::vector<Vertex> ordering_;  // reverse order topological sort of graph_
  size_t current_vertex_ = 0;
  size_t scan_count_ = 0;
  size_t table_functions_ = 0;

  /**
   * Starting from the current vertex, iterate the graph counting the number of execution
   * descriptors remaining before the next required broadcast step. The current vertex is
   * counted as the first step before a broadcast is required; i.e. a return value of 0
   * indicates no additional steps in the graph can be executed without a global
   * broadcast, and a return value of 2 indicates the current vertex and both subsequent
   * vertices can be executed before a global broacast is needed.
   */
  size_t stepsToNextBroadcast() const;

  // The execution descriptors hold the pointers to their results. We need to push them
  // back into this vector as they are created, so we don't lose the intermediate results
  // later.
  std::vector<std::unique_ptr<RaExecutionDesc>> descs_;
};
