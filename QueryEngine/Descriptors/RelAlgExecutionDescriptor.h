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
#include "Shared/toString.h"

class ResultSet;

class ExecutionResult {
 public:
  ExecutionResult();

  ExecutionResult(const std::shared_ptr<ResultSet>& rows,
                  const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(ResultSetPtr&& result, const std::vector<TargetMetaInfo>& targets_meta);

  ExecutionResult(const ExecutionResult& that);

  ExecutionResult(ExecutionResult&& that);

  ExecutionResult(const std::vector<PushedDownFilterInfo>& pushed_down_filter_info,
                  bool filter_push_down_enabled);

  ExecutionResult& operator=(const ExecutionResult& that);

  const std::shared_ptr<ResultSet>& getRows() const { return result_; }

  bool empty() const { return !result_; }

  const ResultSetPtr& getDataPtr() const { return result_; }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

  const std::vector<PushedDownFilterInfo>& getPushedDownFilterInfo() const;

  const bool isFilterPushDownEnabled() const { return filter_push_down_enabled_; }

  void setQueueTime(const int64_t queue_time_ms) {
    CHECK(result_);
    result_->setQueueTime(queue_time_ms);
  }

  std::string toString() const {
    return ::typeName(this) + "(" + ::toString(result_) + ", " +
           ::toString(targets_meta_) + ")";
  }

  enum RType { QueryResult, SimpleResult, Explaination, CalciteDdl };

  std::string getExplanation();
  void updateResultSet(const std::string& query_ra, RType type, bool success = true);
  RType getResultType() const { return type_; }
  void setResultType(RType type) { type_ = type; }
  int64_t getExecutionTime() const { return execution_time_ms_; }
  void setExecutionTime(int64_t execution_time_ms) {
    execution_time_ms_ = execution_time_ms;
  }
  void addExecutionTime(int64_t execution_time_ms) {
    execution_time_ms_ += execution_time_ms;
  }

 private:
  ResultSetPtr result_;
  std::vector<TargetMetaInfo> targets_meta_;
  // filters chosen to be pushed down
  std::vector<PushedDownFilterInfo> pushed_down_filter_info_;
  // whether or not it was allowed to look for filters to push down
  bool filter_push_down_enabled_;

  bool success_;
  uint64_t execution_time_ms_;
  RType type_;
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
                                            nullptr,
                                            0,
                                            0),
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
   * Return the previous execution descriptor in the sequence. If the sequence is made up
   * of 0 or 1 descriptors, returns nullptr.
   */
  RaExecutionDesc* prev();

  /**
   * Returns the index of the next execution descriptor in the graph. If after_broadcast
   * is true, returns the index of the first execution descriptor after the next global
   * broadcast. Returns std::nullopt if no execution descriptors remain in the graph.
   */
  std::optional<size_t> nextStepId(const bool after_broadcast) const;

  bool executionFinished() const;

  RaExecutionDesc* getDescriptor(size_t idx) const {
    CHECK_LT(idx, descs_.size());
    return descs_[idx].get();
  }

  size_t size() const { return descs_.size(); }
  bool empty() const { return descs_.empty(); }

  size_t totalDescriptorsCount() const;

 private:
  DAG graph_;

  std::unordered_set<Vertex> joins_;
  std::vector<Vertex> ordering_;  // reverse order topological sort of graph_
  size_t current_vertex_ = 0;
  size_t scan_count_ = 0;

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
