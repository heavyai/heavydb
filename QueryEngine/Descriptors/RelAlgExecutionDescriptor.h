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

#include "../GroupByAndAggregate.h"
#include "../JoinFilterPushDown.h"

class ResultSet;

class ExecutionResult {
 public:
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

 private:
  ResultSetPtr result_;
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

std::vector<RaExecutionDesc> get_execution_descriptors(const RelAlgNode*);
std::vector<RaExecutionDesc> get_execution_descriptors(
    const std::vector<const RelAlgNode*>&);
