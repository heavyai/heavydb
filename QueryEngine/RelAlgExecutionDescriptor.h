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

#ifndef QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
#define QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H

#include "GroupByAndAggregate.h"
#include "JoinFilterPushDown.h"
#include "RelAlgAbstractInterpreter.h"

class ResultSet;

class ExecutionResult {
 public:
  ExecutionResult(const std::shared_ptr<ResultSet>& rows,
                  const std::vector<TargetMetaInfo>& targets_meta)
      : result_(rows), targets_meta_(targets_meta), filter_push_down_enabled_(false) {}

  ExecutionResult(ResultSetPtr&& result, const std::vector<TargetMetaInfo>& targets_meta)
      : targets_meta_(targets_meta), filter_push_down_enabled_(false) {
    result_ = std::move(result);
  }

  ExecutionResult(const ExecutionResult& that)
      : targets_meta_(that.targets_meta_)
      , pushed_down_filter_info_(that.pushed_down_filter_info_)
      , filter_push_down_enabled_(that.filter_push_down_enabled_) {
    if (!pushed_down_filter_info_.empty() ||
        (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
      return;
    }
    result_ = that.result_;
  }

  ExecutionResult(ExecutionResult&& that)
      : targets_meta_(std::move(that.targets_meta_))
      , pushed_down_filter_info_(std::move(that.pushed_down_filter_info_))
      , filter_push_down_enabled_(std::move(that.filter_push_down_enabled_)) {
    if (!pushed_down_filter_info_.empty() ||
        (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
      return;
    }
    result_ = std::move(that.result_);
  }

  ExecutionResult(const std::vector<PushedDownFilterInfo>& pushed_down_filter_info,
                  bool filter_push_down_enabled)
      : pushed_down_filter_info_(pushed_down_filter_info)
      , filter_push_down_enabled_(filter_push_down_enabled) {}

  ExecutionResult& operator=(const ExecutionResult& that) {
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

  const std::shared_ptr<ResultSet>& getRows() const { return result_; }

  bool empty() const { return !result_; }

  const ResultSetPtr& getDataPtr() const { return result_; }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

  const std::vector<PushedDownFilterInfo>& getPushedDownFilterInfo() const {
    return pushed_down_filter_info_;
  }

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

  void setResult(const ExecutionResult& result) {
    result_ = result;
    body_->setContextData(this);
  }

  const RelAlgNode* getBody() const { return body_; }

 private:
  const RelAlgNode* body_;
  ExecutionResult result_;
};

std::vector<RaExecutionDesc> get_execution_descriptors(const RelAlgNode*);
std::vector<RaExecutionDesc> get_execution_descriptors(
    const std::vector<const RelAlgNode*>&);

#endif  // QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
