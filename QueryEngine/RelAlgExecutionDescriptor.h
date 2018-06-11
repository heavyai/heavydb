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

  ExecutionResult(const IteratorTable& table,
                  const std::vector<TargetMetaInfo>& targets_meta)
      : result_(boost::make_unique<IteratorTable>(table))
      , targets_meta_(targets_meta)
      , filter_push_down_enabled_(false) {}

  ExecutionResult(ResultPtr&& result, const std::vector<TargetMetaInfo>& targets_meta)
      : targets_meta_(targets_meta), filter_push_down_enabled_(false) {
    if (auto rows = boost::get<RowSetPtr>(&result)) {
      result_ = std::move(*rows);
      CHECK(boost::get<RowSetPtr>(result_));
    } else if (auto tab = boost::get<IterTabPtr>(&result)) {
      result_ = std::move(*tab);
      CHECK(boost::get<IterTabPtr>(result_));
    } else {
      CHECK(false);
    }
  }

  ExecutionResult(const ExecutionResult& that)
      : targets_meta_(that.targets_meta_)
      , pushed_down_filter_info_(that.pushed_down_filter_info_)
      , filter_push_down_enabled_(that.filter_push_down_enabled_) {
    if (!pushed_down_filter_info_.empty() ||
        (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
      return;
    }
    if (const auto rows = boost::get<RowSetPtr>(&that.result_)) {
      CHECK(*rows);
      result_ = *rows;
      CHECK(boost::get<RowSetPtr>(result_));
    } else if (const auto tab = boost::get<IterTabPtr>(&that.result_)) {
      CHECK(*tab);
      result_ = boost::make_unique<IteratorTable>(**tab);
      CHECK(boost::get<IterTabPtr>(result_));
    } else {
      CHECK(false);
    }
  }

  ExecutionResult(ExecutionResult&& that)
      : targets_meta_(std::move(that.targets_meta_))
      , pushed_down_filter_info_(std::move(that.pushed_down_filter_info_))
      , filter_push_down_enabled_(std::move(that.filter_push_down_enabled_)) {
    if (!pushed_down_filter_info_.empty() ||
        (filter_push_down_enabled_ && pushed_down_filter_info_.empty())) {
      return;
    }
    if (auto rows = boost::get<RowSetPtr>(&that.result_)) {
      result_ = std::move(*rows);
      CHECK(boost::get<RowSetPtr>(result_));
    } else if (auto tab = boost::get<IterTabPtr>(&that.result_)) {
      result_ = std::move(*tab);
      CHECK(boost::get<IterTabPtr>(result_));
    } else {
      CHECK(false);
    }
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
    if (const auto rows = boost::get<RowSetPtr>(&that.result_)) {
      CHECK(*rows);
      result_ = *rows;
      CHECK(boost::get<RowSetPtr>(result_));
    } else if (const auto tab = boost::get<IterTabPtr>(&that.result_)) {
      CHECK(*tab);
      result_ = boost::make_unique<IteratorTable>(**tab);
      CHECK(boost::get<IterTabPtr>(result_));
    } else {
      CHECK(false);
    }
    targets_meta_ = that.targets_meta_;
    return *this;
  }

  const std::shared_ptr<ResultSet>& getRows() const {
    auto& rows = boost::get<RowSetPtr>(result_);
    CHECK(rows);
    return rows;
  }

  bool empty() const {
    if (auto rows = boost::get<RowSetPtr>(&result_)) {
      return !*rows;
    } else if (auto tab = boost::get<IterTabPtr>(&result_)) {
      return !*tab;
    } else {
      CHECK(false);
    }
    return true;
  }

  const ResultPtr& getDataPtr() const { return result_; }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

  const std::vector<PushedDownFilterInfo>& getPushedDownFilterInfo() const {
    return pushed_down_filter_info_;
  }

  const bool isFilterPushDownEnabled() const { return filter_push_down_enabled_; }

  void setQueueTime(const int64_t queue_time_ms) {
    if (auto rows = boost::get<RowSetPtr>(&result_)) {
      CHECK(*rows);
      (*rows)->setQueueTime(queue_time_ms);
    }
  }

 private:
  ResultPtr result_;
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
