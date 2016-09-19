#ifndef QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
#define QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H

#include "GroupByAndAggregate.h"
#include "RelAlgAbstractInterpreter.h"

class ResultRows;

class ExecutionResult {
 public:
  ExecutionResult(const ResultRows& rows, const std::vector<TargetMetaInfo>& targets_meta)
      : rows_(boost::make_unique<ResultRows>(rows)), targets_meta_(targets_meta) {}

  ExecutionResult(RowSetPtr&& rows, const std::vector<TargetMetaInfo>& targets_meta)
      : rows_(std::move(rows)), targets_meta_(targets_meta) {
    CHECK(rows_);
  }

  ExecutionResult(const ExecutionResult& that)
      : rows_(boost::make_unique<ResultRows>(*that.rows_)), targets_meta_(that.targets_meta_) {}

  ExecutionResult(ExecutionResult&& that)
      : rows_(std::move(that.rows_)), targets_meta_(std::move(that.targets_meta_)) {}

  ExecutionResult& operator=(const ExecutionResult& that) {
    rows_ = boost::make_unique<ResultRows>(*that.rows_);
    CHECK(rows_);
    targets_meta_ = that.targets_meta_;
    return *this;
  }

  const ResultRows& getRows() const {
    CHECK(rows_);
    return *rows_;
  }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

  void setQueueTime(const int64_t queue_time_ms) { rows_->setQueueTime(queue_time_ms); }

 private:
  RowSetPtr rows_;
  std::vector<TargetMetaInfo> targets_meta_;
};

class RaExecutionDesc {
 public:
  RaExecutionDesc(const RelAlgNode* body)
      : body_(body), result_({{}, {}, nullptr, nullptr, {}, ExecutorDeviceType::CPU}, {}) {}

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

#endif  // QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
