#ifndef QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
#define QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H

#include "GroupByAndAggregate.h"
#include "RelAlgAbstractInterpreter.h"

class ResultRows;

class ExecutionResult {
 public:
  ExecutionResult(const ResultRows& rows, const std::vector<TargetMetaInfo>& targets_meta)
      : rows_(rows), targets_meta_(targets_meta) {}

  const ResultRows& getRows() const { return rows_; }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

  void setQueueTime(const int64_t queue_time_ms) { rows_.setQueueTime(queue_time_ms); }

 private:
  ResultRows rows_;
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
