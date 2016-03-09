#ifndef QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
#define QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H

#include "GroupByAndAggregate.h"
#include "RelAlgAbstractInterpreter.h"

class ResultRows;

class ForLoop {
 public:
  ForLoop(const RelAlgNode* node) : node_(node) {}

 private:
  const RelAlgNode* node_;
};

class ExecutionResult {
 public:
  ExecutionResult(const ResultRows& rows, const std::vector<TargetMetaInfo>& targets_meta)
      : rows_(rows), targets_meta_(targets_meta) {}

  const ResultRows& getRows() const { return rows_; }

  const std::vector<TargetMetaInfo>& getTargetsMeta() const { return targets_meta_; }

 private:
  ResultRows rows_;
  std::vector<TargetMetaInfo> targets_meta_;
};

class RaExecutionDesc {
 public:
  RaExecutionDesc(const std::vector<ForLoop>& for_loops, const RelAlgNode* body)
      : for_loops_(for_loops), body_(body), result_({{}, {}, nullptr, nullptr, ExecutorDeviceType::CPU}, {}) {}

  const ExecutionResult& getResult() const { return result_; }

  void setResult(const ExecutionResult& result) {
    result_ = result;
    body_->setContextData(this);
  }

  const RelAlgNode* getBody() const { return body_; }

 private:
  const std::vector<ForLoop> for_loops_;
  const RelAlgNode* body_;
  ExecutionResult result_;
};

std::vector<RaExecutionDesc> get_execution_descriptors(const RelAlgNode*);

#endif  // QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
