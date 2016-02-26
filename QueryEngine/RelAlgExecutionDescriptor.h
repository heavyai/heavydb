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

class RaExecutionDesc {
 public:
  RaExecutionDesc(const std::vector<ForLoop>& for_loops, const RelAlgNode* body)
      : for_loops_(for_loops), body_(body), result_({}, nullptr, nullptr, ExecutorDeviceType::CPU) {}

  const ResultRows& getResult() const { return result_; }

  void setResult(const ResultRows& result) { result_ = result; }

  const RelAlgNode* getBody() const { return body_; }

 private:
  const std::vector<ForLoop> for_loops_;
  const RelAlgNode* body_;
  ResultRows result_;
};

std::list<RaExecutionDesc> get_execution_descriptors(const RelAlgNode*);

#endif  // QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
