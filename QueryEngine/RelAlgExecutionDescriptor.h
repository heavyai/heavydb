#ifndef QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H
#define QUERYENGINE_RELALGEXECUTIONDESCRIPTOR_H

#include "GroupByAndAggregate.h"
#include "RelAlgAbstractInterpreter.h"

class ResultRows;

class ExecutionResult {
 public:
  ExecutionResult(const ResultRows& rows, const std::vector<TargetMetaInfo>& targets_meta)
      : result_(boost::make_unique<ResultRows>(rows)), targets_meta_(targets_meta) {}

  ExecutionResult(const IteratorTable& table, const std::vector<TargetMetaInfo>& targets_meta)
      : result_(boost::make_unique<IteratorTable>(table)), targets_meta_(targets_meta) {}

  ExecutionResult(ResultPtr&& result, const std::vector<TargetMetaInfo>& targets_meta) : targets_meta_(targets_meta) {
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

  ExecutionResult(const ExecutionResult& that) : targets_meta_(that.targets_meta_) {
    if (const auto rows = boost::get<RowSetPtr>(&that.result_)) {
      CHECK(*rows);
      result_ = boost::make_unique<ResultRows>(**rows);
      CHECK(boost::get<RowSetPtr>(result_));
    } else if (const auto tab = boost::get<IterTabPtr>(&that.result_)) {
      CHECK(*tab);
      result_ = boost::make_unique<IteratorTable>(**tab);
      CHECK(boost::get<IterTabPtr>(result_));
    } else {
      CHECK(false);
    }
  }

  ExecutionResult(ExecutionResult&& that) : targets_meta_(std::move(that.targets_meta_)) {
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

  ExecutionResult& operator=(const ExecutionResult& that) {
    if (const auto rows = boost::get<RowSetPtr>(&that.result_)) {
      CHECK(*rows);
      result_ = boost::make_unique<ResultRows>(**rows);
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

  const ResultRows& getRows() const {
    auto& rows = boost::get<RowSetPtr>(result_);
    CHECK(rows);
    return *rows;
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

  void setQueueTime(const int64_t queue_time_ms) {
    if (auto rows = boost::get<RowSetPtr>(&result_)) {
      CHECK(*rows);
      (*rows)->setQueueTime(queue_time_ms);
    }
  }

 private:
  ResultPtr result_;
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
