/*
 * Copyright 2022 HEAVY.AI, Inc.
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

/**
 * @file    SpeculativeTopN.h
 * @brief   Speculative top N algorithm.
 *
 */

#ifndef QUERYENGINE_SPECULATIVETOPN_H
#define QUERYENGINE_SPECULATIVETOPN_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct SpeculativeTopNVal {
  size_t val;
  bool unknown;
};

struct SpeculativeTopNEntry {
  int64_t key;
  size_t val;
  bool unknown;

  bool operator<(const SpeculativeTopNEntry& that) const { return val < that.val; }
  bool operator>(const SpeculativeTopNEntry& that) const { return val > that.val; }
};

class Executor;
class QueryMemoryDescriptor;
class ResultSet;
struct RelAlgExecutionUnit;
class RowSetMemoryOwner;
namespace Analyzer {
class Expr;
}  // namespace Analyzer

class SpeculativeTopNMap {
 public:
  SpeculativeTopNMap();

  SpeculativeTopNMap(const ResultSet& rows,
                     const std::vector<Analyzer::Expr*>& target_exprs,
                     const size_t truncate_n);

  void reduce(SpeculativeTopNMap& that);

  std::shared_ptr<ResultSet> asRows(const RelAlgExecutionUnit& ra_exe_unit,
                                    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                    const QueryMemoryDescriptor& query_mem_desc,
                                    const Executor* executor,
                                    const size_t top_n,
                                    const bool desc) const;

 private:
  std::unordered_map<int64_t, SpeculativeTopNVal> map_;
  size_t unknown_;
};

class SpeculativeTopNFailed : public std::runtime_error {
 public:
  SpeculativeTopNFailed(const std::string& msg)
      : std::runtime_error("SpeculativeTopNFailed: " + msg)
      , failed_during_iteration_(false) {}

  SpeculativeTopNFailed()
      : std::runtime_error("SpeculativeTopNFailed"), failed_during_iteration_(true) {}

  bool failedDuringIteration() const { return failed_during_iteration_; }

 private:
  bool failed_during_iteration_;
};

class SpeculativeTopNBlacklist {
 public:
  void add(const std::shared_ptr<Analyzer::Expr> expr, const bool desc);
  bool contains(const std::shared_ptr<Analyzer::Expr> expr, const bool desc) const;

 private:
  mutable std::mutex mutex_;
  std::vector<std::pair<std::shared_ptr<Analyzer::Expr>, bool>> blacklist_;
};

bool use_speculative_top_n(const RelAlgExecutionUnit&, const QueryMemoryDescriptor&);

#endif  // QUERYENGINE_SPECULATIVETOPN_H
