/**
 * @file    SpeculativeTopN.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Speculative top N algorithm.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef QUERYENGINE_SPECULATIVETOPN_H
#define QUERYENGINE_SPECULATIVETOPN_H

#include <cstddef>
#include <cstdint>
#include <memory>
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
struct QueryMemoryDescriptor;
class ResultRows;
struct RelAlgExecutionUnit;
class RowSetMemoryOwner;
namespace Analyzer {
class Expr;
}  // Analyzer

class SpeculativeTopNMap {
 public:
  SpeculativeTopNMap();

  SpeculativeTopNMap(const ResultRows& rows, const std::vector<Analyzer::Expr*>& target_exprs, const size_t truncate_n);

  void reduce(SpeculativeTopNMap& that);

  ResultRows asRows(const RelAlgExecutionUnit& ra_exe_unit,
                    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                    const QueryMemoryDescriptor& query_mem_desc,
                    const std::vector<int64_t>& init_agg_vals,  // TODO(alex): needed?
                    const Executor* executor,                   // TODO(alex): needed?
                    const size_t top_n,
                    const bool desc) const;

 private:
  std::unordered_map<int64_t, SpeculativeTopNVal> map_;
  size_t unknown_;
};

class SpeculativeTopNFailed : public std::runtime_error {
 public:
  SpeculativeTopNFailed() : std::runtime_error("SpeculativeTopNFailed"){};
};

class SpeculativeTopNBlacklist {
 public:
  void add(const std::shared_ptr<Analyzer::Expr> expr, const bool desc);
  bool contains(const std::shared_ptr<Analyzer::Expr> expr, const bool desc) const;

 private:
  std::vector<std::pair<std::shared_ptr<Analyzer::Expr>, bool>> blacklist_;
};

bool use_speculative_top_n(const RelAlgExecutionUnit&, const QueryMemoryDescriptor&);

#endif  // QUERYENGINE_SPECULATIVETOPN_H
