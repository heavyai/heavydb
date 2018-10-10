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

#include "SpeculativeTopN.h"

#include "RelAlgExecutor.h"
#include "ResultSet.h"

#include <glog/logging.h>

SpeculativeTopNMap::SpeculativeTopNMap() : unknown_(0) {}

SpeculativeTopNMap::SpeculativeTopNMap(const ResultSet& rows,
                                       const std::vector<Analyzer::Expr*>& target_exprs,
                                       const size_t truncate_n)
    : unknown_(0) {
  CHECK_EQ(rows.colCount(), target_exprs.size());
  const bool count_first = dynamic_cast<const Analyzer::AggExpr*>(target_exprs[0]);
  for (size_t i = 0; i < truncate_n + 1; ++i) {
    const auto crt_row = rows.getNextRow(false, false);
    if (crt_row.empty()) {
      break;
    }
    int64_t key{0};
    size_t val{0};
    CHECK_EQ(rows.colCount(), crt_row.size());
    {
      auto scalar_r = boost::get<ScalarTargetValue>(&crt_row[0]);
      CHECK(scalar_r);
      auto p = boost::get<int64_t>(scalar_r);
      CHECK(p);
      if (count_first) {
        val = *p;
      } else {
        key = *p;
      }
    }
    {
      auto scalar_r = boost::get<ScalarTargetValue>(&crt_row[1]);
      CHECK(scalar_r);
      auto p = boost::get<int64_t>(scalar_r);
      CHECK(p);
      if (count_first) {
        key = *p;
      } else {
        val = *p;
      }
    }
    if (i < truncate_n) {
      const auto it_ok = map_.emplace(key, SpeculativeTopNVal{val, false});
      CHECK(it_ok.second);
    } else {
      unknown_ = val;
    }
  }
}

void SpeculativeTopNMap::reduce(SpeculativeTopNMap& that) {
  for (auto& kv : map_) {
    auto& this_entry = kv.second;
    const auto that_it = that.map_.find(kv.first);
    if (that_it != that.map_.end()) {
      const auto& that_entry = that_it->second;
      CHECK(!that_entry.unknown);
      this_entry.val += that_entry.val;
      that.map_.erase(that_it);
    } else {
      this_entry.val += that.unknown_;
      this_entry.unknown = that.unknown_;
    }
  }
  for (const auto& kv : that.map_) {
    const auto it_ok = map_.emplace(
        kv.first, SpeculativeTopNVal{kv.second.val + unknown_, unknown_ != 0});
    CHECK(it_ok.second);
  }
  unknown_ += that.unknown_;
}

std::shared_ptr<ResultSet> SpeculativeTopNMap::asRows(
    const RelAlgExecutionUnit& ra_exe_unit,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const QueryMemoryDescriptor& query_mem_desc,
    const Executor* executor,
    const size_t top_n,
    const bool desc) const {
  std::vector<SpeculativeTopNEntry> vec;
  for (const auto& kv : map_) {
    vec.emplace_back(SpeculativeTopNEntry{kv.first, kv.second.val, kv.second.unknown});
  }
  if (desc) {
    std::sort(vec.begin(), vec.end(), std::greater<SpeculativeTopNEntry>());
  } else {
    std::sort(vec.begin(), vec.end());
  }
  const auto num_rows = std::min(top_n, vec.size());
  for (size_t i = 0; i < num_rows; ++i) {
    if (vec[i].unknown) {
      throw SpeculativeTopNFailed();
    }
  }
  CHECK_EQ(size_t(2), ra_exe_unit.target_exprs.size());
  auto query_mem_desc_rs = query_mem_desc;
  query_mem_desc_rs.setQueryDescriptionType(QueryDescriptionType::GroupByBaselineHash);
  query_mem_desc_rs.setOutputColumnar(false);
  query_mem_desc_rs.setEntryCount(num_rows);
  query_mem_desc_rs.clearAggColWidths();
  query_mem_desc_rs.addAggColWidth({8, 8});
  query_mem_desc_rs.addAggColWidth({8, 8});
  auto rs = std::make_shared<ResultSet>(
      target_exprs_to_infos(ra_exe_unit.target_exprs, query_mem_desc_rs),
      ExecutorDeviceType::CPU,
      query_mem_desc_rs,
      row_set_mem_owner,
      executor);
  auto rs_storage = rs->allocateStorage();
  auto rs_buff = reinterpret_cast<int64_t*>(rs_storage->getUnderlyingBuffer());
  const bool count_first =
      dynamic_cast<const Analyzer::AggExpr*>(ra_exe_unit.target_exprs[0]);
  for (size_t i = 0; i < num_rows; ++i) {
    rs_buff[0] = vec[i].key;
    int64_t col0 = vec[i].key;
    int64_t col1 = vec[i].val;
    if (count_first) {
      std::swap(col0, col1);
    }
    rs_buff[1] = col0;
    rs_buff[2] = col1;
    rs_buff += 3;
  }
  return rs;
}

void SpeculativeTopNBlacklist::add(const std::shared_ptr<Analyzer::Expr> expr,
                                   const bool desc) {
  for (const auto e : blacklist_) {
    CHECK(!(*e.first == *expr) || e.second != desc);
  }
  blacklist_.emplace_back(expr, desc);
}

bool SpeculativeTopNBlacklist::contains(const std::shared_ptr<Analyzer::Expr> expr,
                                        const bool desc) const {
  for (const auto e : blacklist_) {
    if (*e.first == *expr && e.second == desc) {
      return true;
    }
  }
  return false;
}

bool use_speculative_top_n(const RelAlgExecutionUnit& ra_exe_unit,
                           const QueryMemoryDescriptor& query_mem_desc) {
  if (g_cluster) {
    return false;
  }
  if (ra_exe_unit.target_exprs.size() != 2) {
    return false;
  }
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
    if (agg_expr && agg_expr->get_aggtype() != kCOUNT) {
      return false;
    }
  }
  return query_mem_desc.sortOnGpu() && ra_exe_unit.sort_info.limit &&
         ra_exe_unit.sort_info.algorithm == SortAlgorithm::SpeculativeTopN;
}
