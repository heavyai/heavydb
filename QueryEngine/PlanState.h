/*
 * Copyright 2019 OmniSci, Inc.
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

#pragma once

#include <unordered_set>

#include "Analyzer/Analyzer.h"
#include "QueryEngine/Descriptors/InputDescriptors.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"

class Executor;

struct JoinInfo {
  JoinInfo(const std::vector<std::shared_ptr<Analyzer::BinOper>>& equi_join_tautologies,
           const std::vector<std::shared_ptr<HashJoin>>& join_hash_tables)
      : equi_join_tautologies_(equi_join_tautologies)
      , join_hash_tables_(join_hash_tables) {}

  std::vector<std::shared_ptr<Analyzer::BinOper>>
      equi_join_tautologies_;  // expressions we equi-join on are true by
                               // definition when using a hash join; we'll
                               // fold them to true during code generation
  std::vector<std::shared_ptr<HashJoin>> join_hash_tables_;
  std::unordered_set<size_t> sharded_range_table_indices_;
};

struct PlanState {
  using TableId = int;
  using ColumnId = int;
  using HoistedFiltersSet = std::unordered_set<std::shared_ptr<Analyzer::Expr>>;

  struct CompareInputColDescId {
    bool operator()(const InputColDescriptor& lhs, const InputColDescriptor& rhs) const {
      return std::make_pair(lhs.getTableId(), lhs.getColId()) <
             std::make_pair(rhs.getTableId(), rhs.getColId());
    }
  };

  using InputColDescriptorSet = std::set<InputColDescriptor, CompareInputColDescId>;

  PlanState(const bool allow_lazy_fetch,
            const std::vector<InputTableInfo>& query_infos,
            const Executor* executor)
      : allow_lazy_fetch_(allow_lazy_fetch)
      , join_info_({std::vector<std::shared_ptr<Analyzer::BinOper>>{}, {}})
      , query_infos_(query_infos)
      , executor_(executor) {}

  std::vector<int64_t> init_agg_vals_;
  std::vector<Analyzer::Expr*> target_exprs_;
  HoistedFiltersSet hoisted_filters_;
  std::unordered_map<InputColDescriptor, size_t> global_to_local_col_ids_;
  InputColDescriptorSet columns_to_fetch_;
  InputColDescriptorSet columns_to_not_fetch_;
  std::unordered_map<size_t, std::vector<std::shared_ptr<Analyzer::Expr>>>
      left_join_non_hashtable_quals_;
  bool allow_lazy_fetch_;
  JoinInfo join_info_;
  const std::vector<InputTableInfo>& query_infos_;
  std::list<std::shared_ptr<Analyzer::Expr>> simple_quals_;
  const Executor* executor_;

  void allocateLocalColumnIds(
      const std::list<std::shared_ptr<const InputColDescriptor>>& global_col_ids);

  int getLocalColumnId(const Analyzer::ColumnVar* col_var, const bool fetch_column);

  bool isLazyFetchColumn(const Analyzer::Expr* target_expr) const;

  bool isLazyFetchColumn(const InputColDescriptor& col_desc) {
    Analyzer::ColumnVar column(col_desc.getColInfo(), col_desc.getNestLevel());
    return isLazyFetchColumn(&column);
  }

  void addSimpleQual(std::shared_ptr<Analyzer::Expr> simple_qual) {
    simple_quals_.push_back(simple_qual);
  }

  std::list<std::shared_ptr<Analyzer::Expr>> getSimpleQuals() const {
    return simple_quals_;
  }

  void addNonHashtableQualForLeftJoin(size_t idx, std::shared_ptr<Analyzer::Expr> expr);
};
