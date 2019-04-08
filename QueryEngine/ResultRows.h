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

/*
 * @file    ResultRows.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Created on May 9, 2016, 3:45 PM
 */

#ifndef QUERYENGINE_RESULTROWS_H
#define QUERYENGINE_RESULTROWS_H

#include "HyperLogLog.h"
#include "OutputBufferInitialization.h"
#include "QueryMemoryDescriptor.h"
#include "TargetValue.h"

#include "../Analyzer/Analyzer.h"
#include "../Shared/TargetInfo.h"
#include "../StringDictionary/StringDictionaryProxy.h"

#include <glog/logging.h>
#include <boost/noncopyable.hpp>

#include <limits>
#include <list>
#include <mutex>
#include <set>
#include <unordered_set>

extern bool g_bigint_count;

class QueryMemoryDescriptor;
struct RelAlgExecutionUnit;
class RowSetMemoryOwner;

// The legacy way of representing result sets. Don't change it, it's going away.

inline int64_t get_component(const int8_t* group_by_buffer,
                             const size_t comp_sz,
                             const size_t index = 0) {
  int64_t ret = std::numeric_limits<int64_t>::min();
  switch (comp_sz) {
    case 1: {
      ret = group_by_buffer[index];
      break;
    }
    case 2: {
      const int16_t* buffer_ptr = reinterpret_cast<const int16_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    case 4: {
      const int32_t* buffer_ptr = reinterpret_cast<const int32_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    case 8: {
      const int64_t* buffer_ptr = reinterpret_cast<const int64_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    default:
      CHECK(false);
  }
  return ret;
}

inline int64_t get_consistent_frag_size(const std::vector<uint64_t>& frag_offsets) {
  if (frag_offsets.size() < 2) {
    return ssize_t(-1);
  }
  const auto frag_size = frag_offsets[1] - frag_offsets[0];
  for (size_t i = 2; i < frag_offsets.size(); ++i) {
    const auto curr_size = frag_offsets[i] - frag_offsets[i - 1];
    if (curr_size != frag_size) {
      return int64_t(-1);
    }
  }
  return !frag_size ? std::numeric_limits<int64_t>::max()
                    : static_cast<int64_t>(frag_size);
}

inline std::pair<int64_t, int64_t> get_frag_id_and_local_idx(
    const std::vector<uint64_t>& frag_offsets,
    const int64_t global_idx) {
  CHECK_GE(global_idx, int64_t(0));
  for (int64_t frag_id = frag_offsets.size() - 1; frag_id >= 0; --frag_id) {
    const auto frag_off = static_cast<int64_t>(frag_offsets[frag_id]);
    if (frag_off <= global_idx) {
      return {frag_id, global_idx - frag_off};
    }
  }
  return {-1, -1};
}

inline std::vector<int64_t> get_consistent_frags_sizes(
    const std::vector<std::vector<uint64_t>>& frag_offsets) {
  if (frag_offsets.empty()) {
    return {};
  }
  std::vector<int64_t> frag_sizes;
  for (size_t tab_idx = 0; tab_idx < frag_offsets[0].size(); ++tab_idx) {
    std::vector<uint64_t> tab_offs;
    for (auto& offsets : frag_offsets) {
      tab_offs.push_back(offsets[tab_idx]);
    }
    frag_sizes.push_back(get_consistent_frag_size(tab_offs));
  }
  return frag_sizes;
}

template <typename T>
inline std::pair<int64_t, int64_t> get_frag_id_and_local_idx(
    const std::vector<std::vector<T>>& frag_offsets,
    const size_t tab_or_col_idx,
    const int64_t global_idx) {
  CHECK_GE(global_idx, int64_t(0));
  for (int64_t frag_id = frag_offsets.size() - 1; frag_id > 0; --frag_id) {
    CHECK_LT(tab_or_col_idx, frag_offsets[frag_id].size());
    const auto frag_off = static_cast<int64_t>(frag_offsets[frag_id][tab_or_col_idx]);
    if (frag_off < global_idx) {
      return {frag_id, global_idx - frag_off};
    }
  }
  return {-1, -1};
}

inline std::vector<int64_t> get_consistent_frags_sizes(
    const std::vector<Analyzer::Expr*>& target_exprs,
    const std::vector<int64_t>& table_frag_sizes) {
  std::vector<int64_t> col_frag_sizes;
  for (auto expr : target_exprs) {
    if (const auto col_var = dynamic_cast<Analyzer::ColumnVar*>(expr)) {
      if (col_var->get_rte_idx() < 0) {
        CHECK_EQ(-1, col_var->get_rte_idx());
        col_frag_sizes.push_back(int64_t(-1));
      } else {
        col_frag_sizes.push_back(table_frag_sizes[col_var->get_rte_idx()]);
      }
    } else {
      col_frag_sizes.push_back(int64_t(-1));
    }
  }
  return col_frag_sizes;
}

inline std::vector<std::vector<int64_t>> get_col_frag_offsets(
    const std::vector<Analyzer::Expr*>& target_exprs,
    const std::vector<std::vector<uint64_t>>& table_frag_offsets) {
  std::vector<std::vector<int64_t>> col_frag_offsets;
  for (auto& table_offsets : table_frag_offsets) {
    std::vector<int64_t> col_offsets;
    for (auto expr : target_exprs) {
      if (const auto col_var = dynamic_cast<Analyzer::ColumnVar*>(expr)) {
        if (col_var->get_rte_idx() < 0) {
          CHECK_EQ(-1, col_var->get_rte_idx());
          col_offsets.push_back(int64_t(-1));
        } else {
          CHECK_LT(static_cast<size_t>(col_var->get_rte_idx()), table_offsets.size());
          col_offsets.push_back(
              static_cast<int64_t>(table_offsets[col_var->get_rte_idx()]));
        }
      } else {
        col_offsets.push_back(int64_t(-1));
      }
    }
    col_frag_offsets.push_back(col_offsets);
  }
  return col_frag_offsets;
}

typedef std::vector<int64_t> ValueTuple;

class ChunkIter;
class ResultSet;

class RowSetMemoryOwner : boost::noncopyable {
 public:
  void addCountDistinctBuffer(int8_t* count_distinct_buffer,
                              const size_t bytes,
                              const bool system_allocated) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    count_distinct_bitmaps_.emplace_back(
        CountDistinctBitmapBuffer{count_distinct_buffer, bytes, system_allocated});
  }

  void addCountDistinctSet(std::set<int64_t>* count_distinct_set) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    count_distinct_sets_.push_back(count_distinct_set);
  }

  void addGroupByBuffer(int64_t* group_by_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    group_by_buffers_.push_back(group_by_buffer);
  }

  void addVarlenBuffer(void* varlen_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    varlen_buffers_.push_back(varlen_buffer);
  }

  std::string* addString(const std::string& str) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    strings_.emplace_back(str);
    return &strings_.back();
  }

  std::vector<int64_t>* addArray(const std::vector<int64_t>& arr) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    arrays_.emplace_back(arr);
    return &arrays_.back();
  }

  StringDictionaryProxy* addStringDict(std::shared_ptr<StringDictionary> str_dict,
                                       const int dict_id,
                                       const ssize_t generation) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = str_dict_proxy_owned_.find(dict_id);
    if (it != str_dict_proxy_owned_.end()) {
      CHECK_EQ(it->second->getDictionary(), str_dict.get());
      it->second->updateGeneration(generation);
      return it->second;
    }
    StringDictionaryProxy* str_dict_proxy =
        new StringDictionaryProxy(str_dict, generation);
    str_dict_proxy_owned_.emplace(dict_id, str_dict_proxy);
    return str_dict_proxy;
  }

  StringDictionaryProxy* getStringDictProxy(const int dict_id) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = str_dict_proxy_owned_.find(dict_id);
    CHECK(it != str_dict_proxy_owned_.end());
    return it->second;
  }

  void addLiteralStringDictProxy(
      std::shared_ptr<StringDictionaryProxy> lit_str_dict_proxy) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    lit_str_dict_proxy_ = lit_str_dict_proxy;
  }

  StringDictionaryProxy* getLiteralStringDictProxy() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return lit_str_dict_proxy_.get();
  }

  void addColBuffer(const void* col_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    col_buffers_.push_back(const_cast<void*>(col_buffer));
  }

  ~RowSetMemoryOwner() {
    for (const auto& count_distinct_buffer : count_distinct_bitmaps_) {
      if (count_distinct_buffer.system_allocated) {
        free(count_distinct_buffer.ptr);
      }
    }
    for (auto count_distinct_set : count_distinct_sets_) {
      delete count_distinct_set;
    }
    for (auto group_by_buffer : group_by_buffers_) {
      free(group_by_buffer);
    }
    for (auto varlen_buffer : varlen_buffers_) {
      free(varlen_buffer);
    }
    for (auto col_buffer : col_buffers_) {
      free(col_buffer);
    }

    for (auto dict_proxy : str_dict_proxy_owned_) {
      delete dict_proxy.second;
    }
  }

 private:
  struct CountDistinctBitmapBuffer {
    int8_t* ptr;
    const size_t size;
    const bool system_allocated;
  };

  std::vector<CountDistinctBitmapBuffer> count_distinct_bitmaps_;
  std::vector<std::set<int64_t>*> count_distinct_sets_;
  std::vector<int64_t*> group_by_buffers_;
  std::vector<void*> varlen_buffers_;
  std::list<std::string> strings_;
  std::list<std::vector<int64_t>> arrays_;
  std::unordered_map<int, StringDictionaryProxy*> str_dict_proxy_owned_;
  std::shared_ptr<StringDictionaryProxy> lit_str_dict_proxy_;
  std::vector<void*> col_buffers_;
  mutable std::mutex state_mutex_;

  friend class ResultRows;
  friend class ResultSet;
};

inline const Analyzer::AggExpr* cast_to_agg_expr(const Analyzer::Expr* target_expr) {
  return dynamic_cast<const Analyzer::AggExpr*>(target_expr);
}

inline const Analyzer::AggExpr* cast_to_agg_expr(
    const std::shared_ptr<Analyzer::Expr> target_expr) {
  return dynamic_cast<const Analyzer::AggExpr*>(target_expr.get());
}

template <class PointerType>
inline TargetInfo target_info(const PointerType target_expr) {
  const auto agg_expr = cast_to_agg_expr(target_expr);
  bool notnull = target_expr->get_type_info().get_notnull();
  if (!agg_expr) {
    auto target_ti = target_expr ? get_logical_type_info(target_expr->get_type_info())
                                 : SQLTypeInfo(kBIGINT, notnull);
    return {false, kMIN, target_ti, SQLTypeInfo(kNULLT, false), false, false};
  }
  const auto agg_type = agg_expr->get_aggtype();
  const auto agg_arg = agg_expr->get_arg();
  if (!agg_arg) {
    CHECK_EQ(kCOUNT, agg_type);
    CHECK(!agg_expr->get_is_distinct());
    return {true,
            kCOUNT,
            SQLTypeInfo(g_bigint_count ? kBIGINT : kINT, notnull),
            SQLTypeInfo(kNULLT, false),
            false,
            false};
  }

  const auto& agg_arg_ti = agg_arg->get_type_info();
  bool is_distinct{false};
  if (agg_expr->get_aggtype() == kCOUNT) {
    is_distinct = agg_expr->get_is_distinct();
  }

  return {true,
          agg_expr->get_aggtype(),
          agg_type == kCOUNT
              ? SQLTypeInfo((is_distinct || g_bigint_count) ? kBIGINT : kINT, notnull)
              : (agg_type == kAVG ? agg_arg_ti : agg_expr->get_type_info()),
          agg_arg_ti,
          !agg_arg_ti.get_notnull(),
          is_distinct};
}

inline std::vector<TargetInfo> target_exprs_to_infos(
    const std::vector<Analyzer::Expr*>& targets,
    const QueryMemoryDescriptor& query_mem_desc) {
  std::vector<TargetInfo> target_infos;
  for (const auto target_expr : targets) {
    auto target = target_info(target_expr);
    if (query_mem_desc.getQueryDescriptionType() ==
        QueryDescriptionType::NonGroupedAggregate) {
      set_notnull(target, false);
      target.sql_type.set_notnull(false);
    }
    target_infos.push_back(target);
  }
  return target_infos;
}

struct GpuQueryMemory;

class ResultRows {
 public:
  static void inplaceSortGpuImpl(const std::list<Analyzer::OrderEntry>&,
                                 const QueryMemoryDescriptor&,
                                 const GpuQueryMemory&,
                                 Data_Namespace::DataMgr*,
                                 const int);

  static bool reduceSingleRow(const int8_t* row_ptr,
                              const int8_t warp_count,
                              const bool is_columnar,
                              const bool replace_bitmap_ptr_with_bitmap_sz,
                              std::vector<int64_t>& agg_vals,
                              const QueryMemoryDescriptor& query_mem_desc,
                              const std::vector<TargetInfo>& targets,
                              const std::vector<int64_t>& agg_init_vals);
};

#endif  // QUERYENGINE_RESULTROWS_H
