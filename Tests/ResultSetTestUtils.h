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

#ifndef RESULTSETTESTUTILS_H
#define RESULTSETTESTUTILS_H

#include "../QueryEngine/ResultSet.h"
#include "../QueryEngine/RuntimeFunctions.h"
#include "../QueryEngine/TargetValue.h"
#include "../Shared/TargetInfo.h"
#include "../Shared/sqldefs.h"

#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

class NumberGenerator {
 public:
  virtual int64_t getNextValue() = 0;

  virtual void reset() = 0;
};

class EvenNumberGenerator : public NumberGenerator {
 public:
  EvenNumberGenerator() : crt_(0) {}

  int64_t getNextValue() override {
    const auto crt = crt_;
    crt_ += 2;
    return crt;
  }

  void reset() override { crt_ = 0; }

 private:
  int64_t crt_;
};

class ReverseOddOrEvenNumberGenerator : public NumberGenerator {
 public:
  ReverseOddOrEvenNumberGenerator(const int64_t init) : crt_(init), init_(init) {}

  int64_t getNextValue() override {
    const auto crt = crt_;
    crt_ -= 2;
    return crt;
  }

  void reset() override { crt_ = init_; }

 private:
  int64_t crt_;
  int64_t init_;
};

int8_t* advance_to_next_columnar_key_buff(int8_t* key_ptr,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          const size_t key_idx);

int64_t get_empty_key_sentinel(int8_t key_bytes);

void write_key(const int64_t k, int8_t* ptr, const int8_t key_bytes);

void write_int(int8_t* slot_ptr, const int64_t v, const size_t slot_bytes);

void write_fp(int8_t* slot_ptr, const int64_t v, const size_t slot_bytes);

int8_t* fill_one_entry_no_collisions(int8_t* buff,
                                     const QueryMemoryDescriptor& query_mem_desc,
                                     const int64_t v,
                                     const std::vector<TargetInfo>& target_infos,
                                     const bool empty,
                                     const bool null_val = false);

void fill_one_entry_one_col(int8_t* ptr1,
                            const int8_t compact_sz1,
                            int8_t* ptr2,
                            const int8_t compact_sz2,
                            int64_t v,
                            const TargetInfo& target_info,
                            const bool empty_entry,
                            const bool null_val = false);

void fill_one_entry_one_col(int64_t* value_slot,
                            const int64_t v,
                            const TargetInfo& target_info,
                            const size_t entry_count,
                            const bool empty_entry = false,
                            const bool null_val = false);

void fill_one_entry_baseline(int64_t* value_slots,
                             const int64_t v,
                             const std::vector<TargetInfo>& target_infos,
                             const bool empty = false,
                             const bool null_val = false);

void fill_storage_buffer_perfect_hash_colwise(int8_t* buff,
                                              const std::vector<TargetInfo>& target_infos,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              NumberGenerator& generator,
                                              const size_t step = 2);

void fill_storage_buffer_perfect_hash_rowwise(int8_t* buff,
                                              const std::vector<TargetInfo>& target_infos,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              NumberGenerator& generator,
                                              const size_t step = 2);

void fill_storage_buffer_baseline_colwise(int8_t* buff,
                                          const std::vector<TargetInfo>& target_infos,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          NumberGenerator& generator,
                                          const size_t step);

void fill_storage_buffer_baseline_rowwise(int8_t* buff,
                                          const std::vector<TargetInfo>& target_infos,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          NumberGenerator& generator,
                                          const size_t step);

void fill_storage_buffer(int8_t* buff,
                         const std::vector<TargetInfo>& target_infos,
                         const QueryMemoryDescriptor& query_mem_desc,
                         NumberGenerator& generator,
                         const size_t step);

QueryMemoryDescriptor perfect_hash_one_col_desc_small(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes);

QueryMemoryDescriptor perfect_hash_one_col_desc(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes,
    const size_t min_val,
    const size_t max_val,
    std::vector<int8_t> group_column_widths = {8});

QueryMemoryDescriptor perfect_hash_two_col_desc(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes);

QueryMemoryDescriptor baseline_hash_two_col_desc_large(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes);

QueryMemoryDescriptor baseline_hash_two_col_desc_overflow32(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes);

QueryMemoryDescriptor baseline_hash_two_col_desc(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes);

size_t get_slot_count(const std::vector<TargetInfo>& target_infos);

std::unordered_map<size_t, size_t> get_slot_to_target_mapping(
    const std::vector<TargetInfo>& target_infos);

std::vector<TargetInfo> generate_custom_agg_target_infos(std::vector<int8_t> key_columns,
                                                         std::vector<SQLAgg> sql_aggs,
                                                         std::vector<SQLTypes> agg_types,
                                                         std::vector<SQLTypes> arg_types);

template <class T>
inline T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

template <class T>
const T* vptr(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  return boost::get<T>(scalar_r);
}

#endif  // RESULTSETTESTUTILS_H
