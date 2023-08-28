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

#include "Catalog/Catalog.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/MurmurHash1Inl.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/SystemParameters.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <boost/filesystem/operations.hpp>

#include <exception>
#include <memory>
#include <ostream>
#include <set>
#include <vector>

namespace po = boost::program_options;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace Catalog_Namespace;
using namespace TestHelpers;

using QR = QueryRunner::QueryRunner;

const int kNoMatch = -1;
const int kNotPresent = -2;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

namespace {

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_query(const std::string& query_str,
                                     const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, true, true);
}

TargetValue run_simple_query(const std::string& query_str,
                             const ExecutorDeviceType device_type,
                             const bool geo_return_geo_tv = true,
                             const bool allow_loop_joins = true) {
  auto rows = QR::get()->runSQL(query_str, device_type, allow_loop_joins);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  EXPECT_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

struct HashtableInfo {
 public:
  HashtableInfo(int32_t min_, int32_t max_, int32_t hash_entry_count_)
      : min(min_), max(max_), hash_entry_count(hash_entry_count_) {}

  int32_t get_min_val() { return min; }

  int32_t get_max_val() { return max; }

  int32_t get_hash_entry_count() { return hash_entry_count; }

 private:
  int32_t min;
  int32_t max;
  int32_t hash_entry_count;
};

// get necessary info to check a join hashtable
HashtableInfo get_join_hashtable_info(std::vector<int32_t>& inserted_keys) {
  int32_t min = INT32_MAX;
  int32_t max = INT32_MIN;
  for (int32_t v : inserted_keys) {
    if (v < min) {
      min = v;
    }
    if (v > max) {
      max = v;
    }
  }
  // get_bucketized_hash_entry_info for an equi join btw. integer type cols.
  // here, bucket_normalization = 1 and is_bw_eq = false.
  int32_t hash_entry_count = max - min + 1;
  return {min, max, hash_entry_count};
}

// check whether cached join hashtable is correctly built by comparing
// the one with expected join hashtable, especially checking rowID included
// in those hashtables.
// currently this function only supports a hashtable for integer type
// and assume a table is not sharded.
bool check_one_to_one_join_hashtable(std::vector<int32_t>& inserted_keys,
                                     const int32_t* cached_hashtable) {
  auto hashtable_info = get_join_hashtable_info(inserted_keys);
  int32_t min = hashtable_info.get_min_val();
  int32_t max = hashtable_info.get_max_val();
  int32_t rowID = 0;
  for (int32_t v : inserted_keys) {
    int32_t offset = v - min;
    CHECK_GE(offset, 0);
    CHECK_LE(offset, max);
    if (cached_hashtable[offset] != rowID) {
      return false;
    }
    ++rowID;
  }
  return true;
}

bool keys_are_equal(const int32_t* key1,
                    const int32_t* key2,
                    const size_t key_component_count) {
  return memcmp(key1, key2, key_component_count) == 0;
}

int32_t probe_one_to_many_baseline_hashtable(const int32_t* key,
                                             const size_t key_component_count,
                                             const int32_t* cached_hashtable,
                                             const size_t entry_count,
                                             const int32_t rowID) {
  const uint32_t h =
      MurmurHash1Impl(key, key_component_count * sizeof(int32_t), 0) % entry_count;
  uint32_t off = h * key_component_count;
  int32_t base_offset = entry_count * key_component_count;
  bool found_matching_tuple = false;
  if (keys_are_equal(&cached_hashtable[off], key, key_component_count)) {
    int32_t rowID_offset = cached_hashtable[base_offset + h];
    int32_t rowID_cnt = cached_hashtable[base_offset + entry_count + h];
    int32_t rowID_start_offset = base_offset + (2 * entry_count);
    for (int idx = 0; idx < rowID_cnt; idx++) {
      int32_t candidate_rowID = cached_hashtable[rowID_start_offset + rowID_offset + idx];
      if (candidate_rowID == rowID) {
        found_matching_tuple = true;
        break;
      }
    }
    if (found_matching_tuple) {
      return true;
    }
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    off = h_probe * key_component_count;
    if (keys_are_equal(&cached_hashtable[off], key, key_component_count)) {
      int32_t rowID_offset = cached_hashtable[base_offset + h_probe];
      int32_t rowID_cnt = cached_hashtable[base_offset + entry_count + h_probe];
      int32_t rowID_start_offset = base_offset + (2 * entry_count);
      for (int idx = 0; idx < rowID_cnt; idx++) {
        int32_t candidate_rowID =
            cached_hashtable[rowID_start_offset + rowID_offset + idx];
        if (candidate_rowID == rowID) {
          found_matching_tuple = true;
          break;
        }
      }
      if (found_matching_tuple) {
        return true;
      }
    }
    if (cached_hashtable[off] == EMPTY_KEY_32) {
      return false;
    }
    h_probe = (h_probe + 1) % entry_count;
  }
  return false;
}

bool check_one_to_many_baseline_hashtable(std::vector<std::vector<int32_t>>& insert_keys,
                                          const int8_t* cached_hashtable,
                                          size_t entry_count) {
  int rowID = 0;
  for (auto keys : insert_keys) {
    if (!probe_one_to_many_baseline_hashtable(
            keys.data(), keys.size(), (int32_t*)cached_hashtable, entry_count, rowID)) {
      return false;
    }
    ++rowID;
  }
  return true;
}

bool check_one_to_many_join_hashtable(std::vector<int32_t>& inserted_keys,
                                      const int32_t* cached_hashtable) {
  auto hashtable_info = get_join_hashtable_info(inserted_keys);
  int32_t min = hashtable_info.get_min_val();
  int32_t hash_entry_count = hashtable_info.get_hash_entry_count();
  int32_t num_elem = inserted_keys.size();
  int32_t hashtable_size = 2 * hash_entry_count + num_elem;
  int32_t count_buff_start_offset = hash_entry_count;
  int32_t rowID_buff_start_offset = 2 * hash_entry_count;

  int32_t rowID = 0;
  bool has_unmatched_key = false;
  for (int32_t v : inserted_keys) {
    int32_t offset = v - min;
    CHECK_GE(offset, 0);
    CHECK_LT(offset, hashtable_size);
    int32_t PSV = cached_hashtable[offset];  // Prefix Sum Value
    if (PSV != -1) {
      int32_t CV = cached_hashtable[count_buff_start_offset + offset];
      CHECK_GE(CV, 1);  // Count Value
      bool found_matching_key = false;
      for (int32_t idx = 0; idx < CV; idx++) {
        if (cached_hashtable[rowID_buff_start_offset + PSV + idx] == rowID) {
          found_matching_key = true;
          break;
        }
      }
      if (found_matching_key) {
        ++rowID;
        continue;
      } else {
        has_unmatched_key = true;
        break;
      }
    } else {
      has_unmatched_key = true;
      break;
    }
  }
  return !has_unmatched_key;
}

bool compare_keys(const int8_t* entry, const int8_t* key, const size_t key_bytes) {
  for (size_t i = 0; i < key_bytes; ++i) {
    if (entry[i] != key[i]) {
      return false;
    }
  }
  return true;
}

int32_t get_rowID(const int8_t* hash_buff,
                  const uint32_t h,
                  const int8_t* key,
                  const size_t key_bytes) {
  const auto lookup_result_ptr = hash_buff + h * (key_bytes + sizeof(int32_t));
  if (compare_keys(lookup_result_ptr, key, key_bytes)) {
    return *reinterpret_cast<const int32_t*>(lookup_result_ptr + key_bytes);
  }
  if (*reinterpret_cast<const int32_t*>(lookup_result_ptr) != -1) {
    return kNotPresent;
  }
  return kNoMatch;
}

bool probe_one_to_one_baseline_hashtable(const int8_t* cached_hashtable,
                                         const int8_t* key,
                                         const size_t key_bytes,
                                         const size_t entry_count,
                                         const int32_t rowID) {
  if (!entry_count) {
    return false;
  }
  const uint32_t h = MurmurHash1Impl(key, key_bytes, 0) % entry_count;
  int32_t candidate_rowID = get_rowID(cached_hashtable, h, key, key_bytes);
  if (candidate_rowID == rowID) {
    return true;
  }
  uint32_t h_probe = (h + 1) % entry_count;
  while (h_probe != h) {
    candidate_rowID = get_rowID(cached_hashtable, h_probe, key, key_bytes);
    if (candidate_rowID == rowID) {
      return true;
    }
    h_probe = (h_probe + 1) % entry_count;
  }
  return false;
}

// check whether cached baseline hashtable is correctly built.
// this can be done by comparing rowID in the cached hashtable for a given tuple.
// currently this function only supports a hashtable for integer type.
bool check_one_to_one_baseline_hashtable(std::vector<std::vector<int32_t>>& insert_keys,
                                         const int8_t* cached_hashtable,
                                         size_t entry_count) {
  int rowID = 0;
  for (auto keys : insert_keys) {
    if (!probe_one_to_one_baseline_hashtable(cached_hashtable,
                                             (int8_t*)keys.data(),
                                             keys.size() * sizeof(int32_t),
                                             entry_count,
                                             rowID)) {
      return false;
    }
    ++rowID;
  }
  return true;
}

void import_tables_cache_invalidation_for_CPU_one_to_one_join(bool reverse) {
  const std::string drop_table_t1{"DROP TABLE IF EXISTS cache_invalid_t1;"};
  run_ddl_statement(drop_table_t1);

  const std::string drop_table_t2{"DROP TABLE IF EXISTS cache_invalid_t2;"};
  run_ddl_statement(drop_table_t2);

  const std::string create_t1{
      "CREATE TABLE cache_invalid_t1 (id1 int, id2 int, des text encoding dict(32)) with "
      "(fragment_size=2000000);"};
  run_ddl_statement(create_t1);

  const std::string create_t2{
      "CREATE TABLE cache_invalid_t2 (id1 int, id2 int, des text "
      "encoding dict(32)) with (fragment_size=2000000);"};
  run_ddl_statement(create_t2);

  std::vector<std::string> row_insert_sql;
  if (reverse) {
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (1, 1, 'row-0');");
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (0, 0, 'row-1');");
  } else {
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (0, 0, 'row-0');");
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (1, 1, 'row-1');");
  }
  row_insert_sql.emplace_back("INSERT INTO cache_invalid_t2 VALUES (1, 1, 'row-0');");
  row_insert_sql.emplace_back("INSERT INTO cache_invalid_t2 VALUES (1, 1, 'row-1');");
  row_insert_sql.emplace_back("INSERT INTO cache_invalid_t2 VALUES (1, 1, 'row-2');");
  for (std::string insert_str : row_insert_sql) {
    run_query(insert_str, ExecutorDeviceType::CPU);
  }
}

void clearCaches() {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  executor->clearMemory(MemoryLevel::CPU_LEVEL);
  executor->getQueryPlanDagCache().clearQueryPlanCache();
}

void import_tables_cache_invalidation_for_CPU_one_to_many_join(bool reverse) {
  const std::string drop_table_t1{"DROP TABLE IF EXISTS cache_invalid_t1;"};
  run_ddl_statement(drop_table_t1);

  const std::string drop_table_t2{"DROP TABLE IF EXISTS cache_invalid_t2;"};
  run_ddl_statement(drop_table_t2);

  const std::string create_t1{
      "CREATE TABLE cache_invalid_t1 (k1 int, k2 int, v1 int, v2 int);"};
  run_ddl_statement(create_t1);

  const std::string create_t2{"CREATE TABLE cache_invalid_t2 (k1 int, k2 int);"};
  run_ddl_statement(create_t2);

  std::vector<std::string> row_insert_sql;
  if (reverse) {
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (1, 1, 1, 2);");
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (0, 0, 1, 2);");
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (0, 0, 2, 1);");
  } else {
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (0, 0, 1, 2);");
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (0, 0, 2, 1);");
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t1 VALUES (1, 1, 1, 2);");
  }
  for (size_t i = 0; i < 6; ++i) {
    row_insert_sql.emplace_back("INSERT INTO cache_invalid_t2 VALUES (1, 1);");
  }
  for (std::string insert_str : row_insert_sql) {
    run_query(insert_str, ExecutorDeviceType::CPU);
  }
}

std::shared_ptr<HashTable> getCachedHashTable(std::set<QueryPlanHash>& already_visited,
                                              CacheItemType cache_item_type) {
  auto cached_ht = QR::get()->getCachedHashtableWithoutCacheKey(
      already_visited, cache_item_type, 0 /* CPU_DEVICE_IDENTIFIER*/);
  auto cache_key = std::get<0>(cached_ht);
  already_visited.insert(cache_key);
  return std::get<1>(cached_ht);
}
}  // namespace

TEST(Select, DropAndReCreate_OneToOne_HashTable_WithReversedTupleInsertion) {
  // tuple insertion order is controlled by a bool param. of an import function
  import_tables_cache_invalidation_for_CPU_one_to_one_join(false);
  std::set<QueryPlanHash> visited_hashtable_key;

  // (a) baseline hash join, the first run] tuple insertion order: (0, 0) -> (1, 1)
  const auto q1 =
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on t1.id1 "
      "= t2.id1 and t1.id2 = t2.id2;";
  run_query(q1, ExecutorDeviceType::CPU);
  std::vector<std::vector<int32_t>> baseline_hashtable_first_run;
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{1, 1});
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(1));
  std::shared_ptr<BaselineHashTable> cached_q1_ht =
      std::dynamic_pointer_cast<BaselineHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::BASELINE_HT));
  CHECK(check_one_to_one_baseline_hashtable(baseline_hashtable_first_run,
                                            cached_q1_ht->getCpuBuffer(),
                                            cached_q1_ht->getEntryCount()));
  // (b) perfect hash join, the first run] tuple insertion order: 0 -> 1
  const auto q2 =
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on t1.id1 "
      "= t2.id1;";
  run_query(q2, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q2_ht =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(1));
  std::vector<int32_t> perfect_hashtable_first_run{0, 1};
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_first_run,
                                        (int32_t*)cached_q2_ht->getCpuBuffer()));

  const auto q3 =
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on t1.id2 "
      "= t2.id2";
  run_query(q3, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q3_ht =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(2));
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_first_run,
                                        (int32_t*)cached_q3_ht->getCpuBuffer()));

  // the second run --> reversed tuple insertion order compared with the first run
  import_tables_cache_invalidation_for_CPU_one_to_one_join(true);

  // make sure we invalidate all cached hashtables after tables are dropped
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(1));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(2));
  clearCaches();
  visited_hashtable_key.clear();
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(0));

  // (a) baseline hash join, the second run] tuple insertion order: (1, 1) -> (0, 0)
  run_query(q1, ExecutorDeviceType::CPU);
  std::shared_ptr<BaselineHashTable> cached_q1_ht_v2 =
      std::dynamic_pointer_cast<BaselineHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::BASELINE_HT));
  std::vector<std::vector<int32_t>> baseline_hashtable_second_run;
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{1, 1});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(1));
  CHECK(check_one_to_one_baseline_hashtable(baseline_hashtable_second_run,
                                            cached_q1_ht_v2->getCpuBuffer(),
                                            cached_q1_ht_v2->getEntryCount()));

  // (a) perfect hash join, the second run] tuple insertion order: 1 -> 0
  run_query(q2, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q2_ht_v2 =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(1));
  std::vector<int32_t> perfect_hashtable_second_run{1, 0};
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_second_run,
                                        (int32_t*)cached_q2_ht_v2->getCpuBuffer()));

  run_query(q3, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q3_ht_v2 =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(2));
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_second_run,
                                        (int32_t*)cached_q3_ht_v2->getCpuBuffer()));

  run_ddl_statement("DROP TABLE cache_invalid_t1;");
  run_ddl_statement("DROP TABLE cache_invalid_t2;");
  // make sure we invalidate all cached hashtables after tables are dropped
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(1));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(2));
  clearCaches();
  visited_hashtable_key.clear();
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(0));
}

TEST(Select, DropAndReCreate_OneToMany_HashTable_WithReversedTupleInsertion) {
  // tuple insertion order is controlled by a bool param. of an import function
  import_tables_cache_invalidation_for_CPU_one_to_many_join(false);
  std::set<QueryPlanHash> visited_hashtable_key;

  // (a) baseline hash join, the first run]
  // tuple insertion order: (0, 0) -> (0, 0) -> (1,1)
  const auto q1 =
      "select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k1 = t1.k1 and "
      "t0.k2 = t1.k2;";
  run_query(q1, ExecutorDeviceType::CPU);
  std::shared_ptr<BaselineHashTable> cached_q1_ht =
      std::dynamic_pointer_cast<BaselineHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::BASELINE_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(1));
  std::vector<std::vector<int32_t>> baseline_hashtable_first_run;
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{1, 1});
  CHECK(check_one_to_many_baseline_hashtable(baseline_hashtable_first_run,
                                             cached_q1_ht->getCpuBuffer(),
                                             cached_q1_ht->getEntryCount()));

  // (b) perfect hash join, the first run] tuple insertion order: 0 -> 0 -> 1
  const auto q2 =
      "select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k1 = t1.k1;";
  run_query(q2, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q2_ht =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(1));
  std::vector<int32_t> perfect_hashtable_first_run{0, 0, 1};
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_first_run,
                                         (int32_t*)cached_q2_ht->getCpuBuffer()));

  const auto q3 =
      "select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k2 = t1.k2;";
  run_query(q3, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q3_ht =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(2));
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_first_run,
                                         (int32_t*)cached_q3_ht->getCpuBuffer()));

  // [the second run] tuple insertion order: (1, 1) -> (0, 0) -> (0, 0)
  import_tables_cache_invalidation_for_CPU_one_to_many_join(true);

  // make sure we invalidate all cached hashtables after tables are dropped
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(1));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(2));
  clearCaches();
  visited_hashtable_key.clear();
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(0));

  // (a) baseline hash join, the second run] tuple insertion order: (1, 1) -> (0, 0) ->
  // (0, 0)
  run_query(q1, ExecutorDeviceType::CPU);
  std::shared_ptr<BaselineHashTable> cached_q1_ht_v2 =
      std::dynamic_pointer_cast<BaselineHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::BASELINE_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(1));
  std::vector<std::vector<int32_t>> baseline_hashtable_second_run;
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{1, 1});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  CHECK(check_one_to_many_baseline_hashtable(baseline_hashtable_second_run,
                                             cached_q1_ht_v2->getCpuBuffer(),
                                             cached_q1_ht_v2->getEntryCount()));

  // (b) perfect hash join, the second run] tuple insertion order: 1 -> 0 -> 0
  run_query(q2, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q2_ht_v2 =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(1));
  std::vector<int32_t> perfect_hashtable_second_run{1, 0, 0};
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_second_run,
                                         (int32_t*)cached_q2_ht_v2->getCpuBuffer()));

  run_query(q3, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q3_ht_v2 =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(2));
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_second_run,
                                         (int32_t*)cached_q3_ht_v2->getCpuBuffer()));

  run_ddl_statement("DROP TABLE cache_invalid_t1;");
  run_ddl_statement("DROP TABLE cache_invalid_t2;");

  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(1));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(2));
  clearCaches();
  visited_hashtable_key.clear();
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::BASELINE_HT),
            static_cast<size_t>(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                             CacheItemType::PERFECT_HT),
            static_cast<size_t>(0));
}

TEST(Truncate, JoinCacheInvalidationTest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_t1;");
    run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_t2;");

    run_ddl_statement("create table cache_invalid_t1 (k1 text encoding dict(32));");
    run_ddl_statement("create table cache_invalid_t2 (k2 text encoding dict(32));");
    std::vector<std::string> t1_col_val{"1", "2", "3", "4", "5"};
    std::vector<std::string> t2_col_val{"0", "0", "0", "0", "0", "1", "2", "3", "4", "5"};
    for (auto& t1_val : t1_col_val) {
      run_query("insert into cache_invalid_t1 values ('" + t1_val + "');",
                ExecutorDeviceType::CPU);
    }
    for (auto& t2_val : t2_col_val) {
      run_query("insert into cache_invalid_t2 values ('" + t2_val + "');",
                ExecutorDeviceType::CPU);
    }

    auto res_before_truncate = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    EXPECT_EQ(static_cast<uint32_t>(5), res_before_truncate->rowCount());
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    run_ddl_statement("truncate table cache_invalid_t2;");
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));
    auto res_after_truncate = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    EXPECT_EQ(static_cast<uint32_t>(0), res_after_truncate->rowCount());
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(0));

    std::vector<std::string> t2_col_val_v2{
        "1", "2", "3", "4", "5", "0", "0", "0", "0", "0"};
    for (auto& t2_val : t2_col_val) {
      run_query("insert into cache_invalid_t2 values ('" + t2_val + "');",
                ExecutorDeviceType::CPU);
    }

    auto res_before_truncate_v2 = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    EXPECT_EQ(static_cast<uint32_t>(5), res_before_truncate_v2->rowCount());
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    run_ddl_statement("DROP TABLE cache_invalid_t1;");
    run_ddl_statement("DROP TABLE cache_invalid_t2;");
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(0));
    clearCaches();
  }
}

TEST(Truncate, BBoxIntersectCacheInvalidationTest) {
  EXPECT_TRUE(g_enable_bbox_intersect_hashjoin);

  run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_point;");
  run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_poly;");

  run_ddl_statement("CREATE TABLE cache_invalid_point(pt GEOMETRY(point, 4326));");
  run_ddl_statement(
      "CREATE TABLE cache_invalid_poly(poly GEOMETRY(multipolygon, 4326));");

  run_query("INSERT INTO cache_invalid_point VALUES ('POINT(0 0)');",
            ExecutorDeviceType::CPU);
  run_query("INSERT INTO cache_invalid_point VALUES ('POINT(1 1)');",
            ExecutorDeviceType::CPU);
  run_query("INSERT INTO cache_invalid_point VALUES ('POINT(10 10)');",
            ExecutorDeviceType::CPU);

  run_query(
      R"(INSERT INTO cache_invalid_poly VALUES ('MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)))');)",
      ExecutorDeviceType::CPU);

  // GPU does not cache, run on CPU
  {
    auto result = QR::get()->runSQL(
        R"(SELECT count(*) FROM cache_invalid_point a, cache_invalid_poly b WHERE ST_Contains(b.poly, a.pt);)",
        ExecutorDeviceType::CPU);
    EXPECT_EQ(size_t(1), result->rowCount());
    auto row = result->getNextRow(false, false);
    EXPECT_EQ(size_t(1), row.size());
    auto count = boost::get<int64_t>(boost::get<ScalarTargetValue>(row[0]));
    EXPECT_EQ(1, count);  // POINT(1 1)
  }
  EXPECT_EQ(
      QR::get()->getNumberOfCachedItem(
          QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true),
      size_t(2));  // bucket threshold and hash table

  run_ddl_statement("TRUNCATE TABLE cache_invalid_poly");
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::BBOX_INTERSECT_HT,
                                             true),
            size_t(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::BBOX_INTERSECT_HT,
                                             true),
            size_t(2));

  run_query(
      R"(INSERT INTO cache_invalid_poly VALUES ('MULTIPOLYGON(((0 0, 11 0, 11 11, 0 11, 0 0)))');)",
      ExecutorDeviceType::CPU);

  // user provided bucket threshold -- only one additional cache entry
  {
    auto result = QR::get()->runSQL(
        "SELECT /*+ bbox_intersect_bucket_threshold(0.2) */ count(*) FROM "
        "cache_invalid_point "
        "a, cache_invalid_poly b WHERE "
        "ST_Contains(b.poly, a.pt);",
        ExecutorDeviceType::CPU);
    EXPECT_EQ(size_t(1), result->rowCount());
    auto row = result->getNextRow(false, false);
    EXPECT_EQ(size_t(1), row.size());
    auto count = boost::get<int64_t>(boost::get<ScalarTargetValue>(row[0]));
    EXPECT_EQ(2, count);  // POINT(1 1) , POINT(10 10)
  }
  EXPECT_EQ(
      QR::get()->getNumberOfCachedItem(
          QueryRunner::CacheItemStatus::ALL, CacheItemType::BBOX_INTERSECT_HT, true),
      size_t(3));  // bucket threshold and hash table
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::BBOX_INTERSECT_HT,
                                             true),
            size_t(2));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::BBOX_INTERSECT_HT,
                                             true),
            size_t(1));

  run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_point;");
  run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_poly;");
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                             CacheItemType::BBOX_INTERSECT_HT,
                                             true),
            size_t(0));
  EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                             CacheItemType::BBOX_INTERSECT_HT,
                                             true),
            size_t(3));
  clearCaches();
}

TEST(Update, JoinCacheInvalidationTest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists string_join1");
    run_ddl_statement("drop table if exists string_join2");
    run_ddl_statement("create table string_join1 ( t text ) with (vacuum='delayed')");
    run_ddl_statement(
        "create table string_join2 ( t text ) with (vacuum='delayed', "
        "partitions='REPLICATED')");

    run_query("insert into string_join1 values ('muffin')", ExecutorDeviceType::CPU);
    run_query("insert into string_join1 values ('pizza')", ExecutorDeviceType::CPU);
    run_query("insert into string_join1 values ('ice cream')", ExecutorDeviceType::CPU);
    run_query("insert into string_join1 values ('poutine')", ExecutorDeviceType::CPU);
    run_query("insert into string_join1 values ('samosa')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('tomato')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('potato')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('apple')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('orange')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('chutney')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('poutine')", ExecutorDeviceType::CPU);

    run_simple_query(
        "select count(string_join1.t) from string_join1 inner join string_join2 on "
        "string_join1.t = string_join2.t;",
        dt);

    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    run_query("update string_join1 set t='not poutine' where t='poutine';", dt);
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    EXPECT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_query(
            "select count(string_join1.t) from string_join1 inner join string_join2 on "
            "string_join1.t = string_join2.t;",
            dt)));
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    run_ddl_statement("drop table string_join1;");
    run_ddl_statement("drop table string_join2;");
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));
    clearCaches();
  }
}

TEST(Delete, JoinCacheInvalidationTest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists string_join1;");
    run_ddl_statement("drop table if exists string_join2;");
    run_ddl_statement("create table string_join1 ( t text ) with (vacuum='delayed')");
    run_ddl_statement(
        "create table string_join2 ( t text ) with (vacuum='delayed', "
        "partitions='REPLICATED')");

    run_query("insert into string_join1 values ('muffin')", ExecutorDeviceType::CPU);
    run_query("insert into string_join1 values ('pizza')", ExecutorDeviceType::CPU);
    run_query("insert into string_join1 values ('ice cream')", ExecutorDeviceType::CPU);
    run_query("insert into string_join1 values ('poutine')", ExecutorDeviceType::CPU);
    run_query("insert into string_join1 values ('samosa')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('tomato')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('potato')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('apple')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('orange')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('chutney')", ExecutorDeviceType::CPU);
    run_query("insert into string_join2 values ('poutine')", ExecutorDeviceType::CPU);

    run_simple_query(
        "select count(string_join1.t) from string_join1 inner join string_join2 on "
        "string_join1.t = string_join2.t;",
        dt);

    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    run_query("delete from string_join1 where t='poutine';", dt);
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(0));
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    EXPECT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_query(
            "select count(string_join1.t) from string_join1 inner join string_join2 on "
            "string_join1.t = string_join2.t;",
            dt)));
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    run_ddl_statement("drop table string_join1;");
    run_ddl_statement("drop table string_join2;");
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(0));
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));
    clearCaches();
  }
}

TEST(Delete, JoinCacheInvalidationTest_DropTable) {
  // todo: when we support per-table cached hashtable invalidation,
  // then this test should be changed either
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_t1;");
    run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_t2;");
    run_ddl_statement("create table cache_invalid_t1 (k1 text encoding dict(32));");
    run_ddl_statement("create table cache_invalid_t2 (k2 text encoding dict(32));");

    std::vector<std::string> t1_col_val{"1", "2", "3", "4", "5"};
    std::vector<std::string> t2_col_val{"0", "0", "0", "0", "0", "1", "2", "3", "4", "5"};
    for (auto& t1_val : t1_col_val) {
      run_query("insert into cache_invalid_t1 values ('" + t1_val + "');",
                ExecutorDeviceType::CPU);
    }
    for (auto& t2_val : t2_col_val) {
      run_query("insert into cache_invalid_t2 values ('" + t2_val + "');",
                ExecutorDeviceType::CPU);
    }

    auto res = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    EXPECT_EQ(static_cast<uint32_t>(5), res->rowCount());
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::ALL,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    // add and drop dummy table
    run_ddl_statement("create table cache_invalid_t3 (dummy text encoding dict(32));");
    run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_t3;");
    // we drop `cache_invalid_t3` so the cached hashtable built from `cache_invalid_t1`
    // should alive
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    auto res_v2 = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    EXPECT_EQ(static_cast<uint32_t>(5), res_v2->rowCount());
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));

    run_ddl_statement("DROP TABLE cache_invalid_t1;");
    run_ddl_statement("DROP TABLE cache_invalid_t2;");
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::CLEAN_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(0));
    EXPECT_EQ(QR::get()->getNumberOfCachedItem(QueryRunner::CacheItemStatus::DIRTY_ONLY,
                                               CacheItemType::PERFECT_HT),
              static_cast<size_t>(1));
    clearCaches();
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH);

  int err{0};

  // enable bounding box intersection
  g_enable_bbox_intersect_hashjoin = true;

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  QR::reset();
  return err;
}
