/*
 * Copyright 2020 OmniSci, Inc.
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

#include "ArrowSQLRunner/ArrowSQLRunner.h"
#include "TestHelpers.h"

#include "Logger/Logger.h"
#include "QueryEngine/DataRecycler/HashtableRecycler.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/PerfectHashTable.h"
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "QueryEngine/MurmurHash1Inl.h"
#include "QueryEngine/QueryPlanDagExtractor.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/ResultSet.h"
#include "Shared/SystemParameters.h"

#include <gtest/gtest.h>
#include <boost/filesystem/operations.hpp>

#include <exception>
#include <memory>
#include <ostream>
#include <set>
#include <vector>

namespace po = boost::program_options;

using namespace TestHelpers;
using namespace TestHelpers::ArrowSQLRunner;

const int kNoMatch = -1;
const int kNotPresent = -2;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !gpusPresent();
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

extern bool g_enable_overlaps_hashjoin;

namespace {

size_t getNumberOfCachedPerfectHashTables() {
  auto hash_table_cache = PerfectJoinHashTable::getHashTableCache();
  CHECK(hash_table_cache);
  return hash_table_cache->getCurrentNumCachedItems(
      CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
};

size_t getNumberOfCachedBaselineJoinHashTables() {
  auto hash_table_cache = BaselineJoinHashTable::getHashTableCache();
  CHECK(hash_table_cache);
  return hash_table_cache->getCurrentNumCachedItems(
      CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
}

std::tuple<QueryPlanHash,
           std::shared_ptr<HashTable>,
           std::optional<HashtableCacheMetaInfo>>
getCachedHashtableWithoutCacheKey(std::set<size_t>& visited,
                                  CacheItemType hash_table_type,
                                  DeviceIdentifier device_identifier) {
  HashtableRecycler* hash_table_cache{nullptr};
  switch (hash_table_type) {
    case CacheItemType::PERFECT_HT: {
      hash_table_cache = PerfectJoinHashTable::getHashTableCache();
      break;
    }
    case CacheItemType::BASELINE_HT: {
      hash_table_cache = BaselineJoinHashTable::getHashTableCache();
      break;
    }
    default: {
      UNREACHABLE();
      break;
    }
  }
  CHECK(hash_table_cache);
  return hash_table_cache->getCachedHashtableWithoutCacheKey(
      visited, hash_table_type, device_identifier);
}

struct QueryPlanDagInfo {
  std::shared_ptr<const RelAlgNode> root_node;
  std::vector<unsigned> left_deep_trees_id;
  std::unordered_map<unsigned, JoinQualsPerNestingLevel> left_deep_trees_info;
  std::shared_ptr<RelAlgTranslator> rel_alg_translator;
};

QueryPlanDagInfo getQueryInfoForDataRecyclerTest(const std::string& query_str) {
  const auto query_ra = getSqlQueryRelAlg(query_str);
  auto executor = Executor::getExecutor(
      Executor::UNITARY_EXECUTOR_ID, getDataMgr(), getDataMgr()->getBufferProvider());
  executor->setSchemaProvider(getStorage());
  executor->setDatabaseId(TEST_DB_ID);
  auto dag =
      std::make_unique<RelAlgDagBuilder>(query_ra, TEST_DB_ID, getStorage(), nullptr);
  auto ra_executor = RelAlgExecutor(executor.get(),
                                    TEST_DB_ID,
                                    getStorage(),
                                    getDataMgr()->getDataProvider(),
                                    std::move(dag));
  // note that we assume the test for data recycler that needs to have join_info
  // does not contain any ORDER BY clause; this is necessary to create work_unit
  // without actually performing the query
  auto root_node_shared_ptr = ra_executor.getRootRelAlgNodeShPtr();
  auto join_info = ra_executor.getJoinInfo(root_node_shared_ptr.get());
  auto relAlgTranslator = ra_executor.getRelAlgTranslator(root_node_shared_ptr.get());
  return {root_node_shared_ptr, join_info.first, join_info.second, relAlgTranslator};
}

ExtractedPlanDag extractQueryPlanDag(const std::string& query_str) {
  auto query_dag_info = getQueryInfoForDataRecyclerTest(query_str);
  auto data_mgr = getDataMgr();
  auto executor =
      Executor::getExecutor(
          Executor::UNITARY_EXECUTOR_ID, data_mgr, data_mgr->getBufferProvider())
          .get();
  auto extracted_dag_info =
      QueryPlanDagExtractor::extractQueryPlanDag(query_dag_info.root_node.get(),
                                                 executor->getSchemaProvider(),
                                                 std::nullopt,
                                                 query_dag_info.left_deep_trees_info,
                                                 *executor->getTemporaryTables(),
                                                 executor,
                                                 *query_dag_info.rel_alg_translator);
  return extracted_dag_info;
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
  dropTable("cache_invalid_t1");
  createTable(
      "cache_invalid_t1",
      {{"id1", SQLTypeInfo(kINT)}, {"id2", SQLTypeInfo(kINT)}, {"des", dictType()}},
      {2000000});
  if (reverse) {
    insertCsvValues("cache_invalid_t1", "1,1,row-0\n0,0,row-1");
  } else {
    insertCsvValues("cache_invalid_t1", "0,0,row-0\n1,1,row-1");
  }

  dropTable("cache_invalid_t2");
  createTable(
      "cache_invalid_t2",
      {{"id1", SQLTypeInfo(kINT)}, {"id2", SQLTypeInfo(kINT)}, {"des", dictType()}},
      {2000000});
  insertCsvValues("cache_invalid_t2", "1,1,row-0\n1,1,row-1\n1,1,row-2");
}

void import_tables_cache_invalidation_for_CPU_one_to_many_join(bool reverse) {
  dropTable("cache_invalid_t1");
  createTable("cache_invalid_t1",
              {{"k1", SQLTypeInfo(kINT)},
               {"k2", SQLTypeInfo(kINT)},
               {"v1", SQLTypeInfo(kINT)},
               {"v2", SQLTypeInfo(kINT)}});
  if (reverse) {
    insertCsvValues("cache_invalid_t1", "1,1,1,2\n0,0,1,2\n0,0,2,1");
  } else {
    insertCsvValues("cache_invalid_t1", "0,0,1,2\n0,0,2,1\n1,1,1,2");
  }

  dropTable("cache_invalid_t2");
  createTable("cache_invalid_t2", {{"k1", SQLTypeInfo(kINT)}, {"k2", SQLTypeInfo(kINT)}});
  insertCsvValues("cache_invalid_t2", "1,1\n1,1\n1,1\n1,1\n1,1\n1,1");
}

std::shared_ptr<HashTable> getCachedHashTable(std::set<QueryPlanHash>& already_visited,
                                              CacheItemType cache_item_type) {
  auto cached_ht = getCachedHashtableWithoutCacheKey(
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
  run_multiple_agg(q1, ExecutorDeviceType::CPU);
  std::vector<std::vector<int32_t>> baseline_hashtable_first_run;
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{1, 1});
  CHECK_EQ(getNumberOfCachedBaselineJoinHashTables(), (unsigned long)1);
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
  run_multiple_agg(q2, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q2_ht =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)1);
  std::vector<int32_t> perfect_hashtable_first_run{0, 1};
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_first_run,
                                        (int32_t*)cached_q2_ht->getCpuBuffer()));

  const auto q3 =
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on t1.id2 "
      "= t2.id2";
  run_multiple_agg(q3, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q3_ht =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)2);
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_first_run,
                                        (int32_t*)cached_q3_ht->getCpuBuffer()));

  // the second run --> reversed tuple insertion order compared with the first run
  import_tables_cache_invalidation_for_CPU_one_to_one_join(true);

  // make sure we invalidate all cached hashtables after tables are dropped
  CHECK_EQ(getNumberOfCachedBaselineJoinHashTables(), (unsigned long)0);
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)0);
  visited_hashtable_key.clear();

  // (a) baseline hash join, the second run] tuple insertion order: (1, 1) -> (0, 0)
  run_multiple_agg(q1, ExecutorDeviceType::CPU);
  std::shared_ptr<BaselineHashTable> cached_q1_ht_v2 =
      std::dynamic_pointer_cast<BaselineHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::BASELINE_HT));
  std::vector<std::vector<int32_t>> baseline_hashtable_second_run;
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{1, 1});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  CHECK_EQ(getNumberOfCachedBaselineJoinHashTables(), (unsigned long)1);
  CHECK(check_one_to_one_baseline_hashtable(baseline_hashtable_second_run,
                                            cached_q1_ht_v2->getCpuBuffer(),
                                            cached_q1_ht_v2->getEntryCount()));

  // (a) perfect hash join, the second run] tuple insertion order: 1 -> 0
  run_multiple_agg(q2, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q2_ht_v2 =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)1);
  std::vector<int32_t> perfect_hashtable_second_run{1, 0};
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_second_run,
                                        (int32_t*)cached_q2_ht_v2->getCpuBuffer()));

  run_multiple_agg(q3, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q3_ht_v2 =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)2);
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_second_run,
                                        (int32_t*)cached_q3_ht_v2->getCpuBuffer()));

  dropTable("cache_invalid_t1");
  dropTable("cache_invalid_t2");
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
  run_multiple_agg(q1, ExecutorDeviceType::CPU);
  std::shared_ptr<BaselineHashTable> cached_q1_ht =
      std::dynamic_pointer_cast<BaselineHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::BASELINE_HT));
  CHECK_EQ(getNumberOfCachedBaselineJoinHashTables(), (unsigned long)1);
  std::vector<std::vector<int32_t>> baseline_hashtable_first_run;
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{1, 1});
  auto q1_dag_info = extractQueryPlanDag(q1);
  CHECK(check_one_to_many_baseline_hashtable(baseline_hashtable_first_run,
                                             cached_q1_ht->getCpuBuffer(),
                                             cached_q1_ht->getEntryCount()));

  // (b) perfect hash join, the first run] tuple insertion order: 0 -> 0 -> 1
  const auto q2 =
      "select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k1 = t1.k1;";
  run_multiple_agg(q2, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q2_ht =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  auto q2_dag_info = extractQueryPlanDag(q2);
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)1);
  std::vector<int32_t> perfect_hashtable_first_run{0, 0, 1};
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_first_run,
                                         (int32_t*)cached_q2_ht->getCpuBuffer()));

  const auto q3 =
      "select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k2 = t1.k2;";
  run_multiple_agg(q3, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q3_ht =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  auto q3_dag_info = extractQueryPlanDag(q3);
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)2);
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_first_run,
                                         (int32_t*)cached_q3_ht->getCpuBuffer()));

  // [the second run] tuple insertion order: (1, 1) -> (0, 0) -> (0, 0)
  import_tables_cache_invalidation_for_CPU_one_to_many_join(true);
  visited_hashtable_key.clear();

  // make sure we invalidate all cached hashtables after tables are dropped
  CHECK_EQ(getNumberOfCachedBaselineJoinHashTables(), (unsigned long)0);
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)0);

  // (a) baseline hash join, the second run] tuple insertion order: (1, 1) -> (0, 0) ->
  // (0, 0)
  run_multiple_agg(q1, ExecutorDeviceType::CPU);
  std::shared_ptr<BaselineHashTable> cached_q1_ht_v2 =
      std::dynamic_pointer_cast<BaselineHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::BASELINE_HT));
  CHECK_EQ(getNumberOfCachedBaselineJoinHashTables(), (unsigned long)1);
  std::vector<std::vector<int32_t>> baseline_hashtable_second_run;
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{1, 1});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  CHECK(check_one_to_many_baseline_hashtable(baseline_hashtable_second_run,
                                             cached_q1_ht_v2->getCpuBuffer(),
                                             cached_q1_ht_v2->getEntryCount()));

  // (b) perfect hash join, the second run] tuple insertion order: 1 -> 0 -> 0
  run_multiple_agg(q2, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q2_ht_v2 =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)1);
  std::vector<int32_t> perfect_hashtable_second_run{1, 0, 0};
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_second_run,
                                         (int32_t*)cached_q2_ht_v2->getCpuBuffer()));

  run_multiple_agg(q3, ExecutorDeviceType::CPU);
  std::shared_ptr<PerfectHashTable> cached_q3_ht_v2 =
      std::dynamic_pointer_cast<PerfectHashTable>(
          getCachedHashTable(visited_hashtable_key, CacheItemType::PERFECT_HT));
  CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)2);
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_second_run,
                                         (int32_t*)cached_q3_ht_v2->getCpuBuffer()));

  dropTable("cache_invalid_t1");
  dropTable("cache_invalid_t2");
}

TEST(Truncate, JoinCacheInvalidationTest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    dropTable("cache_invalid_t1");
    createTable("cache_invalid_t1", {{"k1", dictType()}});
    insertCsvValues("cache_invalid_t1", "1\n2\n3\n4\n5");

    dropTable("cache_invalid_t2");
    createTable("cache_invalid_t2", {{"k2", dictType()}});
    insertCsvValues("cache_invalid_t2", "0\n0\n0\n0\n0\n1\n2\n3\n4\n5");

    auto res_before_truncate = run_multiple_agg(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(5), res_before_truncate->rowCount());
    CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)1);

    dropTable("cache_invalid_t2");
    createTable("cache_invalid_t2", {{"k2", dictType()}});
    auto res_after_truncate = run_multiple_agg(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(0), res_after_truncate->rowCount());
    CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)0);

    insertCsvValues("cache_invalid_t2", "1\n2\n3\n4\n5\n0\n0\n0\n0\n0");

    auto res_before_truncate_v2 = run_multiple_agg(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(5), res_before_truncate_v2->rowCount());
    CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)1);

    dropTable("cache_invalid_t1");
    dropTable("cache_invalid_t2");
  }
}

TEST(Delete, JoinCacheInvalidationTest_DropTable) {
  // todo: when we support per-table cached hashtable invalidation,
  // then this test should be changed either
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    dropTable("cache_invalid_t1");
    createTable("cache_invalid_t1", {{"k1", dictType()}});
    insertCsvValues("cache_invalid_t1", "1\n2\n3\n4\n5");

    dropTable("cache_invalid_t2");
    createTable("cache_invalid_t2", {{"k2", dictType()}});
    insertCsvValues("cache_invalid_t2", "0\n0\n0\n0\n0\n1\n2\n3\n4\n5");

    auto res = run_multiple_agg(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(5), res->rowCount());
    CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)1);

    // add and drop dummy table
    createTable("cache_invalid_t3", {{"dummy", dictType()}});
    dropTable("cache_invalid_t3");
    // we should have no cached hashtable after dropping a table
    CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)0);

    auto res_v2 = run_multiple_agg(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(5), res_v2->rowCount());
    CHECK_EQ(getNumberOfCachedPerfectHashTables(), (unsigned long)1);

    dropTable("cache_invalid_t1");
    dropTable("cache_invalid_t2");
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  init();

  int err{0};

  // enable overlaps hashjoin
  g_enable_overlaps_hashjoin = true;

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  reset();
  return err;
}
