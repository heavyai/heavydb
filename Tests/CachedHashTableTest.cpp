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

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <csignal>
#include <exception>
#include <memory>
#include <ostream>
#include <set>
#include <vector>
#include "Catalog/Catalog.h"
#include "Catalog/DBObject.h"
#include "DataMgr/DataMgr.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/MurmurHash1Inl.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/UDFCompiler.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/Logger.h"
#include "Shared/MapDParameters.h"
#include "TestHelpers.h"

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
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
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
bool check_one_to_one_join_hashtable(
    std::vector<int32_t>& inserted_keys,
    const std::shared_ptr<std::vector<int32_t>>& cached_hashtable) {
  auto hashtable_info = get_join_hashtable_info(inserted_keys);
  int32_t min = hashtable_info.get_min_val();
  int32_t max = hashtable_info.get_max_val();
  int32_t rowID = 0;
  for (int32_t v : inserted_keys) {
    int32_t offset = v - min;
    CHECK_GE(offset, 0);
    CHECK_LE(offset, max);
    if ((*cached_hashtable)[offset] != rowID) {
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

bool check_one_to_many_baseline_hashtable(
    std::vector<std::vector<int32_t>>& insert_keys,
    const std::shared_ptr<std::vector<int8_t>>& baseline_hashtable,
    size_t entry_count) {
  const int8_t* cached_hashtable = baseline_hashtable->data();
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

bool check_one_to_many_join_hashtable(
    std::vector<int32_t>& inserted_keys,
    const std::shared_ptr<std::vector<int32_t>>& cached_hashtable) {
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
    int32_t PSV = (*cached_hashtable)[offset];  // Prefix Sum Value
    if (PSV != -1) {
      int32_t CV = (*cached_hashtable)[count_buff_start_offset + offset];
      CHECK_GE(CV, 1);  // Count Value
      bool found_matching_key = false;
      for (int32_t idx = 0; idx < CV; idx++) {
        if ((*cached_hashtable)[rowID_buff_start_offset + PSV + idx] == rowID) {
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
bool check_one_to_one_baseline_hashtable(
    std::vector<std::vector<int32_t>>& insert_keys,
    const std::shared_ptr<std::vector<int8_t>>& baseline_hashtable,
    size_t entry_count) {
  int8_t* cached_hashtable = baseline_hashtable->data();
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
    std::string row0{"INSERT INTO cache_invalid_t1 VALUES (1, 1, 'row-0');"};
    std::string row1{"INSERT INTO cache_invalid_t1 VALUES (0, 0, 'row-1');"};
    row_insert_sql.push_back(row0);
    row_insert_sql.push_back(row1);
  } else {
    std::string row0{"INSERT INTO cache_invalid_t1 VALUES (0, 0, 'row-0');"};
    std::string row1{"INSERT INTO cache_invalid_t1 VALUES (1, 1, 'row-1');"};
    row_insert_sql.push_back(row0);
    row_insert_sql.push_back(row1);
  }

  std::string row0{"INSERT INTO cache_invalid_t2 VALUES (1, 1, 'row-0');"};
  std::string row1{"INSERT INTO cache_invalid_t2 VALUES (1, 1, 'row-1');"};
  std::string row2{"INSERT INTO cache_invalid_t2 VALUES (1, 1, 'row-2');"};
  row_insert_sql.push_back(row0);
  row_insert_sql.push_back(row1);
  row_insert_sql.push_back(row2);
  for (std::string insert_str : row_insert_sql) {
    run_query(insert_str, ExecutorDeviceType::CPU);
  }
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
    std::string row0{"INSERT INTO cache_invalid_t1 VALUES (1, 1, 1, 2);"};
    std::string row1{"INSERT INTO cache_invalid_t1 VALUES (0, 0, 1, 2);"};
    std::string row2{"INSERT INTO cache_invalid_t1 VALUES (0, 0, 2, 1);"};
    row_insert_sql.push_back(row0);
    row_insert_sql.push_back(row1);
    row_insert_sql.push_back(row2);
  } else {
    std::string row0{"INSERT INTO cache_invalid_t1 VALUES (0, 0, 1, 2);"};
    std::string row1{"INSERT INTO cache_invalid_t1 VALUES (0, 0, 2, 1);"};
    std::string row2{"INSERT INTO cache_invalid_t1 VALUES (1, 1, 1, 2);"};
    row_insert_sql.push_back(row0);
    row_insert_sql.push_back(row1);
    row_insert_sql.push_back(row2);
  }

  std::string t2_row{"INSERT INTO cache_invalid_t2 VALUES (1, 1);"};
  row_insert_sql.push_back(t2_row);
  row_insert_sql.push_back(t2_row);
  row_insert_sql.push_back(t2_row);
  row_insert_sql.push_back(t2_row);
  row_insert_sql.push_back(t2_row);
  row_insert_sql.push_back(t2_row);
  for (std::string insert_str : row_insert_sql) {
    run_query(insert_str, ExecutorDeviceType::CPU);
  }
}

TEST(Select, DropAndReCreate_OneToOne_HashTable_WithReversedTupleInsertion) {
  // tuple insertion order is controlled by a bool param. of an import function
  import_tables_cache_invalidation_for_CPU_one_to_one_join(false);

  // (a) baseline hash join, the first run] tuple insertion order: (0, 0) -> (1, 1)
  run_query(
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on "
      "t1.id1 = t2.id1 and t1.id2 = t2.id2;",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<int32_t>> baseline_hashtable_first_run;
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{1, 1});
  CHECK_EQ(QR::get()->getNumberOfCachedBaselineJoinHashTables(), (unsigned long)1);
  CHECK(check_one_to_one_baseline_hashtable(
      baseline_hashtable_first_run,
      QR::get()->getCachedBaselineHashTable(0),
      QR::get()->getEntryCntCachedBaselineHashTable(0)));
  // (b) perfect hash join, the first run] tuple insertion order: 0 -> 1
  run_query(
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on "
      "t1.id1 = t2.id1;",
      ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);
  std::vector<int32_t> perfect_hashtable_first_run{0, 1};
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_first_run,
                                        QR::get()->getCachedJoinHashTable(0)));

  run_query(
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on "
      "t1.id2 = t2.id2;",
      ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)2);
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_first_run,
                                        QR::get()->getCachedJoinHashTable(1)));

  // the second run --> reversed tuple insertion order compared with the first run
  import_tables_cache_invalidation_for_CPU_one_to_one_join(true);

  // make sure we invalidate all cached hashtables after tables are dropped
  CHECK_EQ(QR::get()->getNumberOfCachedBaselineJoinHashTables(), (unsigned long)0);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)0);

  // (a) baseline hash join, the second run] tuple insertion order: (1, 1) -> (0, 0)
  run_query(
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on "
      "t1.id1 = t2.id1 and t1.id2 = t2.id2;",
      ExecutorDeviceType::CPU);
  std::vector<std::vector<int32_t>> baseline_hashtable_second_run;
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{1, 1});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  CHECK_EQ(QR::get()->getNumberOfCachedBaselineJoinHashTables(), (unsigned long)1);
  CHECK(check_one_to_one_baseline_hashtable(
      baseline_hashtable_second_run,
      QR::get()->getCachedBaselineHashTable(0),
      QR::get()->getEntryCntCachedBaselineHashTable(0)));

  // (a) perfect hash join, the second run] tuple insertion order: 1 -> 0
  run_query(
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on "
      "t1.id1 = t2.id1;",
      ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);
  std::vector<int32_t> perfect_hashtable_second_run{1, 0};
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_second_run,
                                        QR::get()->getCachedJoinHashTable(0)));

  run_query(
      "SELECT t1.id1, t2.id1 FROM cache_invalid_t1 t1 join cache_invalid_t2 t2 on "
      "t1.id2 = t2.id2;",
      ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)2);
  CHECK(check_one_to_one_join_hashtable(perfect_hashtable_second_run,
                                        QR::get()->getCachedJoinHashTable(0)));

  run_ddl_statement("DROP TABLE cache_invalid_t1;");
  run_ddl_statement("DROP TABLE cache_invalid_t2;");
}

TEST(Select, DropAndReCreate_OneToMany_HashTable_WithReversedTupleInsertion) {
  // tuple insertion order is controlled by a bool param. of an import function
  import_tables_cache_invalidation_for_CPU_one_to_many_join(false);

  // (a) baseline hash join, the first run]
  // tuple insertion order: (0, 0) -> (0, 0) -> (1,1)
  run_query(
      "select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k1 = t1.k1 and "
      "t0.k2 = t1.k2;",
      ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedBaselineJoinHashTables(), (unsigned long)1);
  std::vector<std::vector<int32_t>> baseline_hashtable_first_run;
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_first_run.push_back(std::vector<int32_t>{1, 1});
  CHECK(check_one_to_many_baseline_hashtable(
      baseline_hashtable_first_run,
      QR::get()->getCachedBaselineHashTable(0),
      QR::get()->getEntryCntCachedBaselineHashTable(0)));

  // (b) perfect hash join, the first run] tuple insertion order: 0 -> 0 -> 1
  run_query("select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k1 = t1.k1;",
            ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);
  std::vector<int32_t> perfect_hashtable_first_run{0, 0, 1};
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_first_run,
                                         QR::get()->getCachedJoinHashTable(0)));

  run_query("select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k2 = t1.k2;",
            ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)2);
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_first_run,
                                         QR::get()->getCachedJoinHashTable(0)));

  // [the second run] tuple insertion order: (1, 1) -> (0, 0) -> (0, 0)
  import_tables_cache_invalidation_for_CPU_one_to_many_join(true);

  // make sure we invalidate all cached hashtables after tables are dropped
  CHECK_EQ(QR::get()->getNumberOfCachedBaselineJoinHashTables(), (unsigned long)0);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)0);

  // (a) baseline hash join, the second run] tuple insertion order: (1, 1) -> (0, 0) ->
  // (0, 0)
  run_query(
      "select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k1 = t1.k1 and "
      "t0.k2 = t1.k2;",
      ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedBaselineJoinHashTables(), (unsigned long)1);
  std::vector<std::vector<int32_t>> baseline_hashtable_second_run;
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{1, 1});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  baseline_hashtable_second_run.push_back(std::vector<int32_t>{0, 0});
  CHECK(check_one_to_many_baseline_hashtable(
      baseline_hashtable_second_run,
      QR::get()->getCachedBaselineHashTable(0),
      QR::get()->getEntryCntCachedBaselineHashTable(0)));

  // (b) perfect hash join, the second run] tuple insertion order: 1 -> 0 -> 0
  run_query("select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k1 = t1.k1;",
            ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);
  std::vector<int32_t> perfect_hashtable_second_run{1, 0, 0};
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_second_run,
                                         QR::get()->getCachedJoinHashTable(0)));

  run_query("select * from cache_invalid_t1 t0, cache_invalid_t2 t1 where t0.k2 = t1.k2;",
            ExecutorDeviceType::CPU);
  CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)2);
  CHECK(check_one_to_many_join_hashtable(perfect_hashtable_second_run,
                                         QR::get()->getCachedJoinHashTable(0)));

  run_ddl_statement("DROP TABLE cache_invalid_t1;");
  run_ddl_statement("DROP TABLE cache_invalid_t2;");
}

TEST(Truncate, JoinCacheInvalidationTest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_t1;");
    run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_t2;");

    run_ddl_statement("create table cache_invalid_t1 (k1 text encoding dict(32));");
    run_ddl_statement("create table cache_invalid_t2 (k2 text encoding dict(32));");
    run_query("insert into cache_invalid_t1 values ('1');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t1 values ('2');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t1 values ('3');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t1 values ('4');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t1 values ('5');", ExecutorDeviceType::CPU);

    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('1');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('2');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('3');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('4');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('5');", ExecutorDeviceType::CPU);

    auto res_before_truncate = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(5), res_before_truncate->rowCount());
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);

    run_ddl_statement("truncate table cache_invalid_t2;");
    auto res_after_truncate = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(0), res_after_truncate->rowCount());
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)0);

    run_query("insert into cache_invalid_t2 values ('1');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('2');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('3');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('4');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('5');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);

    auto res_before_truncate_v2 = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(5), res_before_truncate_v2->rowCount());
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);

    run_ddl_statement("DROP TABLE cache_invalid_t1;");
    run_ddl_statement("DROP TABLE cache_invalid_t2;");
  }
}

TEST(Update, JoinCacheInvalidationTest) {
  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }
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

    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);

    run_query("update string_join1 set t='not poutine' where t='poutine';", dt);
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)0);

    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_query(
            "select count(string_join1.t) from string_join1 inner join string_join2 on "
            "string_join1.t = string_join2.t;",
            dt)));
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);

    run_ddl_statement("drop table string_join1;");
    run_ddl_statement("drop table string_join2;");
  }
}

TEST(Delete, JoinCacheInvalidationTest) {
  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }
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

    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);

    run_query("delete from string_join1 where t='poutine';", dt);
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)0);

    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_query(
            "select count(string_join1.t) from string_join1 inner join string_join2 on "
            "string_join1.t = string_join2.t;",
            dt)));
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);

    run_ddl_statement("drop table string_join1;");
    run_ddl_statement("drop table string_join2;");
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
    run_query("insert into cache_invalid_t1 values ('1');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t1 values ('2');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t1 values ('3');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t1 values ('4');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t1 values ('5');", ExecutorDeviceType::CPU);

    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('0');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('1');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('2');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('3');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('4');", ExecutorDeviceType::CPU);
    run_query("insert into cache_invalid_t2 values ('5');", ExecutorDeviceType::CPU);

    auto res = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(5), res->rowCount());
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);

    // add and drop dummy table
    run_ddl_statement("create table cache_invalid_t3 (dummy text encoding dict(32));");
    run_ddl_statement("DROP TABLE IF EXISTS cache_invalid_t3;");
    // we should have no cached hashtable after dropping a table
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)0);

    auto res_v2 = QR::get()->runSQL(
        "select * from cache_invalid_t1, cache_invalid_t2 where k1 = k2;", dt);
    ASSERT_EQ(static_cast<uint32_t>(5), res_v2->rowCount());
    CHECK_EQ(QR::get()->getNumberOfCachedJoinHashTables(), (unsigned long)1);

    run_ddl_statement("DROP TABLE cache_invalid_t1;");
    run_ddl_statement("DROP TABLE cache_invalid_t2;");
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::DEBUG1;
  logger::init(log_options);

  QR::init(BASE_PATH);

  int err{0};

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  QR::reset();
  return err;
}