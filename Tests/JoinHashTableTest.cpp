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

#include "ArrowSQLRunner/ArrowSQLRunner.h"
#include "TestHelpers.h"

#include "DataMgr/DataMgr.h"
#include "DataMgr/DataMgrBufferProvider.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/ResultSet.h"

#include <gtest/gtest.h>
#include <boost/program_options.hpp>

#include <exception>
#include <memory>
#include <ostream>
#include <set>
#include <vector>

using namespace TestHelpers;
using namespace TestHelpers::ArrowSQLRunner;

namespace {
ExecutorDeviceType g_device_type;
}

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

int deviceCount(const ExecutorDeviceType device_type) {
  if (device_type == ExecutorDeviceType::GPU) {
    const auto cuda_mgr = getDataMgr()->getCudaMgr();
    CHECK(cuda_mgr);
    return cuda_mgr->getDeviceCount();
  } else {
    return 1;
  }
}

std::shared_ptr<HashJoin> buildPerfect(std::string_view table1,
                                       std::string_view column1,
                                       std::string_view table2,
                                       std::string_view column2) {
  auto executor = Executor::getExecutor(
      Executor::UNITARY_EXECUTOR_ID, getDataMgr(), getDataMgr()->getBufferProvider());
  CHECK(executor);
  auto storage = getStorage();
  executor->setSchemaProvider(storage);

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(g_device_type);

  ColumnCacheMap column_cache;

  return HashJoin::getSyntheticInstance(TEST_DB_ID,
                                        table1,
                                        column1,
                                        table2,
                                        column2,
                                        memory_level,
                                        HashType::OneToOne,
                                        device_count,
                                        getDataMgr()->getDataProvider(),
                                        column_cache,
                                        executor.get());
}

std::shared_ptr<HashJoin> buildKeyed(std::shared_ptr<Analyzer::BinOper> op) {
  auto executor =
      Executor::getExecutor(TEST_DB_ID, getDataMgr(), getDataMgr()->getBufferProvider());
  CHECK(executor);
  auto storage = getStorage();
  executor->setSchemaProvider(storage);

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(g_device_type);

  ColumnCacheMap column_cache;

  return HashJoin::getSyntheticInstance(op,
                                        memory_level,
                                        HashType::OneToOne,
                                        device_count,
                                        getDataMgr()->getDataProvider(),
                                        column_cache,
                                        executor.get());
}

std::pair<std::string, std::shared_ptr<HashJoin>> checkProperQualDetection(
    std::vector<std::shared_ptr<Analyzer::BinOper>> quals) {
  auto executor =
      Executor::getExecutor(TEST_DB_ID, getDataMgr(), getDataMgr()->getBufferProvider());
  CHECK(executor);
  auto storage = getStorage();
  executor->setSchemaProvider(storage);

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(g_device_type);

  ColumnCacheMap column_cache;

  return HashJoin::getSyntheticInstance(quals,
                                        memory_level,
                                        HashType::OneToOne,
                                        device_count,
                                        getDataMgr()->getDataProvider(),
                                        column_cache,
                                        executor.get());
}

TEST(Build, PerfectOneToOne1) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | perfect one-to-one | payloads 0 1 2 3 4 5 6 7 8 9 |
    const DecodedJoinHashBufferSet s1 = {{{0}, {0}},
                                         {{1}, {1}},
                                         {{2}, {2}},
                                         {{3}, {3}},
                                         {{4}, {4}},
                                         {{5}, {5}},
                                         {{6}, {6}},
                                         {{7}, {7}},
                                         {{8}, {8}},
                                         {{9}, {9}}};

    createTable("table1", {{"nums1", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1\n8");

    createTable("table2", {{"nums2", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n2\n3\n4\n5\n6\n7\n8\n9");

    auto hash_table = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table->getHashType(), HashType::OneToOne);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
  }
}

TEST(Build, PerfectOneToOne2) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | perfect one-to-one | payloads 0 1 2 * 3 4 5 6 * 7 |
    const DecodedJoinHashBufferSet s1 = {{{0}, {0}},
                                         {{1}, {1}},
                                         {{2}, {2}},
                                         {{4}, {3}},
                                         {{5}, {4}},
                                         {{6}, {5}},
                                         {{7}, {6}},
                                         {{9}, {7}}};

    createTable("table1", {{"nums1", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1\n8");

    createTable("table2", {{"nums2", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n2\n4\n5\n6\n7\n9");

    auto hash_table = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table->getHashType(), HashType::OneToOne);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
  }
}

TEST(Build, PerfectOneToMany1) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | perfect one-to-many | offsets 0 2 4 6 8 | counts 2 2 2 2 2 | payloads 0 5 1 6 2 7
    // 3 8 4 9 |
    const DecodedJoinHashBufferSet s1 = {
        {{0}, {0, 5}}, {{1}, {1, 6}}, {{2}, {2, 7}}, {{3}, {3, 8}}, {{4}, {4, 9}}};

    createTable("table1", {{"nums1", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1\n8");

    createTable("table2", {{"nums2", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n2\n3\n4\n0\n1\n2\n3\n4");

    auto hash_table = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
  }
}

TEST(Build, PerfectOneToMany2) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | perfect one-to-many | offsets 0 * 2 4 6 | counts 2 * 2 2 2 | payloads 0 4 1 5 2 6
    // 3 7 |
    const DecodedJoinHashBufferSet s1 = {
        {{0}, {0, 4}}, {{2}, {1, 5}}, {{3}, {2, 6}}, {{4}, {3, 7}}};

    createTable("table1", {{"nums1", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1\n8");

    createTable("table2", {{"nums2", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n2\n3\n4\n0\n2\n3\n4");

    auto hash_table = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
  }
}

TEST(Build, detectProperJoinQual) {
  auto executor =
      Executor::getExecutor(TEST_DB_ID, getDataMgr(), getDataMgr()->getBufferProvider())
          .get();
  CHECK(executor);
  auto storage = getStorage();
  executor->setSchemaProvider(storage);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | perfect one-to-many | offsets 0 2 4 6 8 | counts 2 2 2 2 2 | payloads 0 5 1 6 2 7
    // 3 8 4 9 |
    const DecodedJoinHashBufferSet s1 = {
        {{0}, {0, 5}}, {{1}, {1, 6}}, {{2}, {2, 7}}, {{3}, {3, 8}}, {{4}, {4, 9}}};

    createTable("table1", {{"t11", SQLTypeInfo(kINT)}, {"t12", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1,1\n8,1");

    createTable("table2", {{"t21", SQLTypeInfo(kINT)}, {"t22", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0,1\n1,1\n2,1\n3,1\n4,1\n0,1\n1,1\n2,1\n3,1\n4,1");

    Datum d;
    d.intval = 1;
    SQLTypeInfo ti(kINT, 0, 0, false);
    auto c = std::make_shared<Analyzer::Constant>(ti, false, d);

    // case 1: t12 = 1 AND t11 = t21
    // case 2: 1 = t12 AND t11 = t21
    // case 3: t22 = 1 AND t11 = t21
    // case 4: 1 = t22 AND t11 = t21
    auto t11 = getSyntheticColumnVar(TEST_DB_ID, "table1", "t11", 0, executor);
    auto t21 = getSyntheticColumnVar(TEST_DB_ID, "table2", "t21", 1, executor);
    auto qual2 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, t11, t21);
    auto create_join_qual = [&c, &executor](int case_num) {
      std::shared_ptr<Analyzer::ColumnVar> q1_lhs;
      std::shared_ptr<Analyzer::BinOper> qual1;
      switch (case_num) {
        case 1: {
          q1_lhs = getSyntheticColumnVar(TEST_DB_ID, "table1", "t12", 0, executor);
          qual1 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, c, q1_lhs);
          break;
        }
        case 2: {
          q1_lhs = getSyntheticColumnVar(TEST_DB_ID, "table1", "t12", 0, executor);
          qual1 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, q1_lhs, c);
          break;
        }
        case 3: {
          q1_lhs = getSyntheticColumnVar(TEST_DB_ID, "table2", "t22", 1, executor);
          qual1 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, c, q1_lhs);
          break;
        }
        case 4: {
          q1_lhs = getSyntheticColumnVar(TEST_DB_ID, "table2", "t22", 1, executor);
          qual1 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, q1_lhs, c);
          break;
        }
        default:
          break;
      }
      return qual1;
    };

    for (int i = 1; i <= 4; ++i) {
      auto qual1 = create_join_qual(i);
      std::vector<std::shared_ptr<Analyzer::BinOper>> quals;
      quals.push_back(qual1);
      quals.push_back(qual2);
      auto res = checkProperQualDetection(quals);
      auto hash_table = res.second;
      EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);
      auto s2 = hash_table->toSet(g_device_type, 0);
      EXPECT_EQ(s1, s2);
      EXPECT_TRUE(!res.first.empty());
    }

    dropTable("table1");
    dropTable("table2");
  }
}

TEST(Build, KeyedOneToOne) {
  auto executor =
      Executor::getExecutor(TEST_DB_ID, getDataMgr(), getDataMgr()->getBufferProvider());
  CHECK(executor);
  auto storage = getStorage();
  executor->setSchemaProvider(storage);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | keyed one-to-one | keys * (1,1,1) (3,3,2) (0,0,0) * * |
    const DecodedJoinHashBufferSet s1 = {{{0, 0}, {0}}, {{1, 1}, {1}}, {{3, 3}, {2}}};

    createTable("table1", {{"a1", SQLTypeInfo(kINT)}, {"a2", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1,11\n2,12\n3,13\n4,14");

    createTable("table2", {{"b", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n3");

    auto a1 = getSyntheticColumnVar(TEST_DB_ID, "table1", "a1", 0, executor.get());
    auto a2 = getSyntheticColumnVar(TEST_DB_ID, "table1", "a2", 0, executor.get());
    auto b = getSyntheticColumnVar(TEST_DB_ID, "table2", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    auto et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    auto et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table = buildKeyed(op);

    EXPECT_EQ(hash_table->getHashType(), HashType::OneToOne);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
  }
}

TEST(Build, KeyedOneToMany) {
  auto executor =
      Executor::getExecutor(TEST_DB_ID, getDataMgr(), getDataMgr()->getBufferProvider());
  CHECK(executor);
  auto storage = getStorage();
  executor->setSchemaProvider(storage);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | keyed one-to-many | keys * (1,1) (3,3) (0,0) * * | offsets * 0 1 3 * * | counts *
    // 1 2 1 * * | payloads 1 2 3 0 |
    const DecodedJoinHashBufferSet s1 = {{{0}, {0}}, {{1}, {1}}, {{3}, {2, 3}}};

    createTable("table1", {{"a1", SQLTypeInfo(kINT)}, {"a2", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1,11\n2,12\n3,13\n4,14");

    createTable("table2", {{"b", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n3\n3");

    auto a1 = getSyntheticColumnVar(TEST_DB_ID, "table1", "a1", 0, executor.get());
    auto a2 = getSyntheticColumnVar(TEST_DB_ID, "table1", "a2", 0, executor.get());
    auto b = getSyntheticColumnVar(TEST_DB_ID, "table2", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    auto et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    auto et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table = buildKeyed(op);

    EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
  }
}

TEST(MultiFragment, PerfectOneToOne) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    createTable("table1", {{"nums1", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1\n7");

    createTable("table2", {{"nums2", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n2\n3\n4\n5\n6\n7\n8\n9");

    createTable("table3", {{"nums3", SQLTypeInfo(kINT)}}, {3});
    insertCsvValues("table3", "1\n7");

    createTable("table4", {{"nums4", SQLTypeInfo(kINT)}}, {3});
    insertCsvValues("table4", "0\n1\n2\n3\n4\n5\n6\n7\n8\n9");

    auto hash_table1 = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table1->getHashType(), HashType::OneToOne);

    auto hash_table2 = buildPerfect("table3", "nums3", "table4", "nums4");
    EXPECT_EQ(hash_table2->getHashType(), HashType::OneToOne);

    // | perfect one-to-one | payloads 0 1 2 3 4 5 6 7 8 9 |
    auto s1 = hash_table1->toSet(g_device_type, 0);
    auto s2 = hash_table2->toSet(g_device_type, 0);
    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
    dropTable("table3");
    dropTable("table4");
  }
}

TEST(MultiFragment, PerfectOneToMany) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    createTable("table1", {{"nums1", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1\n7");

    createTable("table2", {{"nums2", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n9");

    createTable("table3", {{"nums3", SQLTypeInfo(kINT)}}, {3});
    insertCsvValues("table3", "1\n7");

    createTable("table4", {{"nums4", SQLTypeInfo(kINT)}}, {3});
    insertCsvValues("table4", "0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n9");

    auto hash_table1 = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table1->getHashType(), HashType::OneToMany);

    auto hash_table2 = buildPerfect("table3", "nums3", "table4", "nums4");
    EXPECT_EQ(hash_table2->getHashType(), HashType::OneToMany);

    // | perfect one-to-many | offsets 0 1 2 3 4 5 6 7 8 9 | counts 1 1 1 1 1 1 1 1 1 2 |
    // payloads 0 1 2 3 4 5 6 7 8 9 10 |
    auto s1 = hash_table1->toSet(g_device_type, 0);
    auto s2 = hash_table2->toSet(g_device_type, 0);
    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
    dropTable("table3");
    dropTable("table4");
  }
}

TEST(MultiFragment, KeyedOneToOne) {
  auto executor =
      Executor::getExecutor(TEST_DB_ID, getDataMgr(), getDataMgr()->getBufferProvider());
  CHECK(executor);
  auto storage = getStorage();
  executor->setSchemaProvider(storage);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    createTable("table1", {{"a1", SQLTypeInfo(kINT)}, {"a2", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1,11\n2,12\n3,13\n4,14");

    createTable("table2", {{"b", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n3");

    auto a1 = getSyntheticColumnVar(TEST_DB_ID, "table1", "a1", 0, executor.get());
    auto a2 = getSyntheticColumnVar(TEST_DB_ID, "table1", "a2", 0, executor.get());
    auto b = getSyntheticColumnVar(TEST_DB_ID, "table2", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    auto et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    auto et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table1 = buildKeyed(op);
    auto baseline = std::dynamic_pointer_cast<BaselineJoinHashTable>(hash_table1);
    CHECK(baseline);
    EXPECT_EQ(hash_table1->getHashType(), HashType::OneToOne);

    createTable("table3", {{"a1", SQLTypeInfo(kINT)}, {"a2", SQLTypeInfo(kINT)}}, {1});
    insertCsvValues("table3", "1,11\n2,12\n3,13\n4,14");

    createTable("table4", {{"b", SQLTypeInfo(kINT)}}, {1});
    insertCsvValues("table4", "0\n1\n3");

    a1 = getSyntheticColumnVar(TEST_DB_ID, "table3", "a1", 0, executor.get());
    a2 = getSyntheticColumnVar(TEST_DB_ID, "table3", "a2", 0, executor.get());
    b = getSyntheticColumnVar(TEST_DB_ID, "table4", "b", 1, executor.get());

    et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table2 = buildKeyed(op);
    EXPECT_EQ(hash_table2->getHashType(), HashType::OneToOne);

    //
    auto s1 = hash_table1->toSet(g_device_type, 0);
    auto s2 = hash_table2->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
    dropTable("table3");
    dropTable("table4");
  }
}

TEST(MultiFragment, KeyedOneToMany) {
  auto executor =
      Executor::getExecutor(TEST_DB_ID, getDataMgr(), getDataMgr()->getBufferProvider());
  CHECK(executor);
  auto storage = getStorage();
  executor->setSchemaProvider(storage);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    createTable("table1", {{"a1", SQLTypeInfo(kINT)}, {"a2", SQLTypeInfo(kINT)}});
    insertCsvValues("table1", "1,11\n2,12\n3,13\n4,14");

    createTable("table2", {{"b", SQLTypeInfo(kINT)}});
    insertCsvValues("table2", "0\n1\n3\n3");

    auto a1 = getSyntheticColumnVar(TEST_DB_ID, "table1", "a1", 0, executor.get());
    auto a2 = getSyntheticColumnVar(TEST_DB_ID, "table1", "a2", 0, executor.get());
    auto b = getSyntheticColumnVar(TEST_DB_ID, "table2", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    auto et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    auto et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table1 = buildKeyed(op);
    EXPECT_EQ(hash_table1->getHashType(), HashType::OneToMany);

    createTable("table3", {{"a1", SQLTypeInfo(kINT)}, {"a2", SQLTypeInfo(kINT)}}, {1});
    insertCsvValues("table3", "1,11\n2,12\n3,13\n4,14");

    createTable("table4", {{"b", SQLTypeInfo(kINT)}}, {1});
    insertCsvValues("table4", "0\n1\n3\n3");

    a1 = getSyntheticColumnVar(TEST_DB_ID, "table3", "a1", 0, executor.get());
    a2 = getSyntheticColumnVar(TEST_DB_ID, "table3", "a2", 0, executor.get());
    b = getSyntheticColumnVar(TEST_DB_ID, "table4", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table2 = buildKeyed(op);
    EXPECT_EQ(hash_table2->getHashType(), HashType::OneToMany);

    //
    auto s1 = hash_table1->toSet(g_device_type, 0);
    auto s2 = hash_table2->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    dropTable("table1");
    dropTable("table2");
    dropTable("table3");
    dropTable("table4");
  }
}

TEST(Other, Regression) {
  createTable("table_a",
              {{"Small_int", SQLTypeInfo(kSMALLINT)},
               {"dest_state", dictType()},
               {"str", kTEXT}});
  insertCsvValues("table_a", "1,testa_1,str1\n1,testa_2,str1");

  createTable("table_b",
              {{"Small_int", SQLTypeInfo(kSMALLINT)},
               {"dest_state", dictType()},
               {"str", kTEXT}});
  insertCsvValues("table_b", "1,testb_1,str1\n2,testb_2,str1");

  EXPECT_THROW(run_multiple_agg(R"(
        SELECT table_a.dest_state AS key0 FROM table_a, table_b WHERE (table_b.dest_state = table_a.Small_int) GROUP BY key0 LIMIT 1000000;
      )",
                                ExecutorDeviceType::CPU),
               std::runtime_error);

  dropTable("table_a");
  dropTable("table_b");
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  init();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  reset();
  return err;
}
