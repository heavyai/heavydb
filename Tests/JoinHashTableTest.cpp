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
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/JoinHashTable/OverlapsJoinHashTable.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/thread_count.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace Catalog_Namespace;
using namespace TestHelpers;

using QR = QueryRunner::QueryRunner;

namespace {
ExecutorDeviceType g_device_type;
}

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

inline auto sql(std::string_view sql_stmts) {
  return QR::get()->runMultipleStatements(std::string(sql_stmts), g_device_type);
}

int deviceCount(const Catalog_Namespace::Catalog* catalog,
                const ExecutorDeviceType device_type) {
  if (device_type == ExecutorDeviceType::GPU) {
    const auto cuda_mgr = catalog->getDataMgr().getCudaMgr();
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
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(catalog.get(), g_device_type);

  ColumnCacheMap column_cache;

  return HashJoin::getSyntheticInstance(table1,
                                        column1,
                                        table2,
                                        column2,
                                        memory_level,
                                        HashType::OneToOne,
                                        device_count,
                                        column_cache,
                                        executor.get());
}

std::shared_ptr<HashJoin> buildKeyed(std::shared_ptr<Analyzer::BinOper> op) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(catalog.get(), g_device_type);

  ColumnCacheMap column_cache;

  return HashJoin::getSyntheticInstance(
      op, memory_level, HashType::OneToOne, device_count, column_cache, executor.get());
}

std::pair<std::string, std::shared_ptr<HashJoin>> checkProperQualDetection(
    std::vector<std::shared_ptr<Analyzer::BinOper>> quals) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(catalog.get(), g_device_type);

  ColumnCacheMap column_cache;

  return HashJoin::getSyntheticInstance(quals,
                                        memory_level,
                                        HashType::OneToOne,
                                        device_count,
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

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (nums1 integer);
      create table table2 (nums2 integer);

      insert into table1 values (1);
      insert into table1 values (8);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (2);
      insert into table2 values (3);
      insert into table2 values (4);
      insert into table2 values (5);
      insert into table2 values (6);
      insert into table2 values (7);
      insert into table2 values (8);
      insert into table2 values (9);
    )");

    auto hash_table = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table->getHashType(), HashType::OneToOne);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
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

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (nums1 integer);
      create table table2 (nums2 integer);

      insert into table1 values (1);
      insert into table1 values (8);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (2);
      insert into table2 values (4);
      insert into table2 values (5);
      insert into table2 values (6);
      insert into table2 values (7);
      insert into table2 values (9);
    )");

    auto hash_table = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table->getHashType(), HashType::OneToOne);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
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

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (nums1 integer);
      create table table2 (nums2 integer);

      insert into table1 values (1);
      insert into table1 values (8);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (2);
      insert into table2 values (3);
      insert into table2 values (4);
      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (2);
      insert into table2 values (3);
      insert into table2 values (4);
    )");

    auto hash_table = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
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

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (nums1 integer);
      create table table2 (nums2 integer);

      insert into table1 values (1);
      insert into table1 values (8);

      insert into table2 values (0);
      insert into table2 values (2);
      insert into table2 values (3);
      insert into table2 values (4);
      insert into table2 values (0);
      insert into table2 values (2);
      insert into table2 values (3);
      insert into table2 values (4);
    )");

    auto hash_table = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

TEST(Build, detectProperJoinQual) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId).get();
  CHECK(executor);
  executor->setCatalog(catalog.get());

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | perfect one-to-many | offsets 0 2 4 6 8 | counts 2 2 2 2 2 | payloads 0 5 1 6 2 7
    // 3 8 4 9 |
    const DecodedJoinHashBufferSet s1 = {
        {{0}, {0, 5}}, {{1}, {1, 6}}, {{2}, {2, 7}}, {{3}, {3, 8}}, {{4}, {4, 9}}};

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (t11 integer, t12 integer);
      create table table2 (t21 integer, t22 integer);

      insert into table1 values (1, 1);
      insert into table1 values (8, 1);

      insert into table2 values (0, 1);
      insert into table2 values (1, 1);
      insert into table2 values (2, 1);
      insert into table2 values (3, 1);
      insert into table2 values (4, 1);
      insert into table2 values (0, 1);
      insert into table2 values (1, 1);
      insert into table2 values (2, 1);
      insert into table2 values (3, 1);
      insert into table2 values (4, 1);
    )");

    Datum d;
    d.intval = 1;
    SQLTypeInfo ti(kINT, 0, 0, false);
    auto c = std::make_shared<Analyzer::Constant>(ti, false, d);

    // case 1: t12 = 1 AND t11 = t21
    // case 2: 1 = t12 AND t11 = t21
    // case 3: t22 = 1 AND t11 = t21
    // case 4: 1 = t22 AND t11 = t21
    auto t11 = getSyntheticColumnVar("table1", "t11", 0, executor);
    auto t21 = getSyntheticColumnVar("table2", "t21", 1, executor);
    auto qual2 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, t11, t21);
    auto create_join_qual = [&c, &executor](int case_num) {
      std::shared_ptr<Analyzer::ColumnVar> q1_lhs;
      std::shared_ptr<Analyzer::BinOper> qual1;
      switch (case_num) {
        case 1: {
          q1_lhs = getSyntheticColumnVar("table1", "t12", 0, executor);
          qual1 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, c, q1_lhs);
          break;
        }
        case 2: {
          q1_lhs = getSyntheticColumnVar("table1", "t12", 0, executor);
          qual1 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, q1_lhs, c);
          break;
        }
        case 3: {
          q1_lhs = getSyntheticColumnVar("table2", "t22", 1, executor);
          qual1 = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, c, q1_lhs);
          break;
        }
        case 4: {
          q1_lhs = getSyntheticColumnVar("table2", "t22", 1, executor);
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
    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

TEST(Build, KeyedOneToOne) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | keyed one-to-one | keys * (1,1,1) (3,3,2) (0,0,0) * * |
    const DecodedJoinHashBufferSet s1 = {{{0, 0}, {0}}, {{1, 1}, {1}}, {{3, 3}, {2}}};

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (a1 integer, a2 integer);
      create table table2 (b integer);

      insert into table1 values (1, 11);
      insert into table1 values (2, 12);
      insert into table1 values (3, 13);
      insert into table1 values (4, 14);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (3);
    )");

    auto a1 = getSyntheticColumnVar("table1", "a1", 0, executor.get());
    auto a2 = getSyntheticColumnVar("table1", "a2", 0, executor.get());
    auto b = getSyntheticColumnVar("table2", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    auto et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    auto et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table = buildKeyed(op);

    EXPECT_EQ(hash_table->getHashType(), HashType::OneToOne);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

TEST(Build, KeyedOneToMany) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | keyed one-to-many | keys * (1,1) (3,3) (0,0) * * | offsets * 0 1 3 * * | counts *
    // 1 2 1 * * | payloads 1 2 3 0 |
    const DecodedJoinHashBufferSet s1 = {{{0}, {0}}, {{1}, {1}}, {{3}, {2, 3}}};

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (a1 integer, a2 integer);
      create table table2 (b integer);

      insert into table1 values (1, 11);
      insert into table1 values (2, 12);
      insert into table1 values (3, 13);
      insert into table1 values (4, 14);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (3);
      insert into table2 values (3);
    )");

    auto a1 = getSyntheticColumnVar("table1", "a1", 0, executor.get());
    auto a2 = getSyntheticColumnVar("table1", "a2", 0, executor.get());
    auto b = getSyntheticColumnVar("table2", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    auto et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    auto et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table = buildKeyed(op);

    EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

TEST(Build, GeoOneToMany1) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | geo one-to-many | keys * (0,2) * * * (1,1) * * * (0,1) (2,1) (2,0) (2,2) (1,2)
    // (0,0) * * (1,0) | offsets * 0 * * * 1 * * * 5 7 9 10 11 13 * * 14 | counts * 1 * *
    // * 4 * * * 2 2 1 1 2 1 * * 2 | payloads 2 0 2 1 3 0 2 1 3 1 3 2 3 0 0 1 |
    // TODO: Fixup above comment to match below
    const DecodedJoinHashBufferSet s1 = {
        {{0}, {0}},    {{0}, {0, 2}}, {{0}, {2}},    {{1}, {0}},          {{1}, {0, 2}},
        {{1}, {2}},    {{2}, {0}},    {{2}, {0, 2}}, {{2}, {2}},          {{3}, {0}},
        {{3}, {0, 2}}, {{3}, {2}},    {{4}, {0, 1}}, {{4}, {0, 1, 2, 3}}, {{4}, {2, 3}},
        {{5}, {1}},    {{5}, {1, 3}}, {{5}, {3}},    {{6}, {1}},          {{6}, {1, 3}},
        {{6}, {3}},    {{7}, {1}},    {{7}, {1, 3}}, {{7}, {3}},          {{8}, {1}},
        {{8}, {1, 3}}, {{8}, {3}}};

    sql(R"(
      drop table if exists my_points;
      drop table if exists my_grid;

      create table my_points (locations geometry(point, 4326) encoding none);
      create table my_grid (cells geometry(multipolygon, 4326) encoding none);

      insert into my_points values ('point(5 5)');
      insert into my_points values ('point(5 25)');
      insert into my_points values ('point(10 5)');

      insert into my_grid values ('multipolygon(((0 0,1 0,1 1,0 1,0 0)))');
      insert into my_grid values ('multipolygon(((1 0,2 0,2 1,1 1,1 0)))');
      insert into my_grid values ('multipolygon(((0 1,1 1,1 2,0 2,0 1)))');
      insert into my_grid values ('multipolygon(((1 1,2 1,2 2,1 2,1 1)))');
    )");

    auto a1 = getSyntheticColumnVar("my_points", "locations", 0, executor.get());
    auto a2 = getSyntheticColumnVar("my_grid", "cells", 1, executor.get());

    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kOVERLAPS, kONE, a1, a2);

    auto memory_level =
        (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                  : Data_Namespace::GPU_LEVEL);

    auto device_count = deviceCount(catalog.get(), g_device_type);

    ColumnCacheMap column_cache;

    auto hash_table = HashJoin::getSyntheticInstance(op,
                                                     memory_level,
                                                     HashType::OneToMany,
                                                     device_count,
                                                     column_cache,
                                                     executor.get());

    EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

TEST(Build, GeoOneToMany2) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    // | geo one-to-many | keys (11,0) (1,1) * * * * (0,0) * (10,0) (10,1) * (1,0) *
    // (11,1) * (0,1) | offsets 0 1 * * * * 2 * 3 4 * 5 * 6 * 7 | counts 1 1 * * * * 1 * 1
    // 1 * 1 * 1 * 1 | payloads 1 0 0 1 1 0 1 0 |
    const DecodedJoinHashBufferSet s1 = {
        {{0}, {0}}, {{1}, {0}}, {{10}, {1}}, {{11}, {1}}};

    sql(R"(
      drop table if exists my_points;
      drop table if exists my_grid;

      create table my_points (locations geometry(point, 4326) encoding none);
      create table my_grid (cells geometry(multipolygon, 4326) encoding none);

      insert into my_points values ('point(5 5)');
      insert into my_points values ('point(5 25)');
      insert into my_points values ('point(10 5)');

      insert into my_grid values ('multipolygon(((0 0,10 0,10 10,0 10,0 0)))');
      insert into my_grid values ('multipolygon(((100 0,110 0,110 10,100 10,100 0)))');
    )");

    auto a1 = getSyntheticColumnVar("my_points", "locations", 0, executor.get());
    auto a2 = getSyntheticColumnVar("my_grid", "cells", 1, executor.get());

    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kOVERLAPS, kONE, a1, a2);

    auto memory_level =
        (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                  : Data_Namespace::GPU_LEVEL);

    auto device_count = deviceCount(catalog.get(), g_device_type);

    ColumnCacheMap column_cache;

    auto hash_table = HashJoin::getSyntheticInstance(op,
                                                     memory_level,
                                                     HashType::OneToMany,
                                                     device_count,
                                                     column_cache,
                                                     executor.get());

    EXPECT_EQ(hash_table->getHashType(), HashType::OneToMany);

    auto s2 = hash_table->toSet(g_device_type, 0);

    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

TEST(MultiFragment, PerfectOneToOne) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (nums1 integer);
      create table table2 (nums2 integer);

      insert into table1 values (1);
      insert into table1 values (7);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (2);
      insert into table2 values (3);
      insert into table2 values (4);
      insert into table2 values (5);
      insert into table2 values (6);
      insert into table2 values (7);
      insert into table2 values (8);
      insert into table2 values (9);
    )");

    sql(R"(
      drop table if exists table3;
      drop table if exists table4;

      create table table3 (nums3 integer) with (fragment_size = 3);
      create table table4 (nums4 integer) with (fragment_size = 3);

      insert into table3 values (1);
      insert into table3 values (7);

      insert into table4 values (0);
      insert into table4 values (1);
      insert into table4 values (2);
      insert into table4 values (3);
      insert into table4 values (4);
      insert into table4 values (5);
      insert into table4 values (6);
      insert into table4 values (7);
      insert into table4 values (8);
      insert into table4 values (9);
    )");

    auto hash_table1 = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table1->getHashType(), HashType::OneToOne);

    auto hash_table2 = buildPerfect("table3", "nums3", "table4", "nums4");
    EXPECT_EQ(hash_table2->getHashType(), HashType::OneToOne);

    // | perfect one-to-one | payloads 0 1 2 3 4 5 6 7 8 9 |
    auto s1 = hash_table1->toSet(g_device_type, 0);
    auto s2 = hash_table2->toSet(g_device_type, 0);
    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
      drop table if exists table3;
      drop table if exists table4;
    )");
  }
}

TEST(MultiFragment, PerfectOneToMany) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (nums1 integer);
      create table table2 (nums2 integer);

      insert into table1 values (1);
      insert into table1 values (7);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (2);
      insert into table2 values (3);
      insert into table2 values (4);
      insert into table2 values (5);
      insert into table2 values (6);
      insert into table2 values (7);
      insert into table2 values (8);
      insert into table2 values (9);
      insert into table2 values (9);
    )");

    sql(R"(
      drop table if exists table3;
      drop table if exists table4;

      create table table3 (nums3 integer) with (fragment_size = 3);
      create table table4 (nums4 integer) with (fragment_size = 3);

      insert into table3 values (1);
      insert into table3 values (7);

      insert into table4 values (0);
      insert into table4 values (1);
      insert into table4 values (2);
      insert into table4 values (3);
      insert into table4 values (4);
      insert into table4 values (5);
      insert into table4 values (6);
      insert into table4 values (7);
      insert into table4 values (8);
      insert into table4 values (9);
      insert into table4 values (9);
    )");

    auto hash_table1 = buildPerfect("table1", "nums1", "table2", "nums2");
    EXPECT_EQ(hash_table1->getHashType(), HashType::OneToMany);

    auto hash_table2 = buildPerfect("table3", "nums3", "table4", "nums4");
    EXPECT_EQ(hash_table2->getHashType(), HashType::OneToMany);

    // | perfect one-to-many | offsets 0 1 2 3 4 5 6 7 8 9 | counts 1 1 1 1 1 1 1 1 1 2 |
    // payloads 0 1 2 3 4 5 6 7 8 9 10 |
    auto s1 = hash_table1->toSet(g_device_type, 0);
    auto s2 = hash_table2->toSet(g_device_type, 0);
    EXPECT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
      drop table if exists table3;
      drop table if exists table4;
    )");
  }
}

TEST(MultiFragment, KeyedOneToOne) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (a1 integer, a2 integer);
      create table table2 (b integer);

      insert into table1 values (1, 11);
      insert into table1 values (2, 12);
      insert into table1 values (3, 13);
      insert into table1 values (4, 14);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (3);
    )");

    auto a1 = getSyntheticColumnVar("table1", "a1", 0, executor.get());
    auto a2 = getSyntheticColumnVar("table1", "a2", 0, executor.get());
    auto b = getSyntheticColumnVar("table2", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    auto et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    auto et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table1 = buildKeyed(op);
    auto baseline = std::dynamic_pointer_cast<BaselineJoinHashTable>(hash_table1);
    CHECK(baseline);
    EXPECT_EQ(hash_table1->getHashType(), HashType::OneToOne);

    sql(R"(
      drop table if exists table3;
      drop table if exists table4;

      create table table3 (a1 integer, a2 integer) with (fragment_size = 1);
      create table table4 (b integer) with (fragment_size = 1);

      insert into table3 values (1, 11);
      insert into table3 values (2, 12);
      insert into table3 values (3, 13);
      insert into table3 values (4, 14);

      insert into table4 values (0);
      insert into table4 values (1);
      insert into table4 values (3);
    )");

    a1 = getSyntheticColumnVar("table3", "a1", 0, executor.get());
    a2 = getSyntheticColumnVar("table3", "a2", 0, executor.get());
    b = getSyntheticColumnVar("table4", "b", 1, executor.get());

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

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
      drop table if exists table3;
      drop table if exists table4;
    )");
  }
}

TEST(MultiFragment, KeyedOneToMany) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);
  CHECK(executor);
  executor->setCatalog(catalog.get());

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    JoinHashTableCacheInvalidator::invalidateCaches();

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (a1 integer, a2 integer);
      create table table2 (b integer);

      insert into table1 values (1, 11);
      insert into table1 values (2, 12);
      insert into table1 values (3, 13);
      insert into table1 values (4, 14);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (3);
      insert into table2 values (3);
    )");

    auto a1 = getSyntheticColumnVar("table1", "a1", 0, executor.get());
    auto a2 = getSyntheticColumnVar("table1", "a2", 0, executor.get());
    auto b = getSyntheticColumnVar("table2", "b", 1, executor.get());

    using VE = std::vector<std::shared_ptr<Analyzer::Expr>>;
    auto et1 = std::make_shared<Analyzer::ExpressionTuple>(VE{a1, a2});
    auto et2 = std::make_shared<Analyzer::ExpressionTuple>(VE{b, b});

    // a1 = b and a2 = b
    auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kEQ, kONE, et1, et2);
    auto hash_table1 = buildKeyed(op);
    EXPECT_EQ(hash_table1->getHashType(), HashType::OneToMany);

    sql(R"(
      drop table if exists table3;
      drop table if exists table4;

      create table table3 (a1 integer, a2 integer) with (fragment_size = 1);
      create table table4 (b integer) with (fragment_size = 1);

      insert into table3 values (1, 11);
      insert into table3 values (2, 12);
      insert into table3 values (3, 13);
      insert into table3 values (4, 14);

      insert into table4 values (0);
      insert into table4 values (1);
      insert into table4 values (3);
      insert into table4 values (3);
    )");

    a1 = getSyntheticColumnVar("table3", "a1", 0, executor.get());
    a2 = getSyntheticColumnVar("table3", "a2", 0, executor.get());
    b = getSyntheticColumnVar("table4", "b", 1, executor.get());

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

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
      drop table if exists table3;
      drop table if exists table4;
    )");
  }
}

TEST(Other, Regression) {
  sql(R"(
      drop table if exists table_a;
      drop table if exists table_b;

      CREATE TABLE table_a (
        Small_int SMALLINT,
        dest_state TEXT ENCODING DICT,
        omnisci_geo_linestring geometry(linestring, 4326)
      );
      CREATE TABLE table_b (
        Small_int SMALLINT,
        dest_state TEXT ENCODING DICT,
        omnisci_geo_linestring geometry(linestring, 4326)
      );
      INSERT INTO table_a VALUES (1, 'testa_1', 'LINESTRING (30 10, 10 30, 40 40)');
      INSERT INTO table_a VALUES (1, 'testa_2', 'LINESTRING (30 10, 10 30, 40 40)');
      INSERT INTO table_b VALUES (1, 'testb_1', 'LINESTRING (30 10, 10 30, 40 40)');
      INSERT INTO table_b VALUES (2, 'testb_2', 'LINESTRING (30 10, 10 30, 40 40)');
    )");

  EXPECT_THROW(sql(R"(
        SELECT table_a.dest_state AS key0 FROM table_a, table_b WHERE (table_b.dest_state = table_a.Small_int) GROUP BY key0 LIMIT 1000000;
      )"),
               std::runtime_error);

  sql(R"(
      drop table if exists table_a;
      drop table if exists table_b;
    )");
}

int main(int argc, char** argv) {
  ::g_enable_overlaps_hashjoin = true;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

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
