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
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/OverlapsJoinHashTable.h"
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

std::shared_ptr<JoinHashTable> buildSyntheticHashJoinTable(std::string_view table1,
                                                           std::string_view column1,
                                                           std::string_view table2,
                                                           std::string_view column2) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(catalog.get(), g_device_type);

  ColumnCacheMap column_cache;

  return JoinHashTable::getSyntheticInstance(table1,
                                             column1,
                                             table2,
                                             column2,
                                             memory_level,
                                             JoinHashTableInterface::HashType::OneToMany,
                                             device_count,
                                             column_cache,
                                             executor.get());
}

std::shared_ptr<BaselineJoinHashTable> buildSyntheticBaselineHashJoinTable(
    std::string_view table1,
    std::string_view column1,
    std::string_view table2,
    std::string_view column2) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(catalog.get(), g_device_type);

  ColumnCacheMap column_cache;

  return BaselineJoinHashTable::getSyntheticInstance(
      table1,
      column1,
      table2,
      column2,
      memory_level,
      JoinHashTableInterface::HashType::OneToMany,
      device_count,
      column_cache,
      executor.get());
}

std::shared_ptr<OverlapsJoinHashTable> buildSyntheticOverlapsHashJoinTable(
    std::string_view table1,
    std::string_view column1,
    std::string_view table2,
    std::string_view column2) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);

  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);

  auto memory_level =
      (g_device_type == ExecutorDeviceType::CPU ? Data_Namespace::CPU_LEVEL
                                                : Data_Namespace::GPU_LEVEL);

  auto device_count = deviceCount(catalog.get(), g_device_type);

  ColumnCacheMap column_cache;

  return OverlapsJoinHashTable::getSyntheticInstance(table1,
                                                     column1,
                                                     table2,
                                                     column2,
                                                     memory_level,
                                                     device_count,
                                                     column_cache,
                                                     executor.get());
}

// 0 1 2 3 4 5 6 7 8 9 | 1 1 1 1 1 1 1 1 1 1 | 0 1 2 3 4 5 6 7 8 9
int32_t hashTable1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
size_t hashTable1_sizes[] = {
    0,
    40,
    80};  // offsetBufferOff(), countBufferOff(), payloadBufferOff()

TEST(Decode, JoinHashTable1) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

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

    auto hptr = reinterpret_cast<const int8_t*>(hashTable1);
    auto s1 = ::decodeJoinHashBuffer(1,
                                     sizeof(*hashTable1),
                                     hptr,
                                     hptr + hashTable1_sizes[0],
                                     hptr + hashTable1_sizes[1],
                                     hptr + hashTable1_sizes[2],
                                     sizeof(hashTable1));

    auto hash_table = buildSyntheticHashJoinTable("table1", "nums1", "table2", "nums2");

    auto s2 = hash_table->decodeJoinHashBuffer(g_device_type, 0);

    ASSERT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

// 0 1 2 3 4 | 1 1 1 1 1 | 0 1 2 3 4
int32_t hashTable2[] = {0, 1, 2, 3, 4, 1, 1, 1, 1, 1, 0, 1, 2, 3, 4};
size_t hashTable2_sizes[] = {
    0,
    20,
    40};  // offsetBufferOff(), countBufferOff(), payloadBufferOff()

TEST(Decode, JoinHashTable2) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (vals1 text);
      create table table2 (vals2 text, shared dictionary (vals2) references table1(vals1));

      insert into table1 values ('a');

      insert into table2 values ('a');
      insert into table2 values ('b');
      insert into table2 values ('c');
      insert into table2 values ('d');
      insert into table2 values ('e');
    )");

    auto hptr = reinterpret_cast<const int8_t*>(hashTable2);
    auto s1 = ::decodeJoinHashBuffer(1,
                                     sizeof(*hashTable2),
                                     hptr,
                                     hptr + hashTable2_sizes[0],
                                     hptr + hashTable2_sizes[1],
                                     hptr + hashTable2_sizes[2],
                                     sizeof(hashTable2));

    auto hash_table = buildSyntheticHashJoinTable("table1", "vals1", "table2", "vals2");
    auto s2 = hash_table->decodeJoinHashBuffer(g_device_type, 0);

    ASSERT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

// (5) * * (1) * * (6) * (7) (9) (4) (8) * * (3) * * (2) (0) * | 0 * * 1 * * 2 * 3 4 5 6 *
// * 7 * * 8 9 * | 1 * * 1 * * 1 * 1 1 1 1 * * 1 * * 1 1 * | 5 1 6 7 9 4 8 3 2 0
int32_t baHashTable1[] = {
    5,          2147483647, 2147483647, 1,  2147483647, 2147483647, 6,
    2147483647, 7,          9,          4,  8,          2147483647, 2147483647,
    3,          2147483647, 2147483647, 2,  0,          2147483647, 0,
    -1,         -1,         1,          -1, -1,         2,          -1,
    3,          4,          5,          6,  -1,         -1,         7,
    -1,         -1,         8,          9,  -1,         1,          0,
    0,          1,          0,          0,  1,          0,          1,
    1,          1,          1,          0,  0,          1,          0,
    0,          1,          1,          0,  5,          1,          6,
    7,          9,          4,          8,  3,          2,          0};
size_t baHashTable1_sizes[] = {
    80,
    160,
    240};  // offsetBufferOff(), countBufferOff(), payloadBufferOff()

TEST(Decode, BaselineJoinHashTable1) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;

      create table table1 (nums1 integer);
      create table table2 (nums2 integer);

      insert into table1 values (0);
      insert into table1 values (1);

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

    auto hptr = reinterpret_cast<const int8_t*>(baHashTable1);
    auto s1 = ::decodeJoinHashBuffer(1,
                                     sizeof(*baHashTable1),
                                     hptr,
                                     hptr + baHashTable1_sizes[0],
                                     hptr + baHashTable1_sizes[1],
                                     hptr + baHashTable1_sizes[2],
                                     sizeof(baHashTable1));

    auto hash_table =
        buildSyntheticBaselineHashJoinTable("table1", "nums1", "table2", "nums2");
    auto s2 = hash_table->decodeJoinHashBuffer(g_device_type, 0);

    ASSERT_EQ(s1, s2);

    sql(R"(
      drop table if exists table1;
      drop table if exists table2;
    )");
  }
}

// * (0,2) * * * (1,1) * * * (0,1) (2,1) (2,2) (2,0) (1,2) (0,0) * * (1,0) | * 0 * * * 1 *
// * * 5 7 9 10 11 13 * * 14 | * 1 * * * 4 * * * 2 2 1 1 2 1 * * 2 | 2 0 2 3 1 0 2 3 1 3 1
// 2 3 0 0 1
int64_t ovHashTable1[] = {9223372036854775807,
                          9223372036854775807,
                          0,
                          2,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          1,
                          1,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          0,
                          1,
                          2,
                          1,
                          2,
                          0,
                          2,
                          2,
                          1,
                          2,
                          0,
                          0,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          1,
                          0,
                          4294967295,
                          -1,
                          8589934591,
                          -1,
                          25769803775,
                          38654705671,
                          47244640266,
                          -4294967283,
                          64424509439,
                          4294967296,
                          0,
                          17179869184,
                          0,
                          8589934592,
                          4294967298,
                          8589934593,
                          1,
                          8589934592,
                          2,
                          8589934593,
                          3,
                          4294967298,
                          4294967299,
                          8589934595,
                          3,
                          4294967296};
size_t ovHashTable1_sizes[] = {
    288,
    360,
    432};  // offsetBufferOff(), countBufferOff(), payloadBufferOff()

TEST(Decode, OverlapsJoinHashTable1) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

    sql(R"(
      drop table if exists my_points;
      drop table if exists my_grid;

      create table my_points (locations geometry(point, 4326) encoding none);
      create table my_grid (cells geometry(multipolygon, 4326) encoding none);

      insert into my_points values ('point(5 5)');
      insert into my_points values ('point(5 25)');
      insert into my_points values ('point(10 5)');

      insert into my_grid values ('multipolygon(((0 0,10 0,10 10,0 10,0 0)))');
      insert into my_grid values ('multipolygon(((10 0,20 0,20 10,10 10,10 0)))');
      insert into my_grid values ('multipolygon(((0 10,10 10,10 20,0 20,0 10)))');
      insert into my_grid values ('multipolygon(((10 10,20 10,20 20,10 20,10 10)))');
    )");

    auto hptr = reinterpret_cast<const int8_t*>(ovHashTable1);
    auto s1 = decodeJoinHashBuffer(2,
                                   sizeof(*ovHashTable1),
                                   hptr,
                                   hptr + ovHashTable1_sizes[0],
                                   hptr + ovHashTable1_sizes[1],
                                   hptr + ovHashTable1_sizes[2],
                                   sizeof(ovHashTable1));

    auto hash_table =
        buildSyntheticOverlapsHashJoinTable("my_points", "locations", "my_grid", "cells");
    auto s2 = hash_table->decodeJoinHashBuffer(g_device_type, 0);

    ASSERT_EQ(s1, s2);

    sql(R"(
      drop table if exists my_points;
      drop table if exists my_grid;
    )");
  }
}

// (11,0) (1,1) * * * * (0,0) * (10,0) (10,1) * (1,0) * (11,1) * (0,1) | 0 1 * * * * 2 * 3
// 4 * 5 * 6 * 7 | 1 1 * * * * 1 * 1 1 * 1 * 1 * 1 | 1 0 0 1 1 0 1 0
int64_t ovHashTable2[] = {11,
                          0,
                          1,
                          1,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          9223372036854775807,
                          0,
                          0,
                          9223372036854775807,
                          9223372036854775807,
                          10,
                          0,
                          10,
                          1,
                          9223372036854775807,
                          9223372036854775807,
                          1,
                          0,
                          9223372036854775807,
                          9223372036854775807,
                          11,
                          1,
                          9223372036854775807,
                          9223372036854775807,
                          0,
                          1,
                          4294967296,
                          -1,
                          -1,
                          -4294967294,
                          17179869187,
                          25769803775,
                          30064771071,
                          34359738367,
                          4294967297,
                          0,
                          0,
                          1,
                          4294967297,
                          4294967296,
                          4294967296,
                          4294967296,
                          1,
                          4294967296,
                          1,
                          1};
size_t ovHashTable2_sizes[] = {
    256,
    320,
    384};  // offsetBufferOff(), countBufferOff(), payloadBufferOff()

TEST(Decode, OverlapsJoinHashTable2) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    g_device_type = dt;

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

    auto hptr = reinterpret_cast<const int8_t*>(ovHashTable2);
    auto s1 = decodeJoinHashBuffer(2,
                                   sizeof(*ovHashTable2),
                                   hptr,
                                   hptr + ovHashTable2_sizes[0],
                                   hptr + ovHashTable2_sizes[1],
                                   hptr + ovHashTable2_sizes[2],
                                   sizeof(ovHashTable2));

    auto hash_table =
        buildSyntheticOverlapsHashJoinTable("my_points", "locations", "my_grid", "cells");
    auto s2 = hash_table->decodeJoinHashBuffer(g_device_type, 0);

    ASSERT_EQ(s1, s2);

    sql(R"(
      drop table if exists my_points;
      drop table if exists my_grid;
    )");
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
