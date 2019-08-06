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

constexpr size_t c_calcite_port = 36279;

using namespace Catalog_Namespace;
using namespace TestHelpers;

using QR = QueryRunner::QueryRunner;

namespace {
ExecutorDeviceType g_device_type;
}

inline auto sql(const std::string& sql_stmts) {
  return QR::get()->runMultipleStatements(sql_stmts, g_device_type);
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

TEST(Memcmp, OverlapsJoinHashTable) {
  g_device_type = ExecutorDeviceType::CPU;

  sql(R"(
    drop table if exists my_points;
    drop table if exists my_grid;
  )");

  sql(R"(
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

  auto catalog = QR::get()->getCatalog();
  auto executor = Executor::getExecutor(catalog->getCurrentDB().dbId);

  auto tmeta1 = catalog->getMetadataForTable("my_points");
  auto tmeta2 = catalog->getMetadataForTable("my_grid");

  SQLTypeInfo ti1{kPOINT, 4326, 4326, false, kENCODING_NONE, 64, kGEOMETRY};
  SQLTypeInfo ti2{kARRAY, 0, 0, true, kENCODING_NONE, 0, kDOUBLE};
  ti2.set_size(32);

  auto a1 = std::make_shared<Analyzer::ColumnVar>(ti1, tmeta1->tableId, 1, 0);
  auto a2 = std::make_shared<Analyzer::ColumnVar>(ti2, tmeta2->tableId, 5, 1);

  auto op = std::make_shared<Analyzer::BinOper>(kBOOLEAN, kOVERLAPS, kONE, a1, a2);

  size_t number_of_join_tables{2};
  std::vector<InputTableInfo> viti(number_of_join_tables);
  viti[0].table_id = tmeta1->tableId;
  viti[0].info = tmeta1->fragmenter->getFragmentsForQuery();
  viti[1].table_id = tmeta2->tableId;
  viti[1].info = tmeta2->fragmenter->getFragmentsForQuery();

  auto memory_level = Data_Namespace::CPU_LEVEL;

  auto device_count = deviceCount(catalog.get(), g_device_type);

  ColumnCacheMap ccm;

  auto hash_table = OverlapsJoinHashTable::getInstance(
      op, viti, memory_level, device_count, ccm, executor.get());

  auto ptr1 =
      reinterpret_cast<const int32_t*>(hash_table->getJoinHashBuffer(g_device_type, 0));
  auto ptr2 = reinterpret_cast<const int32_t*>(
      hash_table->getJoinHashBuffer(g_device_type, 0) + hash_table->offsetBufferOff());
  CHECK_EQ(*ptr1, -1);
  CHECK_EQ(*ptr2, -1);
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
