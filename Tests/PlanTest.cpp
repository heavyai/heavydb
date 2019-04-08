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

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include "../Analyzer/Analyzer.h"
#include "../Catalog/Catalog.h"
#include "../DataMgr/DataMgr.h"
#include "../Parser/ParserNode.h"
#include "../Parser/parser.h"
#include "../Planner/Planner.h"
#include "../QueryRunner/QueryRunner.h"
#include "Shared/MapDParameters.h"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "glog/logging.h"
#include "gtest/gtest.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Planner;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 36279

namespace {
std::unique_ptr<SessionInfo> gsession;

std::shared_ptr<Calcite> g_calcite = nullptr;

void calcite_shutdown_handler() {
  if (g_calcite) {
    g_calcite->close_calcite_server();
  }
}

void mapd_signal_handler(int signal_number) {
  LOG(ERROR) << "Interrupt signal (" << signal_number << ") received.";
  calcite_shutdown_handler();
  // shut down logging force a flush
  google::ShutdownGoogleLogging();
  // terminate program
  if (signal_number == SIGTERM) {
    std::exit(EXIT_SUCCESS);
  } else {
    std::exit(signal_number);
  }
}

void register_signal_handler() {
  std::signal(SIGTERM, mapd_signal_handler);
  std::signal(SIGSEGV, mapd_signal_handler);
  std::signal(SIGABRT, mapd_signal_handler);
}

class SQLTestEnv : public ::testing::Environment {
 public:
  virtual void SetUp() {
    boost::filesystem::path base_path{BASE_PATH};
    CHECK(boost::filesystem::exists(base_path));
    auto system_db_file = base_path / "mapd_catalogs" / MAPD_SYSTEM_DB;
    auto data_dir = base_path / "mapd_data";
    UserMetadata user;
    DBMetadata db;

    register_signal_handler();
    google::InstallFailureFunction(&calcite_shutdown_handler);

    g_calcite = std::make_shared<Calcite>(-1, CALCITEPORT, base_path.string(), 1024);
    {
      MapDParameters mapd_parms;
      auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(
          data_dir.string(), mapd_parms, false, 0);
      auto& sys_cat = SysCatalog::instance();
      sys_cat.init(base_path.string(),
                   dataMgr,
                   {},
                   g_calcite,
                   !boost::filesystem::exists(system_db_file),
                   false,
                   mapd_parms.aggregator,
                   {});
      CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));
      if (!sys_cat.getMetadataForUser("gtest", user)) {
        sys_cat.createUser("gtest", "test!test!", false);
        CHECK(sys_cat.getMetadataForUser("gtest", user));
      }
      if (!sys_cat.getMetadataForDB("gtest_db", db)) {
        sys_cat.createDatabase("gtest_db", user.userId);
        CHECK(sys_cat.getMetadataForDB("gtest_db", db));
      }
    }
    MapDParameters mapd_parms;
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(
        data_dir.string(), mapd_parms, false, 0);
    gsession.reset(new SessionInfo(
        std::make_shared<Catalog_Namespace::Catalog>(base_path.string(),
                                                     db,
                                                     dataMgr,
                                                     std::vector<LeafHostInfo>{},
                                                     g_calcite,
                                                     false),
        user,
        ExecutorDeviceType::GPU,
        ""));
  }
};

inline void run_ddl_statement(const string& input_str) {
  QueryRunner::run_ddl_statement(input_str, gsession);
}

RootPlan* plan_dml(const string& input_str) {
  SQLParser parser;
  list<std::unique_ptr<Parser::Stmt>> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(input_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::DMLStmt* dml = dynamic_cast<Parser::DMLStmt*>(stmt.get());
  CHECK(dml != nullptr);
  Query query;
  dml->analyze(gsession->getCatalog(), query);
  Optimizer optimizer(query, gsession->getCatalog());
  RootPlan* plan = optimizer.optimize();
  return plan;
}
}  // namespace

TEST(ParseAnalyzePlan, Create) {
  ASSERT_NO_THROW(run_ddl_statement("create table if not exists fat (a boolean, b "
                                    "char(5), c varchar(10), d numeric(10,2) "
                                    "encoding rl, e decimal(5,3) encoding sparse(16), f "
                                    "int encoding fixed(16), g smallint, "
                                    "h real, i float, j double, k bigint encoding diff, "
                                    "l text not null encoding dict, m "
                                    "timestamp(0), n time(0), o date);"););
  ASSERT_TRUE(gsession->getCatalog().getMetadataForTable("fat") != nullptr);
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table if not exists skinny (a smallint, b int, c bigint);"););
  ASSERT_TRUE(gsession->getCatalog().getMetadataForTable("skinny") != nullptr);
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table if not exists smallfrag (a int, b text, c bigint) with "
          "(fragment_size = 1000, page_size = 512);"););
  const TableDescriptor* td = gsession->getCatalog().getMetadataForTable("smallfrag");
  EXPECT_TRUE(td->maxFragRows == 1000 && td->fragPageSize == 512);
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table if not exists testdict (a varchar(100) encoding dict(8), c "
          "text encoding dict);"););
  td = gsession->getCatalog().getMetadataForTable("testdict");
  const ColumnDescriptor* cd =
      gsession->getCatalog().getMetadataForColumn(td->tableId, "a");
  const DictDescriptor* dd =
      gsession->getCatalog().getMetadataForDict(cd->columnType.get_comp_param());
  ASSERT_TRUE(dd != nullptr);
  EXPECT_EQ(dd->dictNBits, 8);
}

TEST(ParseAnalyzePlan, Select) {
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select * from fat;")); });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select f.* from fat f;")); });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml("select cast(a as int), d, l from fat;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml("select -1, -1.1, -1e-3, -a from fat;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml("select a, d, l from fat where not 1=0;"));
  });
  EXPECT_NO_THROW(
      { unique_ptr<RootPlan> plan_ptr(plan_dml("select b, d+e, f*g as y from fat;")); });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select b, d+e, f*g as y from fat order by 2 asc null last, 3 desc null first;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select b, d+e, f*g as y from fat order by 2 asc null last, 3 desc null "
                 "first limit 10;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select b, d+e, f*g as y from fat order by 2 asc null last, 3 desc null "
                 "first limit all "
                 "offset 100 rows;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a, d, g from fat where f > 100 and g is null and k <= "
                 "100000000000 and c = "
                 "'xyz';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a, d, g from fat where f > 100 and g is not null or k <= "
                 "100000000000 and c = "
                 "'xyz';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select i, j, k from fat where l like '%whatever%';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select i, j, k from fat where l like '%whatever@%_' escape '@';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select i, j, k from fat where l ilike '%whatever@%_' escape '@';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select i, j, k from fat where l not like '%whatever@%_' escape '@';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select i, j, k from fat where l not ilike '%whatever@%_' escape '@';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select e, f, g from fat where e in (3.5, 133.33, 222.22);"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select e, f, g from fat where e not in (3.5, 133.33, 222.22);"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select e, f, g from fat where e not in (3.5, 133.33, 222.22) or l not like "
        "'%whatever%';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a, b, c from fat where i between 10e5 and 10e6 and j not "
                 "between 10e-4 and "
                 "10e-1;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select case when e between 10 and 20 then 1 when e between 20 and 40 "
                 "then 2 when e is "
                 "null then 100 else 5 end as x, a from fat where case when g > f then "
                 "100 when l like "
                 "'%whatever%' then 200 else 300 end > 100;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select case when e between 10 and 20 then 1 when e between 20 and 40 "
                 "then 2.1 when e is "
                 "null then 100.33 else 5e2 end as x, a from fat where case when g > f "
                 "then 100 when l like "
                 "'%whatever%' then 200 else 300 end > 100;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select case when e between 10 and 20 then i when e between 20 and 40 "
                 "then j when e is "
                 "null then d else 5e2 end as x, a from fat where case when g > f then "
                 "100 when l like "
                 "'%whatever%' then 200 else 300 end > 100;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select count(*), min(a), max(a), avg(b), sum(c), count(distinct b) "
                 "from skinny;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml("select a+b as x from skinny group by x;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a, b, count(*) from skinny group by a, b;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select c, avg(b) from skinny where a > 10 group by c;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select c, avg(b) from skinny where a > 10 group by c having max(a) < 100;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select c, avg(b) from skinny where a > 10 group by c having max(a) < "
                 "100 and count(*) > "
                 "1000;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select count(*)*avg(c) - sum(c) from skinny;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c "
                 "between 100 and 200 "
                 "group by a, b;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c "
                 "between 100 and 200 "
                 "group by a, b having b > 2*a and min(b) > max(a);"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c "
                 "between 100 and 200 "
                 "group by a, b order by x desc null first;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c "
                 "between 100 and 200 "
                 "group by a, b order by x desc null first limit 10 offset 100000000;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select cast(a+b as decimal(10,3)) as x, count(*)*avg(c) - sum(c) as y "
                 "from skinny where c "
                 "between 100 and 200 group by a, b order by x desc null first limit 10 "
                 "offset 100000000;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c "
                 "between 100 and 200 "
                 "group by x, b having x > 10;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select distinct a+b as x, count(*)*avg(c) - sum(c) as y from skinny "
                 "where c between 100 "
                 "and 200 group by x, b having x > 10;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select * from fat where m < timestamp(0) '2015-02-18 13:15:55' and n "
                 ">= time(0) '120000' "
                 "and o <> date '05/06/2014';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select extract(year from date '2015-02-28'), extract(month from date "
        "'2014-12-13'), "
        "extract(day from timestamp(0) '1998-10-24 03:14:55'), extract(dow from date "
        "'1936-02-09'), extract(doy from timestamp(0) '2015-02-18 01:02:11'), "
        "extract(hour from "
        "time(0) '111233'), extract(minute from m), extract(second from n), "
        "extract(epoch from o) "
        "from fat where cast(timestamp(0) '2015-02-18 12:13:14' as int) > 1000;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select * from fat where m >= '1999-09-09T111111' and n <= '222222' and o = "
        "'1996-02-23';"));
  });
  EXPECT_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select AVG(*) from fat;")); },
               std::runtime_error);
}

TEST(ParseAnalyzePlan, Insert) {
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("insert into skinny values (12345, 100000000, 100000000000);"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("insert into skinny select 2*a, 2*b, 2*c from skinny;"));
  });
}

TEST(DISABLED_ParseAnalyzePlan, Views) {
  EXPECT_NO_THROW(
      run_ddl_statement(
          "create view if not exists voo as select * from skinny where a > 15;"););
  EXPECT_NO_THROW(
      run_ddl_statement(
          "create view if not exists moo as select * from skinny where a > 15;"););
  EXPECT_NO_THROW(run_ddl_statement("create view if not exists mic as select c, avg(b) "
                                    "from skinny where a > 10 group by c;"););
  EXPECT_NO_THROW(run_ddl_statement("create view if not exists fatview as select a, d, g "
                                    "from fat where f > 100 and g is not "
                                    "null or k <= 100000000000 and c = 'xyz';"););
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select * from fatview;")); });
}

void drop_views_and_tables() {
  EXPECT_NO_THROW(run_ddl_statement("drop view if exists voo;"));
  EXPECT_NO_THROW(run_ddl_statement("drop view if exists moo;"));
  EXPECT_NO_THROW(run_ddl_statement("drop view if exists goo;"));
  EXPECT_NO_THROW(run_ddl_statement("drop view if exists mic;"));
  EXPECT_NO_THROW(run_ddl_statement("drop view if exists fatview;"));
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists fat;"));
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists skinny;"));
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists smallfrag;"));
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists testdict;"));
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists foxoxoxo;"));
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SQLTestEnv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  drop_views_and_tables();

  return err;
}
