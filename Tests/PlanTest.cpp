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
#include "TestHelpers.h"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Planner;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;
namespace {

inline void run_ddl_statement(const string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

RootPlan* plan_dml(const string& input_str) {
  SQLParser parser;
  list<std::unique_ptr<Parser::Stmt>> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(input_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::DMLStmt* dml = dynamic_cast<Parser::DMLStmt*>(stmt.get());
  CHECK(dml);
  Query query;
  dml->analyze(*QR::get()->getCatalog(), query);
  Optimizer optimizer(query, *QR::get()->getCatalog());
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
  ASSERT_TRUE(QR::get()->getCatalog()->getMetadataForTable("fat") != nullptr);
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table if not exists skinny (a smallint, b int, c bigint);"););
  ASSERT_TRUE(QR::get()->getCatalog()->getMetadataForTable("skinny") != nullptr);
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table if not exists smallfrag (a int, b text, c bigint) with "
          "(fragment_size = 1000, page_size = 512);"););
  const TableDescriptor* td = QR::get()->getCatalog()->getMetadataForTable("smallfrag");
  CHECK(td);
  EXPECT_TRUE(td->maxFragRows == 1000 && td->fragPageSize == 512);
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table if not exists testdict (a varchar(100) encoding dict(8), c "
          "text encoding dict);"););
  td = QR::get()->getCatalog()->getMetadataForTable("testdict");
  CHECK(td);
  const ColumnDescriptor* cd =
      QR::get()->getCatalog()->getMetadataForColumn(td->tableId, "a");
  const DictDescriptor* dd =
      QR::get()->getCatalog()->getMetadataForDict(cd->columnType.get_comp_param());
  ASSERT_TRUE(dd != nullptr);
  EXPECT_EQ(dd->dictNBits, 8);
  const std::string db_name("chelsea");
  ASSERT_NO_THROW(run_ddl_statement("CREATE DATABASE " + db_name + ";"));
  ASSERT_NO_THROW(run_ddl_statement("CREATE DATABASE IF NOT EXISTS " + db_name + ";"));
  Catalog_Namespace::DBMetadata db;
  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
  EXPECT_TRUE(sys_cat.getMetadataForDB(db_name, db));
  EXPECT_EQ(db.dbName, db_name);
  ASSERT_NO_THROW(run_ddl_statement("DROP DATABASE " + db_name + ";"));
  ASSERT_NO_THROW(run_ddl_statement("DROP DATABASE IF EXISTS " + db_name + ";"));
  EXPECT_FALSE(sys_cat.getMetadataForDB(db_name, db));
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
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  drop_views_and_tables();
  QR::reset();

  return err;
}
