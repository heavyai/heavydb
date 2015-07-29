#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include <boost/functional/hash.hpp>
#include "../Catalog/Catalog.h"
#include "../Parser/parser.h"
#include "../Analyzer/Analyzer.h"
#include "../Parser/ParserNode.h"
#include "../DataMgr/DataMgr.h"
#include "../Fragmenter/Fragmenter.h"
#include "PopulateTableRandom.h"
#include "ScanTable.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Fragmenter_Namespace;

#define BASE_PATH "/tmp"

namespace {
std::unique_ptr<SessionInfo> gsession;
;

void run_ddl(const string& input_str) {
  SQLParser parser;
  list<Parser::Stmt*> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(input_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), 1);
  auto stmt = parse_trees.front();
  unique_ptr<Stmt> stmt_ptr(stmt);  // make sure it's deleted
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl != nullptr);
  ddl->execute(*gsession);
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
    {
      auto dataMgr =
          new Data_Namespace::DataMgr(data_dir.string(), false);  // false is for use gpus
      if (!boost::filesystem::exists(system_db_file)) {
        SysCatalog sys_cat(base_path.string(), dataMgr, true);
        sys_cat.initDB();
      }
      SysCatalog sys_cat(base_path.string(), dataMgr);
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
    auto dataMgr = new Data_Namespace::DataMgr(data_dir.string(), false);  // false is for use gpus
    gsession.reset(new SessionInfo(std::make_shared<Catalog>(base_path.string(), db, dataMgr),
                                   user,
                                   ExecutorDeviceType::GPU,
                                   0));
  }
};

bool load_data_test(string table_name, size_t num_rows) {
  vector<size_t> insert_col_hashs =
      populate_table_random(table_name, num_rows, gsession->get_catalog());
  return true;
}
}  // namespace

#define SMALL 10000000
#define LARGE 100000000

TEST(DataLoad, Numbers) {
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers;"););
  ASSERT_NO_THROW(run_ddl(
                      "create table numbers (a smallint, b int, c bigint, d numeric(7,3), e "
                      "double, f float);"););
  EXPECT_TRUE(load_data_test("numbers", LARGE));
  ASSERT_NO_THROW(run_ddl("drop table numbers;"););
}

TEST(DataLoad, Strings) {
  ASSERT_NO_THROW(run_ddl("drop table if exists strings;"););
  ASSERT_NO_THROW(run_ddl("create table strings (x varchar(10), y text);"););
  EXPECT_TRUE(load_data_test("strings", SMALL));
  ASSERT_NO_THROW(run_ddl("drop table strings;"););
}

TEST(StorageSmall, AllTypes) {
  ASSERT_NO_THROW(run_ddl("drop table if exists alltypes;"););
  ASSERT_NO_THROW(
      run_ddl(
          "create table alltypes (a smallint, b int, c bigint, d numeric(7,3), e double, f float, "
          "g timestamp(0), h time(0), i date, x varchar(10), y text);"););
  EXPECT_TRUE(load_data_test("alltypes", SMALL));
  ASSERT_NO_THROW(run_ddl("drop table alltypes;"););
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SQLTestEnv);
  return RUN_ALL_TESTS();
}
