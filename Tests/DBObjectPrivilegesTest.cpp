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

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <thread>
#include <tuple>
#include "../Catalog/Catalog.h"
#include "../Catalog/DBObject.h"
#include "../DataMgr/DataMgr.h"
#include "../QueryEngine/ArrowResultSet.h"
#include "../QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "../QueryEngine/Execute.h"
#include "../QueryRunner/QueryRunner.h"
#include "DBHandlerTestHelpers.h"
#include "Shared/SysDefinitions.h"
#include "Shared/scope.h"
#include "TestHelpers.h"
#include "ThriftHandler/QueryState.h"
#include "gen-cpp/CalciteServer.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace TestHelpers;

using QR = QueryRunner::QueryRunner;

using Catalog_Namespace::DBMetadata;
using Catalog_Namespace::SysCatalog;
using Catalog_Namespace::UserMetadata;

extern size_t g_leaf_count;
std::string g_test_binary_file_path;

namespace {

std::shared_ptr<Calcite> g_calcite;
bool g_aggregator{false};

Catalog_Namespace::UserMetadata g_user;
std::vector<DBObject> privObjects;

auto& sys_cat = Catalog_Namespace::SysCatalog::instance();

inline void run_ddl_statement(const std::string& query) {
  QR::get()->runDDLStatement(query);
}

inline auto sql(std::string_view sql_stmts) {
  return QR::get()->runMultipleStatements(std::string(sql_stmts),
                                          ExecutorDeviceType::CPU);
}

}  // namespace

struct Users {
  void setup_users() {
    if (!sys_cat.getMetadataForUser("Chelsea", g_user)) {
      sys_cat.createUser(
          "Chelsea",
          Catalog_Namespace::UserAlterations{
              "password", /*is_super=*/true, /*default_db=*/"", /*can_login=*/true},
          false);
      CHECK(sys_cat.getMetadataForUser("Chelsea", g_user));
    }
    if (!sys_cat.getMetadataForUser("Arsenal", g_user)) {
      sys_cat.createUser(
          "Arsenal",
          Catalog_Namespace::UserAlterations{
              "password", /*is_super=*/false, /*default_db=*/"", /*can_login=*/true},
          false);
      CHECK(sys_cat.getMetadataForUser("Arsenal", g_user));
    }
    if (!sys_cat.getMetadataForUser("Juventus", g_user)) {
      sys_cat.createUser(
          "Juventus",
          Catalog_Namespace::UserAlterations{
              "password", /*is_super=*/false, /*default_db=*/"", /*can_login=*/true},
          false);
      CHECK(sys_cat.getMetadataForUser("Juventus", g_user));
    }
    if (!sys_cat.getMetadataForUser("Bayern", g_user)) {
      sys_cat.createUser(
          "Bayern",
          Catalog_Namespace::UserAlterations{
              "password", /*is_super=*/false, /*default_db=*/"", /*can_login=*/true},
          false);
      CHECK(sys_cat.getMetadataForUser("Bayern", g_user));
    }
  }
  void drop_users() {
    if (sys_cat.getMetadataForUser("Chelsea", g_user)) {
      sys_cat.dropUser("Chelsea");
      CHECK(!sys_cat.getMetadataForUser("Chelsea", g_user));
    }
    if (sys_cat.getMetadataForUser("Arsenal", g_user)) {
      sys_cat.dropUser("Arsenal");
      CHECK(!sys_cat.getMetadataForUser("Arsenal", g_user));
    }
    if (sys_cat.getMetadataForUser("Juventus", g_user)) {
      sys_cat.dropUser("Juventus");
      CHECK(!sys_cat.getMetadataForUser("Juventus", g_user));
    }
    if (sys_cat.getMetadataForUser("Bayern", g_user)) {
      sys_cat.dropUser("Bayern");
      CHECK(!sys_cat.getMetadataForUser("Bayern", g_user));
    }
  }
  Users() { setup_users(); }
  virtual ~Users() { drop_users(); }
};
struct Roles {
  void setup_roles() {
    if (!sys_cat.getRoleGrantee("OldLady")) {
      sys_cat.createRole("OldLady", false);
      CHECK(sys_cat.getRoleGrantee("OldLady"));
    }
    if (!sys_cat.getRoleGrantee("Gunners")) {
      sys_cat.createRole("Gunners", false);
      CHECK(sys_cat.getRoleGrantee("Gunners"));
    }
    if (!sys_cat.getRoleGrantee("Sudens")) {
      sys_cat.createRole("Sudens", false);
      CHECK(sys_cat.getRoleGrantee("Sudens"));
    }
  }

  void drop_roles() {
    if (sys_cat.getRoleGrantee("OldLady")) {
      sys_cat.dropRole("OldLady");
      CHECK(!sys_cat.getRoleGrantee("OldLady"));
    }
    if (sys_cat.getRoleGrantee("Gunners")) {
      sys_cat.dropRole("Gunners");
      CHECK(!sys_cat.getRoleGrantee("Gunners"));
    }
    if (sys_cat.getRoleGrantee("Sudens")) {
      sys_cat.dropRole("Sudens");
      CHECK(!sys_cat.getRoleGrantee("sudens"));
    }
  }
  Roles() { setup_roles(); }
  virtual ~Roles() { drop_roles(); }
};

struct GrantSyntax : testing::Test {
  Users user_;
  Roles role_;

  void setup_tables() {
    run_ddl_statement("CREATE TABLE IF NOT EXISTS tbl(i INTEGER)");
    run_ddl_statement("CREATE VIEW grantView AS SELECT i FROM tbl;");
  }
  void drop_tables() {
    run_ddl_statement("DROP TABLE IF EXISTS tbl");
    run_ddl_statement("DROP VIEW IF EXISTS grantView");
  }
  explicit GrantSyntax() {
    drop_tables();
    setup_tables();
  }
  ~GrantSyntax() override { drop_tables(); }
};

struct DatabaseObject : testing::Test {
  Catalog_Namespace::UserMetadata user_meta;
  Catalog_Namespace::DBMetadata db_meta;
  Users user_;
  Roles role_;

  explicit DatabaseObject() {}
  ~DatabaseObject() override {}
};

struct TableObject : testing::Test {
  const std::string cquery1 =
      "CREATE TABLE IF NOT EXISTS epl(gp SMALLINT, won SMALLINT);";
  const std::string cquery2 =
      "CREATE TABLE IF NOT EXISTS seriea(gp SMALLINT, won SMALLINT);";
  const std::string cquery3 =
      "CREATE TABLE IF NOT EXISTS bundesliga(gp SMALLINT, won SMALLINT);";
  const std::string dquery1 = "DROP TABLE IF EXISTS epl;";
  const std::string dquery2 = "DROP TABLE IF EXISTS seriea;";
  const std::string dquery3 = "DROP TABLE IF EXISTS bundesliga;";
  Users user_;
  Roles role_;

  void setup_tables() {
    run_ddl_statement(cquery1);
    run_ddl_statement(cquery2);
    run_ddl_statement(cquery3);
  }
  void drop_tables() {
    run_ddl_statement(dquery1);
    run_ddl_statement(dquery2);
    run_ddl_statement(dquery3);
  }
  explicit TableObject() {
    drop_tables();
    setup_tables();
  }
  ~TableObject() override { drop_tables(); }
};

class ViewObject : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("CREATE USER bob (password = 'password', is_super = 'false');");
    run_ddl_statement("CREATE ROLE salesDept;");
    run_ddl_statement("CREATE USER foo (password = 'password', is_super = 'false');");
    run_ddl_statement("GRANT salesDept TO foo;");

    run_ddl_statement("CREATE TABLE bill_table(id integer);");
    run_ddl_statement("CREATE VIEW bill_view AS SELECT id FROM bill_table;");
    run_ddl_statement("CREATE VIEW bill_view_outer AS SELECT id FROM bill_view;");
  }

  void TearDown() override {
    run_ddl_statement("DROP VIEW bill_view_outer;");
    run_ddl_statement("DROP VIEW bill_view;");
    run_ddl_statement("DROP TABLE bill_table");

    run_ddl_statement("DROP USER foo;");
    run_ddl_statement("DROP ROLE salesDept;");
    run_ddl_statement("DROP USER bob;");
  }
};

class DashboardObject : public ::testing::Test {
 protected:
  const std::string dname1 = "ChampionsLeague";
  const std::string dname2 = "Europa";
  const std::string dstate = "active";
  const std::string dhash = "image00";
  const std::string dmeta = "Chelsea are champions";
  int id;
  Users user_;
  Roles role_;

  DashboardDescriptor vd1;

  void setup_dashboards() {
    auto session = QR::get()->getSession();
    CHECK(session);
    auto& cat = session->getCatalog();
    vd1.dashboardName = dname1;
    vd1.dashboardState = dstate;
    vd1.imageHash = dhash;
    vd1.dashboardMetadata = dmeta;
    vd1.userId = session->get_currentUser().userId;
    vd1.user = session->get_currentUser().userName;
    id = cat.createDashboard(vd1);
    sys_cat.createDBObject(
        session->get_currentUser(), dname1, DBObjectType::DashboardDBObjectType, cat, id);
  }

  void drop_dashboards() {
    auto session = QR::get()->getSession();
    CHECK(session);
    auto& cat = session->getCatalog();
    if (cat.getMetadataForDashboard(id)) {
      cat.deleteMetadataForDashboards({id}, session->get_currentUser());
    }
  }

  void SetUp() override {
    drop_dashboards();
    setup_dashboards();
  }

  void TearDown() override { drop_dashboards(); }
};

class DatabaseDdlTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    switchToAdmin();
    createTestUser();
  }

  static void TearDownTestSuite() { dropTestUser(); }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    dropDatabase();
  }

  void TearDown() override {
    switchToAdmin();
    dropDatabase();
    DBHandlerTestFixture::TearDown();
  }

  static void createTestUser() {
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  }

  static void dropTestUser() { sql("DROP USER IF EXISTS test_user;"); }

  static void dropTestUserUnchecked() {
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    UserMetadata test_user;
    sys_catalog.getMetadataForUser("test_user", test_user);
    sys_catalog.dropUserUnchecked("test_user", test_user);
  }

  void createTestDatabase(const std::optional<std::string>& user = std::nullopt) {
    if (user.has_value()) {
      sql("CREATE DATABASE test_database (owner='" + user.value() + "');");
    } else {
      sql("CREATE DATABASE test_database;");
    }
  }

  void dropDatabase() {
    sql("DROP DATABASE IF EXISTS test_database;");
    sql("DROP DATABASE IF EXISTS test_database_new;");
  }

  void assertExpectedDatabase(const Catalog_Namespace::DBMetadata& expected = {},
                              const std::string& database_name = "test_database") {
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    Catalog_Namespace::DBMetadata db;
    sys_catalog.getMetadataForDB(database_name, db);

    ASSERT_EQ(expected.dbId, expected.dbId);
    ASSERT_EQ(expected.dbName, expected.dbName);
    ASSERT_EQ(expected.dbOwner, expected.dbOwner);
  }

  Catalog_Namespace::DBMetadata createDatabaseMetadata(const std::string& dbname,
                                                       const int32_t owner_id) {
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    Catalog_Namespace::DBMetadata db;
    sys_catalog.getMetadataForDB(dbname, db);  // used to fill in the dbId
    db.dbName = dbname;
    db.dbOwner = getCurrentUser().userId;
    return db;
  }

  static Catalog_Namespace::UserMetadata getTestUser() {
    Catalog_Namespace::UserMetadata user;
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    sys_catalog.getMetadataForUser("test_user", user);
    return user;
  }
};

TEST_F(DatabaseDdlTest, ChangeOwner) {
  createTestDatabase();
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
  sql("ALTER DATABASE test_database OWNER TO test_user;");
  assertExpectedDatabase(createDatabaseMetadata("test_database", getTestUser().userId));
}

TEST_F(DatabaseDdlTest, ChangeOwnerPreviousOwnerDropped) {
  if (g_aggregator) {
    LOG(INFO) << "Test not supported in distributed mode due to not being able to drop "
                 "user unchecked.";
    return;
  }
  createTestDatabase("test_user");
  assertExpectedDatabase(createDatabaseMetadata("test_database", getTestUser().userId));
  dropTestUserUnchecked();
  sql("ALTER DATABASE test_database OWNER TO " + shared::kRootUsername + ";");
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
  createTestUser();
}

TEST_F(DatabaseDdlTest, ChangeOwnerLackingCredentials) {
  createTestDatabase();
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
  login("test_user", "test_pass");
  queryAndAssertException(
      "ALTER DATABASE test_database OWNER TO test_user;",
      "Only a super user can change a database's owner. Current user is not a "
      "super-user. Database with name \"test_database\" will not have owner changed.");
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
}

TEST_F(DatabaseDdlTest, ChangeOwnerAsOwner) {
  createTestDatabase("test_user");
  login("test_user", "test_pass");
  assertExpectedDatabase(createDatabaseMetadata("test_database", getTestUser().userId));
  queryAndAssertException(
      "ALTER DATABASE test_database OWNER TO test_user;",
      "Only a super user can change a database's owner. Current user is not a "
      "super-user. Database with name \"test_database\" will not have owner changed.");
  assertExpectedDatabase(createDatabaseMetadata("test_database", getTestUser().userId));
}

TEST_F(DatabaseDdlTest, ChangeOwnerNonExistantDatabase) {
  queryAndAssertException("ALTER DATABASE test_database OWNER TO test_user;",
                          "Database test_database does not exists.");
}

TEST_F(DatabaseDdlTest, ChangeOwnerNonExistantUser) {
  createTestDatabase();
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
  queryAndAssertException("ALTER DATABASE test_database OWNER TO some_user;",
                          "User with username \"some_user\" does not exist. Database "
                          "with name \"test_database\" can not have owner changed.");
}

TEST_F(DatabaseDdlTest, Rename) {
  createTestDatabase();
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
  sql("ALTER DATABASE test_database RENAME TO test_database_new;");
  assertExpectedDatabase(
      createDatabaseMetadata("test_database_new", shared::kRootUserId));
}

TEST_F(DatabaseDdlTest, RenameLackingCredentials) {
  createTestDatabase();
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
  login("test_user", "test_pass");
  queryAndAssertException("ALTER DATABASE test_database RENAME TO test_database_new;",
                          "Only a super user or the owner can rename the database.");
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
}

TEST_F(DatabaseDdlTest, RenameAsOwner) {
  createTestDatabase("test_user");
  login("test_user", "test_pass");
  assertExpectedDatabase(createDatabaseMetadata("test_database", getTestUser().userId));
  sql("ALTER DATABASE test_database RENAME TO test_database_new;");
  assertExpectedDatabase(
      createDatabaseMetadata("test_database_new", getTestUser().userId));
}

TEST_F(DatabaseDdlTest, RenameNonExistantDatabase) {
  queryAndAssertException("ALTER DATABASE test_database OWNER TO test_user;",
                          "Database test_database does not exists.");
}

TEST_F(DatabaseDdlTest, RenameToExistingDatabase) {
  createTestDatabase();
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
  queryAndAssertException("ALTER DATABASE test_database RENAME TO test_database;",
                          "Database test_database already exists.");
  assertExpectedDatabase(createDatabaseMetadata("test_database", shared::kRootUserId));
}

struct ServerObject : public DBHandlerTestFixture {
  Users user_;
  Roles role_;

 protected:
  void SetUp() override {
    if (g_aggregator) {
      LOG(INFO) << "Test fixture not supported in distributed mode.";
      return;
    }
    DBHandlerTestFixture::SetUp();
    sql("CREATE SERVER test_server FOREIGN DATA WRAPPER delimited_file "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  }

  void TearDown() override {
    if (g_aggregator) {
      LOG(INFO) << "Test fixture not supported in distributed mode.";
      return;
    }
    sql("DROP SERVER IF EXISTS test_server;");
    DBHandlerTestFixture::TearDown();
  }
};

TEST_F(GrantSyntax, MultiPrivilegeGrantRevoke) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();
  DBObject tbl_object("tbl", DBObjectType::TableDBObjectType);
  tbl_object.loadKey(cat);
  tbl_object.resetPrivileges();
  auto tbl_object_select = tbl_object;
  auto tbl_object_insert = tbl_object;
  tbl_object_select.setPrivileges(AccessPrivileges::SELECT_FROM_TABLE);
  tbl_object_insert.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE);
  std::vector<DBObject> objects = {tbl_object_select, tbl_object_insert};
  ASSERT_NO_THROW(
      sys_cat.grantDBObjectPrivilegesBatch({"Arsenal", "Juventus"}, objects, cat));
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", objects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", objects), true);
  ASSERT_NO_THROW(
      sys_cat.revokeDBObjectPrivilegesBatch({"Arsenal", "Juventus"}, objects, cat));
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", objects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", objects), false);

  // now the same thing, but with SQL queries
  ASSERT_NO_THROW(
      run_ddl_statement("GRANT SELECT, INSERT ON TABLE tbl TO Arsenal, Juventus"));
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", objects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", objects), true);
  ASSERT_NO_THROW(
      run_ddl_statement("REVOKE SELECT, INSERT ON TABLE tbl FROM Arsenal, Juventus"));
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", objects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", objects), false);
}

TEST_F(GrantSyntax, MultiRoleGrantRevoke) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  std::vector<std::string> roles = {"Gunners", "Sudens"};
  std::vector<std::string> grantees = {"Juventus", "Bayern"};
  auto check_grant = []() {
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Juventus", "Gunners", true), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Bayern", "Gunners", true), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Juventus", "Sudens", true), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Bayern", "Sudens", true), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Arsenal", "Sudens", true), false);
  };
  auto check_revoke = []() {
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Juventus", "Gunners", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Bayern", "Gunners", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Juventus", "Sudens", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("Bayern", "Sudens", true), false);
  };
  ASSERT_NO_THROW(sys_cat.grantRoleBatch(roles, grantees));
  check_grant();
  ASSERT_NO_THROW(sys_cat.revokeRoleBatch(roles, grantees));
  check_revoke();

  // now the same thing, but with SQL queries
  ASSERT_NO_THROW(run_ddl_statement("GRANT Gunners, Sudens TO Juventus, Bayern"));
  check_grant();
  ASSERT_NO_THROW(run_ddl_statement("REVOKE Gunners, Sudens FROM Juventus, Bayern"));
  check_revoke();
}

class InvalidGrantSyntax : public DBHandlerTestFixture {};

TEST_F(InvalidGrantSyntax, InvalidGrantSyntax) {
  std::string error_message;
  error_message =
      "SQL Error: Encountered \"ON\" at line 1, column 23.\nWas expecting one of:\n    "
      "\"ALTER\" ...\n    \"CREATE\" ...\n    \"DELETE\" ...\n    \"DROP\" ...\n    "
      "\"INSERT\" ...\n    \"SELECT\" ...\n    \"TRUNCATE\" ...\n    \"UPDATE\" ...\n    "
      "\"USAGE\" ...\n    \"VIEW\" ...\n    \"ACCESS\" ...\n    \"EDIT\" ...\n    "
      "\"SERVER\" ...\n    \"ALL\" ...\n    ";

  queryAndAssertException("GRANT SELECT, INSERT, ON TABLE tbl TO Arsenal, Juventus;",
                          error_message);
}

TEST_F(InvalidGrantSyntax, InvalidGrantType) {
  std::string error_message;

  error_message = "GRANT failed. Object 'grantView' of type TABLE not found.";
  queryAndAssertException("GRANT SELECT ON TABLE grantView TO Arsenal;", error_message);

  error_message = "REVOKE failed. Object 'grantView' of type TABLE not found.";
  queryAndAssertException("REVOKE SELECT ON TABLE grantView FROM Arsenal;",
                          error_message);

  error_message = "GRANT failed. Object 'tbl' of type VIEW not found.";
  queryAndAssertException("GRANT SELECT ON VIEW tbl TO Arsenal;", error_message);

  error_message = "REVOKE failed. Object 'tbl' of type VIEW not found.";
  queryAndAssertException("REVOKE SELECT ON VIEW tbl FROM Arsenal;", error_message);
}

TEST(UserRoles, InvalidGrantsRevokesTest) {
  run_ddl_statement("CREATE USER Antazin(password = 'password', is_super = 'false');");
  run_ddl_statement("CREATE USER \"Max\"(password = 'password', is_super = 'false');");

  EXPECT_THROW(run_ddl_statement("GRANT Antazin to Antazin;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("REVOKE Antazin from Antazin;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("GRANT Antazin to \"Max\";"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("REVOKE Antazin from \"Max\";"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("GRANT \"Max\" to Antazin;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("REVOKE \"Max\" from Antazin;"), std::runtime_error);

  run_ddl_statement("DROP USER Antazin;");
  run_ddl_statement("DROP USER \"Max\";");
}

TEST(UserRoles, ValidNames) {
  EXPECT_NO_THROW(
      run_ddl_statement("CREATE USER \"dumm.user\" (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER \"dumm.user\";"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE USER vasya (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement(
      "CREATE USER \"vasya.vasya@vasya.com\" (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement(
      "CREATE USER \"vasya ivanov@vasya.ivanov.com\" (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE USER vasya-vasya (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE ROLE developer;"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE ROLE developer-backend;"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE ROLE developer-backend-rendering;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT developer-backend-rendering TO vasya;"));
  EXPECT_NO_THROW(
      run_ddl_statement("GRANT developer-backend TO \"vasya ivanov@vasya.ivanov.com\";"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT developer TO \"vasya.vasya@vasya.com\";"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT developer-backend-rendering TO vasya-vasya;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER vasya;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER \"vasya.vasya@vasya.com\";"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER \"vasya ivanov@vasya.ivanov.com\";"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER vasya-vasya;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP ROLE developer;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP ROLE developer-backend;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP ROLE developer-backend-rendering;"));
}

TEST(UserRoles, RoleHierarchies) {
  // hr prefix here stands for hierarchical roles

  // create objects
  run_ddl_statement("CREATE USER hr_u1 (password = 'u1');");
  run_ddl_statement("CREATE ROLE hr_r1;");
  run_ddl_statement("CREATE ROLE hr_r2;");
  run_ddl_statement("CREATE ROLE hr_r3;");
  run_ddl_statement("CREATE ROLE hr_r4;");
  run_ddl_statement("CREATE TABLE hr_tbl1 (i INTEGER);");

  // check that we can't create cycles
  EXPECT_NO_THROW(run_ddl_statement("GRANT hr_r4 TO hr_r3;"));
  EXPECT_THROW(run_ddl_statement("GRANT hr_r3 TO hr_r4;"), std::runtime_error);
  EXPECT_NO_THROW(run_ddl_statement("GRANT hr_r3 TO hr_r2;"));
  EXPECT_THROW(run_ddl_statement("GRANT hr_r2 TO hr_r4;"), std::runtime_error);

  // make the grant hierarchy
  EXPECT_NO_THROW(run_ddl_statement("GRANT hr_r2 TO hr_r1;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT hr_r1 TO hr_u1;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT SELECT ON TABLE hr_tbl1 TO hr_r1;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT INSERT ON TABLE hr_tbl1 TO hr_r2;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT DELETE ON TABLE hr_tbl1 TO hr_r3;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT UPDATE ON TABLE hr_tbl1 TO hr_r4;"));

  // check that we see privileges gratnted via roles' hierarchy
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();
  AccessPrivileges tbl_privs;
  ASSERT_NO_THROW(tbl_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(tbl_privs.add(AccessPrivileges::INSERT_INTO_TABLE));
  ASSERT_NO_THROW(tbl_privs.add(AccessPrivileges::DELETE_FROM_TABLE));
  ASSERT_NO_THROW(tbl_privs.add(AccessPrivileges::UPDATE_IN_TABLE));
  DBObject tbl1_object("hr_tbl1", DBObjectType::TableDBObjectType);
  tbl1_object.loadKey(cat);
  ASSERT_NO_THROW(tbl1_object.setPrivileges(tbl_privs));
  privObjects.clear();
  privObjects.push_back(tbl1_object);
  EXPECT_EQ(sys_cat.checkPrivileges("hr_u1", privObjects), true);
  // check that when we remove privilege from one role, it's grantees are updated
  EXPECT_NO_THROW(run_ddl_statement("REVOKE DELETE ON TABLE hr_tbl1 FROM hr_r3;"));
  EXPECT_EQ(sys_cat.checkPrivileges("hr_u1", privObjects), false);
  tbl_privs.remove(AccessPrivileges::DELETE_FROM_TABLE);
  ASSERT_NO_THROW(tbl1_object.setPrivileges(tbl_privs));
  privObjects.clear();
  privObjects.push_back(tbl1_object);
  EXPECT_EQ(sys_cat.checkPrivileges("hr_u1", privObjects), true);
  // check that if we remove a role from a middle of hierarchy everythings is fine
  EXPECT_NO_THROW(run_ddl_statement("REVOKE hr_r2 FROM hr_r1;"));
  EXPECT_EQ(sys_cat.checkPrivileges("hr_u1", privObjects), false);
  tbl_privs.remove(AccessPrivileges::UPDATE_IN_TABLE);
  ASSERT_NO_THROW(tbl1_object.setPrivileges(tbl_privs));
  privObjects.clear();
  privObjects.push_back(tbl1_object);
  EXPECT_EQ(sys_cat.checkPrivileges("hr_u1", privObjects), false);
  tbl_privs.remove(AccessPrivileges::INSERT_INTO_TABLE);
  ASSERT_NO_THROW(tbl1_object.setPrivileges(tbl_privs));
  privObjects.clear();
  privObjects.push_back(tbl1_object);
  EXPECT_EQ(sys_cat.checkPrivileges("hr_u1", privObjects), true);

  // clean-up objects
  run_ddl_statement("DROP USER hr_u1;");
  run_ddl_statement("DROP ROLE hr_r1;");
  run_ddl_statement("DROP ROLE hr_r2;");
  run_ddl_statement("DROP ROLE hr_r3;");
  run_ddl_statement("DROP ROLE hr_r4;");
  run_ddl_statement("DROP TABLE hr_tbl1;");
}

TEST(Roles, RecursiveRoleCheckTest) {
  // create roles
  run_ddl_statement("CREATE ROLE regista;");
  run_ddl_statement("CREATE ROLE makelele;");
  run_ddl_statement("CREATE ROLE raumdeuter;");
  run_ddl_statement("CREATE ROLE cdm;");
  run_ddl_statement("CREATE ROLE shadow_cm;");
  run_ddl_statement("CREATE ROLE ngolo_kante;");
  run_ddl_statement("CREATE ROLE thomas_muller;");
  run_ddl_statement("CREATE ROLE jorginho;");
  run_ddl_statement("CREATE ROLE bhagwan;");
  run_ddl_statement("CREATE ROLE messi;");

  // Grant roles
  EXPECT_NO_THROW(run_ddl_statement("GRANT cdm TO regista;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT cdm TO makelele;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT shadow_cm TO raumdeuter;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT regista to jorginho;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT makelele to ngolo_kante;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT raumdeuter to thomas_muller;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT makelele to ngolo_kante;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT makelele to bhagwan;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT regista to bhagwan;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT raumdeuter to bhagwan;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT bhagwan to messi;"));

  auto check_jorginho = [&]() {
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("jorginho", "regista", true), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("jorginho", "makelele", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("jorginho", "raumdeuter", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("jorginho", "cdm", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("jorginho", "cdm", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("jorginho", "shadow_cm", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("jorginho", "bhagwan", false), false);
  };

  auto check_kante = [&]() {
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("ngolo_kante", "regista", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("ngolo_kante", "makelele", true), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("ngolo_kante", "raumdeuter", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("ngolo_kante", "cdm", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("ngolo_kante", "cdm", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("ngolo_kante", "shadow_cm", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("ngolo_kante", "bhagwan", false), false);
  };

  auto check_muller = [&]() {
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("thomas_muller", "regista", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("thomas_muller", "makelele", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("thomas_muller", "raumdeuter", true), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("thomas_muller", "cdm", false), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("thomas_muller", "shadow_cm", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("thomas_muller", "shadow_cm", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("thomas_muller", "bhagwan", false), false);
  };

  auto check_messi = [&]() {
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "regista", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "makelele", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "raumdeuter", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "regista", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "makelele", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "raumdeuter", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "cdm", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "cdm", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "shadow_cm", true), false);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "shadow_cm", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "bhagwan", false), true);
    EXPECT_EQ(sys_cat.isRoleGrantedToGrantee("messi", "bhagwan", true), true);
  };

  auto drop_roles = [&]() {
    run_ddl_statement("DROP ROLE regista;");
    run_ddl_statement("DROP ROLE makelele;");
    run_ddl_statement("DROP ROLE raumdeuter;");
    run_ddl_statement("DROP ROLE cdm;");
    run_ddl_statement("DROP ROLE shadow_cm;");
    run_ddl_statement("DROP ROLE ngolo_kante;");
    run_ddl_statement("DROP ROLE thomas_muller;");
    run_ddl_statement("DROP ROLE jorginho;");
    run_ddl_statement("DROP ROLE bhagwan;");
    run_ddl_statement("DROP ROLE messi;");
  };

  // validate recursive roles
  check_jorginho();
  check_kante();
  check_muller();
  check_messi();

  // cleanup objects
  drop_roles();
}

TEST_F(DatabaseObject, AccessDefaultsTest) {
  auto cat_mapd = sys_cat.getCatalog(shared::kDefaultDbName);
  DBObject mapd_object(shared::kDefaultDbName, DBObjectType::DatabaseDBObjectType);
  privObjects.clear();
  mapd_object.loadKey(*cat_mapd);

  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

TEST_F(DatabaseObject, SqlEditorAccessTest) {
  std::unique_ptr<Catalog_Namespace::SessionInfo> session_juve;

  CHECK(sys_cat.getMetadataForDB(shared::kDefaultDbName, db_meta));
  CHECK(sys_cat.getMetadataForUser("Juventus", user_meta));
  session_juve.reset(new Catalog_Namespace::SessionInfo(
      sys_cat.getCatalog(db_meta.dbName), user_meta, ExecutorDeviceType::GPU, ""));
  auto& cat_mapd = session_juve->getCatalog();
  DBObject mapd_object(shared::kDefaultDbName, DBObjectType::DatabaseDBObjectType);
  privObjects.clear();
  mapd_object.loadKey(cat_mapd);
  mapd_object.setPermissionType(DatabaseDBObjectType);
  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::VIEW_SQL_EDITOR));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::VIEW_SQL_EDITOR));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivilegesBatch(
      {"Chelsea", "Juventus"}, {mapd_object}, cat_mapd));

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::VIEW_SQL_EDITOR));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::VIEW_SQL_EDITOR));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivilegesBatch(
      {"Chelsea", "Juventus"}, {mapd_object}, cat_mapd));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivilegesBatch(
      {"Bayern", "Arsenal"}, {mapd_object}, cat_mapd));

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::VIEW_SQL_EDITOR));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);
}

TEST_F(DatabaseObject, DBLoginAccessTest) {
  std::unique_ptr<Catalog_Namespace::SessionInfo> session_juve;

  CHECK(sys_cat.getMetadataForDB(shared::kDefaultDbName, db_meta));
  CHECK(sys_cat.getMetadataForUser("Bayern", user_meta));
  session_juve.reset(new Catalog_Namespace::SessionInfo(
      sys_cat.getCatalog(db_meta.dbName), user_meta, ExecutorDeviceType::GPU, ""));
  auto& cat_mapd = session_juve->getCatalog();
  DBObject mapd_object(shared::kDefaultDbName, DBObjectType::DatabaseDBObjectType);
  privObjects.clear();
  mapd_object.loadKey(cat_mapd);
  mapd_object.setPermissionType(DatabaseDBObjectType);
  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ACCESS));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ACCESS));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivilegesBatch(
      {"Arsenal", "Bayern"}, {mapd_object}, cat_mapd));

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ACCESS));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ACCESS));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivilegesBatch(
      {"Bayern", "Arsenal"}, {mapd_object}, cat_mapd));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Juventus", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::VIEW_SQL_EDITOR));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

TEST_F(DatabaseObject, TableAccessTest) {
  std::unique_ptr<Catalog_Namespace::SessionInfo> session_ars;

  CHECK(sys_cat.getMetadataForDB(shared::kDefaultDbName, db_meta));
  CHECK(sys_cat.getMetadataForUser("Arsenal", user_meta));
  session_ars.reset(new Catalog_Namespace::SessionInfo(
      sys_cat.getCatalog(db_meta.dbName), user_meta, ExecutorDeviceType::GPU, ""));
  auto& cat_mapd = session_ars->getCatalog();
  AccessPrivileges arsenal_privs;
  AccessPrivileges bayern_privs;
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::CREATE_TABLE));
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::DROP_TABLE));
  ASSERT_NO_THROW(bayern_privs.add(AccessPrivileges::ALTER_TABLE));
  DBObject mapd_object(shared::kDefaultDbName, DBObjectType::DatabaseDBObjectType);
  privObjects.clear();
  mapd_object.loadKey(cat_mapd);
  mapd_object.setPermissionType(TableDBObjectType);
  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(arsenal_privs.remove(AccessPrivileges::CREATE_TABLE));
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::CREATE_TABLE));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(bayern_privs));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(bayern_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Bayern", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(bayern_privs));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);
}

TEST_F(DatabaseObject, ViewAccessTest) {
  std::unique_ptr<Catalog_Namespace::SessionInfo> session_ars;
  CHECK(sys_cat.getMetadataForDB(shared::kDefaultDbName, db_meta));
  CHECK(sys_cat.getMetadataForUser("Arsenal", user_meta));
  session_ars.reset(new Catalog_Namespace::SessionInfo(
      sys_cat.getCatalog(db_meta.dbName), user_meta, ExecutorDeviceType::GPU, ""));
  auto& cat_mapd = session_ars->getCatalog();
  AccessPrivileges arsenal_privs;
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::ALL_VIEW));
  DBObject mapd_object(shared::kDefaultDbName, DBObjectType::DatabaseDBObjectType);
  privObjects.clear();
  mapd_object.loadKey(cat_mapd);
  mapd_object.setPermissionType(ViewDBObjectType);
  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(arsenal_privs.remove(AccessPrivileges::DROP_VIEW));
  ASSERT_NO_THROW(arsenal_privs.remove(AccessPrivileges::TRUNCATE_VIEW));
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ALL_VIEW));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::DROP_VIEW));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::TRUNCATE_VIEW));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
}

TEST_F(DatabaseObject, DashboardAccessTest) {
  std::unique_ptr<Catalog_Namespace::SessionInfo> session_ars;
  CHECK(sys_cat.getMetadataForDB(shared::kDefaultDbName, db_meta));
  CHECK(sys_cat.getMetadataForUser("Arsenal", user_meta));
  session_ars.reset(new Catalog_Namespace::SessionInfo(
      sys_cat.getCatalog(db_meta.dbName), user_meta, ExecutorDeviceType::GPU, ""));
  auto& cat_mapd = session_ars->getCatalog();
  AccessPrivileges arsenal_privs;
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::ALL_DASHBOARD));
  DBObject mapd_object(shared::kDefaultDbName, DBObjectType::DatabaseDBObjectType);
  privObjects.clear();
  mapd_object.loadKey(cat_mapd);
  mapd_object.setPermissionType(DashboardDBObjectType);
  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(arsenal_privs.remove(AccessPrivileges::EDIT_DASHBOARD));
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::CREATE_DASHBOARD));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::DELETE_DASHBOARD));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  mapd_object.resetPrivileges();
  privObjects.clear();
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::EDIT_DASHBOARD));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
}

TEST_F(DatabaseObject, DatabaseAllTest) {
  std::unique_ptr<Catalog_Namespace::SessionInfo> session_ars;

  CHECK(sys_cat.getMetadataForDB(shared::kDefaultDbName, db_meta));
  CHECK(sys_cat.getMetadataForUser("Arsenal", user_meta));
  session_ars.reset(new Catalog_Namespace::SessionInfo(
      sys_cat.getCatalog(db_meta.dbName), user_meta, ExecutorDeviceType::GPU, ""));
  auto& cat_mapd = session_ars->getCatalog();
  AccessPrivileges arsenal_privs;
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::ALL_DATABASE));
  DBObject mapd_object(shared::kDefaultDbName, DBObjectType::DatabaseDBObjectType);
  privObjects.clear();
  mapd_object.loadKey(cat_mapd);
  mapd_object.resetPrivileges();
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  mapd_object.setPermissionType(TableDBObjectType);
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ALL_TABLE));
  privObjects.push_back(mapd_object);
  mapd_object.setPermissionType(ViewDBObjectType);
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ALL_VIEW));
  privObjects.push_back(mapd_object);
  mapd_object.setPermissionType(DashboardDBObjectType);
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ALL_DASHBOARD));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);

  mapd_object.resetPrivileges();
  privObjects.clear();
  mapd_object.setPermissionType(TableDBObjectType);
  ASSERT_NO_THROW(arsenal_privs.remove(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  privObjects.clear();
  arsenal_privs.reset();
  mapd_object.setPermissionType(DashboardDBObjectType);
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::DELETE_DASHBOARD));
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  privObjects.clear();
  mapd_object.setPermissionType(ViewDBObjectType);
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ALL_VIEW));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);

  mapd_object.resetPrivileges();
  privObjects.clear();
  mapd_object.setPermissionType(DashboardDBObjectType);
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::DELETE_DASHBOARD));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);

  mapd_object.resetPrivileges();
  privObjects.clear();
  mapd_object.setPermissionType(TableDBObjectType);
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::SELECT_FROM_TABLE));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);

  mapd_object.resetPrivileges();
  privObjects.clear();
  arsenal_privs.reset();
  mapd_object.setPermissionType(DatabaseDBObjectType);
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::ALL_DATABASE));
  ASSERT_NO_THROW(mapd_object.setPrivileges(arsenal_privs));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", mapd_object, cat_mapd));

  mapd_object.resetPrivileges();
  privObjects.clear();
  mapd_object.setPermissionType(ViewDBObjectType);
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::ALL_VIEW));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);

  mapd_object.resetPrivileges();
  privObjects.clear();
  mapd_object.setPermissionType(TableDBObjectType);
  ASSERT_NO_THROW(mapd_object.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE));
  privObjects.push_back(mapd_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
}

TEST_F(TableObject, AccessDefaultsTest) {
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();
  ASSERT_NO_THROW(sys_cat.grantRole("Sudens", "Bayern"));
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  AccessPrivileges epl_privs;
  AccessPrivileges seriea_privs;
  AccessPrivileges bundesliga_privs;
  ASSERT_NO_THROW(epl_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(seriea_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(bundesliga_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  privObjects.clear();
  DBObject epl_object("epl", DBObjectType::TableDBObjectType);
  DBObject seriea_object("seriea", DBObjectType::TableDBObjectType);
  DBObject bundesliga_object("bundesliga", DBObjectType::TableDBObjectType);
  epl_object.loadKey(cat);
  seriea_object.loadKey(cat);
  bundesliga_object.loadKey(cat);
  ASSERT_NO_THROW(epl_object.setPrivileges(epl_privs));
  ASSERT_NO_THROW(seriea_object.setPrivileges(seriea_privs));
  ASSERT_NO_THROW(bundesliga_object.setPrivileges(bundesliga_privs));
  privObjects.push_back(epl_object);
  privObjects.push_back(seriea_object);
  privObjects.push_back(bundesliga_object);

  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}
TEST_F(TableObject, AccessAfterGrantsTest) {
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();
  ASSERT_NO_THROW(sys_cat.grantRole("Sudens", "Bayern"));
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  AccessPrivileges epl_privs;
  AccessPrivileges seriea_privs;
  AccessPrivileges bundesliga_privs;
  ASSERT_NO_THROW(epl_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(seriea_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(bundesliga_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  privObjects.clear();
  DBObject epl_object("epl", DBObjectType::TableDBObjectType);
  DBObject seriea_object("seriea", DBObjectType::TableDBObjectType);
  DBObject bundesliga_object("bundesliga", DBObjectType::TableDBObjectType);
  epl_object.loadKey(cat);
  seriea_object.loadKey(cat);
  bundesliga_object.loadKey(cat);
  ASSERT_NO_THROW(epl_object.setPrivileges(epl_privs));
  ASSERT_NO_THROW(seriea_object.setPrivileges(seriea_privs));
  ASSERT_NO_THROW(bundesliga_object.setPrivileges(bundesliga_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", epl_object, cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Sudens", bundesliga_object, cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("OldLady", seriea_object, cat));

  privObjects.push_back(epl_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
  privObjects.clear();
  privObjects.push_back(seriea_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
  privObjects.clear();
  privObjects.push_back(bundesliga_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);
}

TEST_F(TableObject, AccessAfterRevokesTest) {
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  ASSERT_NO_THROW(sys_cat.grantRole("Gunners", "Arsenal"));
  AccessPrivileges epl_privs;
  AccessPrivileges seriea_privs;
  AccessPrivileges bundesliga_privs;
  ASSERT_NO_THROW(epl_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(epl_privs.add(AccessPrivileges::INSERT_INTO_TABLE));
  ASSERT_NO_THROW(seriea_privs.add(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(bundesliga_privs.add(AccessPrivileges::ALL_TABLE));
  privObjects.clear();
  DBObject epl_object("epl", DBObjectType::TableDBObjectType);
  DBObject seriea_object("seriea", DBObjectType::TableDBObjectType);
  DBObject bundesliga_object("bundesliga", DBObjectType::TableDBObjectType);
  epl_object.loadKey(cat);
  seriea_object.loadKey(cat);
  bundesliga_object.loadKey(cat);
  ASSERT_NO_THROW(epl_object.setPrivileges(epl_privs));
  ASSERT_NO_THROW(seriea_object.setPrivileges(seriea_privs));
  ASSERT_NO_THROW(bundesliga_object.setPrivileges(bundesliga_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Gunners", epl_object, cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Bayern", bundesliga_object, cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("OldLady", seriea_object, cat));

  privObjects.push_back(epl_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
  privObjects.clear();
  privObjects.push_back(seriea_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
  privObjects.clear();
  privObjects.push_back(bundesliga_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);

  epl_object.resetPrivileges();
  seriea_object.resetPrivileges();
  bundesliga_object.resetPrivileges();
  ASSERT_NO_THROW(epl_privs.remove(AccessPrivileges::SELECT_FROM_TABLE));
  ASSERT_NO_THROW(epl_object.setPrivileges(epl_privs));
  ASSERT_NO_THROW(seriea_object.setPrivileges(seriea_privs));
  ASSERT_NO_THROW(bundesliga_object.setPrivileges(bundesliga_privs));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Gunners", epl_object, cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Bayern", bundesliga_object, cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("OldLady", seriea_object, cat));

  epl_object.resetPrivileges();

  ASSERT_NO_THROW(epl_object.setPrivileges(AccessPrivileges::SELECT_FROM_TABLE));
  privObjects.clear();
  privObjects.push_back(epl_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);

  epl_object.resetPrivileges();
  ASSERT_NO_THROW(epl_object.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE));
  privObjects.clear();
  privObjects.push_back(epl_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);

  seriea_object.resetPrivileges();
  ASSERT_NO_THROW(seriea_object.setPrivileges(AccessPrivileges::SELECT_FROM_TABLE));
  privObjects.clear();
  privObjects.push_back(seriea_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);

  seriea_object.resetPrivileges();
  ASSERT_NO_THROW(seriea_object.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE));
  privObjects.clear();
  privObjects.push_back(seriea_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);

  bundesliga_object.resetPrivileges();
  ASSERT_NO_THROW(bundesliga_object.setPrivileges(AccessPrivileges::ALL_TABLE));
  privObjects.clear();
  privObjects.push_back(bundesliga_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

void testViewPermissions(std::string user, std::string roleToGrant) {
  DBObject bill_view("bill_view", DBObjectType::ViewDBObjectType);
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();
  bill_view.loadKey(cat);
  std::vector<DBObject> privs;

  bill_view.setPrivileges(AccessPrivileges::CREATE_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), false);

  bill_view.setPrivileges(AccessPrivileges::DROP_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), false);

  bill_view.setPrivileges(AccessPrivileges::SELECT_FROM_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), false);

  bill_view.setPrivileges(AccessPrivileges::CREATE_VIEW);
  sys_cat.grantDBObjectPrivileges(roleToGrant, bill_view, cat);
  bill_view.setPrivileges(AccessPrivileges::CREATE_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), true);

  bill_view.setPrivileges(AccessPrivileges::DROP_VIEW);
  sys_cat.grantDBObjectPrivileges(roleToGrant, bill_view, cat);
  bill_view.setPrivileges(AccessPrivileges::DROP_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), true);

  bill_view.setPrivileges(AccessPrivileges::SELECT_FROM_VIEW);
  sys_cat.grantDBObjectPrivileges(roleToGrant, bill_view, cat);
  bill_view.setPrivileges(AccessPrivileges::SELECT_FROM_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), true);

  bill_view.setPrivileges(AccessPrivileges::CREATE_VIEW);
  sys_cat.revokeDBObjectPrivileges(roleToGrant, bill_view, cat);
  bill_view.setPrivileges(AccessPrivileges::CREATE_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), false);

  bill_view.setPrivileges(AccessPrivileges::DROP_VIEW);
  sys_cat.revokeDBObjectPrivileges(roleToGrant, bill_view, cat);
  bill_view.setPrivileges(AccessPrivileges::DROP_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), false);

  bill_view.setPrivileges(AccessPrivileges::SELECT_FROM_VIEW);
  sys_cat.revokeDBObjectPrivileges(roleToGrant, bill_view, cat);
  bill_view.setPrivileges(AccessPrivileges::SELECT_FROM_VIEW);
  privs = {bill_view};
  EXPECT_EQ(sys_cat.checkPrivileges(user, privs), false);
}

TEST_F(ViewObject, UserRoleBobGetsGrants) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }
  testViewPermissions("bob", "bob");
}

TEST_F(ViewObject, GroupRoleFooGetsGrants) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }
  testViewPermissions("foo", "salesDept");
}

TEST_F(ViewObject, CalciteViewResolution) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  auto calciteQueryParsingOption =
      g_calcite->getCalciteQueryParsingOption(true, false, true, false);
  auto calciteOptimizationOption =
      g_calcite->getCalciteOptimizationOption(false, false, {}, false);

  auto query_state1 =
      QR::create_query_state(QR::get()->getSession(), "select * from bill_table");
  TPlanResult result = ::g_calcite->process(query_state1->createQueryStateProxy(),
                                            query_state1->getQueryStr(),
                                            calciteQueryParsingOption,
                                            calciteOptimizationOption);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.primary_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from[0][0], "bill_table");
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.resolved_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from[0][0], "bill_table");

  auto query_state2 =
      QR::create_query_state(QR::get()->getSession(), "select * from bill_view");
  result = ::g_calcite->process(query_state2->createQueryStateProxy(),
                                query_state2->getQueryStr(),
                                calciteQueryParsingOption,
                                calciteOptimizationOption);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.primary_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from[0][0], "bill_view");
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.resolved_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from[0][0], "bill_table");

  auto query_state3 =
      QR::create_query_state(QR::get()->getSession(), "select * from bill_view_outer");
  result = ::g_calcite->process(query_state3->createQueryStateProxy(),
                                query_state3->getQueryStr(),
                                calciteQueryParsingOption,
                                calciteOptimizationOption);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.primary_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from[0][0],
            "bill_view_outer");
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.resolved_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from[0][0], "bill_table");
}

TEST_F(DashboardObject, AccessDefaultsTest) {
  const auto cat = QR::get()->getCatalog();
  ASSERT_NO_THROW(sys_cat.grantRole("Gunners", "Bayern"));
  ASSERT_NO_THROW(sys_cat.grantRole("Sudens", "Arsenal"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(*cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);

  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

TEST_F(DashboardObject, AccessAfterGrantsTest) {
  const auto cat = QR::get()->getCatalog();
  ASSERT_NO_THROW(sys_cat.grantRole("Gunners", "Arsenal"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(*cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);
  ASSERT_NO_THROW(
      sys_cat.grantDBObjectPrivilegesBatch({"Gunners", "Juventus"}, {dash_object}, *cat));

  privObjects.clear();
  privObjects.push_back(dash_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

TEST_F(DashboardObject, AccessAfterRevokesTest) {
  const auto cat = QR::get()->getCatalog();
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  ASSERT_NO_THROW(sys_cat.grantRole("Sudens", "Bayern"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(*cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);
  ASSERT_NO_THROW(
      sys_cat.grantDBObjectPrivilegesBatch({"OldLady", "Arsenal"}, {dash_object}, *cat));

  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("OldLady", dash_object, *cat));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", dash_object, *cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Bayern", dash_object, *cat));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);
}

TEST_F(DashboardObject, GranteesDefaultListTest) {
  const auto cat = QR::get()->getCatalog();
  auto perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int size = static_cast<int>(perms_list.size());
  ASSERT_EQ(size, 0);
}

TEST_F(DashboardObject, GranteesListAfterGrantsTest) {
  const auto cat = QR::get()->getCatalog();
  auto perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int recs1 = static_cast<int>(perms_list.size());
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(*cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);
  ASSERT_NO_THROW(
      sys_cat.grantDBObjectPrivilegesBatch({"OldLady", "Bayern"}, {dash_object}, *cat));
  perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int recs2 = static_cast<int>(perms_list.size());
  ASSERT_NE(recs1, recs2);
  ASSERT_EQ(recs2, 2);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_FALSE(perms_list[1]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::EDIT_DASHBOARD));
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Bayern", dash_object, *cat));
  perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int recs3 = static_cast<int>(perms_list.size());
  ASSERT_EQ(recs3, 2);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));
}

TEST_F(DashboardObject, GranteesListAfterRevokesTest) {
  const auto cat = QR::get()->getCatalog();
  auto perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int recs1 = static_cast<int>(perms_list.size());
  ASSERT_NO_THROW(sys_cat.grantRole("Gunners", "Arsenal"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::EDIT_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(*cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);
  ASSERT_NO_THROW(
      sys_cat.grantDBObjectPrivilegesBatch({"Gunners", "Bayern"}, {dash_object}, *cat));
  perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int recs2 = static_cast<int>(perms_list.size());
  ASSERT_NE(recs1, recs2);
  ASSERT_EQ(recs2, 2);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

  ASSERT_NO_THROW(dash_priv.remove(AccessPrivileges::VIEW_DASHBOARD));
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Gunners", dash_object, *cat));
  perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int recs3 = static_cast<int>(perms_list.size());
  ASSERT_EQ(recs3, 2);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_FALSE(perms_list[0]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Gunners", dash_object, *cat));
  perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int recs4 = static_cast<int>(perms_list.size());
  ASSERT_EQ(recs4, 1);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::EDIT_DASHBOARD));
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Bayern", dash_object, *cat));
  perms_list =
      sys_cat.getMetadataForObject(cat->getCurrentDB().dbId,
                                   static_cast<int>(DBObjectType::DashboardDBObjectType),
                                   id);
  int recs5 = static_cast<int>(perms_list.size());
  ASSERT_EQ(recs1, recs5);
  ASSERT_EQ(recs5, 0);
}

TEST_F(ServerObject, AccessDefaultsTest) {
  if (g_aggregator) {
    LOG(INFO) << "Test not supported in distributed mode.";
    return;
  }
  Catalog_Namespace::Catalog& cat = getCatalog();
  AccessPrivileges server_priv;
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::DROP_SERVER));
  privObjects.clear();
  DBObject server_object("test_server", DBObjectType::ServerDBObjectType);
  server_object.loadKey(cat);
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.push_back(server_object);

  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

TEST_F(ServerObject, AccessAfterGrantsRevokes) {
  if (g_aggregator) {
    LOG(INFO) << "Test not supported in distributed mode.";
    return;
  }
  Catalog_Namespace::Catalog& cat = getCatalog();
  ASSERT_NO_THROW(sys_cat.grantRole("Sudens", "Bayern"));
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  AccessPrivileges server_priv;
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::DROP_SERVER));
  DBObject server_object("test_server", DBObjectType::ServerDBObjectType);
  server_object.loadKey(cat);
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", server_object, cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Sudens", server_object, cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("OldLady", server_object, cat));
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);

  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", server_object, cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Sudens", server_object, cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("OldLady", server_object, cat));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

TEST_F(ServerObject, AccessWithGrantRevokeAllCompound) {
  if (g_aggregator) {
    LOG(INFO) << "Test not supported in distributed mode.";
    return;
  }
  Catalog_Namespace::Catalog& cat = getCatalog();
  ASSERT_NO_THROW(sys_cat.grantRole("Sudens", "Bayern"));
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  AccessPrivileges server_priv;
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::ALL_SERVER));
  DBObject server_object("test_server", DBObjectType::ServerDBObjectType);
  server_object.loadKey(cat);

  // Effectively give all users ALL_SERVER privileges
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", server_object, cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Sudens", server_object, cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("OldLady", server_object, cat));

  // All users should now have ALL_SERVER privileges, check this is true
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);

  // Check that all users have a subset of ALL_SERVER privileges, in this case
  // check that is true for CREATE_SERVER
  server_priv.reset();
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::CREATE_SERVER));
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);

  // Revoke CREATE_SERVER from all users and check that they don't have it
  // anymore (expect super-users)
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", server_object, cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Sudens", server_object, cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("OldLady", server_object, cat));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  // All users should still have DROP_SERVER privileges, check that this is true
  server_priv.reset();
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::DROP_SERVER));
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);

  // All users should still have ALTER_SERVER privileges, check that this is true
  server_priv.reset();
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::ALTER_SERVER));
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);

  // All users should still have SERVER_USAGE privileges, check that this is true
  server_priv.reset();
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::SERVER_USAGE));
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);

  // Revoke ALL_SERVER privileges
  server_priv.reset();
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::ALL_SERVER));
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.clear();
  privObjects.push_back(server_object);
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", server_object, cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Sudens", server_object, cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("OldLady", server_object, cat));

  // Check that after the revoke of ALL_SERVER privileges users no longer
  // have the DROP_SERVER privilege (except super-users)
  server_priv.reset();
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::DROP_SERVER));
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  // users no longer have the ALTER_SERVER privileges, check that this is true
  server_priv.reset();
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::ALTER_SERVER));
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  // users no longer have the SERVER_USAGE privileges, check that this is true
  server_priv.reset();
  ASSERT_NO_THROW(server_priv.add(AccessPrivileges::SERVER_USAGE));
  ASSERT_NO_THROW(server_object.setPrivileges(server_priv));
  privObjects.clear();
  privObjects.push_back(server_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

void create_tables(std::string prefix, int max) {
  const auto cat = QR::get()->getCatalog();
  for (int i = 0; i < max; i++) {
    std::string name = prefix + std::to_string(i);
    run_ddl_statement("CREATE TABLE " + name + " (id integer);");
    auto td = cat->getMetadataForTable(name, false);
    ASSERT_TRUE(td);
    ASSERT_EQ(td->isView, false);
    ASSERT_EQ(td->tableName, name);
  }
}

void create_views(std::string prefix, int max) {
  const auto cat = QR::get()->getCatalog();
  for (int i = 0; i < max; i++) {
    std::string name = "view_" + prefix + std::to_string(i);
    run_ddl_statement("CREATE VIEW " + name + " AS SELECT * FROM " + prefix +
                      std::to_string(i) + ";");
    auto td = cat->getMetadataForTable(name, false);
    ASSERT_TRUE(td);
    ASSERT_EQ(td->isView, true);
    ASSERT_EQ(td->tableName, name);
  }
}

void create_dashboards(std::string prefix, int max) {
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();
  for (int i = 0; i < max; i++) {
    std::string name = "dash_" + prefix + std::to_string(i);
    DashboardDescriptor vd;
    vd.dashboardName = name;
    vd.dashboardState = name;
    vd.imageHash = name;
    vd.dashboardMetadata = name;
    vd.userId = session->get_currentUser().userId;
    ASSERT_EQ(0, session->get_currentUser().userId);
    vd.user = session->get_currentUser().userName;
    cat.createDashboard(vd);

    auto fvd = cat.getMetadataForDashboard(
        std::to_string(session->get_currentUser().userId), name);
    ASSERT_TRUE(fvd);
    ASSERT_EQ(fvd->dashboardName, name);
  }
}

void assert_grants(std::string prefix, int i, bool expected) {
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();

  DBObject tablePermission(prefix + std::to_string(i), DBObjectType::TableDBObjectType);
  try {
    sys_cat.getDBObjectPrivileges("bob", tablePermission, cat);
  } catch (std::runtime_error& e) {
  }
  ASSERT_EQ(
      expected,
      tablePermission.getPrivileges().hasPermission(TablePrivileges::SELECT_FROM_TABLE));

  DBObject viewPermission("view_" + prefix + std::to_string(i),
                          DBObjectType::ViewDBObjectType);
  try {
    sys_cat.getDBObjectPrivileges("bob", viewPermission, cat);
  } catch (std::runtime_error& e) {
  }
  ASSERT_EQ(
      expected,
      viewPermission.getPrivileges().hasPermission(ViewPrivileges::SELECT_FROM_VIEW));

  auto fvd =
      cat.getMetadataForDashboard(std::to_string(session->get_currentUser().userId),
                                  "dash_" + prefix + std::to_string(i));
  DBObject dashPermission(fvd->dashboardId, DBObjectType::DashboardDBObjectType);
  try {
    sys_cat.getDBObjectPrivileges("bob", dashPermission, cat);
  } catch (std::runtime_error& e) {
  }
  ASSERT_EQ(
      expected,
      dashPermission.getPrivileges().hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
}

void check_grant_access(std::string prefix, int max) {
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();

  for (int i = 0; i < max; i++) {
    assert_grants(prefix, i, false);

    auto fvd =
        cat.getMetadataForDashboard(std::to_string(session->get_currentUser().userId),
                                    "dash_" + prefix + std::to_string(i));
    run_ddl_statement("GRANT SELECT ON TABLE " + prefix + std::to_string(i) + " TO bob;");
    run_ddl_statement("GRANT SELECT ON VIEW view_" + prefix + std::to_string(i) +
                      " TO bob;");
    run_ddl_statement("GRANT VIEW ON DASHBOARD " + std::to_string(fvd->dashboardId) +
                      " TO bob;");
    assert_grants(prefix, i, true);

    run_ddl_statement("REVOKE SELECT ON TABLE " + prefix + std::to_string(i) +
                      " FROM bob;");
    run_ddl_statement("REVOKE SELECT ON VIEW view_" + prefix + std::to_string(i) +
                      " FROM bob;");
    run_ddl_statement("REVOKE VIEW ON DASHBOARD " + std::to_string(fvd->dashboardId) +
                      " FROM bob;");
    assert_grants(prefix, i, false);
  }
}

void drop_dashboards(std::string prefix, int max) {
  auto session = QR::get()->getSession();
  CHECK(session);
  auto& cat = session->getCatalog();

  for (int i = 0; i < max; i++) {
    std::string name = "dash_" + prefix + std::to_string(i);
    auto dash = cat.getMetadataForDashboard(
        std::to_string(session->get_currentUser().userId), name);
    ASSERT_TRUE(dash);
    cat.deleteMetadataForDashboards({dash->dashboardId}, session->get_currentUser());
    auto fvd = cat.getMetadataForDashboard(
        std::to_string(session->get_currentUser().userId), name);
    ASSERT_FALSE(fvd);
  }
}

void drop_views(std::string prefix, int max) {
  const auto cat = QR::get()->getCatalog();

  for (int i = 0; i < max; i++) {
    std::string name = "view_" + prefix + std::to_string(i);
    run_ddl_statement("DROP VIEW " + name + ";");
    auto td = cat->getMetadataForTable(name, false);
    ASSERT_FALSE(td);
  }
}

void drop_tables(std::string prefix, int max) {
  const auto cat = QR::get()->getCatalog();

  for (int i = 0; i < max; i++) {
    std::string name = prefix + std::to_string(i);
    run_ddl_statement("DROP TABLE " + name + ";");
    auto td = cat->getMetadataForTable(name, false);
    ASSERT_FALSE(td);
  }
}

void run_concurrency_test(std::string prefix, int max) {
  create_tables(prefix, max);
  create_views(prefix, max);
  create_dashboards(prefix, max);
  check_grant_access(prefix, max);
  drop_dashboards(prefix, max);
  drop_views(prefix, max);
  drop_tables(prefix, max);
}

TEST(Catalog, Concurrency) {
  run_ddl_statement("CREATE USER bob (password = 'password', is_super = 'false');");
  std::string prefix = "for_bob";

  // only a single thread at the moment!
  // because calcite access the sqlite-dbs
  // directly when started in this mode
  int num_threads = 1;
  std::vector<std::shared_ptr<std::thread>> my_threads;

  for (int i = 0; i < num_threads; i++) {
    std::string prefix = "for_bob_" + std::to_string(i) + "_";
    my_threads.push_back(
        std::make_shared<std::thread>(run_concurrency_test, prefix, 100));
  }

  for (auto& thread : my_threads) {
    thread->join();
  }

  run_ddl_statement("DROP USER bob;");
}

TEST(DBObject, LoadKey) {
  static const std::string tbname{"test_tb"};
  static const std::string vwname{"test_vw"};

  // cleanup
  struct CleanupGuard {
    ~CleanupGuard() {
      run_ddl_statement("DROP VIEW IF EXISTS " + vwname + ";");
      run_ddl_statement("DROP TABLE IF EXISTS " + tbname + ";");
    }
  } cleanupGuard;

  // setup
  run_ddl_statement("CREATE TABLE IF NOT EXISTS " + tbname + "(id integer);");
  run_ddl_statement("CREATE VIEW IF NOT EXISTS " + vwname + " AS SELECT id FROM " +
                    tbname + ";");

  // test the LoadKey() function
  auto cat = sys_cat.getCatalog(shared::kDefaultDbName);

  DBObject dbo1(shared::kDefaultDbName, DBObjectType::DatabaseDBObjectType);
  DBObject dbo2(tbname, DBObjectType::TableDBObjectType);
  DBObject dbo3(vwname, DBObjectType::ViewDBObjectType);

  ASSERT_NO_THROW(dbo1.loadKey());
  ASSERT_NO_THROW(dbo2.loadKey(*cat));
  ASSERT_NO_THROW(dbo3.loadKey(*cat));
}

// TODO(Misiu): The SysCatalog tests (and possibly more) have unspecified pre-conditions
// that can cause failures if not identified (such as pre-existing users or dbs).  We
// should clean these tests up to guarantee preconditions and postconditions.  Namely, we
// should check that there are no extra dbs/users at the start of the test suite, and make
// sure users/dbs we intend to create do not exist prior to the test.
TEST(SysCatalog, RenameUser_Basic) {
  using namespace std::string_literals;
  auto username = "chuck"s;
  auto database_name = "nydb"s;
  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    run_ddl_statement("DROP DATABASE nydb;");
    if (rename_successful) {
      run_ddl_statement("DROP USER cryingchuck;");
    } else {
      run_ddl_statement("DROP USER chuck");
    }
  };

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

  run_ddl_statement("DROP DATABASE IF EXISTS nydb");
  run_ddl_statement("DROP USER IF EXISTS chuck");

  run_ddl_statement("CREATE USER chuck (password='password');");
  run_ddl_statement("CREATE DATABASE nydb (owner='chuck');");
  run_ddl_statement("ALTER USER chuck (default_db='nydb')");

  // Check ability to login
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));

  // Rename should be fine
  EXPECT_NO_THROW(run_ddl_statement("ALTER USER chuck RENAME TO cryingchuck;"););

  rename_successful = true;

  // Check if we can login as the new user
  username_out = "cryingchuck"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));
}

TEST(SysCatalog, RenameUser_AlreadyExisting) {
  using namespace std::string_literals;

  auto username = "chuck"s;
  auto new_username = "marcy"s;

  ScopeGuard scope_guard = [] {
    run_ddl_statement("DROP USER chuck;");
    run_ddl_statement("DROP USER marcy;");
  };

  run_ddl_statement("CREATE USER chuck (password='password');");
  run_ddl_statement("CREATE USER marcy (password='password');");

  EXPECT_THROW(run_ddl_statement("ALTER USER chuck RENAME TO marcy;"),
               std::runtime_error);
}

TEST(SysCatalog, RenameUser_UserDoesntExist) {
  using namespace std::string_literals;

  EXPECT_THROW(run_ddl_statement("ALTER USER lemont RENAME TO sanford;"),
               std::runtime_error);
}

namespace {

std::unique_ptr<QR> get_qr_for_user(
    const std::string& user_name,
    const Catalog_Namespace::UserMetadata& user_metadata) {
  auto session = std::make_unique<Catalog_Namespace::SessionInfo>(
      sys_cat.getCatalog(user_name), user_metadata, ExecutorDeviceType::CPU, "");
  return std::make_unique<QR>(std::move(session));
}

}  // namespace

TEST(SysCatalog, RenameUser_AlreadyLoggedInQueryAfterRename) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  auto username = "chuck"s;
  auto database_name = "nydb"s;
  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    run_ddl_statement("DROP DATABASE nydb;");
    if (rename_successful) {
      run_ddl_statement("DROP USER cryingchuck;");
    } else {
      run_ddl_statement("DROP USER chuck");
    }
  };

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

  EXPECT_NO_THROW(run_ddl_statement("CREATE USER chuck (password='password');"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE DATABASE nydb (owner='chuck');"));
  EXPECT_NO_THROW(run_ddl_statement("ALTER USER chuck (default_db='nydb')"));

  // Check ability to login
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));

  auto dt = ExecutorDeviceType::CPU;

  Catalog_Namespace::UserMetadata user_meta2;
  username_out = "chuck"s;
  database_out = "nydb"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta2, false));

  auto chuck_qr = get_qr_for_user("nydb"s, user_meta2);
  EXPECT_NO_THROW(chuck_qr->runDDLStatement("create table chaos ( x integer );"));
  EXPECT_NO_THROW(chuck_qr->runSQL("insert into chaos values ( 1234 );", dt));

  // Rename should be fine
  EXPECT_NO_THROW(run_ddl_statement("ALTER USER chuck RENAME TO cryingchuck;"););

  rename_successful = true;

  // After the rename, can we query with the old session?
  EXPECT_THROW(chuck_qr->runSQL("select x from chaos limit 1;", dt), std::runtime_error);

  Catalog_Namespace::UserMetadata user_meta3;
  username_out = "cryingchuck"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta3, false));

  auto cryingchuck_qr = get_qr_for_user("nydb"s, user_meta3);
  // After the rename, can we query with the new session?
  auto result = cryingchuck_qr->runSQL("select x from chaos limit 1;", dt);
  ASSERT_EQ(result->rowCount(), size_t(1));
  const auto crt_row = result->getNextRow(true, true);
  ASSERT_EQ(crt_row.size(), size_t(1));
  ASSERT_EQ(1234, v<int64_t>(crt_row[0]));
}

TEST(SysCatalog, RenameUser_ReloginWithOldName) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  auto username = "chuck"s;
  auto database_name = "nydb"s;
  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    run_ddl_statement("DROP DATABASE nydb;");
    if (rename_successful) {
      run_ddl_statement("DROP USER cryingchuck;");
    } else {
      run_ddl_statement("DROP USER chuck");
    }
  };

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

  EXPECT_NO_THROW(run_ddl_statement("CREATE USER chuck (password='password');"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE DATABASE nydb (owner='chuck');"));
  EXPECT_NO_THROW(run_ddl_statement("ALTER USER chuck (default_db='nydb')"));

  // Check ability to login
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));

  Catalog_Namespace::UserMetadata user_meta2;
  username_out = "chuck"s;
  database_out = "nydb"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta2, false));

  auto chuck_session = std::make_unique<Catalog_Namespace::SessionInfo>(
      sys_cat.getCatalog("nydb"s), user_meta2, ExecutorDeviceType::CPU, "");
  auto chuck_qr = std::make_unique<QR>(std::move(chuck_session));
  EXPECT_NO_THROW(chuck_qr->runDDLStatement("create table chaos ( x integer );"));
  EXPECT_NO_THROW(
      chuck_qr->runSQL("insert into chaos values ( 1234 );", ExecutorDeviceType::CPU));

  // Rename should be fine
  EXPECT_NO_THROW(run_ddl_statement("ALTER USER chuck RENAME TO cryingchuck;"););

  rename_successful = true;

  Catalog_Namespace::UserMetadata user_meta3;
  EXPECT_THROW(sys_cat.login(database_out, username_out, "password"s, user_meta3, false),
               std::runtime_error);
}

TEST(SysCatalog, RenameUser_CheckPrivilegeTransfer) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  auto rename_successful = false;

  ScopeGuard s = [&rename_successful] {
    run_ddl_statement("DROP DATABASE Ferengi;");
    run_ddl_statement("DROP USER rom;");

    if (rename_successful) {
      run_ddl_statement("DROP USER renamed_quark;");
    } else {
      run_ddl_statement("DROP USER quark;");
    }
  };

  EXPECT_NO_THROW(
      run_ddl_statement("CREATE USER quark (password='password',is_super='false');"));
  EXPECT_NO_THROW(
      run_ddl_statement("CREATE USER rom (password='password',is_super='false');"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE DATABASE Ferengi (owner='rom');"));

  auto database_out = "Ferengi"s;
  auto username_out = "rom"s;
  Catalog_Namespace::UserMetadata user_meta;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));

  auto dt = ExecutorDeviceType::CPU;

  // Log in as rom, create the database tables
  Catalog_Namespace::UserMetadata user_meta2;
  username_out = "rom"s;
  database_out = "Ferengi"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta2, false));

  auto rom_session = std::make_unique<Catalog_Namespace::SessionInfo>(
      sys_cat.getCatalog("Ferengi"s), user_meta2, ExecutorDeviceType::CPU, "");
  auto rom_qr = std::make_unique<QR>(std::move(rom_session));

  EXPECT_NO_THROW(
      rom_qr->runDDLStatement("create table bank_account ( latinum integer );"));
  EXPECT_NO_THROW(
      rom_qr->runDDLStatement("create view riches as select * from bank_account;"));
  EXPECT_NO_THROW(rom_qr->runSQL("insert into bank_account values (1234);", dt));

  EXPECT_NO_THROW(rom_qr->runDDLStatement("grant access on database Ferengi to quark;"));
  EXPECT_NO_THROW(
      rom_qr->runDDLStatement("grant select on table bank_account to quark;"));
  EXPECT_NO_THROW(rom_qr->runDDLStatement("grant select on view riches to quark;"));

  run_ddl_statement("ALTER USER quark RENAME TO renamed_quark;");

  rename_successful = true;

  Catalog_Namespace::UserMetadata user_meta3;
  username_out = "renamed_quark"s;
  database_out = "Ferengi"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta3, false));

  auto renamed_quark_session = std::make_unique<Catalog_Namespace::SessionInfo>(
      sys_cat.getCatalog("Ferengi"s), user_meta3, ExecutorDeviceType::CPU, "");
  auto renamed_quark_qr = std::make_unique<QR>(std::move(renamed_quark_session));

  EXPECT_NO_THROW(renamed_quark_qr->runSQL("select * from bank_account;", dt));
  EXPECT_NO_THROW(renamed_quark_qr->runSQL("select * from riches;", dt));
}

TEST(SysCatalog, RenameUser_SuperUserRenameCheck) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;

  ScopeGuard s = [] {
    run_ddl_statement("DROP DATABASE Ferengi;");
    run_ddl_statement("DROP USER rom;");
    run_ddl_statement("DROP USER quark;");
  };

  run_ddl_statement("CREATE USER quark (password='password',is_super='false');");
  run_ddl_statement("CREATE USER rom (password='password',is_super='false');");
  run_ddl_statement("CREATE DATABASE Ferengi (owner='rom');");

  auto database_out = "Ferengi"s;
  auto username_out = "rom"s;
  Catalog_Namespace::UserMetadata user_meta;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));

  // Log in as rom, create the database tables
  Catalog_Namespace::UserMetadata user_meta2;
  username_out = "rom"s;
  database_out = "Ferengi"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta2, false));

  auto rom_session = std::make_unique<Catalog_Namespace::SessionInfo>(
      sys_cat.getCatalog("Ferengi"s), user_meta2, ExecutorDeviceType::CPU, "");
  auto rom_qr = std::make_unique<QR>(std::move(rom_session));
  EXPECT_THROW(rom_qr->runDDLStatement("ALTER USER quark RENAME TO renamed_quark;"),
               std::runtime_error);
}

TEST(SysCatalog, RenameDatabase_Basic) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  auto username = "magicwand"s;
  auto database_name = "gdpgrowth"s;

  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    if (!rename_successful) {
      run_ddl_statement("DROP DATABASE gdpgrowth;");
    } else {
      run_ddl_statement("DROP DATABASE moregdpgrowth;");
    }
    run_ddl_statement("DROP USER magicwand;");
  };

  run_ddl_statement("CREATE DATABASE gdpgrowth;");
  run_ddl_statement(
      "CREATE USER magicwand (password='threepercent', default_db='gdpgrowth');");

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "threepercent"s, user_meta, false));
  EXPECT_EQ(database_name, database_out);

  rename_successful = true;

  run_ddl_statement("ALTER DATABASE gdpgrowth RENAME TO moregdpgrowth;");

  username_out = username;
  database_out = database_name;

  EXPECT_THROW(
      sys_cat.login(database_out, username_out, "threepercent"s, user_meta, false),
      std::runtime_error);

  username_out = username;
  database_out = "moregdpgrowth";

  // Successfully login
  EXPECT_NO_THROW(
      sys_cat.login(database_out, username_out, "threepercent"s, user_meta, false));
}

TEST(SysCatalog, RenameDatabase_WrongUser) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  auto username = "reader"s;
  auto database_name = "fnews"s;

  ScopeGuard scope_gard = [] {
    run_ddl_statement("DROP DATABASE qworg;");
    run_ddl_statement("DROP DATABASE fnews;");

    run_ddl_statement("DROP USER reader;");
    run_ddl_statement("DROP USER jkyle;");
  };

  run_ddl_statement("CREATE USER reader (password='rabbit');");
  run_ddl_statement("CREATE USER jkyle (password='password');");

  run_ddl_statement("CREATE DATABASE fnews (owner='reader');");
  run_ddl_statement("CREATE DATABASE qworg (owner='jkyle');");

  run_ddl_statement("ALTER USER reader (default_db='fnews');");
  run_ddl_statement("ALTER USER jkyle (default_db='qworg');");

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

  ASSERT_NO_THROW(sys_cat.login(database_out, username_out, "rabbit"s, user_meta, false));
  EXPECT_EQ(database_name, database_out);

  Catalog_Namespace::UserMetadata user_meta2;
  username_out = "jkyle"s;
  database_out = "qworg"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta2, false));

  // Should not be permissable
  auto alternate_qr = get_qr_for_user("qworg"s, user_meta2);
  EXPECT_THROW(alternate_qr->runDDLStatement("ALTER DATABASE fnews RENAME TO cnn;"),
               std::runtime_error);
}

TEST(SysCatalog, RenameDatabase_SuperUser) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  auto username = "maurypovich"s;
  auto database_name = "paternitydb"s;

  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    run_ddl_statement("DROP DATABASE trouble;");
    if (rename_successful) {
      run_ddl_statement("DROP DATABASE nachovater;");
    } else {
      run_ddl_statement("DROP DATABASE paternitydb;");
    }
    run_ddl_statement("DROP USER maurypovich;");
    run_ddl_statement("DROP USER thefather;");
  };

  run_ddl_statement("CREATE USER maurypovich (password='password');");
  run_ddl_statement("CREATE USER thefather (password='password',is_super='true');");

  run_ddl_statement("CREATE DATABASE paternitydb (owner='maurypovich');");
  run_ddl_statement("CREATE DATABASE trouble (owner='thefather');");

  run_ddl_statement("ALTER USER maurypovich (default_db='paternitydb');");
  run_ddl_statement("ALTER USER thefather (default_db='trouble');");

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));
  EXPECT_EQ(database_name, database_out);

  Catalog_Namespace::UserMetadata user_meta2;
  username_out = "thefather"s;
  database_out = "trouble"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta2, false));

  auto alternate_qr = get_qr_for_user("trouble"s, user_meta2);
  EXPECT_NO_THROW(
      alternate_qr->runDDLStatement("ALTER DATABASE paternitydb RENAME TO nachovater;"));

  rename_successful = true;
}

TEST(SysCatalog, RenameDatabase_ExistingDB) {
  using namespace std::string_literals;
  auto username = "rickgrimes"s;
  auto database_name = "zombies"s;

  ScopeGuard scope_guard = [] {
    run_ddl_statement("DROP DATABASE zombies;");
    run_ddl_statement("DROP DATABASE vampires;");
    run_ddl_statement("DROP USER rickgrimes;");
  };

  run_ddl_statement("CREATE DATABASE zombies;");
  run_ddl_statement("CREATE DATABASE vampires;");
  run_ddl_statement(
      "CREATE USER rickgrimes (password='password', default_db='zombies');");

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));
  EXPECT_EQ(database_name, database_out);

  EXPECT_THROW(run_ddl_statement("ALTER DATABASE zombies RENAME TO vampires;"),
               std::runtime_error);
}

TEST(SysCatalog, RenameDatabase_FailedCopy) {
  using namespace std::string_literals;
  auto trash_file_path =
      sys_cat.getCatalogBasePath() + "/" + shared::kCatalogDirectoryName + "/trash";

  ScopeGuard s = [&trash_file_path] {
    boost::filesystem::remove(trash_file_path);
    run_ddl_statement("DROP DATABASE hatchets;");
    run_ddl_statement("DROP USER bury;");
  };

  std::ofstream trash_file(trash_file_path);
  trash_file << "trash!";
  trash_file.close();

  auto username = "bury"s;
  auto database_name = "hatchets"s;

  run_ddl_statement("CREATE DATABASE hatchets;");
  run_ddl_statement("CREATE USER bury (password='password', default_db='hatchets');");

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));
  EXPECT_EQ(database_name, database_out);

  // Check that the file inteferes with the copy operation
  EXPECT_THROW(run_ddl_statement("ALTER DATABASE hatchets RENAME TO trash;"),
               std::runtime_error);

  // Now, check to see if we can log back into the original database
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));
}

TEST(SysCatalog, RenameDatabase_PrivsTest) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  auto rename_successful = false;

  ScopeGuard s = [&rename_successful] {
    if (rename_successful) {
      run_ddl_statement("DROP DATABASE grandnagus;");
    } else {
      run_ddl_statement("DROP DATABASE Ferengi;");
    }
    run_ddl_statement("DROP USER quark;");
    run_ddl_statement("DROP USER rom;");
  };

  run_ddl_statement("CREATE USER quark (password='password',is_super='false');");
  run_ddl_statement("CREATE USER rom (password='password',is_super='false');");
  run_ddl_statement("CREATE DATABASE Ferengi (owner='rom');");

  auto database_out = "Ferengi"s;
  auto username_out = "rom"s;
  Catalog_Namespace::UserMetadata user_meta;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta, false));

  auto dt = ExecutorDeviceType::CPU;

  // Log in as rom, create the database tables
  Catalog_Namespace::UserMetadata user_meta2;
  username_out = "rom"s;
  database_out = "Ferengi"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta2, false));

  auto rom_qr = get_qr_for_user("Ferengi"s, user_meta2);
  EXPECT_NO_THROW(
      rom_qr->runDDLStatement("create table bank_account ( latinum integer );"));
  EXPECT_NO_THROW(
      rom_qr->runDDLStatement("create view riches as select * from bank_account;"));
  EXPECT_NO_THROW(rom_qr->runSQL("insert into bank_account values (1234);", dt));

  EXPECT_NO_THROW(rom_qr->runDDLStatement("grant access on database Ferengi to quark;"));
  EXPECT_NO_THROW(
      rom_qr->runDDLStatement("grant select on table bank_account to quark;"));
  EXPECT_NO_THROW(rom_qr->runDDLStatement("grant select on view riches to quark;"));

  EXPECT_NO_THROW(
      rom_qr->runDDLStatement("ALTER DATABASE Ferengi RENAME TO grandnagus;"));

  // Clear session (similar to what is done in DBHandler after a database rename).
  rom_qr->clearSessionId();

  rename_successful = true;

  Catalog_Namespace::UserMetadata user_meta3;
  username_out = "quark"s;
  database_out = "grandnagus"s;
  ASSERT_NO_THROW(
      sys_cat.login(database_out, username_out, "password"s, user_meta3, false));

  auto quark_qr = get_qr_for_user("grandnagus"s, user_meta3);

  EXPECT_NO_THROW(quark_qr->runSQL("select * from bank_account;", dt));
  EXPECT_NO_THROW(quark_qr->runSQL("select * from riches;", dt));
}

TEST(SysCatalog, DropDatabase_ByOwner) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  const std::string username = "theowner";
  const std::string dbname = "thedb";

  ScopeGuard scope_guard = [&] {
    run_ddl_statement("DROP DATABASE IF EXISTS " + dbname + ";");
    run_ddl_statement("DROP USER IF EXISTS " + username + ";");
  };

  run_ddl_statement("CREATE USER " + username + " (password='password');");
  run_ddl_statement("CREATE DATABASE " + dbname + " (owner='" + username + "');");
  run_ddl_statement("ALTER USER " + username + " (default_db='" + dbname + "');");

  Catalog_Namespace::UserMetadata user_meta;
  std::string username_out{username};
  std::string dbname_out{dbname};

  ASSERT_NO_THROW(sys_cat.login(dbname_out, username_out, "password", user_meta, false));
  EXPECT_EQ(dbname, dbname_out);

  auto qr = get_qr_for_user(dbname, user_meta);
  EXPECT_NO_THROW(qr->runDDLStatement("DROP DATABASE " + dbname + ";"));
}

TEST(SysCatalog, DropDatabase_ByNonOwner) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  const std::string username = "theowner";
  const std::string dbname = "thedb";

  ScopeGuard scope_guard = [&] {
    run_ddl_statement("DROP DATABASE IF EXISTS " + dbname + ";");
    run_ddl_statement("DROP USER IF EXISTS " + username + ";");
    run_ddl_statement("DROP USER IF EXISTS not" + username + ";");
  };

  run_ddl_statement("CREATE USER " + username + " (password='password');");
  run_ddl_statement("CREATE USER not" + username + " (password='password');");
  run_ddl_statement("CREATE DATABASE " + dbname + " (owner='" + username + "');");
  run_ddl_statement("ALTER USER " + username + " (default_db='" + dbname + "');");
  run_ddl_statement("ALTER USER not" + username + " (default_db='" + dbname + "');");

  Catalog_Namespace::UserMetadata user_meta;
  std::string username_out{"not" + username};
  std::string dbname_out{dbname};

  ASSERT_NO_THROW(sys_cat.login(dbname_out, username_out, "password", user_meta, false));
  EXPECT_EQ(dbname, dbname_out);

  auto qr = get_qr_for_user(dbname, user_meta);
  EXPECT_THROW(qr->runDDLStatement("DROP DATABASE " + dbname + ";"), std::runtime_error);
}

TEST(SysCatalog, DropDatabase_BySuperUser) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  const std::string username = "theowner";
  const std::string dbname = "thedb";

  ScopeGuard scope_guard = [&] {
    run_ddl_statement("DROP DATABASE IF EXISTS " + dbname + ";");
    run_ddl_statement("DROP USER IF EXISTS " + username + ";");
    run_ddl_statement("DROP USER IF EXISTS not" + username + ";");
  };

  run_ddl_statement("CREATE USER " + username + " (password='password');");
  run_ddl_statement("CREATE USER not" + username +
                    " (password='password',is_super='true');");
  run_ddl_statement("CREATE DATABASE " + dbname + " (owner='" + username + "');");
  run_ddl_statement("ALTER USER " + username + " (default_db='" + dbname + "');");
  run_ddl_statement("ALTER USER not" + username + " (default_db='" + dbname + "');");

  Catalog_Namespace::UserMetadata user_meta;
  std::string username_out{"not" + username};
  std::string dbname_out{dbname};

  ASSERT_NO_THROW(sys_cat.login(dbname_out, username_out, "password", user_meta, false));
  EXPECT_EQ(dbname, dbname_out);

  auto alternate_qr = get_qr_for_user(dbname, user_meta);
  EXPECT_NO_THROW(alternate_qr->runDDLStatement("DROP DATABASE " + dbname + ";"));
}

TEST(SysCatalog, GetDatabaseList) {
  static const std::string username{"test_user"};
  static const std::string username2{username + "2"};
  static const std::string dbname{"test_db"};
  static const std::string dbname2{dbname + "2"};
  static const std::string dbname3{dbname + "3"};
  static const std::string dbname4{dbname + "4"};

  // cleanup
  struct CleanupGuard {
    ~CleanupGuard() {
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbname4 + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbname3 + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbname2 + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbname + ";");
      run_ddl_statement("DROP USER IF EXISTS " + username2 + ";");
      run_ddl_statement("DROP USER IF EXISTS " + username + ";");
    }
  } cleanupGuard;

  // setup
  run_ddl_statement("CREATE USER " + username + " (password = 'password');");
  run_ddl_statement("CREATE USER " + username2 + " (password = 'password');");
  run_ddl_statement("CREATE DATABASE " + dbname + "(owner='" + username + "');");
  run_ddl_statement("CREATE DATABASE " + dbname2 + "(owner='" + username2 + "');");
  run_ddl_statement("CREATE DATABASE " + dbname3 + "(owner='" + username2 + "');");
  run_ddl_statement("CREATE DATABASE " + dbname4 + "(owner='" + username + "');");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + dbname + " TO " + username2 + ";");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + dbname3 + " TO " + username + ";");

  Catalog_Namespace::UserMetadata user_meta, user_meta2;
  CHECK(sys_cat.getMetadataForUser(username, user_meta));
  CHECK(sys_cat.getMetadataForUser(username2, user_meta2));

  // test database list for arbitrary user #1
  auto dblist = sys_cat.getDatabaseListForUser(user_meta);
  EXPECT_EQ(dblist.front().dbName, dbname);
  EXPECT_EQ(dblist.front().dbOwnerName, username);

  dblist.pop_front();
  EXPECT_EQ(dblist.front().dbName, dbname3);
  EXPECT_EQ(dblist.front().dbOwnerName, username2);

  dblist.pop_front();
  EXPECT_EQ(dblist.front().dbName, dbname4);
  EXPECT_EQ(dblist.front().dbOwnerName, username);

  // test database list for arbitrary user #2
  dblist = sys_cat.getDatabaseListForUser(user_meta2);
  EXPECT_EQ(dblist.front().dbName, dbname);
  EXPECT_EQ(dblist.front().dbOwnerName, username);

  dblist.pop_front();
  EXPECT_EQ(dblist.front().dbName, dbname2);
  EXPECT_EQ(dblist.front().dbOwnerName, username2);

  dblist.pop_front();
  EXPECT_EQ(dblist.front().dbName, dbname3);
  EXPECT_EQ(dblist.front().dbOwnerName, username2);
}

TEST(SysCatalog, LoginWithDefaultDatabase) {
  static const std::string username{"test_user"};
  static const std::string dbname{"test_db"};
  static const std::string dbnamex{dbname + "x"};

  // cleanup
  struct CleanupGuard {
    ~CleanupGuard() {
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbname + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbnamex + ";");
      run_ddl_statement("DROP USER IF EXISTS " + username + ";");
    }
  } cleanupGuard;

  // setup
  run_ddl_statement("CREATE DATABASE " + dbname + ";");
  run_ddl_statement("CREATE DATABASE " + dbnamex + ";");
  run_ddl_statement("CREATE USER " + username +
                    " (password = 'password', default_db = '" + dbnamex + "');");
  Catalog_Namespace::UserMetadata user_meta;

  // test the user's default database
  std::string username2{username};
  std::string dbname2{dbname};
  ASSERT_NO_THROW(sys_cat.login(dbname2, username2, "password", user_meta, false));
  EXPECT_EQ(dbname2, dbname);  // correctly ignored user's default of dbnamex

  username2 = username;
  dbname2.clear();
  ASSERT_NO_THROW(sys_cat.login(dbname2, username2, "password", user_meta, false));
  EXPECT_EQ(dbname2, dbnamex);  // correctly used user's default of dbnamex

  // change the user's default database
  ASSERT_NO_THROW(
      run_ddl_statement("ALTER USER " + username + " (default_db = '" + dbname + "');"));

  // test the user's default database
  username2 = username;
  dbname2.clear();
  ASSERT_NO_THROW(sys_cat.login(dbname2, username2, "password", user_meta, false));
  EXPECT_EQ(dbname2, dbname);  // correctly used user's default of dbname

  // remove the user's default database
  ASSERT_NO_THROW(run_ddl_statement("ALTER USER " + username + " (default_db = NULL);"));

  // test the user's default database
  username2 = username;
  dbname2.clear();
  ASSERT_NO_THROW(sys_cat.login(dbname2, username2, "password", user_meta, false));
  EXPECT_EQ(dbname2,
            shared::kDefaultDbName);  // correctly fell back to system default
                                      // database
}

TEST(SysCatalog, SwitchDatabase) {
  static const std::string username{"test_user"};
  static std::string dbname{"test_db"};
  static std::string dbname2{dbname + "2"};
  static std::string dbname3{dbname + "3"};

  // cleanup
  sql("DROP DATABASE IF EXISTS " + dbname + ";");
  sql("DROP DATABASE IF EXISTS " + dbname2 + ";");
  sql("DROP DATABASE IF EXISTS " + dbname3 + ";");
  sql("DROP USER IF EXISTS " + username + ";");

  // setup
  sql("CREATE USER " + username + " (password = 'password');");
  sql("CREATE DATABASE " + dbname + "(owner='" + username + "');");
  sql("CREATE DATABASE " + dbname2 + "(owner='" + username + "');");
  sql("CREATE DATABASE " + dbname3 + "(owner='" + username + "');");
  sql("REVOKE ACCESS ON DATABASE " + dbname3 + " FROM " + username + ";");

  // test some attempts to switch database
  ASSERT_NO_THROW(sys_cat.switchDatabase(dbname, username));
  ASSERT_NO_THROW(sys_cat.switchDatabase(dbname, username));
  ASSERT_NO_THROW(sys_cat.switchDatabase(dbname2, username));
  ASSERT_THROW(sys_cat.switchDatabase(dbname3, username), std::runtime_error);

  // // distributed test
  // // NOTE(sy): disabling for now due to consistency errors
  // if (DQR* dqr = dynamic_cast<DQR*>(QR::get()); g_aggregator && dqr) {
  //   static const std::string tname{"swdb_test_table"};
  //   LeafAggregator* agg = dqr->getLeafAggregator();
  //   agg->switch_database(dqr->getSession()->get_session_id(), dbname);
  //   sql("CREATE TABLE " + tname + "(i INTEGER);");
  //   ASSERT_NO_THROW(sql("SELECT i FROM " + tname + ";"));
  //   agg->switch_database(dqr->getSession()->get_session_id(), dbname2);
  //   ASSERT_ANY_THROW(agg->leafCatalogConsistencyCheck(*dqr->getSession()));
  //   agg->switch_database(dqr->getSession()->get_session_id(), dbname);
  //   ASSERT_NO_THROW(sql("DROP TABLE " + tname + ";"));
  //   agg->switch_database(dqr->getSession()->get_session_id(), shared::kDefaultDbName);
  // }

  // cleanup
  sql("DROP DATABASE IF EXISTS " + dbname + ";");
  sql("DROP DATABASE IF EXISTS " + dbname2 + ";");
  sql("DROP DATABASE IF EXISTS " + dbname3 + ";");
  sql("DROP USER IF EXISTS " + username + ";");
}

namespace {

void compare_user_lists(const std::vector<std::string>& expected,
                        const std::list<Catalog_Namespace::UserMetadata>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  size_t i = 0;
  for (const auto& user : actual) {
    ASSERT_EQ(expected[i++], user.userName);
  }
}

}  // namespace

TEST(SysCatalog, AllUserMetaTest) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  Users users_;

  static const auto champions = "champions"s;
  static const auto europa = "europa"s;

  struct ExpectedUserLists {
    const std::vector<std::string> super_default = {"admin",
                                                    "Chelsea",
                                                    "Arsenal",
                                                    "Juventus",
                                                    "Bayern"};
    const std::vector<std::string> user_default = {"Arsenal", "Bayern"};
    const std::vector<std::string> user_champions = {"Juventus", "Bayern"};
    const std::vector<std::string> user_europa = {"Arsenal", "Juventus"};
  } expected;

  // cleanup
  struct CleanupGuard {
    ~CleanupGuard() {
      run_ddl_statement("DROP DATABASE IF EXISTS " + champions + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + europa + ";");
    }
  } cleanupGuard;

  run_ddl_statement("DROP DATABASE IF EXISTS " + champions + ";");
  run_ddl_statement("DROP DATABASE IF EXISTS " + europa + ";");
  run_ddl_statement("CREATE DATABASE " + champions + ";");
  run_ddl_statement("CREATE DATABASE " + europa + ";");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + champions + " TO Bayern;");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + champions + " TO Juventus;");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName +
                    " TO Arsenal;");
  run_ddl_statement("GRANT CREATE ON DATABASE " + champions + " TO Juventus;");
  run_ddl_statement("GRANT SELECT ON DATABASE " + europa + " TO Arsenal;");
  run_ddl_statement("GRANT CREATE ON DATABASE " + shared::kDefaultDbName + " TO Bayern;");
  run_ddl_statement("GRANT SELECT ON DATABASE " + europa + " TO Juventus;");

  Catalog_Namespace::UserMetadata user_meta;
  auto db_default(shared::kDefaultDbName);
  auto db_champions(champions);
  auto db_europa(europa);
  auto user_chelsea("Chelsea"s);
  auto user_arsenal("Arsenal"s);
  auto user_bayern("Bayern"s);

  // Super User
  ASSERT_NO_THROW(sys_cat.login(db_default, user_chelsea, "password"s, user_meta, false));
  const auto suser_list = sys_cat.getAllUserMetadata();
  compare_user_lists(expected.super_default, suser_list);

  ASSERT_NO_THROW(
      sys_cat.login(db_champions, user_chelsea, "password"s, user_meta, false));
  const auto suser_list1 = sys_cat.getAllUserMetadata();
  compare_user_lists(expected.super_default, suser_list1);

  ASSERT_NO_THROW(sys_cat.login(db_europa, user_chelsea, "password"s, user_meta, false));
  const auto suser_list2 = sys_cat.getAllUserMetadata();
  compare_user_lists(expected.super_default, suser_list2);

  // Normal User
  Catalog_Namespace::DBMetadata db;
  ASSERT_NO_THROW(sys_cat.getMetadataForDB(db_default, db));
  ASSERT_NO_THROW(sys_cat.login(db_default, user_arsenal, "password"s, user_meta, false));
  const auto nuser_list = sys_cat.getAllUserMetadata(db.dbId);
  compare_user_lists(expected.user_default, nuser_list);

  ASSERT_NO_THROW(sys_cat.getMetadataForDB(db_champions, db));
  ASSERT_NO_THROW(
      sys_cat.login(db_champions, user_bayern, "password"s, user_meta, false));
  const auto nuser_list1 = sys_cat.getAllUserMetadata(db.dbId);
  compare_user_lists(expected.user_champions, nuser_list1);

  ASSERT_NO_THROW(sys_cat.getMetadataForDB(db_europa, db));
  ASSERT_NO_THROW(sys_cat.login(db_europa, user_arsenal, "password"s, user_meta, false));
  const auto nuser_list2 = sys_cat.getAllUserMetadata(db.dbId);
  compare_user_lists(expected.user_europa, nuser_list2);
}

TEST(SysCatalog, RecursiveRolesUserMetaData) {
  if (g_aggregator) {
    LOG(ERROR) << "Test not supported in distributed mode.";
    return;
  }

  using namespace std::string_literals;
  Users users_;
  Roles roles_;

  static const auto champions = "champions"s;
  static const auto europa = "europa"s;
  static const auto london = "london"s;
  static const auto north_london = "north_london"s;
  static const auto munich = "munich"s;
  static const auto turin = "turin"s;

  struct CleanupGuard {
    ~CleanupGuard() {
      run_ddl_statement("DROP ROLE IF EXISTS " + london + ";");
      run_ddl_statement("DROP ROLE IF EXISTS " + north_london + ";");
      run_ddl_statement("DROP ROLE IF EXISTS " + munich + ";");
      run_ddl_statement("DROP ROLE IF EXISTS " + turin + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + champions + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + europa + ";");
    }
  } cleanupGuard;

  struct ExpectedUserLists {
    const std::vector<std::string> user_default = {"Arsenal", "Bayern"};
    const std::vector<std::string> user_champions = {"Juventus", "Bayern"};
    const std::vector<std::string> user_europa = {"Arsenal", "Juventus"};
  } expected;

  run_ddl_statement("CREATE ROLE " + london + ";");
  run_ddl_statement("CREATE ROLE " + north_london + ";");
  run_ddl_statement("CREATE ROLE " + munich + ";");
  run_ddl_statement("CREATE ROLE " + turin + ";");
  run_ddl_statement("DROP DATABASE IF EXISTS " + champions + ";");
  run_ddl_statement("DROP DATABASE IF EXISTS " + europa + ";");
  run_ddl_statement("CREATE DATABASE " + champions + ";");
  run_ddl_statement("CREATE DATABASE " + europa + ";");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + champions + " TO Sudens;");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + champions + " TO OldLady;");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName +
                    " TO Gunners;");
  run_ddl_statement("GRANT CREATE ON DATABASE " + champions + " TO OldLady;");
  run_ddl_statement("GRANT SELECT ON DATABASE " + europa + " TO Gunners;");
  run_ddl_statement("GRANT CREATE ON DATABASE " + shared::kDefaultDbName + " TO Sudens;");
  run_ddl_statement("GRANT SELECT ON DATABASE " + europa + " TO OldLady;");

  Catalog_Namespace::UserMetadata user_meta;
  auto db_default(shared::kDefaultDbName);
  auto db_champions(champions);
  auto db_europa(europa);
  auto user_chelsea("Chelsea"s);
  auto user_arsenal("Arsenal"s);
  auto user_bayern("Bayern"s);
  auto user_juventus("Juventus"s);

  run_ddl_statement("GRANT Gunners to " + london + ";");
  run_ddl_statement("GRANT " + london + " to " + north_london + ";");
  run_ddl_statement("GRANT " + north_london + " to " + user_arsenal + ";");
  run_ddl_statement("GRANT Sudens to " + user_bayern + ";");
  run_ddl_statement("GRANT OldLady to " + user_juventus + ";");

  Catalog_Namespace::DBMetadata db;
  ASSERT_NO_THROW(sys_cat.getMetadataForDB(db_default, db));
  ASSERT_NO_THROW(sys_cat.login(db_default, user_arsenal, "password"s, user_meta, false));
  const auto nuser_list = sys_cat.getAllUserMetadata(db.dbId);
  compare_user_lists(expected.user_default, nuser_list);

  ASSERT_NO_THROW(sys_cat.getMetadataForDB(db_champions, db));
  ASSERT_NO_THROW(
      sys_cat.login(db_champions, user_bayern, "password"s, user_meta, false));
  const auto nuser_list1 = sys_cat.getAllUserMetadata(db.dbId);
  compare_user_lists(expected.user_champions, nuser_list1);

  ASSERT_NO_THROW(sys_cat.getMetadataForDB(db_europa, db));
  ASSERT_NO_THROW(sys_cat.login(db_europa, user_arsenal, "password"s, user_meta, false));
  const auto nuser_list2 = sys_cat.getAllUserMetadata(db.dbId);
  compare_user_lists(expected.user_europa, nuser_list2);
}

TEST(Login, Deactivation) {
  // SysCatalog::login doesn't accept constants
  std::string database = shared::kDefaultDbName;
  std::string active_user = "active_user";
  std::string deactivated_user = "deactivated_user";

  run_ddl_statement(
      "CREATE USER active_user(password = 'password', is_super = 'false', "
      "can_login = 'true');");
  run_ddl_statement(
      "CREATE USER deactivated_user(password = 'password', is_super = 'false', "
      "can_login = 'false');");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + database +
                    " TO active_user, deactivated_user");

  Catalog_Namespace::UserMetadata user_meta;
  ASSERT_NO_THROW(sys_cat.login(database, active_user, "password", user_meta, true));
  ASSERT_THROW(sys_cat.login(database, deactivated_user, "password", user_meta, true),
               std::runtime_error);

  run_ddl_statement("ALTER USER active_user(can_login='false');");
  run_ddl_statement("ALTER USER deactivated_user(can_login='true');");
  ASSERT_NO_THROW(sys_cat.login(database, deactivated_user, "password", user_meta, true));
  ASSERT_THROW(sys_cat.login(database, active_user, "password", user_meta, true),
               std::runtime_error);

  run_ddl_statement("DROP USER active_user;");
  run_ddl_statement("DROP USER deactivated_user;");
}

class GetDbObjectsForGranteeTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("CREATE USER test_user (password = 'test_pass');");
  }

  void TearDown() override {
    sql("DROP USER test_user;");
    DBHandlerTestFixture::TearDown();
  }

  void allOnDatabase(std::string privilege) {
    sql("GRANT " + privilege + " ON DATABASE " + shared::kDefaultDbName +
        " TO test_user;");

    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    std::vector<TDBObject> db_objects{};
    db_handler->get_db_objects_for_grantee(db_objects, session_id, "test_user");

    std::unordered_set<TDBObjectType::type> privilege_types{};
    for (const auto& db_object : db_objects) {
      ASSERT_EQ(shared::kDefaultDbName, db_object.objectName);
      ASSERT_EQ(TDBObjectType::DatabaseDBObjectType, db_object.objectType);
      ASSERT_EQ("test_user", db_object.grantee);

      if (db_object.privilegeObjectType == TDBObjectType::DatabaseDBObjectType) {
        // The first two items represent CREATE and DROP DATABASE privileges, which are
        // not granted
        std::vector<bool> expected_privileges{false, false, true, true};
        ASSERT_EQ(expected_privileges, db_object.privs);
      } else {
        ASSERT_TRUE(std::all_of(db_object.privs.begin(),
                                db_object.privs.end(),
                                [](bool has_privilege) { return has_privilege; }));
      }
      privilege_types.emplace(db_object.privilegeObjectType);
    }

    ASSERT_FALSE(privilege_types.find(TDBObjectType::DatabaseDBObjectType) ==
                 privilege_types.end());
    ASSERT_FALSE(privilege_types.find(TDBObjectType::TableDBObjectType) ==
                 privilege_types.end());
    ASSERT_FALSE(privilege_types.find(TDBObjectType::DashboardDBObjectType) ==
                 privilege_types.end());
    ASSERT_FALSE(privilege_types.find(TDBObjectType::ViewDBObjectType) ==
                 privilege_types.end());
  }
};

TEST_F(GetDbObjectsForGranteeTest, UserWithGrantAllOnDatabase) {
  allOnDatabase("ALL");
}

TEST_F(GetDbObjectsForGranteeTest, UserWithGrantAllPrivilegesOnDatabase) {
  allOnDatabase("ALL PRIVILEGES");
}

TEST(DefaultUser, RoleList) {
  auto* grantee = sys_cat.getGrantee(shared::kRootUsername);
  EXPECT_TRUE(grantee);
  EXPECT_TRUE(grantee->getRoles().empty());
}

class TablePermissionsTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    createDBHandler();
    switchToAdmin();
    createTestUser();
  }

  void runSelectQueryAsUser(const std::string& query,
                            const std::string& username,
                            const std::string& password) {
    Catalog_Namespace::UserMetadata user_meta;
    std::string db_name = shared::kDefaultDbName;
    std::string username_local = username;
    ASSERT_NO_THROW(sys_cat.login(db_name, username_local, password, user_meta, false));
    auto user_qr = get_qr_for_user(db_name, user_meta);
    user_qr->runSQL(query, ExecutorDeviceType::CPU);
  }

  void runSelectQueryAsTestUser(const std::string& query) {
    runSelectQueryAsUser(query, "test_user", "test_pass");
  }

  void runSelectQueryAsTestUserAndAssertException(const std::string& query,
                                                  const std::string& exception) {
    executeLambdaAndAssertException([&, this]() { runSelectQueryAsTestUser(query); },
                                    exception);
  }

  static void TearDownTestSuite() { dropTestUser(); }

  void SetUp() override { DBHandlerTestFixture::SetUp(); }

  void TearDown() override {
    tearDownTable();
    DBHandlerTestFixture::TearDown();
  }

  void runQuery(const std::string& query) {
    std::string first_query_term = query.substr(0, query.find(" "));
    if (to_upper(first_query_term) == "SELECT") {
      // SELECT statements require a different code path due to a conflict with
      // QueryRunner
      runSelectQueryAsTestUser(query);
    } else {
      sql(query);
    }
  }

  void runQueryAndAssertException(const std::string& query,
                                  const std::string& exception) {
    std::string first_query_term = query.substr(0, query.find(" "));
    if (to_upper(first_query_term) == "SELECT") {
      // SELECT statements require a different code path due to a conflict with
      // QueryRunner
      runSelectQueryAsTestUserAndAssertException(query, exception);
    } else {
      queryAndAssertException(query, exception);
    }
  }

  void queryAsTestUserWithNoPrivilegeAndAssertException(const std::string& query,
                                                        const std::string& exception) {
    login("test_user", "test_pass");
    runQueryAndAssertException(query, exception);
  }

  void queryAsTestUserWithPrivilege(const std::string& query,
                                    const std::string& privilege) {
    switchToAdmin();
    sql("GRANT " + privilege + " ON TABLE test_table TO test_user;");
    login("test_user", "test_pass");
    runQuery(query);
  }

  void queryAsTestUserWithPrivilegeAndAssertException(const std::string& query,
                                                      const std::string& privilege,
                                                      const std::string& exception) {
    switchToAdmin();
    sql("GRANT " + privilege + " ON TABLE test_table TO test_user;");
    login("test_user", "test_pass");
    runQueryAndAssertException(query, exception);
  }

  void grantThenRevokePrivilegeToTestUser(const std::string& privilege) {
    switchToAdmin();
    sql("GRANT " + privilege + " ON TABLE test_table TO test_user;");
    sql("REVOKE " + privilege + " ON TABLE test_table FROM test_user;");
  }

  static void createTestUser() {
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  }

  void createTestForeignTable() {
    sql("DROP FOREIGN TABLE IF EXISTS test_table;");
    std::string query{
        "CREATE FOREIGN TABLE test_table (i BIGINT) SERVER default_local_delimited WITH "
        "(file_path = '" +
        getDataFilesPath() + "1.csv');"};
    sql(query);
    is_foreign_table_ = true;
  }

  void createTestTable() {
    sql("DROP TABLE IF EXISTS test_table;");
    std::string query{"CREATE TABLE test_table (i BIGINT);"};
    sql(query);
    sql("INSERT INTO test_table VALUES (1);");
    is_foreign_table_ = false;
  }

  void createTestView() {
    sql("DROP VIEW IF EXISTS test_view");
    createTestTable();
    sql("CREATE VIEW test_view AS SELECT * FROM test_table");
  }

  void tearDownTable() {
    loginAdmin();
    if (is_foreign_table_) {
      sql("DROP FOREIGN TABLE IF EXISTS test_table;");
    } else {
      sql("DROP TABLE IF EXISTS test_table;");
    }
    sql("DROP TABLE IF EXISTS renamed_test_table;");
    sql("DROP VIEW IF EXISTS test_view");
  }

  static void dropTestUser() { sql("DROP USER IF EXISTS test_user;"); }

 private:
  bool is_foreign_table_;

  static std::string getDataFilesPath() {
    return boost::filesystem::canonical(g_test_binary_file_path +
                                        "/../../Tests/FsiDataFiles")
               .string() +
           "/";
  }
};

class ForeignTableAndTablePermissionsTest
    : public TablePermissionsTest,
      public testing::WithParamInterface<ddl_utils::TableType> {
 protected:
  void SetUp() override {
    TablePermissionsTest::SetUp();
    if (g_aggregator && GetParam() == ddl_utils::TableType::FOREIGN_TABLE) {
      LOG(INFO) << "Test fixture not supported in distributed mode.";
      GTEST_SKIP();
      return;
    }
    switch (GetParam()) {
      case ddl_utils::TableType::FOREIGN_TABLE:
        createTestForeignTable();
        break;
      case ddl_utils::TableType::TABLE:
        createTestTable();
        break;
      default:
        UNREACHABLE();
    }
  }
};

class ForeignTablePermissionsTest : public TablePermissionsTest {
 protected:
  void SetUp() override {
    TablePermissionsTest::SetUp();
    if (g_aggregator) {
      LOG(INFO) << "Test not supported in distributed mode.";
      GTEST_SKIP();
    }
  }
};

INSTANTIATE_TEST_SUITE_P(ForeignTableAndTablePermissionsTest,
                         ForeignTableAndTablePermissionsTest,
                         ::testing::Values(ddl_utils::TableType::FOREIGN_TABLE,
                                           ddl_utils::TableType::TABLE));

TEST_F(ForeignTablePermissionsTest, ForeignTableGrantRevokeDropPrivilege) {
  std::string privilege{"DROP"};
  std::string query{"DROP FOREIGN TABLE test_table;"};
  std::string no_privilege_exception{
      "Foreign table \"test_table\" will not be dropped. User has no DROP "
      "TABLE privileges."};
  createTestForeignTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(TablePermissionsTest, TableGrantRevokeDropPrivilege) {
  std::string privilege{"DROP"};
  std::string query{"DROP TABLE test_table;"};
  std::string no_privilege_exception{
      "Table test_table will not be dropped. User has no proper privileges."};
  createTestTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_P(ForeignTableAndTablePermissionsTest, GrantRevokeSelectPrivilege) {
  if (g_aggregator) {
    // TODO: select queries as a user currently do not work in distributed
    // mode for regular tables (DistributedQueryRunner::init can not be run
    // more than once.)
    LOG(INFO) << "Test not supported in distributed mode.";
    GTEST_SKIP();
  }
  std::string privilege{"SELECT"};
  std::string query{"SELECT * FROM test_table;"};
  std::string no_privilege_exception{
      "Violation of access privileges: user test_user has no proper privileges for "
      "object test_table"};
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(ForeignTablePermissionsTest, ForeignTableGrantRevokeDeletePrivilege) {
  std::string privilege{"DELETE"};
  std::string query{"DELETE FROM test_table WHERE i = 1;"};
  std::string no_privilege_exception{
      "Violation of access privileges: user test_user has no proper "
      "privileges for "
      "object test_table"};
  std::string query_exception{
      "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not "
      "supported for foreign tables."};
  createTestForeignTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilegeAndAssertException(query, privilege, query_exception);
}

TEST_F(TablePermissionsTest, TableGrantRevokeDeletePrivilege) {
  std::string privilege{"DELETE"};
  std::string query{"DELETE FROM test_table WHERE i = 1;"};
  std::string no_privilege_exception{
      "Violation of access privileges: user test_user has no proper "
      "privileges for "
      "object test_table"};
  createTestTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(ForeignTablePermissionsTest, ForeignTableGrantRevokeInsertPrivilege) {
  std::string privilege{"INSERT"};
  std::string query{"INSERT INTO test_table VALUES (2);"};
  std::string no_privilege_exception{"User has no insert privileges on test_table."};
  std::string query_exception{
      "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not "
      "supported for foreign tables."};
  createTestForeignTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilegeAndAssertException(query, privilege, query_exception);
}

TEST_F(TablePermissionsTest, TableGrantRevokeInsertPrivilege) {
  std::string privilege{"INSERT"};
  std::string query{"INSERT INTO test_table VALUES (2);"};
  std::string no_privilege_exception{"User has no insert privileges on test_table."};
  createTestTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(ForeignTablePermissionsTest, ForeignTableGrantRevokeTruncatePrivilege) {
  std::string privilege{"TRUNCATE"};
  std::string query{"TRUNCATE TABLE test_table;"};
  std::string no_privilege_exception{
      "Table test_table will not be truncated. User test_user has no proper "
      "privileges."};
  std::string query_exception{
      "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not "
      "supported for foreign tables."};
  createTestForeignTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilegeAndAssertException(query, privilege, query_exception);
}

TEST_F(TablePermissionsTest, TableGrantRevokeTruncatePrivilege) {
  std::string privilege{"TRUNCATE"};
  std::string query{"TRUNCATE TABLE test_table;"};
  std::string no_privilege_exception{
      "Table test_table will not be truncated. User test_user has no proper "
      "privileges."};
  createTestTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(ForeignTablePermissionsTest, ForeignTableGrantRevokeUpdatePrivilege) {
  std::string privilege{"UPDATE"};
  std::string query{"UPDATE test_table SET i = 2 WHERE i = 1;"};
  std::string no_privilege_exception{
      "Violation of access privileges: user test_user has no proper "
      "privileges for "
      "object test_table"};
  std::string query_exception{
      "DELETE, INSERT, TRUNCATE, OR UPDATE commands are not "
      "supported for foreign tables."};
  createTestForeignTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilegeAndAssertException(query, privilege, query_exception);
}

TEST_F(TablePermissionsTest, TableGrantRevokeUpdatePrivilege) {
  std::string privilege{"UPDATE"};
  std::string query{"UPDATE test_table SET i = 2 WHERE i = 1;"};
  std::string no_privilege_exception{
      "Violation of access privileges: user test_user has no proper "
      "privileges for "
      "object test_table"};
  createTestTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_P(ForeignTableAndTablePermissionsTest, GrantRevokeShowCreateTablePrivilege) {
  if (g_aggregator && GetParam() == ddl_utils::TableType::FOREIGN_TABLE) {
    LOG(INFO) << "Test not supported in distributed mode.";
    return;
  }
  std::string privilege{"DROP"};
  std::string query{"SHOW CREATE TABLE test_table;"};
  std::string no_privilege_exception{"Table/View test_table does not exist."};
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(TablePermissionsTest, TableGrantRevokeAlterTablePrivilege) {
  std::string privilege{"ALTER"};
  std::string query{"ALTER TABLE test_table RENAME COLUMN i TO j;"};
  std::string no_privilege_exception{
      "Current user does not have the privilege to alter table: test_table"};
  createTestTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(ForeignTablePermissionsTest, TableGrantRevokeAlterForeignTablePrivilege) {
  std::string privilege{"ALTER"};
  std::string query{
      "ALTER FOREIGN TABLE test_table SET (refresh_update_type = 'append');"};
  std::string no_privilege_exception{
      "Current user does not have the privilege to alter foreign table: "
      "test_table"};
  createTestForeignTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  grantThenRevokePrivilegeToTestUser(privilege);
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(TablePermissionsTest, TableRenameTablePrivilege) {
  std::string privilege{"ALTER"};
  std::string query{"RENAME TABLE test_table TO renamed_test_table;"};
  std::string no_privilege_exception{
      "Current user does not have the privilege to alter table: test_table"};
  createTestTable();
  queryAsTestUserWithNoPrivilegeAndAssertException(query, no_privilege_exception);
  queryAsTestUserWithPrivilege(query, privilege);
}

TEST_F(ForeignTablePermissionsTest, ForeignTableAllPrivileges) {
  createTestForeignTable();
  sql("GRANT ALL ON TABLE test_table TO test_user;");
  login("test_user", "test_pass");
  runQuery("SHOW CREATE TABLE test_table;");
  runQuery("SELECT * FROM test_table;");
  runQuery("ALTER FOREIGN TABLE test_table SET (refresh_update_type = 'append');");
  runQuery("DROP FOREIGN TABLE test_table;");
}

TEST_F(TablePermissionsTest, TableAllPrivileges) {
  createTestTable();
  sql("GRANT ALL ON TABLE test_table TO test_user;");
  login("test_user", "test_pass");
  runQuery("SHOW CREATE TABLE test_table;");
  runQuery("UPDATE test_table SET i = 2 WHERE i = 1;");
  runQuery("TRUNCATE TABLE test_table;");
  runQuery("INSERT INTO test_table VALUES (2);");
  runQuery("DELETE FROM test_table WHERE i = 2;");
  if (!g_aggregator) {
    // TODO: select queries as a user currently do not work in distributed
    // mode for regular tables (DistributedQueryRunner::init can not be run
    // more than once.)
    runQuery("SELECT * FROM test_table;");
  }
  runQuery("ALTER TABLE test_table RENAME COLUMN i TO j;");
  runQuery("DROP TABLE test_table;");
}

TEST_F(TablePermissionsTest, ShowCreateView) {
  createTestView();
  queryAsTestUserWithNoPrivilegeAndAssertException(
      "SHOW CREATE TABLE test_view", "Table/View test_view does not exist.");
  switchToAdmin();
  sql("GRANT ALL ON VIEW test_view TO test_user;");
  queryAsTestUserWithNoPrivilegeAndAssertException(
      "SHOW CREATE TABLE test_view", "Not enough privileges to show the view SQL");
  switchToAdmin();
  sql("GRANT ALL ON TABLE test_table TO test_user;");
  login("test_user", "test_pass");
  TQueryResult result;
  EXPECT_NO_THROW(sql(result, "SHOW CREATE TABLE test_view;"));
  EXPECT_EQ("CREATE VIEW test_view AS SELECT * FROM test_table;",
            result.row_set.columns[0].data.str_col[0]);
  switchToAdmin();
  sql("REVOKE ALL ON VIEW test_view FROM test_user;");
  queryAsTestUserWithNoPrivilegeAndAssertException(
      "SHOW CREATE TABLE test_view", "Table/View test_view does not exist.");
}

TEST_F(ForeignTablePermissionsTest, ForeignTableGrantRevokeCreateTablePrivilege) {
  login("test_user", "test_pass");
  executeLambdaAndAssertException([this] { createTestForeignTable(); },
                                  "Foreign table \"test_table\" will not be "
                                  "created. User has no CREATE TABLE privileges.");

  switchToAdmin();
  sql("GRANT CREATE TABLE ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  sql("REVOKE CREATE TABLE ON DATABASE " + shared::kDefaultDbName + " FROM test_user;");
  login("test_user", "test_pass");
  executeLambdaAndAssertException([this] { createTestForeignTable(); },
                                  "Foreign table \"test_table\" will not be "
                                  "created. User has no CREATE TABLE privileges.");

  switchToAdmin();
  sql("GRANT CREATE TABLE ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  login("test_user", "test_pass");
  createTestForeignTable();

  // clean up permissions
  switchToAdmin();
  sql("REVOKE CREATE TABLE ON DATABASE " + shared::kDefaultDbName + " FROM test_user;");
}

TEST_F(ForeignTablePermissionsTest, ForeignTableRefreshOwner) {
  sql("GRANT CREATE TABLE ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  login("test_user", "test_pass");
  createTestForeignTable();
  runQuery("REFRESH FOREIGN TABLES test_table;");
  // clean up permissions
  switchToAdmin();
  sql("REVOKE CREATE TABLE ON DATABASE " + shared::kDefaultDbName + " FROM test_user;");
}

TEST_F(ForeignTablePermissionsTest, ForeignTableRefreshSuperUser) {
  sql("GRANT CREATE TABLE ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  login("test_user", "test_pass");
  createTestForeignTable();
  switchToAdmin();
  runQuery("REFRESH FOREIGN TABLES test_table;");
  // clean up permissions
  sql("REVOKE CREATE TABLE ON DATABASE " + shared::kDefaultDbName + " FROM test_user;");
}

TEST_F(ForeignTablePermissionsTest, ForeignTableRefreshNonOwner) {
  createTestForeignTable();
  sql("GRANT ALL ON TABLE test_table TO test_user;");
  login("test_user", "test_pass");
  runQueryAndAssertException(
      "REFRESH FOREIGN TABLES test_table;",
      "REFRESH FOREIGN TABLES failed on table \"test_table\". It can only be "
      "executed by super user or owner of the object.");
}

class ServerPrivApiTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    createDBHandler();
    switchToAdmin();
    createTestUser("test_user");
    createTestUser("test_user_2");
  }

  static void TearDownTestSuite() {
    dropTestUser("test_user");
    dropTestUser("test_user_2");
  }

  void SetUp() override {
    if (g_aggregator) {
      LOG(INFO) << "Test fixture not supported in distributed mode.";
      GTEST_SKIP();
      return;
    }
    DBHandlerTestFixture::SetUp();
    loginAdmin();
    dropServer();
    createTestServer();
  }

  void TearDown() override {
    loginAdmin();
    dropServer();
    revokeTestUserServerPrivileges("test_user");
    revokeTestUserServerPrivileges("test_user_2");
  }
  static void createTestUser(std::string name) {
    sql("CREATE USER  " + name + " (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName + " TO  " + name + ";");
  }

  static void dropTestUser(std::string name) { sql("DROP USER IF EXISTS " + name + ";"); }

  void revokeTestUserServerPrivileges(std::string name) {
    sql("REVOKE ALL ON DATABASE " + shared::kDefaultDbName + " FROM " + name + ";");
    sql("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName + " TO " + name + ";");
  }
  void createTestServer() {
    sql("CREATE SERVER test_server FOREIGN DATA WRAPPER delimited_file "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
  }

  void dropServer() { sql("DROP SERVER IF EXISTS test_server;"); }

  void assertExpectedDBObj(std::vector<TDBObject>& db_objs,
                           std::string name,
                           TDBObjectType::type obj_type,
                           std::vector<bool> privs,
                           std::string grantee_name,
                           TDBObjectType::type priv_type) {
    bool obj_found = false;
    for (const auto& db_obj : db_objs) {
      if ((db_obj.objectName == name) && (db_obj.objectType == obj_type) &&
          (db_obj.privs == privs) && (db_obj.grantee == grantee_name) &&
          (db_obj.privilegeObjectType == priv_type)) {
        obj_found = true;
        break;
      };
    }
    ASSERT_TRUE(obj_found);
  }

  void assertDBAccessObj(std::vector<TDBObject>& db_objs) {
    assertExpectedDBObj(db_objs,
                        shared::kDefaultDbName,
                        TDBObjectType::DatabaseDBObjectType,
                        {0, 0, 0, 1},
                        "test_user",
                        TDBObjectType::DatabaseDBObjectType);
  }
  void assertSuperAccessObj(std::vector<TDBObject>& db_objs) {
    assertExpectedDBObj(db_objs,
                        "super",
                        TDBObjectType::AbstractDBObjectType,
                        {1, 1, 1, 1},
                        "admin",
                        TDBObjectType::ServerDBObjectType);
  }
};

TEST_F(ServerPrivApiTest, CreateForGrantee) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT CREATE SERVER ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_objects_for_grantee(priv_objs, session_id, "test_user");
  ASSERT_EQ(priv_objs.size(), 2u);
  assertDBAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      shared::kDefaultDbName,
                      TDBObjectType::DatabaseDBObjectType,
                      {1, 0, 0, 0},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, DropForGrantee) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT DROP SERVER ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_objects_for_grantee(priv_objs, session_id, "test_user");
  ASSERT_EQ(priv_objs.size(), 2u);
  assertDBAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      shared::kDefaultDbName,
                      TDBObjectType::DatabaseDBObjectType,
                      {0, 1, 0, 0},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, AlterForGrantee) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT ALTER SERVER ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_objects_for_grantee(priv_objs, session_id, "test_user");
  ASSERT_EQ(priv_objs.size(), 2u);
  assertDBAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      shared::kDefaultDbName,
                      TDBObjectType::DatabaseDBObjectType,
                      {0, 0, 1, 0},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, AlterOnServerGrantee) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT ALTER ON SERVER test_server TO test_user;");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_objects_for_grantee(priv_objs, session_id, "test_user");
  ASSERT_EQ(priv_objs.size(), 2u);
  assertDBAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      "test_server",
                      TDBObjectType::ServerDBObjectType,
                      {0, 0, 1, 0},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, UsageForGrantee) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT SERVER USAGE ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_objects_for_grantee(priv_objs, session_id, "test_user");
  ASSERT_EQ(priv_objs.size(), 2u);
  assertDBAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      shared::kDefaultDbName,
                      TDBObjectType::DatabaseDBObjectType,
                      {0, 0, 0, 1},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, UsageOnServerGrantee) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT USAGE ON SERVER test_server TO test_user;");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_objects_for_grantee(priv_objs, session_id, "test_user");
  ASSERT_EQ(priv_objs.size(), 2u);
  assertDBAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      "test_server",
                      TDBObjectType::ServerDBObjectType,
                      {0, 0, 0, 1},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, GetDBObjNonSuser) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT CREATE SERVER ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  login("test_user", "test_pass");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_objects_for_grantee(priv_objs, session_id, "test_user");
  ASSERT_EQ(priv_objs.size(), 2u);
  assertDBAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      shared::kDefaultDbName,
                      TDBObjectType::DatabaseDBObjectType,
                      {1, 0, 0, 0},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, GetDBObjNoAccess) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT CREATE SERVER ON DATABASE " + shared::kDefaultDbName + " TO test_user_2;");
  login("test_user", "test_pass");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_objects_for_grantee(priv_objs, session_id, "test_user_2");
  // no privs returned
  ASSERT_EQ(priv_objs.size(), 0u);
}

TEST_F(ServerPrivApiTest, AlterOnServerObjectPrivs) {
  sql("GRANT ALTER ON SERVER test_server TO test_user;");
  std::vector<TDBObject> priv_objs;
  login("test_user", "test_pass");
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  db_handler->get_db_object_privs(
      priv_objs, session_id, "test_server", TDBObjectType::ServerDBObjectType);
  ASSERT_EQ(priv_objs.size(), 1u);
  assertExpectedDBObj(priv_objs,
                      "test_server",
                      TDBObjectType::ServerDBObjectType,
                      {0, 0, 1, 0},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, AlterOnServerObjectPrivsSuper) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT ALTER ON SERVER test_server TO test_user;");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_object_privs(
      priv_objs, session_id, "test_server", TDBObjectType::ServerDBObjectType);
  ASSERT_EQ(priv_objs.size(), 2u);
  // Suser access obj returned when calling as suser
  assertSuperAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      "test_server",
                      TDBObjectType::ServerDBObjectType,
                      {0, 0, 1, 0},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}
TEST_F(ServerPrivApiTest, UsageOnServerObjectPrivs) {
  sql("GRANT USAGE ON SERVER test_server TO test_user;");
  std::vector<TDBObject> priv_objs;
  login("test_user", "test_pass");
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  db_handler->get_db_object_privs(
      priv_objs, session_id, "test_server", TDBObjectType::ServerDBObjectType);
  ASSERT_EQ(priv_objs.size(), 1u);
  assertExpectedDBObj(priv_objs,
                      "test_server",
                      TDBObjectType::ServerDBObjectType,
                      {0, 0, 0, 1},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, UsageOnServerObjectPrivsSuper) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  sql("GRANT USAGE ON SERVER test_server TO test_user;");
  std::vector<TDBObject> priv_objs;
  db_handler->get_db_object_privs(
      priv_objs, session_id, "test_server", TDBObjectType::ServerDBObjectType);
  ASSERT_EQ(priv_objs.size(), 2u);
  // Suser access obj returned when calling as suser
  assertSuperAccessObj(priv_objs);
  assertExpectedDBObj(priv_objs,
                      "test_server",
                      TDBObjectType::ServerDBObjectType,
                      {0, 0, 0, 1},
                      "test_user",
                      TDBObjectType::ServerDBObjectType);
}

TEST_F(ServerPrivApiTest, ShowServerRequiresPermission) {
  login("test_user", "test_pass");
  queryAndAssertException("SHOW CREATE SERVER test_server",
                          "Foreign server test_server does not exist.");
  loginAdmin();
  sql("GRANT USAGE ON SERVER test_server TO test_user;");
  login("test_user", "test_pass");
  ASSERT_NO_THROW(sql("SHOW CREATE SERVER test_server"));
}

TEST(Temporary, Users) {
  auto user_cleanup = [] {
    if (sys_cat.getMetadataForUser("username1", g_user)) {
      sys_cat.dropUser("username1");
    }
    CHECK(!sys_cat.getMetadataForUser("username1", g_user));

    if (sys_cat.getMetadataForUser("username2", g_user)) {
      sys_cat.dropUser("username2");
    }
    CHECK(!sys_cat.getMetadataForUser("username2", g_user));
  };
  user_cleanup();

  auto read_only = g_read_only;
  g_read_only = true;
  ScopeGuard scope_guard = [&] {
    g_read_only = read_only;
    user_cleanup();
  };

  sys_cat.createUser(
      "username1",
      Catalog_Namespace::UserAlterations{
          "password1", /*is_super=*/true, /*dbname=*/"", /*can_login=*/true},
      /*is_temporary=*/true);
  CHECK(sys_cat.getMetadataForUser("username1", g_user));

  EXPECT_TRUE(g_user.is_temporary);
  EXPECT_EQ(g_user.can_login, true);

  EXPECT_NO_THROW(sys_cat.alterUser(
      "username1",
      Catalog_Namespace::UserAlterations{
          /*password=*/{}, /*is_super=*/{}, /*default_db=*/{}, /*can_login=*/false}));
  CHECK(sys_cat.getMetadataForUser("username1", g_user));
  EXPECT_EQ(g_user.can_login, false);

  EXPECT_NO_THROW(sys_cat.renameUser("username1", "username2"));
  CHECK(sys_cat.getMetadataForUser("username2", g_user));
  EXPECT_EQ(g_user.userName, "username2");
}

class ObjectDescriptorCleanupTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    sql("DROP TABLE IF EXISTS test_table;");
    sql("CREATE TABLE test_table (i INTEGER);");
  }

  void TearDown() override { sql("DROP TABLE IF EXISTS test_table;"); }

  void assertNumObjectDesciptors(size_t num_descriptors) {
    auto td = getCatalog().getMetadataForTable("test_table", false);
    ASSERT_EQ(sys_cat
                  .getMetadataForObject(getCatalog().getDatabaseId(),
                                        DBObjectType::TableDBObjectType,
                                        td->tableId)
                  .size(),
              num_descriptors);
  }

  void assertExpectedDescriptorRole() {
    assertNumObjectDesciptors(1);
    auto td = getCatalog().getMetadataForTable("test_table", false);
    auto object_desc = sys_cat.getMetadataForObject(
        getCatalog().getDatabaseId(), DBObjectType::TableDBObjectType, td->tableId)[0];
    ASSERT_EQ(object_desc->roleName, "test_role");
    ASSERT_EQ(object_desc->roleType, false);
    ASSERT_TRUE(
        object_desc->privs.hasPermission(AccessPrivileges::SELECT_FROM_TABLE.privileges));
  }

  void assertExpectedDescriptorUser(std::string username = "test_user") {
    assertNumObjectDesciptors(1);
    auto td = getCatalog().getMetadataForTable("test_table", false);
    auto object_desc = sys_cat.getMetadataForObject(
        getCatalog().getDatabaseId(), DBObjectType::TableDBObjectType, td->tableId)[0];
    ASSERT_EQ(object_desc->roleName, username);
    ASSERT_EQ(object_desc->roleType, true);
    ASSERT_TRUE(
        object_desc->privs.hasPermission(AccessPrivileges::SELECT_FROM_TABLE.privileges));
  }
};

TEST_F(ObjectDescriptorCleanupTest, DeleteObjectDescriptorOnDropRole) {
  sql("CREATE ROLE test_role;");
  assertNumObjectDesciptors(0);
  sql("GRANT SELECT ON TABLE test_table TO test_role");
  assertExpectedDescriptorRole();
  sql("DROP ROLE test_role;");
  assertNumObjectDesciptors(0);
}

TEST_F(ObjectDescriptorCleanupTest, DeleteObjectDescriptorOnDropUser) {
  sql("CREATE USER test_user;");
  assertNumObjectDesciptors(0);
  sql("GRANT SELECT ON TABLE test_table TO test_user");
  assertExpectedDescriptorUser();
  sql("DROP USER test_user;");
  assertNumObjectDesciptors(0);
}

TEST_F(ObjectDescriptorCleanupTest, DeleteObjectDescriptorFromRenamedUser) {
  sql("CREATE USER test_user");
  assertNumObjectDesciptors(0);
  sql("GRANT SELECT ON TABLE test_table TO test_user");
  assertExpectedDescriptorUser();
  sql("ALTER USER test_user RENAME TO test_user_renamed;");
  assertExpectedDescriptorUser("test_user_renamed");
  sql("DROP USER test_user_renamed;");
  assertNumObjectDesciptors(0);
}

class ReassignOwnedTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    createDBHandler();
    createTestUser("test_user_1", "test_pass");
    createTestUser("test_user_2", "test_pass");
    createTestUser("test_user_3", "test_pass");
    createTestUser("all_permissions_test_user", "test_pass");
    sql("GRANT ALL ON DATABASE " + shared::kDefaultDbName +
        " TO all_permissions_test_user;");
  }

  static void TearDownTestSuite() {
    loginAdmin();
    dropTestUser("test_user_1");
    dropTestUser("test_user_2");
    dropTestUser("test_user_3");
    dropTestUser("all_permissions_test_user");
  }

  void SetUp() override { dropAllDatabaseObjects(); }

  void TearDown() override { dropAllDatabaseObjects(); }

  void dropAllDatabaseObjects() {
    loginAdmin();
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    auto dashboards = getCatalog().getAllDashboardsMetadata();
    for (const auto dashboard : dashboards) {
      db_handler->delete_dashboard(session_id, dashboard->dashboardId);
    }
    sql("DROP TABLE IF EXISTS test_table_1;");
    sql("DROP TABLE IF EXISTS test_table_2;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table_1;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table_2;");
    sql("DROP VIEW IF EXISTS test_view_1;");
    sql("DROP VIEW IF EXISTS test_view_2;");
    sql("DROP SERVER IF EXISTS test_server_1;");
    sql("DROP SERVER IF EXISTS test_server_2;");
    sql("DROP DATABASE IF EXISTS test_db;");
  }

  static int32_t createTestUser(const std::string& user_name, const std::string& pass) {
    sql("CREATE USER " + user_name + " (password = '" + pass + "');");
    sql("GRANT ACCESS, CREATE TABLE, CREATE VIEW, CREATE DASHBOARD ON "
        "DATABASE " +
        shared::kDefaultDbName + " TO " + user_name + ";");
    sql("GRANT CREATE SERVER ON DATABASE " + shared::kDefaultDbName + " TO " + user_name +
        ";");
    UserMetadata user_metadata{};
    SysCatalog::instance().getMetadataForUser(user_name, user_metadata);
    return user_metadata.userId;
  }

  static void dropTestUser(const std::string& user_name) {
    sql("DROP USER IF EXISTS " + user_name + ";");
  }

  std::string getForeignTableFilePath() {
    return getDataFilesPath() + "../../Tests/FsiDataFiles/1.csv";
  }

  void createDatabaseObjects(const std::string& name_suffix) {
    sql("CREATE TABLE test_table_" + name_suffix + " (i INTEGER);");
    sql("CREATE FOREIGN TABLE test_foreign_table_" + name_suffix +
        " (i INTEGER) SERVER default_local_delimited WITH "
        "(file_path='" +
        getForeignTableFilePath() + "', header='true');");
    sql("CREATE VIEW test_view_" + name_suffix + " AS SELECT * FROM test_table_" +
        name_suffix + ";");
    sql("CREATE SERVER test_server_" + name_suffix +
        " FOREIGN DATA WRAPPER delimited_file "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    db_handler->create_dashboard(
        session_id, "test_dashboard_" + name_suffix, "state", "image", "metadata");
  }

  void assertDatabaseObjectsOwnership(const std::string& user_name,
                                      const std::string& name_suffix) {
    UserMetadata user;
    SysCatalog::instance().getMetadataForUser(user_name, user);
    auto user_id = user.userId;
    const auto& catalog = getCatalog();

    auto td = catalog.getMetadataForTable("test_table_" + name_suffix, false);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(user_id, td->userId);

    auto foreign_td = catalog.getForeignTable("test_foreign_table_" + name_suffix);
    ASSERT_NE(foreign_td, nullptr);
    ASSERT_EQ(user_id, foreign_td->userId);

    auto view = catalog.getMetadataForTable("test_view_" + name_suffix, false);
    ASSERT_NE(view, nullptr);
    ASSERT_EQ(user_id, view->userId);

    auto dashboard = catalog.getMetadataForDashboard(std::to_string(user_id),
                                                     "test_dashboard_" + name_suffix);
    ASSERT_NE(dashboard, nullptr);
    ASSERT_EQ(user_id, dashboard->userId);
    ASSERT_EQ(user_name, dashboard->user);

    // Permission entries are not added for super users
    if (!user.isSuper) {
      ASSERT_TRUE(SysCatalog::instance().verifyDBObjectOwnership(
          user, DBObject{td->tableId, TableDBObjectType}, catalog));
      ASSERT_TRUE(SysCatalog::instance().verifyDBObjectOwnership(
          user, DBObject{foreign_td->tableId, TableDBObjectType}, catalog));
      ASSERT_TRUE(SysCatalog::instance().verifyDBObjectOwnership(
          user, DBObject{view->tableId, ViewDBObjectType}, catalog));
      ASSERT_TRUE(SysCatalog::instance().verifyDBObjectOwnership(
          user, DBObject{dashboard->dashboardId, DashboardDBObjectType}, catalog));

      assertObjectRoleDescriptor(
          DBObjectType::TableDBObjectType, foreign_td->tableId, user_id);
      assertObjectRoleDescriptor(DBObjectType::TableDBObjectType, td->tableId, user_id);
      assertObjectRoleDescriptor(DBObjectType::ViewDBObjectType, view->tableId, user_id);
      assertObjectRoleDescriptor(
          DBObjectType::DashboardDBObjectType, dashboard->dashboardId, user_id);
    }

    auto server = catalog.getForeignServer("test_server_" + name_suffix);
    ASSERT_NE(server, nullptr);
    ASSERT_EQ(user_id, server->user_id);

    if (!user.isSuper) {
      ASSERT_TRUE(SysCatalog::instance().verifyDBObjectOwnership(
          user, DBObject{server->id, ServerDBObjectType}, catalog));
      assertObjectRoleDescriptor(DBObjectType::ServerDBObjectType, server->id, user_id);
    }
  }

  void assertObjectRoleDescriptor(DBObjectType object_type,
                                  int32_t object_id,
                                  int32_t owner_id) {
    auto object_type_id = static_cast<int>(object_type);
    int found_objects = 0;
    for (const auto object : SysCatalog::instance().getMetadataForObject(
             getCatalog().getDatabaseId(), object_type_id, object_id)) {
      if (object->objectType == object_type_id && object->objectId == object_id) {
        if (object->objectOwnerId != owner_id) {
          FAIL() << "ObjectRoleDescriptor found with wrong owner id,  object type: "
                 << object_type_id << ", object id: " << object_id
                 << ", owner id: " << object->objectOwnerId
                 << ", expected owner id: " << owner_id
                 << " name: " << object->objectName;
        }
        found_objects++;
      }
    }
    if (found_objects == 0) {
      FAIL() << "No ObjectRoleDescriptor found for object type: " << object_type_id
             << ", object id: " << object_id << ", owner id: " << owner_id;
    }
  }

  void assertDatabasePermissions(const std::string& user_name,
                                 const std::string& dbname,
                                 bool has_object_permissions) {
    auto db = SysCatalog::instance().getDB(dbname);
    ASSERT_EQ(db.has_value(), true);
    assertObjectPermissions(user_name,
                            db->dbId,
                            AccessPrivileges(DatabasePrivileges::ACCESS |
                                             DatabasePrivileges::VIEW_SQL_EDITOR),
                            DBObjectType::DatabaseDBObjectType,
                            has_object_permissions,
                            true,
                            dbname);
  }

  void assertObjectPermissions(const std::string& user_name,
                               const std::string& name_suffix,
                               bool has_object_permissions,
                               const std::string& owner_user_name) {
    const auto& catalog = getCatalog();

    auto foreign_td = catalog.getForeignTable("test_foreign_table_" + name_suffix);
    ASSERT_NE(foreign_td, nullptr);
    assertObjectPermissions(user_name,
                            foreign_td->tableId,
                            AccessPrivileges::ALL_TABLE,
                            DBObjectType::TableDBObjectType,
                            has_object_permissions);

    auto td = catalog.getMetadataForTable("test_table_" + name_suffix, false);
    ASSERT_NE(td, nullptr);
    assertObjectPermissions(user_name,
                            td->tableId,
                            AccessPrivileges::ALL_TABLE,
                            DBObjectType::TableDBObjectType,
                            has_object_permissions);

    auto view = catalog.getMetadataForTable("test_view_" + name_suffix, false);
    ASSERT_NE(view, nullptr);
    assertObjectPermissions(user_name,
                            view->tableId,
                            AccessPrivileges::ALL_VIEW,
                            DBObjectType::ViewDBObjectType,
                            has_object_permissions);

    auto server = catalog.getForeignServer("test_server_" + name_suffix);
    ASSERT_NE(server, nullptr);
    assertObjectPermissions(user_name,
                            server->id,
                            AccessPrivileges::ALL_SERVER,
                            DBObjectType::ServerDBObjectType,
                            has_object_permissions);

    UserMetadata user;
    SysCatalog::instance().getMetadataForUser(owner_user_name, user);
    auto dashboard = catalog.getMetadataForDashboard(std::to_string(user.userId),
                                                     "test_dashboard_" + name_suffix);
    ASSERT_NE(dashboard, nullptr);
    assertObjectPermissions(user_name,
                            dashboard->dashboardId,
                            AccessPrivileges::ALL_DASHBOARD,
                            DBObjectType::DashboardDBObjectType,
                            has_object_permissions);
  }

  DBObject createDBObject(int32_t object_id,
                          DBObjectType object_type,
                          const bool is_db,
                          std::string object_name) {
    if (!is_db) {
      return DBObject{object_id, object_type};
    }
    CHECK(!object_name.empty());
    return DBObject{object_name, object_type};
  }

  void assertObjectPermissions(const std::string& user_name,
                               int32_t object_id,
                               const AccessPrivileges& privilege_type,
                               DBObjectType object_type,
                               bool has_object_permissions,
                               const bool is_db = false,
                               std::string object_name = {}) {
    AccessPrivileges privileges;
    privileges.add(privilege_type);
    DBObject object = createDBObject(object_id, object_type, is_db, object_name);
    object.loadKey(getCatalog());
    object.setPrivileges(privileges);
    ASSERT_EQ(SysCatalog::instance().checkPrivileges(user_name, {object}),
              has_object_permissions);
  }

  void assertNoDashboardsForUsers(const std::vector<std::string>& user_names) {
    const auto& catalog = getCatalog();
    for (const auto& user_name : user_names) {
      UserMetadata user;
      SysCatalog::instance().getMetadataForUser(user_name, user);
      for (const auto dashboard : catalog.getAllDashboardsMetadata()) {
        ASSERT_NE(dashboard->userId, user.userId);
        ASSERT_NE(dashboard->user, user_name);
        ASSERT_EQ(catalog.getMetadataForDashboard(std::to_string(user.userId),
                                                  dashboard->dashboardName),
                  nullptr);
      }
    }
  }

  void assertDatabaseOwnership(const std::string& db_name, const std::string& user_name) {
    UserMetadata user;
    SysCatalog::instance().getMetadataForUser(user_name, user);

    DBMetadata database;
    SysCatalog::instance().getMetadataForDB(db_name, database);
    ASSERT_EQ(user.userId, database.dbOwner);

    auto& catalog = *SysCatalog::instance().getCatalog(database.dbId);
    ASSERT_EQ(db_name, catalog.name());

    // Permission entries are not added for super users
    if (!user.isSuper) {
      ASSERT_TRUE(SysCatalog::instance().verifyDBObjectOwnership(
          user, DBObject{db_name, DatabaseDBObjectType}, catalog));
      assertObjectRoleDescriptor(DBObjectType::DatabaseDBObjectType, -1, user.userId);
    }
  }

  static std::string getDataFilesPath() {
    return boost::filesystem::canonical(g_test_binary_file_path +
                                        "/../../Tests/FsiDataFiles")
               .string() +
           "/";
  }
};

TEST_F(ReassignOwnedTest, SuperUser) {
  login("test_user_1", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  login("test_user_2", "test_pass");
  createDatabaseObjects("2");
  assertDatabaseObjectsOwnership("test_user_2", "2");
  assertObjectPermissions("test_user_2", "2", true, "test_user_2");

  loginAdmin();
  sql("REASSIGN OWNED BY test_user_1, test_user_2 TO test_user_3;");

  assertDatabaseObjectsOwnership("test_user_3", "1");
  assertDatabaseObjectsOwnership("test_user_3", "2");
  assertObjectPermissions("test_user_3", "1", true, "test_user_3");
  assertObjectPermissions("test_user_3", "2", true, "test_user_3");
  assertNoDashboardsForUsers({"test_user_1", "test_user_2"});

  // Assert that old owners no longer have permissions to database objects
  assertObjectPermissions("test_user_1", "1", false, "test_user_3");
  assertObjectPermissions("test_user_2", "2", false, "test_user_3");
}

TEST_F(ReassignOwnedTest, UserWithAllPermissions) {
  login("all_permissions_test_user", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("all_permissions_test_user", "1");
  assertObjectPermissions(
      "all_permissions_test_user", "1", true, "all_permissions_test_user");

  loginAdmin();
  sql("REASSIGN OWNED BY all_permissions_test_user TO test_user_1;");

  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");
  assertNoDashboardsForUsers({"all_permissions_test_user"});

  // Assert that old owner still has permissions to database objects because of previous
  // grant of all permissions
  assertObjectPermissions("all_permissions_test_user", "1", true, "test_user_1");
}

TEST_F(ReassignOwnedTest, ReassignToSameUser) {
  login("test_user_1", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  loginAdmin();
  sql("REASSIGN OWNED BY test_user_1 TO test_user_1;");

  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");
}

TEST_F(ReassignOwnedTest, DatabaseOwner) {
  loginAdmin();
  sql("CREATE DATABASE test_db (owner = 'test_user_1');");

  login("test_user_1", "test_pass", "test_db");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");
  assertDatabaseOwnership("test_db", "test_user_1");

  login("admin", "HyperInteractive", "test_db");
  sql("REASSIGN OWNED BY test_user_1 TO test_user_2;");

  assertDatabaseObjectsOwnership("test_user_2", "1");
  assertObjectPermissions("test_user_2", "1", true, "test_user_2");
  assertNoDashboardsForUsers({"test_user_1"});

  // Assert that the old owner still owns the database
  assertDatabaseOwnership("test_db", "test_user_1");

  // Assert that the old owner still has permissions to database objects because of
  // database ownership
  assertObjectPermissions("test_user_1", "1", true, "test_user_2");
}

TEST_F(ReassignOwnedTest, MultipleDatabases) {
  // Create objects in the default database
  login("test_user_1", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  loginAdmin();
  sql("CREATE DATABASE test_db;");
  sql("GRANT ALL ON DATABASE test_db TO test_user_1;");

  // Create objects in the new database
  login("test_user_1", "test_pass", "test_db");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  login("admin", "HyperInteractive", "test_db");
  sql("REASSIGN OWNED BY test_user_1 TO test_user_2;");

  assertDatabaseObjectsOwnership("test_user_2", "1");
  assertObjectPermissions("test_user_2", "1", true, "test_user_2");
  assertNoDashboardsForUsers({"test_user_1"});

  // Assert that ownership in the default database is not changed
  loginAdmin();
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");
  assertNoDashboardsForUsers({"test_user_2"});
}

TEST_F(ReassignOwnedTest, MultipleDatabasesWithAll) {
  // Create objects in the default database
  login("test_user_1", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  loginAdmin();
  sql("CREATE DATABASE test_db;");
  sql("GRANT ALL ON DATABASE test_db TO test_user_1;");
  assertDatabasePermissions("test_user_1", "test_db", true);
  // Check that default database permissions ALL is not given to either user
  assertDatabasePermissions("test_user_1", shared::kDefaultDbName, false);
  assertDatabasePermissions("test_user_2", shared::kDefaultDbName, false);
  // Check database ownership
  assertDatabaseOwnership(shared::kDefaultDbName, shared::kRootUsername);
  assertDatabaseOwnership("test_db", shared::kRootUsername);

  // Create objects in the new database
  login("test_user_1", "test_pass", "test_db");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  login("admin", "HyperInteractive", "test_db");
  sql("REASSIGN ALL OWNED BY test_user_1 TO test_user_2;");

  assertDatabaseObjectsOwnership("test_user_2", "1");
  assertObjectPermissions("test_user_2", "1", true, "test_user_2");
  assertNoDashboardsForUsers({"test_user_1"});

  // Assert that ownership in the default database **has** changed
  loginAdmin();
  assertDatabaseObjectsOwnership("test_user_2", "1");
  assertObjectPermissions("test_user_2", "1", true, "test_user_2");
  assertNoDashboardsForUsers({"test_user_1"});

  // test_user_2 should not have permissions to database and test_user_1 should maintain
  // original permissions on database
  assertDatabasePermissions("test_user_1", "test_db", true);
  assertDatabasePermissions("test_user_2", "test_db", false);

  // Default database should also have permissions remained unchanged
  assertDatabasePermissions("test_user_1", shared::kDefaultDbName, false);
  assertDatabasePermissions("test_user_2", shared::kDefaultDbName, false);

  // Check database ownership has not changed
  assertDatabaseOwnership(shared::kDefaultDbName, shared::kRootUsername);
  assertDatabaseOwnership("test_db", shared::kRootUsername);
}

TEST_F(ReassignOwnedTest, ReassignToSuperUser) {
  login("test_user_1", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  loginAdmin();
  sql("REASSIGN OWNED BY test_user_1 TO admin;");

  assertDatabaseObjectsOwnership("admin", "1");
  assertObjectPermissions("admin", "1", true, "admin");
  assertNoDashboardsForUsers({"test_user_1"});

  // Assert that the old owner no longer has permissions to database objects
  assertObjectPermissions("test_user_1", "1", false, "admin");
}

TEST_F(ReassignOwnedTest, ReassignFromSuperUser) {
  loginAdmin();
  // Use a different database for this test case in order to avoid changing database
  // objects that are created outside the scope of this test in the default database.
  sql("CREATE DATABASE test_db;");

  login("admin", "HyperInteractive", "test_db");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("admin", "1");
  assertObjectPermissions("admin", "1", true, "admin");

  sql("REASSIGN OWNED BY admin TO test_user_1;");

  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");
  assertNoDashboardsForUsers({"admin"});

  // Assert that the super user/admin still owns the database
  assertDatabaseOwnership("test_db", "admin");

  // Assert that the super user/admin still has permissions to database objects
  assertObjectPermissions("admin", "1", true, "test_user_1");
}

TEST_F(ReassignOwnedTest, NonSuperUser) {
  login("test_user_1", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  login("test_user_2", "test_pass");
  queryAndAssertException("REASSIGN OWNED BY test_user_1 TO test_user_2;",
                          "Only super users can reassign ownership of database objects.");
}

TEST_F(ReassignOwnedTest, NonExistentOldOwner) {
  login("test_user_1", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  loginAdmin();
  queryAndAssertException(
      "REASSIGN OWNED BY test_user_1, nonexistent_user TO test_user_2;",
      "User with username \"nonexistent_user\" does not exist.");
}

TEST_F(ReassignOwnedTest, NonExistentNewOwner) {
  login("test_user_1", "test_pass");
  createDatabaseObjects("1");
  assertDatabaseObjectsOwnership("test_user_1", "1");
  assertObjectPermissions("test_user_1", "1", true, "test_user_1");

  loginAdmin();
  queryAndAssertException("REASSIGN OWNED BY test_user_1 TO nonexistent_user;",
                          "User with username \"nonexistent_user\" does not exist.");
}

class AlterServerOwnerTest : public ReassignOwnedTest {
 protected:
  void SetUp() override {
    if (g_aggregator) {
      LOG(INFO) << "Test fixture not supported in distributed mode.";
      GTEST_SKIP();
    }
    ReassignOwnedTest::dropAllDatabaseObjects();
  }

  static void createServer() {
    sql("CREATE SERVER test_server_1 FOREIGN DATA WRAPPER delimited_file WITH "
        "(storage_type "
        "= 'LOCAL_FILE', base_path = '/test_path/');");
    // grant alter on server to other user to create additional ObjectRoleDescriptor that
    // needs to be updated
    sql("GRANT ALTER ON SERVER test_server_1 TO test_user_2;");
  }

  void verifyOwnership(std::string user_name) {
    UserMetadata user;
    SysCatalog::instance().getMetadataForUser(user_name, user);
    auto user_id = user.userId;
    auto& catalog = getCatalog();
    auto server = catalog.getForeignServer("test_server_1");
    ASSERT_NE(server, nullptr);
    ASSERT_EQ(user_id, server->user_id);
    if (!user.isSuper) {
      ASSERT_TRUE(SysCatalog::instance().verifyDBObjectOwnership(
          user, DBObject{server->id, DBObjectType::ServerDBObjectType}, catalog));
    }
    // Assumes additional privileges have been granted if owner is superuser
    assertObjectRoleDescriptor(DBObjectType::ServerDBObjectType, server->id, user_id);
  }
};

TEST_F(AlterServerOwnerTest, ToRegularUser) {
  createServer();
  verifyOwnership("admin");
  sql("ALTER SERVER test_server_1 OWNER TO test_user_1;");
  verifyOwnership("test_user_1");
}

TEST_F(AlterServerOwnerTest, ToSuperUser) {
  login("test_user_1", "test_pass");
  createServer();
  verifyOwnership("test_user_1");
  loginAdmin();
  sql("ALTER SERVER test_server_1 OWNER TO admin;");
  verifyOwnership("admin");
}

TEST(SyncUserWithRemoteProvider, DEFAULT_DB) {
  run_ddl_statement("DROP DATABASE IF EXISTS db1;");
  run_ddl_statement("DROP DATABASE IF EXISTS db2;");
  ScopeGuard dbguard = [] {
    run_ddl_statement("DROP DATABASE IF EXISTS db1;");
    run_ddl_statement("DROP DATABASE IF EXISTS db2;");
  };
  run_ddl_statement("CREATE DATABASE db1;");
  run_ddl_statement("CREATE DATABASE db2;");

  auto db1 = sys_cat.getDB("db1");
  ASSERT_TRUE(db1);
  auto db2 = sys_cat.getDB("db2");
  ASSERT_TRUE(db2);

  run_ddl_statement("DROP USER IF EXISTS u1;");
  ScopeGuard u1guard = [] { run_ddl_statement("DROP USER IF EXISTS u1;"); };
  Catalog_Namespace::UserAlterations alts;

  // create u1 w/db1
  alts.default_db = "db1";
  sys_cat.syncUserWithRemoteProvider("u1", {}, alts);
  auto u1 = sys_cat.getUser("u1");
  ASSERT_TRUE(u1);
  ASSERT_EQ(u1->defaultDbId, db1->dbId);

  // alter u1 w/db2
  alts.default_db = "db2";
  sys_cat.syncUserWithRemoteProvider("u1", {}, alts);
  u1 = sys_cat.getUser("u1");
  ASSERT_TRUE(u1);
  ASSERT_EQ(u1->defaultDbId, db2->dbId);

  // alter u1 no-op
  alts.default_db = std::nullopt;
  sys_cat.syncUserWithRemoteProvider("u1", {}, alts);
  u1 = sys_cat.getUser("u1");
  ASSERT_TRUE(u1);
  ASSERT_EQ(u1->defaultDbId, db2->dbId);

  // alter u1 to clear out the default_db
  alts.default_db = "";
  sys_cat.syncUserWithRemoteProvider("u1", {}, alts);
  u1 = sys_cat.getUser("u1");
  ASSERT_TRUE(u1);
  ASSERT_EQ(u1->defaultDbId, -1);
}

TEST(SyncUserWithRemoteProvider, IS_SUPER) {
  run_ddl_statement("DROP USER IF EXISTS u1;");
  run_ddl_statement("DROP USER IF EXISTS u2;");
  ScopeGuard uguard = [] {
    run_ddl_statement("DROP USER IF EXISTS u1;");
    run_ddl_statement("DROP USER IF EXISTS u2;");
  };
  Catalog_Namespace::UserAlterations alts;

  // create u1 non-super
  alts.is_super = false;
  sys_cat.syncUserWithRemoteProvider("u1", {}, alts);
  auto u1 = sys_cat.getUser("u1");
  ASSERT_TRUE(u1);
  ASSERT_EQ(u1->isSuper, false);

  // alter u1 no-op
  alts.is_super = std::nullopt;
  sys_cat.syncUserWithRemoteProvider("u1", {}, alts);
  u1 = sys_cat.getUser("u1");
  ASSERT_TRUE(u1);
  ASSERT_EQ(u1->isSuper, false);

  // alter u1 super
  alts.is_super = true;
  sys_cat.syncUserWithRemoteProvider("u1", {}, alts);
  u1 = sys_cat.getUser("u1");
  ASSERT_TRUE(u1);
  ASSERT_EQ(u1->isSuper, true);

  // create u2 super
  sys_cat.syncUserWithRemoteProvider("u2", {}, alts);
  auto u2 = sys_cat.getUser("u2");
  ASSERT_TRUE(u2);
  ASSERT_EQ(u2->isSuper, true);

  // alter u2 non-super
  alts.is_super = false;
  sys_cat.syncUserWithRemoteProvider("u2", {}, alts);
  u2 = sys_cat.getUser("u2");
  ASSERT_TRUE(u2);
  ASSERT_EQ(u2->isSuper, false);
}

class DropUserTest : public DBHandlerTestFixture {};

TEST_F(DropUserTest, DropAdmin) {
  queryAndAssertException("DROP USER admin;",
                          "Cannot drop user. User admin is required to exist.");
}

class CreateDropDatabaseTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop database if exists orphan_db");
    sql("drop user if exists test_admin_user");
  }
  void TearDown() override {
    loginAdmin();
    sql("drop database if exists orphan_db");
    sql("drop user if exists test_admin_user");
    DBHandlerTestFixture::TearDown();
  }
  // Drops a user while skipping the normal checks (like if the user owns a db).  Used to
  // create db states that are no longer valid used for legacy testsing.
  static void dropUserUnchecked(const std::string& user_name) {
    CHECK(!isDistributedMode()) << "Can't manipulate syscat directly in distributed mode";
    auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
    Catalog_Namespace::UserMetadata user;
    CHECK(sys_cat.getMetadataForUser(user_name, user));
    sys_cat.dropUserUnchecked(user_name, user);
  }
};

TEST_F(CreateDropDatabaseTest, OrphanedDB) {
  sql("create user test_admin_user (password = 'password', is_super = 'true')");
  login("test_admin_user", "password");
  sql("create database orphan_db");
  login("admin", "HyperInteractive", "orphan_db");
  sqlAndCompareResult("show databases",
                      {{"heavyai", "admin"},
                       {"information_schema", "admin"},
                       {"orphan_db", "test_admin_user"}});
  sql("create table temp (i int)");
  queryAndAssertException(
      "drop user test_admin_user",
      "Cannot drop user. User test_admin_user owns database orphan_db");
}

// We should no longer be able to generate an orphaned db, but in case we do, we should
// still be able to show it as a super-user.
TEST_F(CreateDropDatabaseTest, LegacyOrphanedDB) {
  if (isDistributedMode()) {
    GTEST_SKIP() << "Can not manipulate syscat directly in distributed mode.";
  }
  sql("create user test_admin_user (password = 'password', is_super = 'true')");
  login("test_admin_user", "password");
  sql("create database orphan_db");
  login("admin", "HyperInteractive", "orphan_db");
  sqlAndCompareResult("show databases",
                      {{"heavyai", "admin"},
                       {"information_schema", "admin"},
                       {"orphan_db", "test_admin_user"}});

  dropUserUnchecked("test_admin_user");

  sqlAndCompareResult("show databases",
                      {{"heavyai", "admin"},
                       {"information_schema", "admin"},
                       {"orphan_db", "<DELETED>"}});
}

class DatabaseCaseSensitiveTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() {
    switchToAdmin();
    sql("CREATE USER test_user (password = 'test_pass');");
    sql("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName + " TO test_user;");
  }

  static void TearDownTestSuite() {
    switchToAdmin();
    sql("DROP USER IF EXISTS test_user;");
  }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    switchToAdmin();
    sql("DROP DATABASE IF EXISTS test_db;");
    sql("CREATE DATABASE test_db;");
    login(shared::kRootUsername, shared::kDefaultRootPasswd, "test_db");
    sql("CREATE TABLE test_db_table (test_db_table_col INTEGER);");
    sql("GRANT ACCESS ON DATABASE test_db TO test_user;");
    switchToAdmin();
  }

  void TearDown() override {
    switchToAdmin();
    sql("DROP DATABASE IF EXISTS test_db;");
    sql("DROP DATABASE IF EXISTS test_db_2;");
    DBHandlerTestFixture::TearDown();
  }

  void assertDatabaseExists(const std::string& db_name) {
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    Catalog_Namespace::DBMetadata db_metadata;
    ASSERT_TRUE(sys_catalog.getMetadataForDB(db_name, db_metadata));
  }

  void assertDatabaseDoesNotExist(const std::string& db_name) {
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    Catalog_Namespace::DBMetadata db_metadata;
    ASSERT_FALSE(sys_catalog.getMetadataForDB(db_name, db_metadata));
  }

  void assertDatabaseOwner(const std::string& db_name, const std::string& owner) {
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    Catalog_Namespace::UserMetadata user_metadata;
    ASSERT_TRUE(sys_catalog.getMetadataForUser(owner, user_metadata));

    Catalog_Namespace::DBMetadata db_metadata;
    ASSERT_TRUE(sys_catalog.getMetadataForDB(db_name, db_metadata));

    ASSERT_EQ(user_metadata.userId, db_metadata.dbOwner);
  }
};

TEST_F(DatabaseCaseSensitiveTest, Create) {
  queryAndAssertException("CREATE DATABASE TEST_db;", "Database TEST_db already exists.");
}

TEST_F(DatabaseCaseSensitiveTest, Drop) {
  sql("DROP DATABASE test_DB;");
  assertDatabaseDoesNotExist("test_db");
}

TEST_F(DatabaseCaseSensitiveTest, Rename) {
  sql("ALTER DATABASE TEST_DB RENAME TO test_DB_2;");
  assertDatabaseDoesNotExist("TEST_DB");
  assertDatabaseExists("test_db_2");
}

TEST_F(DatabaseCaseSensitiveTest, ChangeOwner) {
  sql("ALTER DATABASE TEST_db OWNER TO test_user;");
  assertDatabaseOwner("test_db", "test_user");
}

TEST_F(DatabaseCaseSensitiveTest, Grant) {
  login("test_user", "test_pass", "TEST_dB");
  queryAndAssertException(
      "CREATE TABLE test_table (i INTEGER);",
      "Table test_table will not be created. User has no create privileges.");

  switchToAdmin();
  sql("GRANT CREATE TABLE ON DATABASE test_DB TO test_user;");

  login("test_user", "test_pass", "TEST_dB");
  sql("CREATE TABLE test_table (i INTEGER);");
}

TEST_F(DatabaseCaseSensitiveTest, Revoke) {
  sql("GRANT CREATE TABLE ON DATABASE test_db TO test_user;");

  login("test_user", "test_pass", "test_DB");
  sql("CREATE TABLE test_table (i INTEGER);");

  switchToAdmin();
  sql("REVOKE CREATE TABLE ON DATABASE TEST_db FROM test_user;");

  login("test_user", "test_pass", "test_DB");
  queryAndAssertException(
      "CREATE TABLE test_table_2 (i INTEGER);",
      "Table test_table_2 will not be created. User has no create privileges.");
}

TEST_F(DatabaseCaseSensitiveTest, SwitchDatabase) {
  login("test_user", "test_pass");
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  TSessionInfo session_info;
  db_handler->get_session_info(session_info, session_id);
  ASSERT_EQ(session_info.database, shared::kDefaultDbName);

  db_handler->switch_database(session_id, "TEST_db");

  db_handler->get_session_info(session_info, session_id);
  ASSERT_EQ(session_info.database, "test_db");
}

TEST_F(DatabaseCaseSensitiveTest, GetTablesForDatabase) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  std::vector<std::string> table_names;
  db_handler->get_tables_for_database(table_names, session_id, "TEST_db");
  ASSERT_EQ(table_names, std::vector<std::string>{"test_db_table"});
}

TEST_F(DatabaseCaseSensitiveTest, GetTableDetailsForDatabase) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  TTableDetails table_details;
  db_handler->get_table_details_for_database(
      table_details, session_id, "test_db_table", "TEST_db");
  ASSERT_EQ(table_details.row_desc.size(), size_t(1));
  ASSERT_EQ(table_details.row_desc[0].col_name, "test_db_table_col");
}

TEST_F(DatabaseCaseSensitiveTest, GetInternalTableDetailsForDatabase) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  TTableDetails table_details;
  db_handler->get_internal_table_details_for_database(
      table_details, session_id, "test_db_table", "TEST_DB");
  ASSERT_EQ(table_details.row_desc.size(), size_t(2));
  ASSERT_EQ(table_details.row_desc[0].col_name, "test_db_table_col");
  ASSERT_EQ(table_details.row_desc[1].col_name, "rowid");
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()("test-help",
                     "Print all DBObjectPrivilegesTest specific options (for gtest "
                     "options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: DBObjectPrivilegesTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  logger::init(log_options);
  DBHandlerTestFixture::createDBHandler();
  QR::init(BASE_PATH);

  g_calcite = QR::get()->getCatalog()->getCalciteMgr();

  // get dirname of test binary
  g_test_binary_file_path = boost::filesystem::canonical(argv[0]).parent_path().string();

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
