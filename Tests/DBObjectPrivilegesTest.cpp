#include <gtest/gtest.h>
#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <thread>
#include <tuple>
#include "../Catalog/Catalog.h"
#include "../Catalog/DBObject.h"
#include "../DataMgr/DataMgr.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ArrowResultSet.h"
#include "../QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "../QueryEngine/Execute.h"
#include "../QueryRunner/QueryRunner.h"
#include "Shared/MapDParameters.h"
#include "Shared/scope.h"
#include "TestHelpers.h"
#include "ThriftHandler/QueryState.h"
#include "gen-cpp/CalciteServer.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;
namespace {
std::shared_ptr<Calcite> g_calcite;

Catalog_Namespace::UserMetadata user;
std::vector<DBObject> privObjects;

auto& sys_cat = Catalog_Namespace::SysCatalog::instance();

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

inline void run_ddl_statement(const std::string& query) {
  QR::get()->runDDLStatement(query);
}

}  // namespace

struct Users {
  void setup_users() {
    if (!sys_cat.getMetadataForUser("Chelsea", user)) {
      sys_cat.createUser("Chelsea", "password", true, "");
      CHECK(sys_cat.getMetadataForUser("Chelsea", user));
    }
    if (!sys_cat.getMetadataForUser("Arsenal", user)) {
      sys_cat.createUser("Arsenal", "password", false, "");
      CHECK(sys_cat.getMetadataForUser("Arsenal", user));
    }
    if (!sys_cat.getMetadataForUser("Juventus", user)) {
      sys_cat.createUser("Juventus", "password", false, "");
      CHECK(sys_cat.getMetadataForUser("Juventus", user));
    }
    if (!sys_cat.getMetadataForUser("Bayern", user)) {
      sys_cat.createUser("Bayern", "password", false, "");
      CHECK(sys_cat.getMetadataForUser("Bayern", user));
    }
  }
  void drop_users() {
    if (sys_cat.getMetadataForUser("Chelsea", user)) {
      sys_cat.dropUser("Chelsea");
      CHECK(!sys_cat.getMetadataForUser("Chelsea", user));
    }
    if (sys_cat.getMetadataForUser("Arsenal", user)) {
      sys_cat.dropUser("Arsenal");
      CHECK(!sys_cat.getMetadataForUser("Arsenal", user));
    }
    if (sys_cat.getMetadataForUser("Juventus", user)) {
      sys_cat.dropUser("Juventus");
      CHECK(!sys_cat.getMetadataForUser("Juventus", user));
    }
    if (sys_cat.getMetadataForUser("Bayern", user)) {
      sys_cat.dropUser("Bayern");
      CHECK(!sys_cat.getMetadataForUser("Bayern", user));
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

  void setup_tables() { run_ddl_statement("CREATE TABLE IF NOT EXISTS tbl(i INTEGER)"); }
  void drop_tables() { run_ddl_statement("DROP TABLE IF EXISTS tbl"); }
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

struct ViewObject : testing::Test {
  void setup_objects() {
    run_ddl_statement("CREATE USER bob (password = 'password', is_super = 'false');");
    run_ddl_statement("CREATE ROLE salesDept;");
    run_ddl_statement("CREATE USER foo (password = 'password', is_super = 'false');");
    run_ddl_statement("GRANT salesDept TO foo;");

    run_ddl_statement("CREATE TABLE bill_table(id integer);");
    run_ddl_statement("CREATE VIEW bill_view AS SELECT id FROM bill_table;");
    run_ddl_statement("CREATE VIEW bill_view_outer AS SELECT id FROM bill_view;");
  }

  void remove_objects() {
    run_ddl_statement("DROP VIEW bill_view_outer;");
    run_ddl_statement("DROP VIEW bill_view;");
    run_ddl_statement("DROP TABLE bill_table");

    run_ddl_statement("DROP USER foo;");
    run_ddl_statement("DROP ROLE salesDept;");
    run_ddl_statement("DROP USER bob;");
  }
  explicit ViewObject() { setup_objects(); }
  ~ViewObject() override { remove_objects(); }
};

struct DashboardObject : testing::Test {
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
      cat.deleteMetadataForDashboard(id);
    }
  }

  explicit DashboardObject() {
    drop_dashboards();
    setup_dashboards();
  }

  ~DashboardObject() override { drop_dashboards(); }
};

TEST_F(GrantSyntax, MultiPrivilegeGrantRevoke) {
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

TEST(UserRoles, InvalidGrantsRevokesTest) {
  run_ddl_statement("CREATE USER Antazin(password = 'password', is_super = 'false');");
  run_ddl_statement("CREATE USER Max(password = 'password', is_super = 'false');");

  EXPECT_THROW(run_ddl_statement("GRANT Antazin to Antazin;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("REVOKE Antazin from Antazin;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("GRANT Antazin to Max;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("REVOKE Antazin from Max;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("GRANT Max to Antazin;"), std::runtime_error);
  EXPECT_THROW(run_ddl_statement("REVOKE Max from Antazin;"), std::runtime_error);

  run_ddl_statement("DROP USER Antazin;");
  run_ddl_statement("DROP USER Max;");
}

TEST(UserRoles, ValidNames) {
  EXPECT_NO_THROW(
      run_ddl_statement("CREATE USER \"dumm.user\" (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER \"dumm.user\";"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE USER vasya (password = 'password');"));
  EXPECT_NO_THROW(
      run_ddl_statement("CREATE USER vasya.vasya@vasya.com (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement(
      "CREATE USER \"vasya ivanov\"@vasya.ivanov.com (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE USER vasya-vasya (password = 'password');"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE ROLE developer;"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE ROLE developer-backend;"));
  EXPECT_NO_THROW(run_ddl_statement("CREATE ROLE developer-backend-rendering;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT developer-backend-rendering TO vasya;"));
  EXPECT_NO_THROW(
      run_ddl_statement("GRANT developer-backend TO \"vasya ivanov\"@vasya.ivanov.com;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT developer TO vasya.vasya@vasya.com;"));
  EXPECT_NO_THROW(run_ddl_statement("GRANT developer-backend-rendering TO vasya-vasya;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER vasya;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER vasya.vasya@vasya.com;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP USER \"vasya ivanov\"@vasya.ivanov.com;"));
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
  auto cat_mapd = Catalog_Namespace::Catalog::get(OMNISCI_DEFAULT_DB);
  DBObject mapd_object(OMNISCI_DEFAULT_DB, DBObjectType::DatabaseDBObjectType);
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

  CHECK(sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, db_meta));
  CHECK(sys_cat.getMetadataForUser("Juventus", user_meta));
  session_juve.reset(
      new Catalog_Namespace::SessionInfo(Catalog_Namespace::Catalog::get(db_meta.dbName),
                                         user_meta,
                                         ExecutorDeviceType::GPU,
                                         ""));
  auto& cat_mapd = session_juve->getCatalog();
  DBObject mapd_object(OMNISCI_DEFAULT_DB, DBObjectType::DatabaseDBObjectType);
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

  CHECK(sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, db_meta));
  CHECK(sys_cat.getMetadataForUser("Bayern", user_meta));
  session_juve.reset(
      new Catalog_Namespace::SessionInfo(Catalog_Namespace::Catalog::get(db_meta.dbName),
                                         user_meta,
                                         ExecutorDeviceType::GPU,
                                         ""));
  auto& cat_mapd = session_juve->getCatalog();
  DBObject mapd_object(OMNISCI_DEFAULT_DB, DBObjectType::DatabaseDBObjectType);
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

  CHECK(sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, db_meta));
  CHECK(sys_cat.getMetadataForUser("Arsenal", user_meta));
  session_ars.reset(
      new Catalog_Namespace::SessionInfo(Catalog_Namespace::Catalog::get(db_meta.dbName),
                                         user_meta,
                                         ExecutorDeviceType::GPU,
                                         ""));
  auto& cat_mapd = session_ars->getCatalog();
  AccessPrivileges arsenal_privs;
  AccessPrivileges bayern_privs;
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::CREATE_TABLE));
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::DROP_TABLE));
  ASSERT_NO_THROW(bayern_privs.add(AccessPrivileges::ALTER_TABLE));
  DBObject mapd_object(OMNISCI_DEFAULT_DB, DBObjectType::DatabaseDBObjectType);
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
  CHECK(sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, db_meta));
  CHECK(sys_cat.getMetadataForUser("Arsenal", user_meta));
  session_ars.reset(
      new Catalog_Namespace::SessionInfo(Catalog_Namespace::Catalog::get(db_meta.dbName),
                                         user_meta,
                                         ExecutorDeviceType::GPU,
                                         ""));
  auto& cat_mapd = session_ars->getCatalog();
  AccessPrivileges arsenal_privs;
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::ALL_VIEW));
  DBObject mapd_object(OMNISCI_DEFAULT_DB, DBObjectType::DatabaseDBObjectType);
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
  CHECK(sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, db_meta));
  CHECK(sys_cat.getMetadataForUser("Arsenal", user_meta));
  session_ars.reset(
      new Catalog_Namespace::SessionInfo(Catalog_Namespace::Catalog::get(db_meta.dbName),
                                         user_meta,
                                         ExecutorDeviceType::GPU,
                                         ""));
  auto& cat_mapd = session_ars->getCatalog();
  AccessPrivileges arsenal_privs;
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::ALL_DASHBOARD));
  DBObject mapd_object(OMNISCI_DEFAULT_DB, DBObjectType::DatabaseDBObjectType);
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

  CHECK(sys_cat.getMetadataForDB(OMNISCI_DEFAULT_DB, db_meta));
  CHECK(sys_cat.getMetadataForUser("Arsenal", user_meta));
  session_ars.reset(
      new Catalog_Namespace::SessionInfo(Catalog_Namespace::Catalog::get(db_meta.dbName),
                                         user_meta,
                                         ExecutorDeviceType::GPU,
                                         ""));
  auto& cat_mapd = session_ars->getCatalog();
  AccessPrivileges arsenal_privs;
  ASSERT_NO_THROW(arsenal_privs.add(AccessPrivileges::ALL_DATABASE));
  DBObject mapd_object(OMNISCI_DEFAULT_DB, DBObjectType::DatabaseDBObjectType);
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
  testViewPermissions("bob", "bob");
}

TEST_F(ViewObject, GroupRoleFooGetsGrants) {
  testViewPermissions("foo", "salesDept");
}

TEST_F(ViewObject, CalciteViewResolution) {
  auto query_state1 =
      QR::create_query_state(QR::get()->getSession(), "select * from bill_table");
  TPlanResult result = ::g_calcite->process(query_state1->createQueryStateProxy(),
                                            query_state1->getQueryStr(),
                                            {},
                                            true,
                                            false,
                                            false);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.primary_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from[0], "bill_table");
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.resolved_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from[0], "bill_table");

  auto query_state2 =
      QR::create_query_state(QR::get()->getSession(), "select * from bill_view");
  result = ::g_calcite->process(query_state2->createQueryStateProxy(),
                                query_state2->getQueryStr(),
                                {},
                                true,
                                false,
                                false);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.primary_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from[0], "bill_view");
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.resolved_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from[0], "bill_table");

  auto query_state3 =
      QR::create_query_state(QR::get()->getSession(), "select * from bill_view_outer");
  result = ::g_calcite->process(query_state3->createQueryStateProxy(),
                                query_state3->getQueryStr(),
                                {},
                                true,
                                false,
                                false);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.primary_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.primary_accessed_objects.tables_selected_from[0], "bill_view_outer");
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from.size(), (size_t)1);
  EXPECT_EQ(result.resolved_accessed_objects.tables_inserted_into.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_updated_in.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_deleted_from.size(), (size_t)0);
  EXPECT_EQ(result.resolved_accessed_objects.tables_selected_from[0], "bill_table");
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

    cat.deleteMetadataForDashboard(std::to_string(session->get_currentUser().userId),
                                   name);
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
  auto cat = Catalog_Namespace::Catalog::get(OMNISCI_DEFAULT_DB);

  DBObject dbo1(OMNISCI_DEFAULT_DB, DBObjectType::DatabaseDBObjectType);
  DBObject dbo2(tbname, DBObjectType::TableDBObjectType);
  DBObject dbo3(vwname, DBObjectType::ViewDBObjectType);

  ASSERT_NO_THROW(dbo1.loadKey());
  ASSERT_NO_THROW(dbo2.loadKey(*cat));
  ASSERT_NO_THROW(dbo3.loadKey(*cat));
}

TEST(SysCatalog, RenameUser_Basic) {
  using namespace std::string_literals;
  auto username = "chuck"s;
  auto database_name = "nydb"s;
  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    if (rename_successful) {
      run_ddl_statement("DROP USER cryingchuck;");
    } else {
      run_ddl_statement("DROP USER chuck");
    }
    run_ddl_statement("DROP DATABASE nydb;");
  };

  Catalog_Namespace::UserMetadata user_meta;
  auto username_out(username);
  auto database_out(database_name);

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
      Catalog_Namespace::Catalog::get(user_name),
      user_metadata,
      ExecutorDeviceType::CPU,
      "");
  return std::make_unique<QR>(std::move(session));
}

}  // namespace

TEST(SysCatalog, RenameUser_AlreadyLoggedInQueryAfterRename) {
  using namespace std::string_literals;
  auto username = "chuck"s;
  auto database_name = "nydb"s;
  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    if (rename_successful) {
      run_ddl_statement("DROP USER cryingchuck;");
    } else {
      run_ddl_statement("DROP USER chuck");
    }
    run_ddl_statement("DROP DATABASE nydb;");
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
  using namespace std::string_literals;
  auto username = "chuck"s;
  auto database_name = "nydb"s;
  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    if (rename_successful) {
      run_ddl_statement("DROP USER cryingchuck;");
    } else {
      run_ddl_statement("DROP USER chuck");
    }
    run_ddl_statement("DROP DATABASE nydb;");
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
      Catalog_Namespace::Catalog::get("nydb"s), user_meta2, ExecutorDeviceType::CPU, "");
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
  using namespace std::string_literals;
  auto rename_successful = false;

  ScopeGuard s = [&rename_successful] {
    run_ddl_statement("DROP USER rom;");

    if (rename_successful) {
      run_ddl_statement("DROP USER renamed_quark;");
    } else {
      run_ddl_statement("DROP USER quark;");
    }
    run_ddl_statement("DROP DATABASE Ferengi;");
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
      Catalog_Namespace::Catalog::get("Ferengi"s),
      user_meta2,
      ExecutorDeviceType::CPU,
      "");
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
      Catalog_Namespace::Catalog::get("Ferengi"s),
      user_meta3,
      ExecutorDeviceType::CPU,
      "");
  auto renamed_quark_qr = std::make_unique<QR>(std::move(renamed_quark_session));

  EXPECT_NO_THROW(renamed_quark_qr->runSQL("select * from bank_account;", dt));
  EXPECT_NO_THROW(renamed_quark_qr->runSQL("select * from riches;", dt));
}

TEST(SysCatalog, RenameUser_SuperUserRenameCheck) {
  using namespace std::string_literals;

  ScopeGuard s = [] {
    run_ddl_statement("DROP USER rom;");
    run_ddl_statement("DROP USER quark;");

    run_ddl_statement("DROP DATABASE Ferengi;");
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
      Catalog_Namespace::Catalog::get("Ferengi"s),
      user_meta2,
      ExecutorDeviceType::CPU,
      "");
  auto rom_qr = std::make_unique<QR>(std::move(rom_session));
  EXPECT_THROW(rom_qr->runDDLStatement("ALTER USER quark RENAME TO renamed_quark;"),
               std::runtime_error);
}

TEST(SysCatalog, RenameDatabase_Basic) {
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
  using namespace std::string_literals;
  auto username = "reader"s;
  auto database_name = "fnews"s;

  ScopeGuard scope_gard = [] {
    run_ddl_statement("DROP USER reader;");
    run_ddl_statement("DROP USER jkyle;");

    run_ddl_statement("DROP DATABASE qworg;");
    run_ddl_statement("DROP DATABASE fnews;");
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
  using namespace std::string_literals;
  auto username = "maurypovich"s;
  auto database_name = "paternitydb"s;

  auto rename_successful = false;

  ScopeGuard scope_guard = [&rename_successful] {
    run_ddl_statement("DROP USER maurypovich;");
    run_ddl_statement("DROP USER thefather;");
    run_ddl_statement("DROP DATABASE trouble;");
    if (rename_successful) {
      run_ddl_statement("DROP DATABASE nachovater;");
    } else {
      run_ddl_statement("DROP DATABASE paternitydb;");
    }
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
  auto trash_file_path = sys_cat.getBasePath() + "/mapd_catalogs/trash";

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
  using namespace std::string_literals;
  auto rename_successful = false;

  ScopeGuard s = [&rename_successful] {
    run_ddl_statement("DROP USER quark;");
    run_ddl_statement("DROP USER rom;");

    if (rename_successful) {
      run_ddl_statement("DROP DATABASE grandnagus;");
    } else {
      run_ddl_statement("DROP DATABASE Ferengi;");
    }
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
      run_ddl_statement("DROP USER " + username2 + ";");
      run_ddl_statement("DROP USER " + username + ";");
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
      run_ddl_statement("DROP USER " + username + ";");
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
            OMNISCI_DEFAULT_DB);  // correctly fell back to system default database
}

TEST(SysCatalog, SwitchDatabase) {
  static const std::string username{"test_user"};
  static std::string dbname{"test_db"};
  static std::string dbname2{dbname + "2"};
  static std::string dbname3{dbname + "3"};

  // cleanup
  struct CleanupGuard {
    ~CleanupGuard() {
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbname3 + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbname2 + ";");
      run_ddl_statement("DROP DATABASE IF EXISTS " + dbname + ";");
      run_ddl_statement("DROP USER " + username + ";");
    }
  } cleanupGuard;

  // setup
  run_ddl_statement("CREATE USER " + username + " (password = 'password');");
  run_ddl_statement("CREATE DATABASE " + dbname + "(owner='" + username + "');");
  run_ddl_statement("CREATE DATABASE " + dbname2 + "(owner='" + username + "');");
  run_ddl_statement("CREATE DATABASE " + dbname3 + "(owner='" + username + "');");
  run_ddl_statement("REVOKE ACCESS ON DATABASE " + dbname3 + " FROM " + username + ";");

  // test some attempts to switch database
  ASSERT_NO_THROW(sys_cat.switchDatabase(dbname, username));
  ASSERT_NO_THROW(sys_cat.switchDatabase(dbname, username));
  ASSERT_NO_THROW(sys_cat.switchDatabase(dbname2, username));
  ASSERT_THROW(sys_cat.switchDatabase(dbname3, username), std::runtime_error);
}

namespace {

void compare_user_lists(const std::vector<std::string>& expected,
                        const std::list<Catalog_Namespace::UserMetadata>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  size_t i = 0;
  for (const auto user : actual) {
    ASSERT_EQ(expected[i++], user.userName);
  }
}

}  // namespace

TEST(SysCatalog, AllUserMetaTest) {
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

  run_ddl_statement("DROP DATABASE IF EXISTS " + champions + ";");
  run_ddl_statement("DROP DATABASE IF EXISTS " + europa + ";");
  run_ddl_statement("CREATE DATABASE " + champions + ";");
  run_ddl_statement("CREATE DATABASE " + europa + ";");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + champions + " TO Bayern;");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + champions + " TO Juventus;");
  run_ddl_statement("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO Arsenal;");
  run_ddl_statement("GRANT CREATE ON DATABASE " + champions + " TO Juventus;");
  run_ddl_statement("GRANT SELECT ON DATABASE " + europa + " TO Arsenal;");
  run_ddl_statement("GRANT CREATE ON DATABASE " + OMNISCI_DEFAULT_DB + " TO Bayern;");
  run_ddl_statement("GRANT SELECT ON DATABASE " + europa + " TO Juventus;");

  Catalog_Namespace::UserMetadata user_meta;
  auto db_default(OMNISCI_DEFAULT_DB);
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
      run_ddl_statement("DROP ROLE " + london + ";");
      run_ddl_statement("DROP ROLE " + north_london + ";");
      run_ddl_statement("DROP ROLE " + munich + ";");
      run_ddl_statement("DROP ROLE " + turin + ";");
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
  run_ddl_statement("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO Gunners;");
  run_ddl_statement("GRANT CREATE ON DATABASE " + champions + " TO OldLady;");
  run_ddl_statement("GRANT SELECT ON DATABASE " + europa + " TO Gunners;");
  run_ddl_statement("GRANT CREATE ON DATABASE " + OMNISCI_DEFAULT_DB + " TO Sudens;");
  run_ddl_statement("GRANT SELECT ON DATABASE " + europa + " TO OldLady;");

  Catalog_Namespace::UserMetadata user_meta;
  auto db_default(OMNISCI_DEFAULT_DB);
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

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH);

  g_calcite = QR::get()->getCatalog()->getCalciteMgr();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
