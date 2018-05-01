#include "../Parser/parser.h"
#include "../Catalog/Catalog.h"
#include "../Catalog/DBObject.h"
#include "../DataMgr/DataMgr.h"
#include "../QueryRunner/QueryRunner.h"
#include <boost/filesystem/operations.hpp>
#include <gtest/gtest.h>
#include <glog/logging.h>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 39093

namespace {
std::shared_ptr<Calcite> g_calcite;
std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;
Catalog_Namespace::SysCatalog& sys_cat = Catalog_Namespace::SysCatalog::instance();
;
Catalog_Namespace::UserMetadata user;
Catalog_Namespace::DBMetadata db;
std::vector<DBObject> privObjects;

class DBObjectPermissionsEnv : public ::testing::Environment {
 public:
  virtual void SetUp() {
    std::string db_name{MAPD_SYSTEM_DB};
    std::string user_name{MAPD_ROOT_USER};
    boost::filesystem::path base_path{BASE_PATH};
    if (!boost::filesystem::exists(base_path)) {
      boost::filesystem::create_directory(base_path);
    }
    CHECK(boost::filesystem::exists(base_path));
    auto system_db_file = base_path / "mapd_catalogs" / "mapd";
    auto data_dir = base_path / "mapd_data";
    g_calcite = std::make_shared<Calcite>(-1, CALCITEPORT, base_path.string(), 1024);
    {
      auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
      CHECK(boost::filesystem::exists(system_db_file));
      sys_cat.init(base_path.string(), dataMgr, {}, g_calcite, false, true);
      CHECK(sys_cat.getMetadataForDB(db_name, db));
      auto cat = Catalog_Namespace::Catalog::get(db_name);
      if (cat == nullptr) {
        cat = std::make_shared<Catalog_Namespace::Catalog>(
            base_path.string(), db, dataMgr, std::vector<LeafHostInfo>{}, g_calcite);
        Catalog_Namespace::Catalog::set(db_name, cat);
      }
      CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));
    }
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
    g_session.reset(
        new Catalog_Namespace::SessionInfo(std::make_shared<Catalog_Namespace::Catalog>(
                                               base_path.string(), db, dataMgr, std::vector<LeafHostInfo>{}, g_calcite),
                                           user,
                                           ExecutorDeviceType::GPU,
                                           ""));
  }
};

inline void run_ddl_statement(const std::string& query) {
  QueryRunner::run_ddl_statement(query, g_session);
}
}  // namespace

struct Users {
  void setup_users() {
    if (!sys_cat.getMetadataForUser("Chelsea", user)) {
      sys_cat.createUser("Chelsea", "password", true);
      CHECK(sys_cat.getMetadataForUser("Chelsea", user));
    }
    if (!sys_cat.getMetadataForUser("Arsenal", user)) {
      sys_cat.createUser("Arsenal", "password", false);
      CHECK(sys_cat.getMetadataForUser("Arsenal", user));
    }
    if (!sys_cat.getMetadataForUser("Juventus", user)) {
      sys_cat.createUser("Juventus", "password", false);
      CHECK(sys_cat.getMetadataForUser("Juventus", user));
    }
    if (!sys_cat.getMetadataForUser("Bayern", user)) {
      sys_cat.createUser("Bayern", "password", false);
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
  explicit Users() { setup_users(); }
  virtual ~Users() { drop_users(); }
};
struct Roles {
  void setup_roles() {
    if (!sys_cat.getMetadataForRole("OldLady")) {
      sys_cat.createRole("OldLady", false);
      CHECK(sys_cat.getMetadataForRole("OldLady"));
    }
    if (!sys_cat.getMetadataForRole("Gunners")) {
      sys_cat.createRole("Gunners", false);
      CHECK(sys_cat.getMetadataForRole("Gunners"));
    }
    if (!sys_cat.getMetadataForRole("Sudens")) {
      sys_cat.createRole("Sudens", false);
      CHECK(sys_cat.getMetadataForRole("Sudens"));
    }
  }

  void drop_roles() {
    if (sys_cat.getMetadataForRole("OldLady")) {
      sys_cat.dropRole("OldLady");
      CHECK(!sys_cat.getMetadataForRole("OldLady"));
    }
    if (sys_cat.getMetadataForRole("Gunners")) {
      sys_cat.dropRole("Gunners");
      CHECK(!sys_cat.getMetadataForRole("Gunners"));
    }
    if (sys_cat.getMetadataForRole("Sudens")) {
      sys_cat.dropRole("Sudens");
      CHECK(!sys_cat.getMetadataForRole("sudens"));
    }
  }
  explicit Roles() { setup_roles(); }
  virtual ~Roles() { drop_roles(); }
};
struct TableStruct {
  std::string cquery1 = "CREATE TABLE IF NOT EXISTS epl(gp SMALLINT, won SMALLINT);";
  std::string cquery2 = "CREATE TABLE IF NOT EXISTS seriea(gp SMALLINT, won SMALLINT);";
  std::string cquery3 = "CREATE TABLE IF NOT EXISTS bundesliga(gp SMALLINT, won SMALLINT);";
  std::string dquery1 = "DROP TABLE IF EXISTS epl;";
  std::string dquery2 = "DROP TABLE IF EXISTS seriea;";
  std::string dquery3 = "DROP TABLE IF EXISTS bundesliga;";

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
  Users* user;
  Roles* role;
  explicit TableStruct() {
    user = new Users();
    role = new Roles();
    drop_tables();
    setup_tables();
  }
  virtual ~TableStruct() {
    drop_tables();
    delete user;
    delete role;
  }
};

struct TableObject : testing::Test {
  TableStruct* table_;
  TableObject() { table_ = new TableStruct(); }
  virtual ~TableObject() { delete table_; }
};

struct ViewObject : testing::Test {
  ViewObject() {
    run_ddl_statement("CREATE USER bob (password = 'password', is_super = 'false');");
    run_ddl_statement("CREATE ROLE salesDept;");
    run_ddl_statement("CREATE USER foo (password = 'password', is_super = 'false');");
    run_ddl_statement("GRANT salesDept TO foo;");

    run_ddl_statement("CREATE TABLE bill_table(id integer);");
    run_ddl_statement("CREATE VIEW bill_view AS SELECT id FROM bill_table;");
    run_ddl_statement("CREATE VIEW bill_view_outer AS SELECT id FROM bill_view;");
  }
  virtual ~ViewObject() {
    run_ddl_statement("DROP VIEW bill_view_outer;");
    run_ddl_statement("DROP VIEW bill_view;");
    run_ddl_statement("DROP TABLE bill_table");

    run_ddl_statement("DROP USER foo;");
    run_ddl_statement("DROP ROLE salesDept;");
    run_ddl_statement("DROP USER bob;");
  }
};

struct DashboardStruct {
  std::string dname1 = "ChampionsLeague";
  std::string dname2 = "Europa";
  std::string dstate = "active";
  std::string dhash = "image00";
  std::string dmeta = "Chelsea are champions";
  int id;

  FrontendViewDescriptor vd1;
  void setup_dashboards() {
    auto& gcat = g_session->get_catalog();
    vd1.viewName = dname1;
    vd1.viewState = dstate;
    vd1.imageHash = dhash;
    vd1.viewMetadata = dmeta;
    vd1.userId = g_session->get_currentUser().userId;
    vd1.user = g_session->get_currentUser().userName;
    id = gcat.createFrontendView(vd1);
    sys_cat.createDBObject(g_session->get_currentUser(), dname1, DBObjectType::DashboardDBObjectType, gcat, id);
  }

  void drop_dashboards() {
    auto& gcat = g_session->get_catalog();
    if (gcat.getMetadataForDashboard(id)) {
      gcat.deleteMetadataForDashboard(id);
    }
  }
  Users* user;
  Roles* role;

  explicit DashboardStruct() {
    user = new Users();
    role = new Roles();
    drop_dashboards();
    setup_dashboards();
  }
  virtual ~DashboardStruct() {
    drop_dashboards();
    delete user;
    delete role;
  }
};

struct DashboardObject : testing::Test {
  DashboardStruct* dash_;
  DashboardObject() { dash_ = new DashboardStruct(); }
  virtual ~DashboardObject() { delete dash_; }
};

TEST_F(TableObject, AccessDefaultsTest) {
  auto& g_cat = g_session->get_catalog();
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
  epl_object.loadKey(g_cat);
  seriea_object.loadKey(g_cat);
  bundesliga_object.loadKey(g_cat);
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
  auto& g_cat = g_session->get_catalog();
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
  epl_object.loadKey(g_cat);
  seriea_object.loadKey(g_cat);
  bundesliga_object.loadKey(g_cat);
  ASSERT_NO_THROW(epl_object.setPrivileges(epl_privs));
  ASSERT_NO_THROW(seriea_object.setPrivileges(seriea_privs));
  ASSERT_NO_THROW(bundesliga_object.setPrivileges(bundesliga_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", epl_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Sudens", bundesliga_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("OldLady", seriea_object, g_cat));

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
  auto& g_cat = g_session->get_catalog();
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
  epl_object.loadKey(g_cat);
  seriea_object.loadKey(g_cat);
  bundesliga_object.loadKey(g_cat);
  ASSERT_NO_THROW(epl_object.setPrivileges(epl_privs));
  ASSERT_NO_THROW(seriea_object.setPrivileges(seriea_privs));
  ASSERT_NO_THROW(bundesliga_object.setPrivileges(bundesliga_privs));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Gunners", epl_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Bayern", bundesliga_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("OldLady", seriea_object, g_cat));

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
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Gunners", epl_object, g_cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Bayern", bundesliga_object, g_cat));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("OldLady", seriea_object, g_cat));

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
  auto& cat = g_session->get_catalog();
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
  TPlanResult result = ::g_calcite->process(*g_session, "select * from bill_table", true, false);
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

  result = ::g_calcite->process(*g_session, "select * from bill_view", true, false);
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

  result = ::g_calcite->process(*g_session, "select * from bill_view_outer", true, false);
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
  auto& g_cat = g_session->get_catalog();
  ASSERT_NO_THROW(sys_cat.grantRole("Gunners", "Bayern"));
  ASSERT_NO_THROW(sys_cat.grantRole("Sudens", "Arsenal"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(dash_->id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(g_cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);

  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

TEST_F(DashboardObject, AccessAfterGrantsTest) {
  auto& g_cat = g_session->get_catalog();
  ASSERT_NO_THROW(sys_cat.grantRole("Gunners", "Arsenal"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(dash_->id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(g_cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Gunners", dash_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Juventus", dash_object, g_cat));

  privObjects.clear();
  privObjects.push_back(dash_object);
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);
}

TEST_F(DashboardObject, AccessAfterRevokesTest) {
  auto& g_cat = g_session->get_catalog();
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  ASSERT_NO_THROW(sys_cat.grantRole("Sudens", "Bayern"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(dash_->id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(g_cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("OldLady", dash_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Arsenal", dash_object, g_cat));

  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("OldLady", dash_object, g_cat));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), false);

  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Arsenal", dash_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Bayern", dash_object, g_cat));
  EXPECT_EQ(sys_cat.checkPrivileges("Chelsea", privObjects), true);
  EXPECT_EQ(sys_cat.checkPrivileges("Arsenal", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Juventus", privObjects), false);
  EXPECT_EQ(sys_cat.checkPrivileges("Bayern", privObjects), true);
}

TEST_F(DashboardObject, GranteesDefaultListTest) {
  auto& g_cat = g_session->get_catalog();
  auto perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int size = static_cast<int>(perms_list.size());
  ASSERT_EQ(size, 0);
}

TEST_F(DashboardObject, GranteesListAfterGrantsTest) {
  auto& g_cat = g_session->get_catalog();
  auto perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int recs1 = static_cast<int>(perms_list.size());
  ASSERT_NO_THROW(sys_cat.grantRole("OldLady", "Juventus"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(dash_->id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(g_cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("OldLady", dash_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Bayern", dash_object, g_cat));
  perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int recs2 = static_cast<int>(perms_list.size());
  ASSERT_NE(recs1, recs2);
  ASSERT_EQ(recs2, 2);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_FALSE(perms_list[1]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::EDIT_DASHBOARD));
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Bayern", dash_object, g_cat));
  perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int recs3 = static_cast<int>(perms_list.size());
  ASSERT_EQ(recs3, 2);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));
}

TEST_F(DashboardObject, GranteesListAfterRevokesTest) {
  auto& g_cat = g_session->get_catalog();
  auto perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int recs1 = static_cast<int>(perms_list.size());
  ASSERT_NO_THROW(sys_cat.grantRole("Gunners", "Arsenal"));
  AccessPrivileges dash_priv;
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::EDIT_DASHBOARD));
  privObjects.clear();
  DBObject dash_object(dash_->id, DBObjectType::DashboardDBObjectType);
  dash_object.loadKey(g_cat);
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  privObjects.push_back(dash_object);
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Gunners", dash_object, g_cat));
  ASSERT_NO_THROW(sys_cat.grantDBObjectPrivileges("Chelsea", dash_object, g_cat));
  perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int recs2 = static_cast<int>(perms_list.size());
  ASSERT_NE(recs1, recs2);
  ASSERT_EQ(recs2, 2);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

  ASSERT_NO_THROW(dash_priv.remove(AccessPrivileges::VIEW_DASHBOARD));
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Gunners", dash_object, g_cat));
  perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int recs3 = static_cast<int>(perms_list.size());
  ASSERT_EQ(recs3, 2);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_FALSE(perms_list[0]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[1]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::VIEW_DASHBOARD));
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Gunners", dash_object, g_cat));
  perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int recs4 = static_cast<int>(perms_list.size());
  ASSERT_EQ(recs4, 1);
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
  ASSERT_TRUE(perms_list[0]->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

  ASSERT_NO_THROW(dash_priv.add(AccessPrivileges::EDIT_DASHBOARD));
  ASSERT_NO_THROW(dash_object.setPrivileges(dash_priv));
  ASSERT_NO_THROW(sys_cat.revokeDBObjectPrivileges("Chelsea", dash_object, g_cat));
  perms_list = sys_cat.getMetadataForObject(
      g_cat.get_currentDB().dbId, static_cast<int>(DBObjectType::DashboardDBObjectType), dash_->id);
  int recs5 = static_cast<int>(perms_list.size());
  ASSERT_EQ(recs1, recs5);
  ASSERT_EQ(recs5, 0);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  testing::AddGlobalTestEnvironment(new DBObjectPermissionsEnv);
  return RUN_ALL_TESTS();
}
