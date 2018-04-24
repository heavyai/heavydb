#include "../Parser/parser.h"
#include "../Catalog/Catalog.h"
#include "../Catalog/DBObject.h"
#include "../DataMgr/DataMgr.h"
#include <boost/filesystem/operations.hpp>
#include <gtest/gtest.h>
#include <glog/logging.h>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 39093

namespace {
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
    auto calcite = std::make_shared<Calcite>(-1, CALCITEPORT, base_path.string(), 1024);
    {
      auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
      CHECK(boost::filesystem::exists(system_db_file));
      sys_cat.init(base_path.string(), dataMgr, {}, calcite, false, true);
      CHECK(sys_cat.getMetadataForDB(db_name, db));
      auto cat = Catalog_Namespace::Catalog::get(db_name);
      if (cat == nullptr) {
        cat = std::make_shared<Catalog_Namespace::Catalog>(
            base_path.string(), db, dataMgr, std::vector<LeafHostInfo>{}, calcite);
        Catalog_Namespace::Catalog::set(db_name, cat);
      }
      CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));
    }
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
    g_session.reset(
        new Catalog_Namespace::SessionInfo(std::make_shared<Catalog_Namespace::Catalog>(
                                               base_path.string(), db, dataMgr, std::vector<LeafHostInfo>{}, calcite),
                                           user,
                                           ExecutorDeviceType::GPU,
                                           ""));
  }
};

void run_ddl(const std::string& query) {
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  CHECK_EQ(parser.parse(query, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
  CHECK(ddl != nullptr);
  ddl->execute(*g_session);
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
    run_ddl(cquery1);
    run_ddl(cquery2);
    run_ddl(cquery3);
  }
  void drop_tables() {
    run_ddl(dquery1);
    run_ddl(dquery2);
    run_ddl(dquery3);
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

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  testing::AddGlobalTestEnvironment(new DBObjectPermissionsEnv);
  return RUN_ALL_TESTS();
}
