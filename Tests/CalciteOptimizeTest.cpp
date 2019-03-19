#include <glog/logging.h>
#include <gtest/gtest.h>
#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <thread>
#include <tuple>
#include "../Catalog/Catalog.h"
#include "../Catalog/DBObject.h"
#include "../DataMgr/DataMgr.h"
#include "../Parser/parser.h"
#include "../QueryRunner/QueryRunner.h"
#include "Shared/MapDParameters.h"
#include "gen-cpp/CalciteServer.h"

#ifdef BASE_PATH
constexpr char c_base_path[] = BASE_PATH;
#else
constexpr char c_base_path[] = "./tmp";
#endif

constexpr size_t c_calcite_port = 36279;

namespace {
std::shared_ptr<Calcite> g_calcite;
std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;
Catalog_Namespace::SysCatalog& sys_cat = Catalog_Namespace::SysCatalog::instance();
Catalog_Namespace::UserMetadata user;
Catalog_Namespace::DBMetadata db;
std::vector<DBObject> privObjects;

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

class CalciteOptimizeEnv : public ::testing::Environment {
 public:
  void SetUp() override {
    std::string db_name{MAPD_DEFAULT_DB};
    std::string user_name{MAPD_ROOT_USER};
    boost::filesystem::path base_path{c_base_path};
    if (!boost::filesystem::exists(base_path)) {
      boost::filesystem::create_directory(base_path);
    }
    CHECK(boost::filesystem::exists(base_path));
    auto system_db_file = base_path / "mapd_catalogs" / MAPD_DEFAULT_DB;
    auto data_dir = base_path / "mapd_data";

    register_signal_handler();
    google::InstallFailureFunction(&calcite_shutdown_handler);

    g_calcite = std::make_shared<Calcite>(-1, c_calcite_port, base_path.string(), 1024);
    {
      MapDParameters mapd_parms;
      auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(
          data_dir.string(), mapd_parms, false, 0);
      CHECK(boost::filesystem::exists(system_db_file));
      sys_cat.init(
          base_path.string(), dataMgr, {}, g_calcite, false, mapd_parms.aggregator, {});
      CHECK(sys_cat.getMetadataForDB(db_name, db));
      CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));
    }
    MapDParameters mapd_parms;
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(
        data_dir.string(), mapd_parms, false, 0);
    g_session.reset(new Catalog_Namespace::SessionInfo(
        Catalog_Namespace::Catalog::get(db_name), user, ExecutorDeviceType::GPU, ""));
  }
};

inline void run_ddl_statement(const std::string& query) {
  QueryRunner::run_ddl_statement(query, g_session);
}
}  // namespace

struct ViewObject : testing::Test {
  void setup_objects() {
    run_ddl_statement("CREATE TABLE table1(i1 integer, i2 integer);");
    run_ddl_statement("CREATE VIEW view_table1 AS SELECT i1, i2 FROM table1;");
    run_ddl_statement("CREATE VIEW view_view_table1 AS SELECT i1, i2 FROM view_table1;");
    run_ddl_statement("CREATE TABLE shape_table (block_group_id INT)");
    run_ddl_statement(
        "CREATE TABLE attribute_table( block_group_id INT, segment_name TEXT ENCODING "
        "DICT(8), segment_type TEXT ENCODING DICT(8), agg_column TEXT ENCODING DICT(8))");
    run_ddl_statement(
        "CREATE VIEW attribute_view AS select "
        "rowid,block_group_id,segment_name,segment_type,agg_column from attribute_table");
    run_ddl_statement(
        "CREATE VIEW shape_view AS select rowid, block_group_id from shape_table");
  }

  void remove_objects() {
    run_ddl_statement("DROP VIEW view_view_table1;");
    run_ddl_statement("DROP VIEW view_table1;");
    run_ddl_statement("DROP TABLE table1");
    run_ddl_statement("DROP VIEW attribute_view");
    run_ddl_statement("DROP VIEW shape_view");
    run_ddl_statement("DROP TABLE shape_table");
    run_ddl_statement("DROP TABLE attribute_table");
  }

  explicit ViewObject() { setup_objects(); }
  ~ViewObject() override { remove_objects(); }
};

TEST_F(ViewObject, BasicTest) {
  TPlanResult tresult =
      ::g_calcite->process(*g_session, "select i1 from table1", {}, true, false, false);

  TPlanResult vresult = ::g_calcite->process(
      *g_session, "select i1 from view_view_table1", {}, true, false, false);

  EXPECT_NE(vresult.plan_result, tresult.plan_result);

  TPlanResult ovresult = ::g_calcite->process(
      *g_session, "select i1 from view_view_table1", {}, true, false, true);

  EXPECT_EQ(ovresult.plan_result, tresult.plan_result);

  TPlanResult tab_result = ::g_calcite->process(
      *g_session,
      "SELECT shape_table.rowid FROM shape_table, attribute_table WHERE "
      "shape_table.block_group_id = attribute_table.block_group_id",
      {},
      true,
      false,
      true);
  TPlanResult view_result = ::g_calcite->process(
      *g_session,
      "SELECT shape_view.rowid FROM shape_view, attribute_view WHERE "
      "shape_view.block_group_id = attribute_view.block_group_id",
      {},
      true,
      false,
      true);
  EXPECT_EQ(tab_result.plan_result, view_result.plan_result);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  testing::AddGlobalTestEnvironment(new CalciteOptimizeEnv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
