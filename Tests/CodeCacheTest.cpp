#include "DBHandlerTestHelpers.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryRunner/QueryRunner.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern size_t g_code_cache_max_num_items;
extern size_t g_gpu_code_cache_max_size_in_bytes;
extern bool g_is_test_env;

using namespace TestHelpers;
using QR = QueryRunner::QueryRunner;

class BaseTestFixture : public DBHandlerTestFixture {
 protected:
  void SetUp() override {}

  void TearDown() override {}

  const char* table_schema = "(x int, y int);";

  void buildTable(const std::string& table_name) {
    sql("DROP TABLE IF EXISTS " + table_name + ";");
    sql("CREATE TABLE " + table_name + " " + table_schema);
    ValuesGenerator gen(table_name);
    for (size_t i = 0; i < 10; i++) {
      sql(gen(i, i * 10));
    }
  }
};

class GPUCodeCacheTest : public BaseTestFixture {
 protected:
  void SetUp() override {
    dt = TExecuteMode::type::GPU;
    try {
      if (!getCatalog().getDataMgr().gpusPresent()) {
        GTEST_SKIP() << "GPU not available. Skipping test.";
      }
      qe_instance = QueryEngine::getInstance();
      qe_instance->gpu_code_accessor->clear();
    } catch (...) {
      LOG(WARNING) << "Cannot get the QueryEngine instance, aborting.";
      std::rethrow_exception(std::current_exception());
    }
    BaseTestFixture::SetUp();
    buildTable("gpu_code_cache_t");
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS gpu_code_cache_t;");
    BaseTestFixture::TearDown();
  }

  TExecuteMode::type dt;
  std::shared_ptr<QueryEngine> qe_instance;
};

TEST_F(GPUCodeCacheTest, EvictionMetricType_ByteSize) {
  setExecuteMode(dt);
  ScopeGuard reset = [this] {
    qe_instance->gpu_code_accessor->resetCache(g_gpu_code_cache_max_size_in_bytes);
  };
  size_t const new_cache_size = 1000000;
  qe_instance->gpu_code_accessor->resetCache(new_cache_size);
  sql("SELECT x FROM gpu_code_cache_t WHERE y = 0;");
  EXPECT_LE(qe_instance->gpu_code_accessor->getCacheSize(), new_cache_size);
  auto const cache_metric1 = qe_instance->gpu_code_accessor->getCodeCacheMetric();
  sql("SELECT x FROM gpu_code_cache_t WHERE y = 0 AND x > 0;");
  EXPECT_LE(qe_instance->gpu_code_accessor->getCacheSize(), new_cache_size);
  auto const cache_metric2 = qe_instance->gpu_code_accessor->getCodeCacheMetric();
  EXPECT_LT(cache_metric1.evict_count, cache_metric2.evict_count);
}

TEST_F(GPUCodeCacheTest, DuplicatedQuery) {
  setExecuteMode(dt);
  sql("SELECT x FROM gpu_code_cache_t;");
  auto const sz1 = qe_instance->gpu_code_accessor->getCacheSize();
  sql("SELECT x FROM gpu_code_cache_t;");
  auto const sz2 = qe_instance->gpu_code_accessor->getCacheSize();
  EXPECT_EQ(sz1, sz2);
  sql("SELECT x,y FROM gpu_code_cache_t group by 1,2;");
  auto const sz3 = qe_instance->gpu_code_accessor->getCacheSize();
  EXPECT_LT(sz2, sz3);
}

class CPUCodeCacheTest : public BaseTestFixture {
 protected:
  void SetUp() override {
    dt = TExecuteMode::type::CPU;
    try {
      qe_instance = QueryEngine::getInstance();
      qe_instance->cpu_code_accessor->clear();
    } catch (...) {
      LOG(WARNING) << "Cannot get the QueryEngine instance, aborting.";
      std::rethrow_exception(std::current_exception());
    }
    BaseTestFixture::SetUp();
    buildTable("cpu_code_cache_t");
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS cpu_code_cache_t;");
    BaseTestFixture::TearDown();
  }

  TExecuteMode::type dt;
  std::shared_ptr<QueryEngine> qe_instance;
};

TEST_F(CPUCodeCacheTest, DuplicatedQuery) {
  setExecuteMode(dt);
  sql("SELECT x FROM cpu_code_cache_t;");
  auto const sz1 = qe_instance->cpu_code_accessor->getCacheSize();
  sql("SELECT x FROM cpu_code_cache_t;");
  auto const sz2 = qe_instance->cpu_code_accessor->getCacheSize();
  EXPECT_EQ(sz1, sz2);
  sql("SELECT x,y FROM cpu_code_cache_t group by 1,2;");
  auto const sz3 = qe_instance->cpu_code_accessor->getCacheSize();
  EXPECT_LT(sz2, sz3);
}

TEST_F(CPUCodeCacheTest, EvictionMetricType_EntryCount) {
  setExecuteMode(dt);
  ScopeGuard reset = [this] {
    qe_instance->cpu_code_accessor->resetCache(g_code_cache_max_num_items);
  };
  qe_instance->cpu_code_accessor->resetCache(1);
  sql("SELECT x FROM cpu_code_cache_t WHERE y = 0;");
  EXPECT_EQ(qe_instance->cpu_code_accessor->getCacheSize(), size_t(1));
  auto const cache_metric1 = qe_instance->cpu_code_accessor->getCodeCacheMetric();
  sql("SELECT x FROM cpu_code_cache_t WHERE y = 0 AND x > 0;");
  EXPECT_EQ(qe_instance->cpu_code_accessor->getCacheSize(), size_t(1));
  auto const cache_metric2 = qe_instance->cpu_code_accessor->getCodeCacheMetric();
  EXPECT_LT(cache_metric1.evict_count, cache_metric2.evict_count);
}

int main(int argc, char* argv[]) {
  g_is_test_env = true;
  ScopeGuard reset = [] { g_is_test_env = false; };
  TestHelpers::init_logger_stderr_only(argc, argv);
  namespace po = boost::program_options;
  po::options_description desc("Options");
  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(),
            vm);
  po::notify(vm);

  int err{0};
  try {
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
