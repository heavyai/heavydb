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

#include "DBHandlerTestHelpers.h"
#include "Logger/Logger.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/Execute.h"
#include "QueryRunner/QueryRunner.h"
#include "TestHelpers.h"
#include "gen-cpp/heavy_types.h"

#include <array>
#include <future>
#include <sstream>
#include <vector>

#ifdef TBB_PREVIEW_WAITING_FOR_WORKERS
#include <tbb/global_control.h>
#endif

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using NumExecutors = int;

bool g_keep_data{false};
NumExecutors g_max_num_executors{4};
size_t g_num_tables{25};

extern bool g_is_test_env;
extern bool g_enable_executor_resource_mgr;

using namespace TestHelpers;

#define SKIP_NO_GPU()                                        \
  if (skipTests(dt)) {                                       \
    CHECK(dt == TExecuteMode::type::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    return;                                                  \
  }

namespace {
// Return vector of powers of 2 up to max_num_executors.
std::vector<NumExecutors> num_executors_vec(NumExecutors const max_num_executors) {
  std::vector<NumExecutors> vec;
  for (NumExecutors i = 1; i <= max_num_executors; i *= 2) {
    vec.push_back(i);
  }
  return vec;
}

using Param = std::tuple<TExecuteMode::type, NumExecutors>;
enum ParamName { DT = 0, NUM_EXECUTORS };

// Return false on error
bool finalize_tbb_private_server_threads() {
#ifdef TBB_PREVIEW_WAITING_FOR_WORKERS  // set when ENABLE_TSAN
#if 2021006 <= TBB_VERSION_MAJOR * 1000 + TBB_VERSION_MINOR
  auto handle = oneapi::tbb::task_scheduler_handle{oneapi::tbb::attach{}};
#else
  auto handle = tbb::task_scheduler_handle::get();
#endif
  return tbb::finalize(handle, std::nothrow_t{});
#else
  return true;
#endif
}
}  // namespace

class BaseTestFixture : public DBHandlerTestFixture,
                        public ::testing::WithParamInterface<Param> {
 protected:
  Param param_;

  void SetUp() override {
    param_ = GetParam();
    Executor::nukeCacheOfExecutors();
    resizeDispatchQueue(std::get<NUM_EXECUTORS>(param_));
  }

  void TearDown() override { finalize_tbb_private_server_threads(); }

  bool skipTests(const TExecuteMode::type device_type) {
#ifdef HAVE_CUDA
    return device_type == TExecuteMode::type::GPU &&
           !(getCatalog().getDataMgr().gpusPresent());
#else
    return device_type == TExecuteMode::type::GPU;
#endif
  }

  const char* table_schema = R"(
    (
        i64 BIGINT,
        i32 INT,
        i16 SMALLINT,
        i8 TINYINT,
        d DOUBLE,
        f FLOAT,
        i1 BOOLEAN,
        str TEXT ENCODING DICT(32),
        arri64 BIGINT[]
    ) WITH (FRAGMENT_SIZE=2);
)";

  void runSqlExecuteTest(const std::string& table_name, const TExecuteMode::type dt) {
    auto check_returned_rows = [this](const std::string& query, const size_t num_rows) {
      TQueryResult result_set;
      sql(result_set, query);
      EXPECT_EQ(getRowCount(result_set), num_rows);
    };

    setExecuteMode(dt);

    // basic queries
    sqlAndCompareResult("SELECT COUNT(*) FROM " + table_name + " WHERE i32 < 50;",
                        {{i(15)}});
    sqlAndCompareResult("SELECT MIN(d) FROM " + table_name + " WHERE i1 IS NOT NULL;",
                        {{1.0001}});

    // Simple query with a sort
    sqlAndCompareResult("SELECT i64 FROM " + table_name + " ORDER BY i64 LIMIT 1;",
                        {{i(0)}});

    // complex queries
    check_returned_rows("SELECT d, f, COUNT(*) FROM " + table_name + " GROUP BY d, f;",
                        6);
    check_returned_rows("SELECT d, f, COUNT(*) FROM " + table_name +
                            " GROUP BY d, f ORDER BY f DESC NULLS LAST LIMIT 5;",
                        5);
    sqlAndCompareResult("SELECT COUNT(*) FROM " + table_name + " WHERE str like 'hello';",
                        {{i(10)}});
    sqlAndCompareResult(
        "SELECT COUNT(*) FROM " + table_name + " WHERE str ilike 'hello';", {{i(10)}});
    sqlAndCompareResult(
        "SELECT COUNT(*) FROM " + table_name + " WHERE str REGEXP '^[a-z]+r$';",
        {{i(0)}});
    check_returned_rows(
        "SELECT approx_count_distinct(d), approx_count_distinct(str), i64, i32, "
        "i16 FROM " +
            table_name + " WHERE i32 < 50 GROUP BY i64, i32, i16 ORDER BY i64, i32, i16;",
        5);

    // multi-step
    check_returned_rows(
        "SELECT d, f, COUNT(*) FROM " + table_name + " GROUP BY d, f HAVING d < f;", 5);

    // joins
    sqlAndCompareResult("SELECT COUNT(*) FROM " + table_name +
                            " a INNER JOIN (SELECT i32 FROM " + table_name +
                            " GROUP BY i32) b on a.i64 = b.i32;",
                        {{i(1)}});
  }

  void buildTable(const std::string& table_name) {
    sql("DROP TABLE IF EXISTS " + table_name + ";");

    sql("CREATE TABLE " + table_name + " " + table_schema);
    ValuesGenerator gen(table_name);
    for (size_t i = 0; i < 10; i++) {
      sql(gen(100, 10, 2, 1, 1.0001, 1.1, "'true'", "'hello'", "{100, 200}"));
    }
    for (size_t i = 0; i < 5; i++) {
      // Note - Windows template processing can't proccess an embedded
      // ternary operator.
      std::string row_alternate = (i % 2 == 0) ? "{NULL, 200}" : "{100, NULL}";
      sql(gen(500, 50, "NULL", 5, 5.0001, 5.1, "'false'", "'world'", row_alternate));
    }
    for (size_t i = 0; i < 5; i++) {
      sql(gen(100 * i,
              10 * i,
              2 * i,
              1 * i,
              1.0001 * static_cast<float>(i),
              1.1 * static_cast<float>(i),
              "NULL",
              "NULL",
              "{" + std::to_string(100 * i) + "," + std::to_string(200 * i) + "}"));
    }
  }
};

struct TestNameSuffix {
  // NOTE: test names must be non-empty, unique, and may only contain ASCII alphanumeric
  // characters. [Use of underscore must follow specific restrictions. See url for info.]
  // https://google.github.io/googletest/advanced.html#specifying-names-for-value-parameterized-test-parameters
  std::string operator()(
      ::testing::TestParamInfo<BaseTestFixture::ParamType> const& info) const {
    std::ostringstream oss;
    oss << std::get<DT>(info.param) << "_executors_"
        << std::get<NUM_EXECUTORS>(info.param);
    return oss.str();
  }
};

class SingleTableTestEnv : public BaseTestFixture {
 protected:
  void SetUp() override {
    BaseTestFixture::SetUp();
    buildTable("test_parallel");

    if (!skipTests(TExecuteMode::type::GPU)) {
      // warm up the PTX JIT
      setExecuteMode(TExecuteMode::type::GPU);
      sql("SELECT COUNT(*) FROM test_parallel;");
    }
  }

  void TearDown() override {
    if (!g_keep_data) {
      sql("DROP TABLE IF EXISTS test_parallel;");
    }
    BaseTestFixture::TearDown();
  }
};

TEST_P(SingleTableTestEnv, SingleTableTest) {
  auto const num_executors = std::get<NUM_EXECUTORS>(param_);
  TExecuteMode::type const dt = std::get<DT>(param_);
  {
    SKIP_NO_GPU();
    setExecuteMode(dt);
    std::vector<std::future<void>> worker_threads;
    auto execution_time = measure<>::execution([&]() {
      for (NumExecutors w = 0; w < num_executors; w++) {
        worker_threads.push_back(std::async(
            std::launch::async,
            [this](const std::string& table_name, const TExecuteMode::type dt) {
              runSqlExecuteTest(table_name, dt);
            },
            "test_parallel",
            dt));
      }
      for (auto& t : worker_threads) {
        t.get();
      }
    });
    LOG(ERROR) << "Finished execution with " << num_executors << " executors, "
               << execution_time << " ms.";
  }
}

class MultiTableTestEnv : public BaseTestFixture {
 protected:
  void SetUp() override {
    BaseTestFixture::SetUp();
    for (size_t i = 0; i < g_num_tables; i++) {
      buildTable("test_parallel_" + std::to_string(i));
    }

    if (!skipTests(TExecuteMode::type::GPU)) {
      // warm up the PTX JIT
      setExecuteMode(TExecuteMode::type::GPU);
      sql("SELECT COUNT(*) FROM test_parallel_0;");
    }
  }

  void TearDown() override {
    if (!g_keep_data) {
      for (size_t i = 0; i < g_num_tables; i++) {
        sql("DROP TABLE IF EXISTS test_parallel_" + std::to_string(i) + ";");
      }
    }
    BaseTestFixture::TearDown();
  }
};

TEST_P(MultiTableTestEnv, MultiTableTest) {
  auto const num_executors = std::get<NUM_EXECUTORS>(param_);
  TExecuteMode::type const dt = std::get<DT>(param_);
  {
    SKIP_NO_GPU();
    setExecuteMode(dt);
    std::vector<std::future<void>> worker_threads;

    // use fewer tables on CPU, as speedup isn't as large
    auto num_tables = dt == TExecuteMode::type::CPU ? 2 : g_num_tables;
    auto execution_time = measure<>::execution([&]() {
      for (size_t w = 0; w < num_tables; w++) {
        worker_threads.push_back(std::async(
            std::launch::async,
            [this](const std::string& table_name, const TExecuteMode::type dt) {
              runSqlExecuteTest(table_name, dt);
            },
            "test_parallel_" + std::to_string(w),
            dt));
      }
      for (auto& t : worker_threads) {
        t.get();
      }
    });
    LOG(ERROR) << "Finished execution with " << g_num_tables << " tables, "
               << num_executors << " executors, " << execution_time << " ms.";
  }
}

class UpdateDeleteTestEnv : public BaseTestFixture {
 protected:
  void SetUp() override {
    BaseTestFixture::SetUp();
    for (int i = 0; i < num_tables; i++) {
      buildTable("test_parallel_" + std::to_string(i));
    }

    if (!skipTests(TExecuteMode::type::GPU)) {
      setExecuteMode(TExecuteMode::type::GPU);
      // warm up the PTX JIT
      sql("SELECT COUNT(*) FROM test_parallel_0;");
    }
  }

  void TearDown() override {
    if (!g_keep_data) {
      for (int i = 0; i < num_tables; i++) {
        sql("DROP TABLE IF EXISTS test_parallel_" + std::to_string(i) + ";");
      }
    }
    BaseTestFixture::TearDown();
  }

 public:
  int num_tables{g_max_num_executors * 2};
};

TEST_P(UpdateDeleteTestEnv, Delete_OneTable) {
  const size_t iterations = 5;

  TExecuteMode::type const dt = std::get<DT>(param_);
  {
    SKIP_NO_GPU();
    setExecuteMode(dt);

    for (size_t i = 0; i < iterations; i++) {
      std::vector<std::future<void>> worker_threads;
      auto execution_time = measure<>::execution([&]() {
        // three readers, one writer
        for (size_t j = 0; j < 4; j++) {
          worker_threads.push_back(std::async(
              std::launch::async,
              [this, j](const std::string& table_name, const TExecuteMode::type dt) {
                if (j == 0) {
                  // run insert, then delete
                  sql("INSERT INTO " + table_name +
                      " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);");
                  sql("DELETE FROM " + table_name + " WHERE i64 = 1;");
                  sqlAndCompareResult(
                      "SELECT COUNT(*) FROM " + table_name + " WHERE i64 = 1;",
                      {{int64_t(0)}});
                } else {
                  // run select
                  sqlAndCompareResult(
                      "SELECT MIN(d) FROM " + table_name + " WHERE i1 IS NOT NULL;",
                      {{1.0001}});
                }
              },
              "test_parallel_0",
              dt));
        }
        for (auto& t : worker_threads) {
          t.get();
        }
      });
      LOG(ERROR) << "Finished execution of iteration " << i << " , " << execution_time
                 << " ms.";
    }
  }
}

TEST_P(UpdateDeleteTestEnv, Delete_TwoTables) {
  TExecuteMode::type const dt = std::get<DT>(param_);
  {
    SKIP_NO_GPU();
    setExecuteMode(dt);

    std::vector<std::future<void>> worker_threads;
    // three readers, one writer
    for (size_t j = 0; j < 4; j++) {
      worker_threads.push_back(std::async(
          std::launch::async,
          [this, j](const std::string& select_table_name,
                    const std::string& delete_table_name,
                    const TExecuteMode::type dt) {
            if (j == 0) {
              // run insert, then delete
              sql("INSERT INTO " + delete_table_name +
                  " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);");
              sql("DELETE FROM " + delete_table_name + " WHERE i64 = 1;");
              sqlAndCompareResult(
                  "SELECT COUNT(*) FROM " + delete_table_name + " WHERE i64 = 1;",
                  {{i(0)}});
            } else {
              // run select
              runSqlExecuteTest(select_table_name, dt);
            }
          },
          "test_parallel_0",
          "test_parallel_1",
          dt));
    }
    for (auto& t : worker_threads) {
      t.get();
    }
  }
}

TEST_P(UpdateDeleteTestEnv, Update_OneTable) {
  const size_t iterations = 5;

  TExecuteMode::type const dt = std::get<DT>(param_);
  {
    SKIP_NO_GPU();
    setExecuteMode(dt);

    for (size_t i = 0; i < iterations; i++) {
      std::vector<std::future<void>> worker_threads;
      auto execution_time = measure<>::execution([&]() {
        // three readers, one writer
        for (size_t j = 0; j < 4; j++) {
          worker_threads.push_back(std::async(
              std::launch::async,
              [this, j](const std::string& table_name, const TExecuteMode::type dt) {
                if (j == 0) {
                  // run insert, then delete
                  sql("INSERT INTO " + table_name +
                      " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);");
                  sql("UPDATE " + table_name + " SET i64 = -1 WHERE i64 = 1;");
                  sqlAndCompareResult(
                      "SELECT COUNT(*) FROM " + table_name + " WHERE i64 = 1;",
                      {{int64_t(0)}});
                } else {
                  // run select
                  sqlAndCompareResult(
                      "SELECT MIN(d) FROM " + table_name + " WHERE i1 IS NOT NULL;",
                      {{1.0001}});
                }
              },
              "test_parallel_0",
              dt));
        }
        for (auto& t : worker_threads) {
          t.get();
        }
      });
      LOG(ERROR) << "Finished execution of iteration " << i << " , " << execution_time
                 << " ms.";
    }
  }
}

TEST_P(UpdateDeleteTestEnv, Update_OneTableVarlen) {
  const size_t iterations = 5;

  TExecuteMode::type const dt = std::get<DT>(param_);
  {
    SKIP_NO_GPU();
    setExecuteMode(dt);

    for (size_t i = 0; i < iterations; i++) {
      std::vector<std::future<void>> worker_threads;
      auto execution_time = measure<>::execution([&]() {
        // three readers, one writer
        for (size_t j = 0; j < 4; j++) {
          worker_threads.push_back(std::async(
              std::launch::async,
              [this, j](const std::string& table_name, const TExecuteMode::type dt) {
                if (j == 0) {
                  // run insert, then delete
                  sql("INSERT INTO " + table_name +
                      " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);");
                  sql("UPDATE " + table_name +
                      " SET arri64 = ARRAY[1,2,3] WHERE i64 = 1;");
                  sqlAndCompareResult(
                      "SELECT COUNT(*) FROM " + table_name + " WHERE i64 = 1;",
                      {{int64_t(1)}});
                  sql("UPDATE " + table_name +
                      " SET arri64 = ARRAY[1,2,3] WHERE i64 = 1;");
                  sql("DELETE FROM " + table_name + " WHERE i64 = 1;");
                } else {
                  // run select
                  sqlAndCompareResult(
                      "SELECT MIN(d) FROM " + table_name + " WHERE i1 IS NOT NULL;",
                      {{1.0001}});
                }
              },
              "test_parallel_0",
              dt));
        }
        for (auto& t : worker_threads) {
          t.get();
        }
      });
      LOG(ERROR) << "Finished execution of iteration " << i << " , " << execution_time
                 << " ms.";
    }
  }
}

TEST_P(UpdateDeleteTestEnv, Update_TwoTables) {
  TExecuteMode::type const dt = std::get<DT>(param_);
  {
    SKIP_NO_GPU();
    setExecuteMode(dt);

    std::vector<std::future<void>> worker_threads;
    // three readers, one writer
    for (size_t j = 0; j < 4; j++) {
      worker_threads.push_back(std::async(
          std::launch::async,
          [this, j](const std::string& select_table_name,
                    const std::string& delete_table_name,
                    const TExecuteMode::type dt) {
            if (j == 0) {
              // run insert, then delete
              sql("INSERT INTO " + delete_table_name +
                  " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);");
              sql("UPDATE " + delete_table_name + " SET i64 = -1 WHERE i64 = 1;");
              sqlAndCompareResult(
                  "SELECT COUNT(*) FROM " + delete_table_name + " WHERE i64 = 1;",
                  {{int64_t(0)}});
            } else {
              // run select
              runSqlExecuteTest(select_table_name, dt);
            }
          },
          "test_parallel_0",
          "test_parallel_1",
          dt));
    }
    for (auto& t : worker_threads) {
      t.get();
    }
  }
}

TEST_P(UpdateDeleteTestEnv, UpdateDelete_TwoTables) {
  TExecuteMode::type const dt = std::get<DT>(param_);
  {
    SKIP_NO_GPU();
    setExecuteMode(dt);

    std::vector<std::future<void>> worker_threads;
    // three readers, one delete, one update
    for (size_t j = 0; j < 5; j++) {
      worker_threads.push_back(std::async(
          std::launch::async,
          [this, j](const std::string& update_table_name,
                    const std::string& delete_table_name,
                    const TExecuteMode::type dt) {
            if (j == 0) {
              // run insert, then delete
              sql("INSERT INTO " + delete_table_name +
                  " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);");
              sql("DELETE FROM " + delete_table_name + " WHERE i64 = 1;");
              sqlAndCompareResult(
                  "SELECT COUNT(*) FROM " + delete_table_name + " WHERE i64 = 1;",
                  {{int64_t(0)}});
            } else if (j == 1) {
              sql("UPDATE " + update_table_name + " SET i32 = -1 WHERE i64 = 500;");
            } else {
              // run select
              sqlAndCompareResult(
                  "SELECT MIN(d) FROM " + update_table_name + " WHERE i1 IS NOT NULL;",
                  {{1.0001}});
            }
          },
          "test_parallel_0",
          "test_parallel_1",
          dt));
    }
    for (auto& t : worker_threads) {
      t.get();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    SingleTableTest,
    SingleTableTestEnv,
    ::testing::Combine(::testing::Values(TExecuteMode::type::CPU,
                                         TExecuteMode::type::GPU),
                       ::testing::ValuesIn(num_executors_vec(g_max_num_executors))),
    TestNameSuffix{});

INSTANTIATE_TEST_SUITE_P(
    MultiTableTest,
    MultiTableTestEnv,
    ::testing::Combine(::testing::Values(TExecuteMode::type::CPU,
                                         TExecuteMode::type::GPU),
                       ::testing::ValuesIn(num_executors_vec(g_max_num_executors))),
    TestNameSuffix{});

INSTANTIATE_TEST_SUITE_P(UpdateDeleteTest,
                         UpdateDeleteTestEnv,
                         ::testing::Combine(::testing::Values(TExecuteMode::type::CPU,
                                                              TExecuteMode::type::GPU),
                                            ::testing::Values(g_max_num_executors)),
                         TestNameSuffix{});

int main(int argc, char* argv[]) {
  g_is_test_env = true;
  g_enable_executor_resource_mgr = false;

  TestHelpers::init_logger_stderr_only(argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");
  desc.add_options()(
      "num-executors",
      po::value<NumExecutors>(&g_max_num_executors)->default_value(g_max_num_executors),
      "Maximum number of parallel executors to test (most tests will start with 1 "
      "executor and increase by doubling until max is reached).");

  desc.add_options()(
      "enable-executor-resource-mgr",
      po::value<bool>(&g_enable_executor_resource_mgr)->implicit_value(true),
      "Enable executor resource manager to track execution resources and "
      "selectively gate concurrency based on resource availability.");

  desc.add_options()(
      "max-num-gpu-per-query",
      po::value<int>(&g_max_num_gpu_per_query)->default_value(g_max_num_gpu_per_query),
      "Sets the maximum number of GPUs that each query can utilize. If this value "
      "exceeds the total number of GPUs available in the system, it will automatically "
      "be adjusted to the system's limit. The default value is 0, which indicates that "
      "the query engine will attempt to use all available GPUs.");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(),
            vm);
  po::notify(vm);

  if (vm.count("keep-data")) {
    g_keep_data = true;
  }

  if (g_enable_executor_resource_mgr) {
#ifdef HAVE_TSAN
    // Parallel executor tests allocate too much memory to
    // instrument under TSAN, so only enable the ExecutorResourceMgr
    // (i.e. parallel kernel execution) if TSAN is off.
    // Note that some parallel kernel execution is tested under TSAN
    // in ConcurrencyTest, although before we turn on parallel kernel execution
    // by default we probably will want to find a way to moderate resource requirements
    // to run this test under TSAN as well.
    // Todo(todd): Find way to run this test with parallel kernel execution on
    // and TSAN enabled
    return 0;
#endif
  }

  if (vm.count("max-num-gpu-per-query")) {
    g_max_num_gpu_per_query = vm["max-num-gpu-per-query"].as<int>();
    if (g_max_num_gpu_per_query > 0) {
      std::cout << "Set the maximum # GPUs per query: " << g_max_num_gpu_per_query
                << std::endl;
    }
  }

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
