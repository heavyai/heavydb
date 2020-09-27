/*
 * Copyright 2020 OmniSci, Inc.
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

#include "TestHelpers.h"

#include "Logger/Logger.h"

#include "../QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "../QueryEngine/Execute.h"
#include "../QueryRunner/QueryRunner.h"

#include <array>
#include <future>
#include <string>
#include <vector>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

bool g_keep_data{false};
size_t g_max_num_executors{8};
size_t g_num_tables{25};

extern bool g_is_test_env;

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

inline void run_ddl_statement(const std::string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

TargetValue run_simple_agg(const std::string& query_str, const ExecutorDeviceType dt) {
  auto rows = QR::get()
                  ->runSelectQuery(query_str,
                                   dt,
                                   /*hoist_literals=*/true,
                                   /*allow_loop_joins=*/false,
                                   /*just_explain=*/false)
                  ->getRows();
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
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

void run_sql_execute_test(const std::string& table_name, const ExecutorDeviceType dt) {
  auto check_returned_rows = [](const auto& result_set, const size_t num_rows) {
    CHECK(result_set);
    EXPECT_EQ(result_set->rowCount(), num_rows);
  };

  // basic queries
  EXPECT_EQ(15,
            v<int64_t>(run_simple_agg(
                "SELECT COUNT(*) FROM " + table_name + " WHERE i32 < 50;", dt)));
  EXPECT_DOUBLE_EQ(
      double(1.0001),
      v<double>(run_simple_agg(
          "SELECT MIN(d) FROM " + table_name + " WHERE i1 IS NOT NULL;", dt)));

  // Simple query with a sort
  EXPECT_EQ(0,
            v<int64_t>(run_simple_agg(
                "SELECT i64 FROM " + table_name + " ORDER BY i64 LIMIT 1;", dt)));

  // complex queries
  EXPECT_NO_THROW(check_returned_rows(
      QR::get()
          ->runSelectQuery("SELECT d, f, COUNT(*) FROM " + table_name + " GROUP BY d, f;",
                           dt,
                           /*hoist_literals=*/true,
                           /*allow_loop_joins=*/false,
                           /*just_explain=*/false)
          ->getRows(),
      6));
  EXPECT_NO_THROW(check_returned_rows(
      QR::get()
          ->runSelectQuery("SELECT d, f, COUNT(*) FROM " + table_name +
                               " GROUP BY d, f ORDER BY f DESC NULLS LAST LIMIT 5;",
                           dt,
                           /*hoist_literals=*/true,
                           /*allow_loop_joins=*/false,
                           /*just_explain=*/false)
          ->getRows(),
      5));
  EXPECT_NO_THROW(check_returned_rows(
      QR::get()
          ->runSelectQuery(
              "SELECT approx_count_distinct(d), approx_count_distinct(str), i64, i32, "
              "i16 FROM " +
                  table_name +
                  " WHERE i32 < 50 GROUP BY i64, i32, i16 ORDER BY i64, i32, i16;",
              dt,
              /*hoist_literals=*/true,
              /*allow_loop_joins=*/false,
              /*just_explain=*/false)
          ->getRows(),
      5));

  // multi-step
  EXPECT_NO_THROW(check_returned_rows(
      QR::get()
          ->runSelectQuery(
              "SELECT d, f, COUNT(*) FROM " + table_name + " GROUP BY d, f HAVING d < f;",
              dt,
              /*hoist_literals=*/true,
              /*allow_loop_joins=*/false,
              /*just_explain=*/false)
          ->getRows(),
      5));

  // joins
  EXPECT_EQ(1,
            v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM " + table_name +
                                          " a INNER JOIN (SELECT i32 FROM " + table_name +
                                          " GROUP BY i32) b on a.i64 = b.i32;",
                                      dt)));
}

void build_table(const std::string& table_name) {
  run_ddl_statement("DROP TABLE IF EXISTS " + table_name + ";");

  run_ddl_statement("CREATE TABLE " + table_name + " " + table_schema);
  ValuesGenerator gen(table_name);
  for (size_t i = 0; i < 10; i++) {
    QR::get()->runSQL(gen(100, 10, 2, 1, 1.0001, 1.1, "'true'", "'hello'", "{100, 200}"),
                      ExecutorDeviceType::CPU);
  }
  for (size_t i = 0; i < 5; i++) {
    QR::get()->runSQL(gen(500,
                          50,
                          "NULL",
                          5,
                          5.0001,
                          5.1,
                          "'false'",
                          "'world'",
                          i % 2 == 0 ? "{NULL, 200}" : "{100, NULL}"),
                      ExecutorDeviceType::CPU);
  }
  for (size_t i = 0; i < 5; i++) {
    QR::get()->runSQL(
        gen(100 * i,
            10 * i,
            2 * i,
            1 * i,
            1.0001 * static_cast<float>(i),
            1.1 * static_cast<float>(i),
            "NULL",
            "NULL",
            "{" + std::to_string(100 * i) + "," + std::to_string(200 * i) + "}"),
        ExecutorDeviceType::CPU);
  }
}

class SingleTableTestEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    build_table("test_parallel");

    if (!skip_tests(ExecutorDeviceType::GPU)) {
      // warm up the PTX JIT
      run_simple_agg("SELECT COUNT(*) FROM test_parallel;", ExecutorDeviceType::GPU);
    }
  }

  void TearDown() override {
    if (!g_keep_data) {
      run_ddl_statement("DROP TABLE IF EXISTS test_parallel;");
    }
  }
};

TEST_F(SingleTableTestEnv, AllTables) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    for (size_t i = 1; i <= g_max_num_executors; i *= 2) {
      QR::get()->resizeDispatchQueue(i);
      std::vector<std::future<void>> worker_threads;
      auto execution_time = measure<>::execution([&]() {
        for (size_t w = 0; w < i; w++) {
          worker_threads.push_back(
              std::async(std::launch::async, run_sql_execute_test, "test_parallel", dt));
        }
        for (auto& t : worker_threads) {
          t.get();
        }
      });
      LOG(ERROR) << "Finished execution with " << i << " executors, " << execution_time
                 << " ms.";
    }
  }
}

class MultiTableTestEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    for (size_t i = 0; i < g_num_tables; i++) {
      build_table("test_parallel_" + std::to_string(i));
    }

    if (!skip_tests(ExecutorDeviceType::GPU)) {
      // warm up the PTX JIT
      run_simple_agg("SELECT COUNT(*) FROM test_parallel_0;", ExecutorDeviceType::GPU);
    }
  }

  void TearDown() override {
    if (!g_keep_data) {
      for (size_t i = 0; i < g_num_tables; i++) {
        run_ddl_statement("DROP TABLE IF EXISTS test_parallel_" + std::to_string(i) +
                          ";");
      }
    }
  }
};

TEST_F(MultiTableTestEnv, AllTables) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (size_t i = 1; i <= g_max_num_executors; i *= 2) {
      QR::get()->resizeDispatchQueue(i);
      std::vector<std::future<void>> worker_threads;

      // use fewer tables on CPU, as speedup isn't as large
      auto num_tables = dt == ExecutorDeviceType::CPU ? 2 : g_num_tables;
      auto execution_time = measure<>::execution([&]() {
        for (size_t w = 0; w < num_tables; w++) {
          worker_threads.push_back(std::async(std::launch::async,
                                              run_sql_execute_test,
                                              "test_parallel_" + std::to_string(w),
                                              dt));
        }
        for (auto& t : worker_threads) {
          t.get();
        }
      });
      LOG(ERROR) << "Finished execution with " << g_num_tables << " tables, " << i
                 << " executors, " << execution_time << " ms.";
    }
  }
}

class UpdateDeleteTestEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    for (size_t i = 0; i < g_max_num_executors * 2; i++) {
      build_table("test_parallel_" + std::to_string(i));
    }

    if (!skip_tests(ExecutorDeviceType::GPU)) {
      // warm up the PTX JIT
      run_simple_agg("SELECT COUNT(*) FROM test_parallel_0;", ExecutorDeviceType::GPU);
    }
  }

  void TearDown() override {
    if (!g_keep_data) {
      for (size_t i = 0; i < g_max_num_executors * 2; i++) {
        run_ddl_statement("DROP TABLE IF EXISTS test_parallel_" + std::to_string(i) +
                          ";");
      }
    }
  }

 public:
  size_t num_tables{g_max_num_executors * 2};
};

TEST_F(UpdateDeleteTestEnv, Delete_OneTable) {
  const size_t iterations = 5;
  QR::get()->resizeDispatchQueue(g_max_num_executors);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    for (size_t i = 0; i < iterations; i++) {
      std::vector<std::future<void>> worker_threads;
      auto execution_time = measure<>::execution([&]() {
        // three readers, one writer
        for (size_t j = 0; j < 4; j++) {
          worker_threads.push_back(std::async(
              std::launch::async,
              [j](const std::string& table_name, const ExecutorDeviceType dt) {
                if (j == 0) {
                  // run insert, then delete
                  QR::get()->runSQL(
                      "INSERT INTO " + table_name +
                          " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);",
                      ExecutorDeviceType::CPU);
                  QR::get()->runSQL("DELETE FROM " + table_name + " WHERE i64 = 1;",
                                    ExecutorDeviceType::CPU);
                  EXPECT_EQ(
                      0,
                      v<int64_t>(run_simple_agg(
                          "SELECT COUNT(*) FROM " + table_name + " WHERE i64 = 1;", dt)));
                } else {
                  // run select
                  EXPECT_DOUBLE_EQ(
                      double(1.0001),
                      v<double>(run_simple_agg(
                          "SELECT MIN(d) FROM " + table_name + " WHERE i1 IS NOT NULL;",
                          dt)));
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

TEST_F(UpdateDeleteTestEnv, Delete_TwoTables) {
  QR::get()->resizeDispatchQueue(g_max_num_executors);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    std::vector<std::future<void>> worker_threads;
    // three readers, one writer
    for (size_t j = 0; j < 4; j++) {
      worker_threads.push_back(std::async(
          std::launch::async,
          [j](const std::string& select_table_name,
              const std::string& delete_table_name,
              const ExecutorDeviceType dt) {
            if (j == 0) {
              // run insert, then delete
              QR::get()->runSQL(
                  "INSERT INTO " + delete_table_name +
                      " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);",
                  ExecutorDeviceType::CPU);
              QR::get()->runSQL("DELETE FROM " + delete_table_name + " WHERE i64 = 1;",
                                ExecutorDeviceType::CPU);
              EXPECT_EQ(
                  0,
                  v<int64_t>(run_simple_agg(
                      "SELECT COUNT(*) FROM " + delete_table_name + " WHERE i64 = 1;",
                      dt)));
            } else {
              // run select
              run_sql_execute_test(select_table_name, dt);
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

TEST_F(UpdateDeleteTestEnv, Update_OneTable) {
  const size_t iterations = 5;
  QR::get()->resizeDispatchQueue(g_max_num_executors);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    for (size_t i = 0; i < iterations; i++) {
      std::vector<std::future<void>> worker_threads;
      auto execution_time = measure<>::execution([&]() {
        // three readers, one writer
        for (size_t j = 0; j < 4; j++) {
          worker_threads.push_back(std::async(
              std::launch::async,
              [i, j](const std::string& table_name, const ExecutorDeviceType dt) {
                if (j == 0) {
                  // run insert, then delete
                  QR::get()->runSQL(
                      "INSERT INTO " + table_name +
                          " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);",
                      ExecutorDeviceType::CPU);
                  QR::get()->runSQL(
                      "UPDATE " + table_name + " SET i64 = -1 WHERE i64 = 1;",
                      ExecutorDeviceType::CPU);
                  EXPECT_EQ(
                      0,
                      v<int64_t>(run_simple_agg(
                          "SELECT COUNT(*) FROM " + table_name + " WHERE i64 = 1;", dt)));
                } else {
                  // run select
                  EXPECT_DOUBLE_EQ(
                      double(1.0001),
                      v<double>(run_simple_agg(
                          "SELECT MIN(d) FROM " + table_name + " WHERE i1 IS NOT NULL;",
                          dt)));
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

TEST_F(UpdateDeleteTestEnv, Update_OneTableVarlen) {
  const size_t iterations = 5;
  QR::get()->resizeDispatchQueue(g_max_num_executors);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    for (size_t i = 0; i < iterations; i++) {
      std::vector<std::future<void>> worker_threads;
      auto execution_time = measure<>::execution([&]() {
        // three readers, one writer
        for (size_t j = 0; j < 4; j++) {
          worker_threads.push_back(std::async(
              std::launch::async,
              [j](const std::string& table_name, const ExecutorDeviceType dt) {
                if (j == 0) {
                  // run insert, then delete
                  QR::get()->runSQL(
                      "INSERT INTO " + table_name +
                          " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);",
                      ExecutorDeviceType::CPU);
                  QR::get()->runSQL("UPDATE " + table_name +
                                        " SET arri64 = ARRAY[1,2,3] WHERE i64 = 1;",
                                    ExecutorDeviceType::CPU);
                  EXPECT_EQ(
                      1,
                      v<int64_t>(run_simple_agg(
                          "SELECT COUNT(*) FROM " + table_name + " WHERE i64 = 1;", dt)));
                  QR::get()->runSQL("UPDATE " + table_name +
                                        " SET arri64 = ARRAY[1,2,3] WHERE i64 = 1;",
                                    ExecutorDeviceType::CPU);
                  QR::get()->runSQL("DELETE FROM " + table_name + " WHERE i64 = 1;",
                                    ExecutorDeviceType::CPU);
                } else {
                  // run select
                  EXPECT_DOUBLE_EQ(
                      double(1.0001),
                      v<double>(run_simple_agg(
                          "SELECT MIN(d) FROM " + table_name + " WHERE i1 IS NOT NULL;",
                          dt)));
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

TEST_F(UpdateDeleteTestEnv, Update_TwoTables) {
  QR::get()->resizeDispatchQueue(g_max_num_executors);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    std::vector<std::future<void>> worker_threads;
    // three readers, one writer
    for (size_t j = 0; j < 4; j++) {
      worker_threads.push_back(std::async(
          std::launch::async,
          [j](const std::string& select_table_name,
              const std::string& delete_table_name,
              const ExecutorDeviceType dt) {
            if (j == 0) {
              // run insert, then delete
              QR::get()->runSQL(
                  "INSERT INTO " + delete_table_name +
                      " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);",
                  ExecutorDeviceType::CPU);
              QR::get()->runSQL(
                  "UPDATE " + delete_table_name + " SET i64 = -1 WHERE i64 = 1;",
                  ExecutorDeviceType::CPU);
              EXPECT_EQ(
                  0,
                  v<int64_t>(run_simple_agg(
                      "SELECT COUNT(*) FROM " + delete_table_name + " WHERE i64 = 1;",
                      dt)));
            } else {
              // run select
              run_sql_execute_test(select_table_name, dt);
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

TEST_F(UpdateDeleteTestEnv, UpdateDelete_TwoTables) {
  QR::get()->resizeDispatchQueue(g_max_num_executors);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    std::vector<std::future<void>> worker_threads;
    // three readers, one delete, one update
    for (size_t j = 0; j < 5; j++) {
      worker_threads.push_back(std::async(
          std::launch::async,
          [j](const std::string& update_table_name,
              const std::string& delete_table_name,
              const ExecutorDeviceType dt) {
            if (j == 0) {
              // run insert, then delete
              QR::get()->runSQL(
                  "INSERT INTO " + delete_table_name +
                      " VALUES(1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);",
                  ExecutorDeviceType::CPU);
              QR::get()->runSQL("DELETE FROM " + delete_table_name + " WHERE i64 = 1;",
                                ExecutorDeviceType::CPU);
              EXPECT_EQ(
                  0,
                  v<int64_t>(run_simple_agg(
                      "SELECT COUNT(*) FROM " + delete_table_name + " WHERE i64 = 1;",
                      dt)));
            } else if (j == 1) {
              QR::get()->runSQL(
                  "UPDATE " + update_table_name + " SET i32 = -1 WHERE i64 = 500;", dt);
            } else {
              // run select
              EXPECT_DOUBLE_EQ(
                  double(1.0001),
                  v<double>(run_simple_agg("SELECT MIN(d) FROM " + update_table_name +
                                               " WHERE i1 IS NOT NULL;",
                                           dt)));
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

int main(int argc, char* argv[]) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");
  desc.add_options()(
      "num-executors",
      po::value<size_t>(&g_max_num_executors)->default_value(g_max_num_executors),
      "Maximum number of parallel executors to test (most tests will start with 1 "
      "executor and increase by doubling until max is reached).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("keep-data")) {
    g_keep_data = true;
  }

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
