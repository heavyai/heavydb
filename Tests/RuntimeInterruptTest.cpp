/*
 * Copyright 2020, OmniSci, Inc.
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

#include <gtest/gtest.h>
#include <boost/program_options.hpp>
#include <exception>
#include <future>
#include <stdexcept>

#include "Catalog/Catalog.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/Logger.h"
#include "Shared/StringTransform.h"

using QR = QueryRunner::QueryRunner;
unsigned INTERRUPT_CHECK_FREQ = 10;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

bool g_cpu_only{false};
// nested loop over 1M * 1M
std::string test_query_large{"SELECT count(1) FROM t_large t1, t_large t2;"};
// nested loop over 100k * 100k
std::string test_query_medium{"SELECT count(1) FROM t_medium t1, t_medium t2;"};
// nested loop over 1k * 1k
std::string test_query_small{"SELECT count(1) FROM t_small t1, t_small t2;"};

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

namespace {

std::shared_ptr<ResultSet> run_query(const std::string& query_str,
                                     std::shared_ptr<Executor> executor,
                                     const ExecutorDeviceType device_type,
                                     const std::string& session_id = "") {
  if (session_id.length() != 32) {
    LOG(ERROR) << "Incorrect or missing session info.";
  }
  return QR::get()->runSQLWithAllowingInterrupt(
      query_str, executor, session_id, device_type, INTERRUPT_CHECK_FREQ);
}

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

int create_and_populate_table() {
  try {
    run_ddl_statement("DROP TABLE IF EXISTS t_large;");
    run_ddl_statement("DROP TABLE IF EXISTS t_medium;");
    run_ddl_statement("DROP TABLE IF EXISTS t_small;");
    run_ddl_statement("CREATE TABLE t_large (x int not null);");
    run_ddl_statement("CREATE TABLE t_medium (x int not null);");
    run_ddl_statement("CREATE TABLE t_small (x int not null);");

    // write a temporary datafile used in the test
    // because "INSERT INTO ..." stmt for this takes too much time
    // and add pre-generated dataset increases meaningless LOC of this test code
    const auto file_path_small =
        boost::filesystem::path("../../Tests/Import/datafiles/interrupt_table_small.txt");
    if (boost::filesystem::exists(file_path_small)) {
      boost::filesystem::remove(file_path_small);
    }
    std::ofstream small_out(file_path_small.string());
    for (int i = 0; i < 1000; i++) {
      if (small_out.is_open()) {
        small_out << "1\n";
      }
    }
    small_out.close();

    const auto file_path_medium = boost::filesystem::path(
        "../../Tests/Import/datafiles/interrupt_table_medium.txt");
    if (boost::filesystem::exists(file_path_medium)) {
      boost::filesystem::remove(file_path_medium);
    }
    std::ofstream medium_out(file_path_medium.string());
    for (int i = 0; i < 100000; i++) {
      if (medium_out.is_open()) {
        medium_out << "1\n";
      }
    }
    medium_out.close();

    const auto file_path_large =
        boost::filesystem::path("../../Tests/Import/datafiles/interrupt_table_large.txt");
    if (boost::filesystem::exists(file_path_large)) {
      boost::filesystem::remove(file_path_large);
    }
    std::ofstream large_out(file_path_large.string());
    for (int i = 0; i < 1000000; i++) {
      if (large_out.is_open()) {
        large_out << "1\n";
      }
    }
    large_out.close();

    std::string import_small_table_str{
        "COPY t_small FROM "
        "'../../Tests/Import/datafiles/interrupt_table_small.txt' WITH "
        "(header='false')"};
    std::string import_medium_table_str{
        "COPY t_medium FROM "
        "'../../Tests/Import/datafiles/interrupt_table_medium.txt' "
        "WITH (header='false')"};
    std::string import_large_table_str{
        "COPY t_large FROM "
        "'../../Tests/Import/datafiles/interrupt_table_large.txt' WITH "
        "(header='false')"};
    run_ddl_statement(import_small_table_str);
    run_ddl_statement(import_medium_table_str);
    run_ddl_statement(import_large_table_str);

  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table";
    return -1;
  }
  return 0;
}

int drop_table() {
  try {
    run_ddl_statement("DROP TABLE IF EXISTS t_large;");
    run_ddl_statement("DROP TABLE IF EXISTS t_medium;");
    run_ddl_statement("DROP TABLE IF EXISTS t_small;");
    boost::filesystem::remove("../../Tests/Import/datafiles/interrupt_table_small.txt");
    boost::filesystem::remove("../../Tests/Import/datafiles/interrupt_table_medium.txt");
    boost::filesystem::remove("../../Tests/Import/datafiles/interrupt_table_large.txt");
  } catch (...) {
    LOG(ERROR) << "Failed to drop table";
    return -1;
  }
  return 0;
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

}  // namespace

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

TEST(Interrupt, Kill_RunningQuery) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    auto executor = QR::get()->getExecutor();
    bool startQueryExec = false;
    executor->enableRuntimeQueryInterrupt(INTERRUPT_CHECK_FREQ);
    std::shared_ptr<ResultSet> res1 = nullptr;
    std::exception_ptr exception_ptr = nullptr;
    try {
      SKIP_NO_GPU();
      std::string query_session = generate_random_string(32);
      // we first run the query as async function call
      auto query_thread1 = std::async(std::launch::async, [&] {
        std::shared_ptr<ResultSet> res = nullptr;
        try {
          res = run_query(test_query_large, executor, dt, query_session);
        } catch (...) {
          exception_ptr = std::current_exception();
        }
        return res;
      });

      // wait until our server starts to process the first query
      std::string curRunningSession = "";
      while (!startQueryExec) {
        mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        curRunningSession = executor->getCurrentQuerySession(session_read_lock);
        if (curRunningSession == query_session) {
          startQueryExec = true;
        }
        session_read_lock.unlock();
      }
      // then, after query execution is started, we try to interrupt the running query
      // by providing the interrupt signal with the running session info
      executor->interrupt(query_session, query_session);
      res1 = query_thread1.get();
      if (exception_ptr != nullptr) {
        std::rethrow_exception(exception_ptr);
      } else {
        // when we reach here, it means the query is finished without interrupted
        // due to some reasons, i.e., very fast query execution
        // so, instead, we check whether query result is correct
        CHECK_EQ(1, (int64_t)res1.get()->rowCount());
        auto crt_row = res1.get()->getNextRow(false, false);
        auto ret_val = v<int64_t>(crt_row[0]);
        CHECK_EQ((int64_t)1000000 * 1000000, ret_val);
      }
    } catch (const std::runtime_error& e) {
      std::string expected_err_msg = "Query execution has been interrupted";
      std::string received_err_msg = e.what();
      bool check = (expected_err_msg == received_err_msg) ? true : false;
      CHECK(check);
    }
  }
}

TEST(Interrupt, Kill_PendingQuery) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    std::future<std::shared_ptr<ResultSet>> query_thread1;
    std::future<std::shared_ptr<ResultSet>> query_thread2;
    auto executor = QR::get()->getExecutor();
    executor->enableRuntimeQueryInterrupt(INTERRUPT_CHECK_FREQ);
    bool startQueryExec = false;
    std::exception_ptr exception_ptr1 = nullptr;
    std::exception_ptr exception_ptr2 = nullptr;
    std::shared_ptr<ResultSet> res1 = nullptr;
    std::shared_ptr<ResultSet> res2 = nullptr;
    try {
      SKIP_NO_GPU();
      std::string session1 = generate_random_string(32);
      std::string session2 = generate_random_string(32);
      // we first run the query as async function call
      query_thread1 = std::async(std::launch::async, [&] {
        std::shared_ptr<ResultSet> res = nullptr;
        try {
          res = run_query(test_query_medium, executor, dt, session1);
        } catch (...) {
          exception_ptr1 = std::current_exception();
        }
        return res;
      });
      // make sure our server recognizes a session for running query correctly
      std::string curRunningSession = "";
      while (!startQueryExec) {
        mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
        curRunningSession = executor->getCurrentQuerySession(session_read_lock);
        if (curRunningSession == session1) {
          startQueryExec = true;
        }
        session_read_lock.unlock();
      }
      query_thread2 = std::async(std::launch::async, [&] {
        std::shared_ptr<ResultSet> res = nullptr;
        try {
          // run pending query as async call
          res = run_query(test_query_medium, executor, dt, session2);
        } catch (...) {
          exception_ptr2 = std::current_exception();
        }
        return res;
      });
      // then, we try to interrupt the pending query
      // by providing the interrupt signal with the pending query's session info
      if (startQueryExec) {
        executor->interrupt(session2, session2);
      }
      res2 = query_thread2.get();
      res1 = query_thread1.get();
      if (exception_ptr2 != nullptr) {
        // pending query throws an runtime exception due to query interrupt
        std::rethrow_exception(exception_ptr2);
      }
      if (exception_ptr1 != nullptr) {
        // running query should never return the runtime exception
        CHECK(false);
      }
    } catch (const std::runtime_error& e) {
      // catch exception due to runtime query interrupt
      // and compare thrown message to confirm that
      // this exception comes from our interrupt request
      std::string expected_err_msg =
          "Query execution has been interrupted (pending query)";
      std::string received_err_msg = e.what();
      bool check = (expected_err_msg == received_err_msg) ? true : false;
      CHECK(check);
    }
    // check running query's result
    CHECK_EQ(1, (int64_t)res1.get()->rowCount());
    auto crt_row = res1.get()->getNextRow(false, false);
    auto ret_val = v<int64_t>(crt_row[0]);
    CHECK_EQ((int64_t)100000 * 100000, ret_val);
  }
}

TEST(Interrupt, Make_PendingQuery_Run) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    std::future<std::shared_ptr<ResultSet>> query_thread1;
    std::future<std::shared_ptr<ResultSet>> query_thread2;
    auto executor = QR::get()->getExecutor();
    executor->enableRuntimeQueryInterrupt(INTERRUPT_CHECK_FREQ);
    bool startQueryExec = false;
    std::exception_ptr exception_ptr1 = nullptr;
    std::exception_ptr exception_ptr2 = nullptr;
    std::shared_ptr<ResultSet> res1 = nullptr;
    std::shared_ptr<ResultSet> res2 = nullptr;
    try {
      SKIP_NO_GPU();
      std::string session1 = generate_random_string(32);
      std::string session2 = generate_random_string(32);
      // we first run the query as async function call
      query_thread1 = std::async(std::launch::async, [&] {
        std::shared_ptr<ResultSet> res = nullptr;
        try {
          res = run_query(test_query_large, executor, dt, session1);
        } catch (...) {
          exception_ptr1 = std::current_exception();
        }
        return res;
      });
      // make sure our server recognizes a session for running query correctly
      std::string curRunningSession = "";
      while (!startQueryExec) {
        mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
        curRunningSession = executor->getCurrentQuerySession(session_read_lock);
        if (curRunningSession == session1) {
          startQueryExec = true;
        }
        session_read_lock.unlock();
      }
      query_thread2 = std::async(std::launch::async, [&] {
        std::shared_ptr<ResultSet> res = nullptr;
        try {
          // run pending query as async call
          res = run_query(test_query_small, executor, dt, session2);
        } catch (...) {
          exception_ptr2 = std::current_exception();
        }
        return res;
      });
      // then, we try to interrupt the running query
      // by providing the interrupt signal with the running query's session info
      // so we can expect that running query session releases all H/W resources and locks,
      // and so pending query takes them for its query execution (becomes running query)
      if (startQueryExec) {
        executor->interrupt(session1, session1);
      }
      res2 = query_thread2.get();
      res1 = query_thread1.get();
      if (exception_ptr1 != nullptr) {
        std::rethrow_exception(exception_ptr1);
      }
      if (exception_ptr2 != nullptr) {
        // pending query should never return the runtime exception
        // because it is executed after running query is interrupted
        CHECK(false);
      }
    } catch (const std::runtime_error& e) {
      // catch exception due to runtime query interrupt
      // and compare thrown message to confirm that
      // this exception comes from our interrupt request
      std::string expected_err_msg = "Query execution has been interrupted";
      std::string received_err_msg = e.what();
      bool check = (expected_err_msg == received_err_msg) ? true : false;
      CHECK(check);
    }
    // check running query's result
    CHECK_EQ(1, (int64_t)res2.get()->rowCount());
    auto crt_row = res2.get()->getNextRow(false, false);
    auto ret_val = v<int64_t>(crt_row[0]);
    CHECK_EQ((int64_t)1000 * 1000, ret_val);
  }
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  TestHelpers::init_logger_stderr_only(argc, argv);

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = create_and_populate_table();
    err = RUN_ALL_TESTS();
    err = drop_table();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}