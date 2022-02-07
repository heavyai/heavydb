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
#include "Logger/Logger.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/StringTransform.h"
#include "Shared/scope.h"

using QR = QueryRunner::QueryRunner;
unsigned PENDING_QUERY_INTERRUPT_CHECK_FREQ = 10;
double RUNNING_QUERY_INTERRUPT_CHECK_FREQ = 0.9;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_runtime_query_interrupt;
extern bool g_enable_non_kernel_time_query_interrupt;

bool g_cpu_only{false};
// nested loop over 1M * 1M
std::string test_query_large{
    "SELECT count(1) FROM t_large t1, t_large t2 where t1.x = t2.x;"};
// nested loop over 100k * 100k
std::string test_query_medium{
    "SELECT count(1) FROM t_medium t1, t_medium t2 where t1.x = t2.x;"};
// nested loop over 1k * 1k
std::string test_query_small{
    "SELECT count(1) FROM t_small t1, t_small t2 where t1.x = t2.x;"};

std::string pending_query_interrupted_msg =
    "Query execution has been interrupted (pending query)";
std::string running_query_interrupted_msg =
    "Query execution failed with error Query execution has been interrupted";

namespace {

std::shared_ptr<ResultSet> run_query(const std::string& query_str,
                                     const ExecutorDeviceType device_type,
                                     const std::string& session_id = "",
                                     const std::string& session_name = "admin") {
  if (session_id.length() != 32) {
    LOG(ERROR) << "Incorrect or missing session info.";
  }
  return QR::get()->runSQLWithAllowingInterrupt(query_str,
                                                session_id,
                                                session_name,
                                                device_type,
                                                0.5,
                                                PENDING_QUERY_INTERRUPT_CHECK_FREQ);
}

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

int create_and_populate_table() {
  try {
    run_ddl_statement("DROP TABLE IF EXISTS t_very_large_agg;");
    run_ddl_statement("DROP TABLE IF EXISTS t_very_large_parquet;");
    run_ddl_statement("DROP TABLE IF EXISTS t_very_large_csv;");
    run_ddl_statement("DROP TABLE IF EXISTS t_very_large;");
    run_ddl_statement("DROP TABLE IF EXISTS t_large;");
    run_ddl_statement("DROP TABLE IF EXISTS t_medium;");
    run_ddl_statement("DROP TABLE IF EXISTS t_small;");
    run_ddl_statement(
        "CREATE TABLE t_very_large_csv (x int not null, y int not null, z int not "
        "null);");
    run_ddl_statement(
        "CREATE TABLE t_very_large_parquet (x int not null, y int not null, z int not "
        "null);");
    run_ddl_statement(
        "CREATE TABLE t_very_large (x int not null, y int not null, z int not null);");
    run_ddl_statement(
        "CREATE TABLE t_very_large_agg (k1 int not null, k2 int not null, i int not "
        "null) WITH (fragment_size = 1000000);");
    run_ddl_statement("CREATE TABLE t_large (x int not null);");
    run_ddl_statement("CREATE TABLE t_medium (x int not null);");
    run_ddl_statement("CREATE TABLE t_small (x int not null);");

    // write a temporary datafile used in the test
    // because "INSERT INTO ..." stmt for this takes too much time
    // and add pre-generated dataset increases meaningless LOC of this test code
    const auto file_path_small =
        boost::filesystem::path("../../Tests/Import/datafiles/interrupt_table_small.csv");
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
        "../../Tests/Import/datafiles/interrupt_table_medium.csv");
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
        boost::filesystem::path("../../Tests/Import/datafiles/interrupt_table_large.csv");
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

    const auto file_path_very_large = boost::filesystem::path(
        "../../Tests/Import/datafiles/interrupt_table_very_large.csv");
    if (boost::filesystem::exists(file_path_very_large)) {
      boost::filesystem::remove(file_path_very_large);
    }
    std::ofstream very_large_out(file_path_very_large.string());
    for (int i = 0; i < 10000000; i++) {
      if (very_large_out.is_open()) {
        very_large_out << "1,1,1\n2,2,2\n3,3,3\n4,4,4\n5,5,5\n";
      }
    }
    very_large_out.close();

    const auto file_path_very_large_agg = boost::filesystem::path(
        "../../Tests/Import/datafiles/interrupt_table_very_large_agg.csv");
    if (boost::filesystem::exists(file_path_very_large_agg)) {
      boost::filesystem::remove(file_path_very_large_agg);
    }
    std::ofstream very_large_agg_out(file_path_very_large_agg.string());
    for (int i = 0; i < 10000000; i++) {
      if (very_large_agg_out.is_open()) {
        very_large_agg_out << rand() % 1000 << "," << rand() % 1000 << ","
                           << rand() % 100000 << "\n";
      }
    }
    very_large_agg_out.close();

    std::string import_small_table_str{
        "COPY t_small FROM "
        "'../../Tests/Import/datafiles/interrupt_table_small.csv' WITH "
        "(header='false')"};
    std::string import_medium_table_str{
        "COPY t_medium FROM "
        "'../../Tests/Import/datafiles/interrupt_table_medium.csv' "
        "WITH (header='false')"};
    std::string import_large_table_str{
        "COPY t_large FROM "
        "'../../Tests/Import/datafiles/interrupt_table_large.csv' WITH "
        "(header='false')"};
    std::string import_very_large_table_str{
        "COPY t_very_large FROM "
        "'../../Tests/Import/datafiles/interrupt_table_very_large.csv' WITH "
        "(header='false')"};
    std::string import_very_large_agg_table_str{
        "COPY t_very_large_agg FROM "
        "'../../Tests/Import/datafiles/interrupt_table_very_large_agg.csv' WITH "
        "(header='false')"};
    run_ddl_statement(import_small_table_str);
    run_ddl_statement(import_medium_table_str);
    run_ddl_statement(import_large_table_str);
    run_ddl_statement(import_very_large_table_str);
    run_ddl_statement(import_very_large_agg_table_str);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table";
    return -1;
  }
  return 0;
}

int drop_table() {
  try {
    run_ddl_statement("DROP TABLE IF EXISTS t_very_large_csv;");
    run_ddl_statement("DROP TABLE IF EXISTS t_very_large_parquet;");
    run_ddl_statement("DROP TABLE IF EXISTS t_very_large;");
    run_ddl_statement("DROP TABLE IF EXISTS t_large;");
    run_ddl_statement("DROP TABLE IF EXISTS t_medium;");
    run_ddl_statement("DROP TABLE IF EXISTS t_small;");
    // keep interrupt_table_very_large.parquet
    boost::filesystem::remove("../../Tests/Import/datafiles/interrupt_table_small.csv");
    boost::filesystem::remove("../../Tests/Import/datafiles/interrupt_table_medium.csv");
    boost::filesystem::remove("../../Tests/Import/datafiles/interrupt_table_large.csv");
    boost::filesystem::remove(
        "../../Tests/Import/datafiles/interrupt_table_very_large.csv");
    boost::filesystem::remove(
        "../../Tests/Import/datafiles/interrupt_table_very_large_agg.csv");
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

TEST(Interrupt, Kill_RunningQuery) {
  auto dt = ExecutorDeviceType::CPU;
  // assume a single executor is allowed for this test
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  bool startQueryExec = false;
  executor->enableRuntimeQueryInterrupt(RUNNING_QUERY_INTERRUPT_CHECK_FREQ,
                                        PENDING_QUERY_INTERRUPT_CHECK_FREQ);
  std::shared_ptr<ResultSet> res1 = nullptr;
  std::exception_ptr exception_ptr = nullptr;
  std::vector<size_t> assigned_executor_ids;
  try {
    std::string query_session = generate_random_string(32);
    // we first run the query as async function call
    auto query_thread1 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        res = run_query(test_query_large, dt, query_session, "session1");
      } catch (...) {
        exception_ptr = std::current_exception();
      }
      return res;
    });

    while (assigned_executor_ids.empty()) {
      assigned_executor_ids = executor->getExecutorIdsRunningQuery(query_session);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK_EQ(assigned_executor_ids.size(), static_cast<size_t>(1));

    // wait until our server starts to process the first query
    std::string curRunningSession = "";
    size_t assigned_executor_id = *assigned_executor_ids.begin();
    auto assigned_executor_ptr = Executor::getExecutor(assigned_executor_id);
    CHECK(assigned_executor_ptr);
    while (!startQueryExec) {
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      curRunningSession =
          assigned_executor_ptr->getCurrentQuerySession(session_read_lock);
      if (curRunningSession == query_session) {
        startQueryExec = true;
      }
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
    std::string received_err_msg = e.what();
    auto check_interrupted_msg = received_err_msg.find("interrupted");
    CHECK(check_interrupted_msg != std::string::npos) << received_err_msg;
  }
}

TEST(Interrupt, Check_Query_Runs_After_Interruption) {
  // this test checks whether we successfully clear the interrupt status
  // after killing the running query
  // so as to run the next query without any issue (that is issued due to the previous
  // interruption status)
  auto dt = ExecutorDeviceType::CPU;
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  bool startQueryExec = false;
  std::vector<size_t> assigned_executor_ids;
  executor->enableRuntimeQueryInterrupt(RUNNING_QUERY_INTERRUPT_CHECK_FREQ,
                                        PENDING_QUERY_INTERRUPT_CHECK_FREQ);
  std::shared_ptr<ResultSet> res1 = nullptr;
  std::shared_ptr<ResultSet> res2 = nullptr;
  std::exception_ptr exception_ptr1 = nullptr;
  std::exception_ptr exception_ptr2 = nullptr;
  try {
    std::string query_session = generate_random_string(32);
    // we first run the query as async function call
    auto query_thread1 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        res = run_query(test_query_large, dt, query_session, "session1");
      } catch (...) {
        exception_ptr1 = std::current_exception();
      }
      return res;
    });

    while (assigned_executor_ids.empty()) {
      assigned_executor_ids = executor->getExecutorIdsRunningQuery(query_session);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK_EQ(assigned_executor_ids.size(), static_cast<size_t>(1));

    // wait until our server starts to process the first query
    std::string curRunningSession = "";
    size_t assigned_executor_id = *assigned_executor_ids.begin();
    auto assigned_executor_ptr = Executor::getExecutor(assigned_executor_id);
    CHECK(assigned_executor_ptr);
    while (!startQueryExec) {
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      curRunningSession =
          assigned_executor_ptr->getCurrentQuerySession(session_read_lock);
      if (curRunningSession == query_session) {
        startQueryExec = true;
      }
    }
    // then, after query execution is started, we try to interrupt the running query
    // by providing the interrupt signal with the running session info
    executor->interrupt(query_session, query_session);
    res1 = query_thread1.get();
    if (exception_ptr1 != nullptr) {
      startQueryExec = false;
      auto query_thread2 = std::async(std::launch::async, [&] {
        std::shared_ptr<ResultSet> res = nullptr;
        try {
          res = run_query(test_query_small, dt, query_session, "session1");
        } catch (...) {
          exception_ptr2 = std::current_exception();
        }
        return res;
      });
      res2 = query_thread2.get();
      std::rethrow_exception(exception_ptr1);
    }
  } catch (const std::runtime_error& e) {
    // the first SELECT query fails due to interruption
    CHECK(exception_ptr1);
    std::string received_err_msg = e.what();
    auto check_interrupted_msg = received_err_msg.find("interrupted");
    CHECK(check_interrupted_msg != std::string::npos) << received_err_msg;

    // the second SELECT query that is fired after interrupting the first query
    // should be successfully evaluated
    CHECK(!exception_ptr2);
    CHECK_EQ(1, (int64_t)res2.get()->rowCount());
    auto crt_row = res2.get()->getNextRow(false, false);
    auto ret_val = v<int64_t>(crt_row[0]);
    CHECK_EQ((int64_t)1000 * 1000, ret_val);
  }
}

TEST(Interrupt, Kill_PendingQuery) {
  auto dt = ExecutorDeviceType::CPU;
  std::future<std::shared_ptr<ResultSet>> query_thread1;
  std::future<std::shared_ptr<ResultSet>> query_thread2;
  QR::get()->resizeDispatchQueue(2);
  std::vector<size_t> assigned_executor_ids_for_session1;
  std::vector<size_t> assigned_executor_ids_for_session2;
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  executor->enableRuntimeQueryInterrupt(RUNNING_QUERY_INTERRUPT_CHECK_FREQ,
                                        PENDING_QUERY_INTERRUPT_CHECK_FREQ);
  bool startQueryExec = false;
  std::exception_ptr exception_ptr1 = nullptr;
  std::exception_ptr exception_ptr2 = nullptr;
  std::shared_ptr<ResultSet> res1 = nullptr;
  std::shared_ptr<ResultSet> res2 = nullptr;
  try {
    std::string session1 = generate_random_string(32);
    std::string session2 = generate_random_string(32);
    // we first run the query as async function call
    query_thread1 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        res = run_query(test_query_medium, dt, session1, "session1");
      } catch (...) {
        exception_ptr1 = std::current_exception();
      }
      return res;
    });

    while (assigned_executor_ids_for_session1.empty()) {
      assigned_executor_ids_for_session1 = executor->getExecutorIdsRunningQuery(session1);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK_EQ(assigned_executor_ids_for_session1.size(), static_cast<size_t>(1));

    // make sure our server recognizes a session for running query correctly
    std::string curRunningSession = "";
    size_t assigned_executor_id_for_session1 =
        *assigned_executor_ids_for_session1.begin();
    auto assigned_executor_ptr_for_session1 =
        Executor::getExecutor(assigned_executor_id_for_session1);
    CHECK(assigned_executor_ptr_for_session1);
    while (!startQueryExec) {
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
      curRunningSession =
          assigned_executor_ptr_for_session1->getCurrentQuerySession(session_read_lock);
      if (curRunningSession == session1) {
        startQueryExec = true;
      }
    }
    query_thread2 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        // run pending query as async call
        res = run_query(test_query_medium, dt, session2, "session2");
      } catch (...) {
        exception_ptr2 = std::current_exception();
      }
      return res;
    });
    bool s2_enrolled = false;
    while (assigned_executor_ids_for_session2.empty()) {
      assigned_executor_ids_for_session2 = executor->getExecutorIdsRunningQuery(session2);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK_EQ(assigned_executor_ids_for_session2.size(), static_cast<size_t>(1));
    size_t assigned_executor_id_for_session2 =
        *assigned_executor_ids_for_session2.begin();
    auto assigned_executor_ptr_for_session2 =
        Executor::getExecutor(assigned_executor_id_for_session2);
    CHECK(assigned_executor_ptr_for_session2);
    while (!s2_enrolled) {
      {
        mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
        s2_enrolled = assigned_executor_ptr_for_session2->checkIsQuerySessionEnrolled(
            session2, session_read_lock);
      }
      if (s2_enrolled) {
        break;
      }
    }
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
    std::string received_err_msg = e.what();
    auto check_interrupted_msg = received_err_msg.find("interrupted");
    CHECK(check_interrupted_msg != std::string::npos) << received_err_msg;
    std::cout << received_err_msg << "\n";
  }
  // check running query's result
  CHECK_EQ(1, (int64_t)res1.get()->rowCount());
  auto crt_row = res1.get()->getNextRow(false, false);
  auto ret_val = v<int64_t>(crt_row[0]);
  CHECK_EQ((int64_t)100000 * 100000, ret_val);
}

TEST(Interrupt, Make_PendingQuery_Run) {
  auto dt = ExecutorDeviceType::CPU;
  std::future<std::shared_ptr<ResultSet>> query_thread1;
  std::future<std::shared_ptr<ResultSet>> query_thread2;
  QR::get()->resizeDispatchQueue(1);
  std::vector<size_t> assigned_executor_ids_for_session1;
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  executor->enableRuntimeQueryInterrupt(RUNNING_QUERY_INTERRUPT_CHECK_FREQ,
                                        PENDING_QUERY_INTERRUPT_CHECK_FREQ);
  bool startQueryExec = false;
  std::exception_ptr exception_ptr1 = nullptr;
  std::exception_ptr exception_ptr2 = nullptr;
  std::shared_ptr<ResultSet> res1 = nullptr;
  std::shared_ptr<ResultSet> res2 = nullptr;
  try {
    std::string session1 = generate_random_string(32);
    std::string session2 = generate_random_string(32);
    // we first run the query as async function call
    query_thread1 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        res = run_query(test_query_large, dt, session1, "session1");
      } catch (...) {
        exception_ptr1 = std::current_exception();
      }
      return res;
    });

    while (assigned_executor_ids_for_session1.empty()) {
      assigned_executor_ids_for_session1 = executor->getExecutorIdsRunningQuery(session1);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK_EQ(assigned_executor_ids_for_session1.size(), static_cast<size_t>(1));

    // make sure our server recognizes a session for running query correctly
    std::string curRunningSession = "";
    size_t assigned_executor_id_for_session1 =
        *assigned_executor_ids_for_session1.begin();
    auto assigned_executor_ptr_for_session1 =
        Executor::getExecutor(assigned_executor_id_for_session1);
    CHECK(assigned_executor_ptr_for_session1);
    while (!startQueryExec) {
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
      curRunningSession =
          assigned_executor_ptr_for_session1->getCurrentQuerySession(session_read_lock);
      if (curRunningSession == session1) {
        startQueryExec = true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    query_thread2 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        // run pending query as async call
        res = run_query(test_query_small, dt, session2, "session2");
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
    std::string received_err_msg = e.what();
    auto check_interrupted_msg = received_err_msg.find("interrupted");
    CHECK(check_interrupted_msg != std::string::npos) << received_err_msg;
  }
  // check running query's result
  CHECK_EQ(1, (int64_t)res2.get()->rowCount());
  auto crt_row = res2.get()->getNextRow(false, false);
  auto ret_val = v<int64_t>(crt_row[0]);
  CHECK_EQ((int64_t)1000 * 1000, ret_val);
}

TEST(Interrupt, Interrupt_Session_Running_Multiple_Queries) {
  // Session1 fires four queries under four parallel executors
  // Let say Session1's query Q1 runs, then remaining queries (Q2~Q4) become pending query
  // and they are waiting to get the executor lock that is held by 1
  // now this test checks an interrupt request on the Session1
  // can kill not only running but also pending queries simultaneously
  auto dt = ExecutorDeviceType::CPU;
  std::future<std::shared_ptr<ResultSet>> query_thread1;
  std::future<std::shared_ptr<ResultSet>> query_thread2;
  std::future<std::shared_ptr<ResultSet>> query_thread3;
  std::future<std::shared_ptr<ResultSet>> query_thread4;
  std::future<std::shared_ptr<ResultSet>> query_thread5;
  QR::get()->resizeDispatchQueue(4);
  std::vector<size_t> assigned_executor_ids_for_session1;
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  executor->enableRuntimeQueryInterrupt(RUNNING_QUERY_INTERRUPT_CHECK_FREQ,
                                        PENDING_QUERY_INTERRUPT_CHECK_FREQ);
  bool startQueryExec = false;
  std::atomic<bool> catchInterruption(false);
  std::atomic<bool> detect_time_out = false;
  std::exception_ptr exception_ptr1 = nullptr;
  std::exception_ptr exception_ptr2 = nullptr;
  std::exception_ptr exception_ptr3 = nullptr;
  std::exception_ptr exception_ptr4 = nullptr;
  std::exception_ptr exception_ptr5 = nullptr;
  std::shared_ptr<ResultSet> res1 = nullptr;
  std::shared_ptr<ResultSet> res2 = nullptr;
  std::shared_ptr<ResultSet> res3 = nullptr;
  std::shared_ptr<ResultSet> res4 = nullptr;
  std::shared_ptr<ResultSet> res5 = nullptr;
  std::future_status q1_status;
  std::future_status q2_status;
  std::future_status q3_status;
  std::future_status q4_status;

  try {
    std::string session1 = generate_random_string(32);
    // we first run the query as async function call
    query_thread1 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        res = run_query(test_query_large, dt, session1, "session1");
      } catch (...) {
        exception_ptr1 = std::current_exception();
      }
      return res;
    });

    while (assigned_executor_ids_for_session1.empty()) {
      assigned_executor_ids_for_session1 = executor->getExecutorIdsRunningQuery(session1);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK_EQ(assigned_executor_ids_for_session1.size(), static_cast<size_t>(1));

    // make sure our server recognizes a session for running query correctly
    std::string curRunningSession = "";
    size_t assigned_executor_id_for_session1 =
        *assigned_executor_ids_for_session1.begin();
    auto assigned_executor_ptr_for_session1 =
        Executor::getExecutor(assigned_executor_id_for_session1);
    CHECK(assigned_executor_ptr_for_session1);
    while (!startQueryExec) {
      mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
      curRunningSession =
          assigned_executor_ptr_for_session1->getCurrentQuerySession(session_read_lock);
      if (curRunningSession.compare(session1) == 0) {
        startQueryExec = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(startQueryExec);

    query_thread2 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        // run pending query as async call
        res = run_query(test_query_large, dt, session1, "session1");
      } catch (...) {
        exception_ptr2 = std::current_exception();
      }
      return res;
    });
    query_thread3 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        // run pending query as async call
        res = run_query(test_query_large, dt, session1, "session1");
      } catch (...) {
        exception_ptr3 = std::current_exception();
      }
      return res;
    });
    query_thread4 = std::async(std::launch::async, [&] {
      std::shared_ptr<ResultSet> res = nullptr;
      try {
        // run pending query as async call
        res = run_query(test_query_large, dt, session1, "session1");
      } catch (...) {
        exception_ptr4 = std::current_exception();
      }
      return res;
    });

    // then, we try to interrupt the running query
    // by providing the interrupt signal with the running query's session info
    // so we can expect that running query session releases all H/W resources and locks,
    // and so pending query takes them for its query execution (becomes running query)
    int queue_size = -1;
    bool send_interrupt_signal = false;
    while (startQueryExec) {
      // check all Q1~Q4 of Session1 are enrolled in the session map
      {
        mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
        queue_size = executor->getQuerySessionInfo(session1, session_read_lock).size();
      }
      if (queue_size == 4) {
        executor->interrupt(session1, session1);
        send_interrupt_signal = true;
        break;
      }
    }
    CHECK(send_interrupt_signal);

    auto check_interrup_msg = [&catchInterruption](const std::string& msg,
                                                   bool is_pending_query) {
      auto check_interrupted_msg = msg.find("interrupted");
      CHECK(check_interrupted_msg != std::string::npos) << msg;
      catchInterruption.store(true);
    };

    auto get_query_status_with_timeout =
        [&detect_time_out](std::future<std::shared_ptr<ResultSet>>& thread,
                           std::future_status& status,
                           std::shared_ptr<ResultSet> res,
                           size_t timeout_sec) {
          do {
            status = thread.wait_for(std::chrono::seconds(timeout_sec));
            if (status == std::future_status::timeout) {
              detect_time_out.store(true);
              res = thread.get();
              break;
            } else if (status == std::future_status::ready) {
              res = thread.get();
            }
          } while (status != std::future_status::ready);
        };

    get_query_status_with_timeout(query_thread1, q1_status, res1, 60);
    get_query_status_with_timeout(query_thread2, q2_status, res2, 60);
    get_query_status_with_timeout(query_thread3, q3_status, res3, 60);
    get_query_status_with_timeout(query_thread4, q4_status, res4, 60);

    if (exception_ptr1 != nullptr) {
      try {
        std::rethrow_exception(exception_ptr1);
      } catch (const std::runtime_error& e) {
        check_interrup_msg(e.what(), false);
      }
    }
    if (exception_ptr2 != nullptr) {
      try {
        std::rethrow_exception(exception_ptr2);
      } catch (const std::runtime_error& e) {
        check_interrup_msg(e.what(), true);
      }
    }
    if (exception_ptr3 != nullptr) {
      try {
        std::rethrow_exception(exception_ptr3);
      } catch (const std::runtime_error& e) {
        check_interrup_msg(e.what(), true);
      }
    }
    if (exception_ptr4 != nullptr) {
      try {
        std::rethrow_exception(exception_ptr4);
      } catch (const std::runtime_error& e) {
        check_interrup_msg(e.what(), true);
      }
    }

    if (catchInterruption.load()) {
      {
        mapd_shared_lock<mapd_shared_mutex> session_read_lock(executor->getSessionLock());
        queue_size = executor->getQuerySessionInfo(session1, session_read_lock).size();
      }
      CHECK_EQ(queue_size, 0);
      throw std::runtime_error("SUCCESS");
    }

    if (detect_time_out.load()) {
      throw std::runtime_error("TIME_OUT");
    }

  } catch (const std::runtime_error& e) {
    // catch exception due to runtime query interrupt
    // and compare thrown message to confirm that
    // this exception comes from our interrupt request
    std::string received_err_msg = e.what();
    if (received_err_msg.compare("TIME_OUT") == 0) {
      // catch time_out scenario, so returns immediately to avoid
      // unexpected hangs of our jenkins
      return;
    } else if (received_err_msg.compare("SUCCESS") == 0) {
      // make sure we interrupt the query correctly
      CHECK(catchInterruption.load());
      // if a query is interrupted, its resultset ptr should be nullptr
      CHECK(!res1);  // for Q1 of Session1
      CHECK(!res2);  // for Q2 of Session1
      CHECK(!res3);  // for Q3 of Session1
      CHECK(!res4);  // for Q4 of Session1

      // check the current queue is empty
      // if so, it should be available to schedule the next query
      std::string session2 = generate_random_string(32);
      query_thread5 = std::async(std::launch::async, [&] {
        std::shared_ptr<ResultSet> res = nullptr;
        try {
          res = run_query(test_query_small, dt, session2, "session2");
        } catch (...) {
          exception_ptr5 = std::current_exception();
        }
        return res;
      });
      res5 = query_thread5.get();
      CHECK(!exception_ptr5);
      CHECK_EQ(1, (int64_t)res5.get()->rowCount());
      auto crt_row = res5.get()->getNextRow(false, false);
      auto ret_val = v<int64_t>(crt_row[0]);
      CHECK_EQ((int64_t)1000 * 1000, ret_val);
    }
  }
}

TEST(Non_Kernel_Time_Interrupt, Interrupt_COPY_statement_CSV) {
  std::atomic<bool> catchInterruption(false);
  std::atomic<bool> detect_time_out(false);
  std::string import_very_large_table_str{
      "COPY t_very_large_csv FROM "
      "'../../Tests/Import/datafiles/interrupt_table_very_large.csv' WITH "
      "(header='false')"};

  auto check_interrup_msg = [&catchInterruption](const std::string& msg,
                                                 bool is_pending_query) {
    auto check_interrupted_msg = msg.find("interrupted");
    CHECK(check_interrupted_msg != std::string::npos) << msg;
    catchInterruption.store(true);
  };

  try {
    std::string session1 = generate_random_string(32);
    QR::get()->addSessionId(session1, "session1", ExecutorDeviceType::CPU);

    auto interrupt_thread = std::async(std::launch::async, [&] {
      // make sure our server recognizes a session for running query correctly
      QuerySessionStatus::QueryStatus curRunningSessionStatus;
      bool startQueryExec = false;
      auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
      int cnt = 0;
      while (cnt < 6000) {
        {
          mapd_shared_lock<mapd_shared_mutex> session_read_lock(
              executor->getSessionLock());
          curRunningSessionStatus =
              executor->getQuerySessionStatus(session1, session_read_lock);
        }
        if (curRunningSessionStatus == QuerySessionStatus::RUNNING_IMPORTER) {
          startQueryExec = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++cnt;
        if (cnt == 6000) {
          std::cout << "Detect timeout while performing COPY stmt on csv table"
                    << std::endl;
          detect_time_out.store(true);
          return;
        }
      }
      CHECK(startQueryExec);
      executor->interrupt(session1, session1);
      return;
    });

    try {
      QR::get()->runDDLStatement(import_very_large_table_str);
    } catch (const QueryExecutionError& e) {
      if (e.getErrorCode() == Executor::ERR_INTERRUPTED) {
        catchInterruption.store(true);
      } else {
        throw e;
      }
    } catch (const std::runtime_error& e) {
      check_interrup_msg(e.what(), false);
    } catch (...) {
      throw;
    }

    if (catchInterruption.load()) {
      std::cout << "Detect interrupt request while performing COPY stmt on csv table"
                << std::endl;
      std::shared_ptr<ResultSet> res = run_query("SELECT COUNT(1) FROM t_very_large_csv",
                                                 ExecutorDeviceType::CPU,
                                                 session1,
                                                 "session1");
      CHECK_EQ(1, (int64_t)res.get()->rowCount());
      auto crt_row = res.get()->getNextRow(false, false);
      auto ret_val = v<int64_t>(crt_row[0]);
      CHECK_EQ((int64_t)0, ret_val);
      return;
    }
    if (detect_time_out.load()) {
      return;
    }
  } catch (...) {
    CHECK(false);
  }
}

TEST(Non_Kernel_Time_Interrupt, Interrupt_COPY_statement_Parquet) {
  std::atomic<bool> catchInterruption(false);
  std::atomic<bool> detect_time_out(false);
  std::string import_very_large_parquet_table_str{
      "COPY t_very_large_parquet FROM "
      "'../../Tests/Import/datafiles/interrupt_table_very_large.parquet' WITH "
      "(header='false', parquet='true')"};

  try {
    std::string session1 = generate_random_string(32);
    QR::get()->addSessionId(session1, "session1", ExecutorDeviceType::CPU);
    auto check_interrup_msg = [&catchInterruption](const std::string& msg,
                                                   bool is_pending_query) {
      auto check_interrupted_msg = msg.find("interrupted");
      CHECK(check_interrupted_msg != std::string::npos) << msg;
      catchInterruption.store(true);
    };

    auto interrupt_thread = std::async(std::launch::async, [&] {
      // make sure our server recognizes a session for running query correctly
      QuerySessionStatus::QueryStatus curRunningSessionStatus;
      bool startQueryExec = false;
      auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
      int cnt = 0;
      while (cnt < 6000) {
        {
          mapd_shared_lock<mapd_shared_mutex> session_read_lock(
              executor->getSessionLock());
          curRunningSessionStatus =
              executor->getQuerySessionStatus(session1, session_read_lock);
        }
        if (curRunningSessionStatus == QuerySessionStatus::RUNNING_IMPORTER) {
          startQueryExec = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++cnt;
        if (cnt == 6000) {
          std::cout << "Detect timeout while performing COPY stmt on csv table"
                    << std::endl;
          detect_time_out.store(true);
          return;
        }
      }
      CHECK(startQueryExec);
      executor->interrupt(session1, session1);
      return;
    });

    try {
      QR::get()->runDDLStatement(import_very_large_parquet_table_str);
    } catch (const QueryExecutionError& e) {
      if (e.getErrorCode() == Executor::ERR_INTERRUPTED) {
        catchInterruption.store(true);
      } else {
        throw e;
      }
    } catch (const std::runtime_error& e) {
      check_interrup_msg(e.what(), false);
    } catch (...) {
      throw;
    }

    if (catchInterruption.load()) {
      std::cout << "Detect interrupt request while performing COPY stmt on parquet table"
                << std::endl;
      std::shared_ptr<ResultSet> res =
          run_query("SELECT COUNT(1) FROM t_very_large_parquet",
                    ExecutorDeviceType::CPU,
                    session1,
                    "session1");
      CHECK_EQ(1, (int64_t)res.get()->rowCount());
      auto crt_row = res.get()->getNextRow(false, false);
      auto ret_val = v<int64_t>(crt_row[0]);
      CHECK_EQ((int64_t)0, ret_val);
      return;
    }
    if (detect_time_out.load()) {
      return;
    }
  } catch (...) {
    CHECK(false);
  }
}

TEST(Non_Kernel_Time_Interrupt, Interrupt_During_Reduction) {
  std::atomic<bool> catchInterruption(false);
  std::atomic<bool> detect_time_out(false);
  std::vector<size_t> assigned_executor_ids;
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  bool keep_global_columnar_flag = g_enable_columnar_output;
  g_enable_columnar_output = true;
  ScopeGuard reset_flag = [keep_global_columnar_flag] {
    g_enable_columnar_output = keep_global_columnar_flag;
  };

  auto check_interrup_msg = [&catchInterruption](const std::string& msg,
                                                 bool is_pending_query) {
    auto check_interrupted_msg = msg.find("interrupted");
    std::string expected_msg{
        "Query execution has interrupted during result set reduction"};
    CHECK((check_interrupted_msg != std::string::npos) || expected_msg.compare(msg) == 0)
        << msg;
    catchInterruption.store(true);
    std::cout << msg << std::endl;
  };

  try {
    std::string session1 = generate_random_string(32);
    QR::get()->addSessionId(session1, "session1", ExecutorDeviceType::CPU);

    auto interrupt_thread = std::async(std::launch::async, [&] {
      while (assigned_executor_ids.empty()) {
        assigned_executor_ids = executor->getExecutorIdsRunningQuery(session1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      std::string curRunningSession{""};
      bool startReduction = false;
      QuerySessionStatus::QueryStatus qss = QuerySessionStatus::QueryStatus::UNDEFINED;
      size_t assigned_executor_id = *assigned_executor_ids.begin();
      auto assigned_executor_ptr = Executor::getExecutor(assigned_executor_id);
      CHECK(assigned_executor_ptr);
      int cnt = 0;
      while (cnt < 6000) {
        {
          mapd_shared_lock<mapd_shared_mutex> session_read_lock(
              executor->getSessionLock());
          curRunningSession =
              assigned_executor_ptr->getCurrentQuerySession(session_read_lock);
          if (curRunningSession.compare(session1) == 0) {
            qss = assigned_executor_ptr
                      ->getQuerySessionInfo(curRunningSession, session_read_lock)[0]
                      .getQueryStatus();
            if (qss == QuerySessionStatus::QueryStatus::RUNNING_REDUCTION) {
              startReduction = true;
              break;
            }
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++cnt;
        if (cnt == 6000) {
          std::cout << "Detect timeout while performing reduction" << std::endl;
          detect_time_out.store(true);
          return;
        }
      }
      CHECK(startReduction);
      assigned_executor_ptr->interrupt(session1, session1);
      return;
    });

    try {
      run_query("SELECT k1, k2, AVG(i) FROM t_very_large_agg GROUP BY k1, k2",
                ExecutorDeviceType::CPU,
                session1);
      CHECK_EQ(assigned_executor_ids.size(), static_cast<size_t>(1));
    } catch (const QueryExecutionError& e) {
      if (e.getErrorCode() == Executor::ERR_INTERRUPTED) {
        // timing issue... the query is interrupted before entering the reduction
        catchInterruption.store(true);
      } else {
        throw e;
      }
    } catch (const std::runtime_error& e) {
      check_interrup_msg(e.what(), false);
    } catch (...) {
      throw;
    }

    if (catchInterruption.load()) {
      std::cout << "Detect interrupt request while performing reduction" << std::endl;
      return;
    }
    if (detect_time_out.load()) {
      return;
    }
  } catch (...) {
    CHECK(false);
  }
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  TestHelpers::init_logger_stderr_only(argc, argv);
  g_enable_runtime_query_interrupt = true;
  g_enable_non_kernel_time_query_interrupt = true;

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
