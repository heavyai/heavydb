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
#include <sys/socket.h>
#include <unistd.h>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>
#include "DBHandlerTestHelpers.h"
#include "Shared/ThriftClient.h"
#include "Tests/TestHelpers.h"
#include "ThriftHandler/DBHandler.h"
#include "gen-cpp/Heavy.h"

std::filesystem::path binary_path;

using namespace apache::thrift::protocol;
namespace bp = boost::process;

namespace {
// Finds a free port on the system.
int32_t get_free_port() {
  auto sock = socket(AF_INET, SOCK_STREAM, 0);  // create a socket
  CHECK(sock >= 0);
  struct sockaddr_in serv_addr;
  memset(&serv_addr, 0, sizeof(serv_addr));  // zero-initialize
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  // Because the try to bind to address 0, bind will return an arbitrary free socket.
  CHECK_GE(bind(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)), 0);
  socklen_t serv_len = sizeof(serv_addr);
  // getsockname will populate serv_addr with socket bind() assigned to sock.
  CHECK_GE(getsockname(sock, (struct sockaddr*)&serv_addr, &serv_len), 0);
  return serv_addr.sin_port;
}
}  // namespace

struct ServerPorts {
  int32_t server = 0;
  int32_t http = 0;
  int32_t calcite = 0;
  int32_t http_binary = 0;

  // Default constructor will find free ports on the system for each field.
  ServerPorts() {
    server = get_free_port();
    http = get_free_port();
    calcite = get_free_port();
    http_binary = get_free_port();
  }
};

struct Connection {
  std::shared_ptr<ThriftClientConnection> client_conn;
  std::shared_ptr<TTransport> transport;
  std::shared_ptr<TBinaryProtocol> protocol;
  std::shared_ptr<HeavyClient> client;
  TSessionId session;

  Connection(const std::string& host,
             const int32_t port,
             const std::string& cert = "",
             const std::string& user = shared::kRootUsername,
             const std::string& pwd = shared::kDefaultRootPasswd,
             const std::string& db = shared::kDefaultDbName) {
    while (true) {
      try {
        client_conn = std::make_shared<ThriftClientConnection>();
        transport = client_conn->open_buffered_client_transport(host, port, cert);
        transport->open();
        protocol = std::make_shared<TBinaryProtocol>(transport);
        client = std::make_shared<HeavyClient>(protocol);
        client->connect(session, user, pwd, db);
        std::cout << "connected to server on port " << port << "\n";
        break;
      } catch (...) {
        // Keep trying to connect until we are successful.
      }
    }
  }

  TQueryResult sql_execute(const std::string& sql) {
    TQueryResult result;
    client->sql_execute(result, session, sql, false, "", -1, -1);
    return result;
  }
};

class CloudEnvironment : public ::testing::Environment {
 public:
  constexpr static char localhost[] = "localhost";
  constexpr static char base_path[] = "./tmp";

  void SetUp() override {
    ASSERT_TRUE(boost::filesystem::exists(base_path));

    ServerPorts ports_1, ports_2;

    // Start servers on separate processes.
    child_processes_.emplace_back(startServerCommand(ports_1, base_path));
    child_processes_.emplace_back(startServerCommand(ports_2, base_path));

    std::cout << "waiting for servers to start...\n";
    std::this_thread::sleep_for(20'000ms);

    // Connect clients to servers.
    servers_.emplace_back(Connection(localhost, ports_1.server));
    servers_.emplace_back(Connection(localhost, ports_2.server));

    // When DBHandlers are destroyed they will shutdown the associated server.
    std::cout << "finishing connecting to servers.\n";
  }

  void TearDown() override {
    // Shutdown created servers.
    for (auto it = child_processes_.begin(); it != child_processes_.end();) {
      auto pid = it->id();
      int32_t exit_status;
      kill(pid, SIGTERM);
      waitpid(pid, &exit_status, 0);
      it = child_processes_.erase(it);
    }
  }

  std::string startServerCommand(const ServerPorts& ports,
                                 const std::string& base_path) const {
    // ServerPorts ports;
    std::stringstream ss;
    ss << binary_path.parent_path().string() << "/../bin/heavydb --port " << ports.server
       << " --http-port " << ports.http << " --calcite-port " << ports.calcite
       << " --http-binary-port " << ports.http_binary
       << " --multi-instance true --log-severity=DEBUG2 " << base_path;
    std::cout << "starting server: " << ss.str() << "\n";
    return ss.str();
  }

  std::vector<Connection> servers_;
  std::vector<bp::child> child_processes_;
};

CloudEnvironment* g_cloud_env;

class TableTest : public ::testing::Test {
 protected:
  constexpr static char select_table_0[] = "SELECT * FROM test_table0;",
                        select_table_1[] = "SELECT * FROM test_table1;",
                        drop_table_0[] = "DROP TABLE IF EXISTS test_table0;",
                        drop_table_1[] = "DROP TABLE IF EXISTS test_table1;",
                        create_table_0[] = "CREATE TABLE test_table0 (i int);",
                        not_found_table_0[] = "Object 'test_table0' not found",
                        not_found_table_1[] = "Object 'test_table1' not found";

  void SetUp() override { TearDown(); }

  void TearDown() override {
    executeOnServer(drop_table_0);
    executeOnServer(drop_table_1);
  }

  TQueryResult executeOnServer(const std::string& query, size_t server = 0) {
    return g_cloud_env->servers_[server].sql_execute(query);
  }

  void sqlAndAssertResult(
      const std::string& sql,
      const std::vector<std::vector<NullableTargetValue>>& expected_result,
      size_t server = 0) {
    DBHandlerTestFixture::assertResultSetEqual(expected_result,
                                               executeOnServer(sql, server));
  }

  void sqlAndCompareException(const std::string& sql,
                              const std::string& exception_string,
                              size_t server = 0) {
    try {
      executeOnServer(sql, server);
      FAIL() << "expected exception with text: '" << exception_string << "'.";
    } catch (const TDBException& e) {
      std::string err_msg = e.what();
      if (err_msg.find(exception_string) == std::string::npos) {
        throw;
      }
    } catch (const std::runtime_error& e) {
      std::string err_msg = e.what();
      if (err_msg.find(exception_string) == std::string::npos) {
        throw;
      }
    }
  }
};

// TODO(Misiu): Abstract tests so that we can permute our various read/write statements.
TEST_F(TableTest, Write) {
  // Setup
  executeOnServer(create_table_0);

  // Preconditions
  sqlAndAssertResult(select_table_0, {});
  sqlAndAssertResult(select_table_0, {}, 1);

  // Alter table on one server
  executeOnServer("INSERT INTO test_table0 values (1);");

  // Verify insert is reflected on both servers.
  sqlAndAssertResult(select_table_0, {{1L}}, 1);
  sqlAndAssertResult(select_table_0, {{1L}}, 0);
}

TEST_F(TableTest, Drop) {
  // Setup
  executeOnServer(create_table_0);

  // Preconditions
  sqlAndAssertResult(select_table_0, {}, 0);
  sqlAndAssertResult(select_table_0, {}, 1);

  // Create a new table
  executeOnServer("DROP TABLE test_table0;");

  // Verify create is reflected on both servers.
  sqlAndCompareException(select_table_0, not_found_table_0, 1);
  sqlAndCompareException(select_table_0, not_found_table_0, 0);
}

TEST_F(TableTest, Create) {
  // Preconditions
  sqlAndCompareException(select_table_0, not_found_table_0, 0);
  sqlAndCompareException(select_table_0, not_found_table_0, 1);

  // Create a new table
  executeOnServer(create_table_0);

  // Verify create is reflected on both servers.
  sqlAndAssertResult(select_table_0, {}, 1);
  sqlAndAssertResult(select_table_0, {}, 0);
}

TEST_F(TableTest, Alter) {
  GTEST_SKIP() << "Test currently causes issues with server teardown.";
  // Setup test
  executeOnServer(create_table_0);

  // Preconditions
  sqlAndAssertResult(select_table_0, {}, 0);
  sqlAndAssertResult(select_table_0, {}, 1);

  sqlAndCompareException(select_table_1, not_found_table_1, 0);
  sqlAndCompareException(select_table_1, not_found_table_1, 1);

  // Alter table on one server
  executeOnServer("ALTER TABLE test_table0 rename to test_table1;");

  // Verify insert is reflected on both servers.
  sqlAndCompareException(select_table_0, not_found_table_0, 1);
  sqlAndCompareException(select_table_0, not_found_table_0, 0);

  sqlAndAssertResult(select_table_1, {}, 1);
  sqlAndAssertResult(select_table_1, {}, 0);
}

void shutdown_children(int32_t signum) {
  g_cloud_env->TearDown();  // shut down children.
  signal(signum, SIG_DFL);  // reset to default handler.
  raise(signum);            // raise signal with default handler.
}

void register_signals() {
  // Before we handle any signals we need to shut down the child processes.
  for (auto sig : {SIGABRT, SIGFPE, SIGILL, SIGINT, SIGSEGV, SIGTERM}) {
    signal(sig, shutdown_children);
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  binary_path = std::filesystem::canonical(argv[0]);
  testing::InitGoogleTest(&argc, argv);

  int err = 0;
  try {
    g_cloud_env = dynamic_cast<CloudEnvironment*>(
        testing::AddGlobalTestEnvironment(new CloudEnvironment));
    register_signals();
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
