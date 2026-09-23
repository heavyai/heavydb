/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>
#include <boost/process.hpp>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <thread>
#include <type_traits>

#include <thrift/protocol/TBinaryProtocol.h>

#include "Shared/SysDefinitions.h"
#include "Shared/ThriftClient.h"
#include "Tests/DBHandlerTestHelpers.h"  // TODO(Misiu): Remove the dep when possible.
#include "ThriftHandler/DBHandler.h"
#include "gen-cpp/Heavy.h"

using apache::thrift::protocol::TBinaryProtocol;
using apache::thrift::transport::TTransport;

extern bool g_enable_fsi;

std::filesystem::path binary_path;

// This is an odd hack that is needed because the Go compiler has some issues with our C++
// program linking a Go library compiled from a C++ file containg C functions.  This gets
// overwritten later in linking, but we need a default definition to avoid a linking
// error.
extern "C" void etcd_onMemberUpdate_thunk(void* self,
                                          char* member_name,
                                          char* hostip_cstr,
                                          int port) {}

namespace bp = boost::process;

using RowSet = std::vector<NullableTargetValue>;
using ExpectedResultSet = std::vector<RowSet>;
using ServerList = std::vector<size_t>;
using Results = std::vector<TQueryResult>;

// Used to print our variant types (NullableTargetValues) as a string.
class PrintVisitor : boost::static_visitor<> {
 public:
  // TODO(Misiu): Does not currently support all types (GeoTargetValue,
  // ArrayTargetValue, etc...).

  template <typename T>
  std::string operator()(const T& val) const {
    throw std::runtime_error("Unknown value type: "s + typeid(val).name());
    return "error";  // Unreachable but necessary for compiler.
  }

  template <typename... Args>
  std::string operator()(const boost::variant<Args...>& val) const {
    return boost::apply_visitor(PrintVisitor(), val);
  }

  std::string operator()(const int64_t val) const { return std::to_string(val); }
  std::string operator()(const double val) const { return std::to_string(val); }
  std::string operator()(const float val) const { return std::to_string(val); }
  std::string operator()(const std::string& val) const { return val; }
  std::string operator()(const void* val) const { return "NULL"; }
};

// Object types we are interested in testing.
enum class ObjectType {
  Table,
  View,
  Database,
  Dashboard,
  User,
  Role,
  ForeignTable,
  Server,
  CustomExpression
};

namespace {
// Finds a free port on the system.
int32_t get_free_port(const std::set<int32_t>& banned_ports = {}) {
  int32_t port = 0;
  // bind() may return a port reserved by the system, so we want to make sure the port we
  // got is above 1023 (a registered port, not a system port).
  while (port < 1024 || banned_ports.count(port) > 0) {
    auto sock = socket(AF_INET, SOCK_STREAM, 0);  // create a socket
    CHECK(sock >= 0);
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));  // zero-initialize
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    // Because the try to bind to address 0, bind will return an arbitrary free socket.
    CHECK_GE(
        bind(sock, reinterpret_cast<struct sockaddr*>(&serv_addr), sizeof(serv_addr)), 0);
    socklen_t serv_len = sizeof(serv_addr);
    // getsockname will populate serv_addr with socket bind() assigned to sock.
    CHECK_GE(getsockname(sock, reinterpret_cast<struct sockaddr*>(&serv_addr), &serv_len),
             0);
    port = serv_addr.sin_port;
  }
  return port;
}

// Used to explicitly cast type for the sake of expected result set creation.
constexpr int64_t i(int64_t i) {
  return i;
}

// Turn a vector of strings into a result set (for comparing the results of thrift
// functions that don't return a result set).
TQueryResult create_result(const std::vector<std::string>& cols) {
  TQueryResult result;
  result.row_set.row_desc.emplace_back(TColumnType{});
  result.row_set.row_desc[0].col_type.type = TDatumType::STR;
  result.row_set.rows.emplace_back(TRow{});
  for (auto col : cols) {
    TDatum datum;
    datum.is_null = false;
    datum.val.str_val = col;
    result.row_set.rows[0].cols.emplace_back(datum);
  }
  return result;
}

int32_t extract_result(const TQueryResult& result) {
  return std::stoi(result.row_set.rows[0].cols[0].val.str_val);
}

// Creates a file with contents based on a result set.  Used for import or FSI source
// files.
void create_file(const std::string& path, const ExpectedResultSet& values) {
  std::ofstream fs;
  fs.open(path);
  for (auto row_it = values.begin(); row_it != values.end(); ++row_it) {
    for (auto col_it = row_it->begin(); col_it != row_it->end(); ++col_it) {
      fs << boost::apply_visitor(PrintVisitor(), *col_it);
      if (std::next(col_it) != row_it->end()) {
        fs << ",";
      }
    }
    if (std::next(row_it) != values.end()) {
      fs << "\n";
    }
  }
  fs.close();
}

// Used for printing enum types.  Text is used in sql statements (create, drop, etc...).
std::string to_string(const ObjectType& type) {
  switch (type) {
    case (ObjectType::Table):
      return "TABLE";
    case (ObjectType::View):
      return "VIEW";
    case (ObjectType::Database):
      return "DATABASE";
    case (ObjectType::Dashboard):
      return "DASHBOARD";
    case (ObjectType::User):
      return "USER";
    case (ObjectType::Role):
      return "ROLE";
    case (ObjectType::ForeignTable):
      return "FOREIGN TABLE";
    case (ObjectType::Server):
      return "SERVER";
    case (ObjectType::CustomExpression):
      return "CUSTOM EXPRESSION";
  }
  return "UNKNOWN";
}

// TODO(Misiu): These functions can all be made constexpr in c++20
std::string create(const ObjectType& type,
                   const std::string& name,
                   const std::optional<std::string>& properties = {}) {
  std::stringstream ss;
  ss << "CREATE " << to_string(type) << " " << name;
  if (properties.has_value()) {
    ss << " " << properties.value();
  }
  ss << ";";
  return ss.str();
}

std::string drop(const ObjectType& type, const std::string& name, bool exists = false) {
  std::stringstream ss;
  ss << "DROP " << to_string(type) << (exists ? " IF EXISTS " : " ") << name << ";";
  return ss.str();
}

std::string alter(const ObjectType& type,
                  const std::string& name,
                  const std::string& new_name) {
  std::stringstream ss;
  ss << "ALTER " << to_string(type) << " " << name << " RENAME TO " << new_name << ";";
  return ss.str();
}

std::string itas(const std::string& name, const std::string& source) {
  std::stringstream ss;
  ss << "INSERT INTO " << name << " SELECT * FROM " << source << ";";
  return ss.str();
}

std::string ctas(const std::string& name, const std::string& source) {
  return create(ObjectType::Table, name, " AS SELECT * FROM " + source);
}

std::string select(const std::string& name, const std::string& order = "ORDER BY i") {
  std::stringstream ss;
  ss << "SELECT * FROM " << name << " " << order << ";";
  return ss.str();
}

std::string group_by(const std::string& name) {
  std::stringstream ss;
  ss << "SELECT i, COUNT(s) FROM " << name << " GROUP BY i ORDER BY i;";
  return ss.str();
}

std::string join(const std::pair<std::string, std::string>& names) {
  auto& [name_0, name_1] = names;
  std::stringstream ss;
  ss << "SELECT t1.i, t1.s FROM " << name_0 << " AS t1 JOIN " << name_1
     << " AS t2 ON t1.i = t2.i ORDER BY t1.i;";
  return ss.str();
}

// Helper function to create a string out of a single row of NullableTargetValues by
std::string insert(const RowSet& row) {
  if (row.empty()) {
    throw std::runtime_error("Can not insert empty row set.");
  }
  std::stringstream ss;
  ss << "(" << boost::apply_visitor(PrintVisitor(), row[0]);
  for (auto col_itr = ++(row.begin()); col_itr != row.end(); ++col_itr) {
    ss << ", " << boost::apply_visitor(PrintVisitor(), *col_itr);
  }
  ss << ")";
  return ss.str();
}

// Returns an sql statement that inserts an ExpectedResultSet by converting it to strings.
std::string insert(const std::string& name, const ExpectedResultSet& values) {
  if (values.empty()) {
    throw std::runtime_error("Can not insert empty set into table: '" + name + "'.");
  }
  std::stringstream ss;
  ss << "INSERT INTO " << name << " VALUES " << insert(values[0]);
  for (auto row_itr = ++(values.begin()); row_itr != values.end(); ++row_itr) {
    ss << ", " << insert(*row_itr);
  }
  ss << ";";
  return ss.str();
}

std::string copy(const std::string& name,
                 const std::string& file,
                 const ExpectedResultSet& values) {
  std::stringstream ss;
  ss << "COPY " << name << " FROM '" << file << "' WITH (HEADER='false');";
  return ss.str();
}

std::string delete_all(const std::string& name) {
  std::stringstream ss;
  ss << "DELETE FROM " << name << ";";
  return ss.str();
}

// Wrapper that executes a given function with the expectation that it throws an exception
// containing specific text.
template <typename Func, typename... Args>
void run_and_catch(const std::string& exception_text, Func func, Args... args) {
  try {
    func(args...);
    throw std::runtime_error("expected exception with text: '" + exception_text + "',");
  } catch (const TDBException& e) {  // thrift error
    std::string err_msg = e.what();
    if (err_msg.find(exception_text) == std::string::npos) {
      throw;
    }
  } catch (const std::runtime_error& e) {  // standard error
    std::string err_msg = e.what();
    if (err_msg.find(exception_text) == std::string::npos) {
      throw;
    }
  } catch (const std::exception& e) {  // error thrown by custom expressions
    std::string err_msg = e.what();
    if (err_msg.find(exception_text) == std::string::npos) {
      throw;
    }
  }
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
    http = get_free_port({server});
    calcite = get_free_port({server, http});
    http_binary = get_free_port({server, http, calcite});
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
        std::this_thread::sleep_for(1000ms);
        // Keep trying to connect until we are successful.
      }
    }
  }

  TQueryResult sqlExecute(const std::string& sql) {
    TQueryResult result;
    client->sql_execute(result, session, sql, false, "", -1, -1);
    return result;
  }

  // TODO(Misiu): Update this to use a custom schema.
  void createTable(const std::string& name) {
    TColumnType c1, c2;
    c1.col_name = "i";
    c1.col_type.type = TDatumType::INT;
    c1.col_id = 1;
    c2.col_name = "s";
    c2.col_type.type = TDatumType::STR;
    c2.col_id = 2;
    c2.col_type.encoding = TEncodingType::DICT;
    TRowDescriptor row_desc = {c1, c2};
    client->create_table(session, name, row_desc);
  }

  void loadTable(const std::string& name, const RowSet& row) {
    std::vector<TStringRow> string_rows;
    auto& string_row = string_rows.emplace_back(TStringRow{});
    for (auto& col : row) {
      auto& string_val = string_row.cols.emplace_back();
      string_val.str_val = boost::apply_visitor(PrintVisitor(), col);
    }
    client->load_table(session, name, string_rows, {});
  }

  void importTable(const std::string& name, const std::string& file) {
    TCopyParams copy_params;
    copy_params.has_header = TImportHeaderRow::NO_HEADER;
    client->import_table(session, name, file, copy_params);
  }

  std::vector<TDBInfo> getDatabases() {
    std::vector<TDBInfo> dbs;
    client->get_databases(dbs, session);
    return dbs;
  }

  std::vector<TDashboard> getDashboards() {
    std::vector<TDashboard> dashboards;
    client->get_dashboards(dashboards, session);
    return dashboards;
  }

  TDashboard getDashboard(int32_t id) {
    TDashboard dashboard;
    client->get_dashboard(dashboard, session, id);
    return dashboard;
  }

  int32_t createDashboard(const std::string& name) {
    return client->create_dashboard(session, name, "", "", "");
  }

  void deleteDashboard(int32_t id) { client->delete_dashboard(session, id); }

  std::vector<std::string> getUsers() {
    std::vector<std::string> users;
    client->get_users(users, session);
    return users;
  }

  std::vector<std::string> getRoles() {
    std::vector<std::string> roles;
    client->get_roles(roles, session);
    return roles;
  }

  std::vector<TCustomExpression> getCustomExpressions() {
    std::vector<TCustomExpression> expressions;
    client->get_custom_expressions(expressions, session);
    return expressions;
  }

  int32_t createCustomExpression(const std::string& name, const std::string& table_name) {
    TCustomExpression custom_expr;
    custom_expr.name = name;
    custom_expr.expression_json = "test_expr_json";
    custom_expr.data_source_type = TDataSourceType::type::TABLE;
    custom_expr.data_source_name = table_name;
    return client->create_custom_expression(session, custom_expr);
  }

  void deleteCustomExpressions(const std::vector<int32_t>& ids, bool is_soft_delete) {
    client->delete_custom_expressions(session, ids, is_soft_delete);
  }
};

class CloudEnvironment : public ::testing::Environment {
 public:
  constexpr static char localhost[] = "localhost";
  constexpr static char base_path[] = "./tmp";
  constexpr static size_t num_servers = 2;  // TODO(Misiu): Parameterize.

  void SetUp() override {
    ASSERT_TRUE(std::filesystem::exists(base_path));
    g_enable_fsi = false;  // Temporarily unsupported.

    std::vector<ServerPorts> ports;
    for (size_t i = 0; i < num_servers; ++i) {
      std::string server_name = "server_" + std::to_string(i);
      std::string log_file_name = server_name + ".log";
      std::string disk_cache_name = server_name + "_cache";
      std::filesystem::remove(binary_path.parent_path().string() + "/" + base_path +
                              "/log/" + log_file_name);
      std::filesystem::remove(binary_path.parent_path().string() + "/" + base_path + "/" +
                              disk_cache_name);
      ports.emplace_back(ServerPorts());
      // Start servers on separate processes.
      child_processes_.emplace_back(
          startServerCommand(ports[i], base_path, log_file_name, disk_cache_name));
    }

    std::cout << "waiting for servers to start...\n";
    std::this_thread::sleep_for(20000ms);

    // Connect clients to servers.
    for (size_t i = 0; i < num_servers; ++i) {
      servers_.emplace_back(Connection(localhost, ports[i].server));
    }

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
                                 const std::string& base_path,
                                 const std::string& log_file_name,
                                 const std::string& disk_cache_name) const {
    // ServerPorts ports;
    std::stringstream ss;
    ss << binary_path.parent_path().string() << "/../bin/heavydb --port " << ports.server
       << " --http-port " << ports.http << " --calcite-port " << ports.calcite
       << " --http-binary-port " << ports.http_binary
       << " --multi-instance true --allowed-import-paths=[\"/\"] --disk-cache-path="
       << disk_cache_name << " --log-severity=DEBUG2 --log-file-name=" << log_file_name
       << " " << base_path;
    std::cout << "starting server: " << ss.str() << "\n";
    return ss.str();
  }

  std::vector<size_t> getServerIndexes() const {
    std::vector<size_t> indexes;
    size_t i = 0;
    for (auto server : servers_) {
      indexes.emplace_back(i++);
    }
    return indexes;
  }

  std::vector<Connection> servers_;
  std::vector<bp::child> child_processes_;
};

CloudEnvironment* g_cloud_env;
using MultiServerResults = std::array<TQueryResult, CloudEnvironment::num_servers>;

// A bunch of anonymous namespace functions require the CloudEnvironment to be defined, so
// we have a second section here.
namespace {
// These functions wrap Connection member functions with a sever for convenience.
TQueryResult sql_execute(const std::string& query, size_t server = 0) {
  return g_cloud_env->servers_[server].sqlExecute(query);
}

TQueryResult create_table(const std::string& name, size_t server = 0) {
  g_cloud_env->servers_[server].createTable(name);
  return {};
}

TQueryResult load_table(const std::string& name, const RowSet& rows, size_t server = 0) {
  g_cloud_env->servers_[server].loadTable(name, rows);
  return {};
}

TQueryResult import_table(const std::string& name,
                          const std::string& file_name,
                          size_t server = 0) {
  g_cloud_env->servers_[server].importTable(name, file_name);
  return {};
}

TQueryResult get_databases(size_t server = 0) {
  auto dbs = g_cloud_env->servers_[server].getDatabases();
  std::vector<std::string> db_names;
  for (auto db : dbs) {
    db_names.emplace_back(db.db_name);
  }
  return create_result(db_names);
}

TQueryResult get_dashboards(size_t server = 0) {
  auto dashboards = g_cloud_env->servers_[server].getDashboards();
  std::vector<std::string> dash_names;
  for (auto dash : dashboards) {
    dash_names.emplace_back(dash.dashboard_name);
  }
  return create_result(dash_names);
}

TQueryResult get_dashboard(int32_t id, size_t server = 0) {
  auto dash = g_cloud_env->servers_[server].getDashboard(id);
  return create_result({dash.dashboard_name});
}

TQueryResult create_dashboard(const std::string& name, size_t server = 0) {
  std::vector<std::string> res;
  res.emplace_back(std::to_string(g_cloud_env->servers_[server].createDashboard(name)));
  return create_result(res);
}

TQueryResult delete_dashboard(int32_t str_id, size_t server = 0) {
  g_cloud_env->servers_[server].deleteDashboard(str_id);
  return create_result({{"0"}});
}

TQueryResult get_roles(size_t server = 0) {
  auto roles = g_cloud_env->servers_[server].getRoles();
  return create_result(roles);
}

TQueryResult get_users(size_t server = 0) {
  auto users = g_cloud_env->servers_[server].getUsers();
  return create_result(users);
}

TQueryResult get_custom_expressions(size_t server = 0) {
  auto custom_expressions = g_cloud_env->servers_[server].getCustomExpressions();
  std::vector<std::string> exp_names;
  for (auto& exp : custom_expressions) {
    exp_names.emplace_back(exp.name);
  }
  return create_result(exp_names);
}

TQueryResult create_custom_expression(const std::string& name,
                                      const std::string& table_name,
                                      size_t server = 0) {
  std::vector<std::string> res;
  res.emplace_back(std::to_string(
      g_cloud_env->servers_[server].createCustomExpression(name, table_name)));
  return create_result(res);
}

TQueryResult delete_custom_expression(int32_t id, size_t server = 0) {
  std::vector<int32_t> ids{id};
  g_cloud_env->servers_[server].deleteCustomExpressions(ids, false);
  return create_result({{"0"}});  // id = 0 implies id does not exist.
}

TQueryResult delete_custom_expressions(const std::vector<int32_t>& ids,
                                       bool is_soft_delete,
                                       size_t server = 0) {
  g_cloud_env->servers_[server].deleteCustomExpressions(ids, is_soft_delete);
  return create_result({{"0"}});
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

MultiServerResults run_on_all_servers(const std::string& sql) {
  MultiServerResults results;
  for (auto server : g_cloud_env->getServerIndexes()) {
    results[server] = sql_execute(sql, server);
  }
  return results;
}

template <typename Func, typename... Args>
MultiServerResults run_on_all_servers(Func func, Args... args) {
  MultiServerResults results;
  for (auto server : g_cloud_env->getServerIndexes()) {
    results[server] = func(args..., server);
  }
  return results;
}

// TODO(Misiu): Move this functionality out of DBHandlerTestFixtures.  We should not need
// to depend on DBHandler at all.  It would also be nice to have an expect version rather
// than an assert.
void assert_results_equal(const ExpectedResultSet& expected,
                          const MultiServerResults& results) {
  for (auto& result : results) {
    EXPECT_NO_THROW(DBHandlerTestFixture::assertResultSetEqual(expected, result));
  }
}

void assert_unordered_equal(const std::unordered_set<std::string>& server_names,
                            const MultiServerResults& results) {
  for (auto& result : results) {
    ASSERT_EQ(result.row_set.is_columnar, false);
    ASSERT_EQ(result.row_set.row_desc[0].col_type.type, TDatumType::STR);
    ASSERT_EQ(result.row_set.row_desc[0].col_name, "server_name");
    const size_t num_rows = result.row_set.rows.size();
    ASSERT_EQ(num_rows, server_names.size());
    std::unordered_set<std::string> result_set;
    for (size_t i = 0; i < num_rows; ++i) {
      ASSERT_TRUE(server_names.count(result.row_set.rows[i].cols[0].val.str_val) > 0);
      result_set.insert(result.row_set.rows[i].cols[0].val.str_val);
    }
    ASSERT_EQ(result_set.size(), server_names.size());
  }
}

template <typename... Args>
void catch_on_all_servers(Args... args) {
  for (auto server : g_cloud_env->getServerIndexes()) {
    EXPECT_NO_THROW(run_and_catch(args..., server));
  }
}

}  // namespace

class TableTest : public ::testing::Test {
 public:
  static constexpr char name[] = "obj_0";
  static constexpr char name_1[] = "obj_1";
  static constexpr char schema[] = "(i int, s text)";

  virtual ObjectType type() { return ObjectType::Table; }

  void SetUp() override { TearDown(); }

  void TearDown() override {
    sql_execute(drop(type(), name, true));
    sql_execute(drop(type(), name_1, true));
  }
};

TEST_F(TableTest, Create) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_results_equal({}, run_on_all_servers(select(name)));
}

TEST_F(TableTest, CreateAlreadyExists) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  catch_on_all_servers("already exists", sql_execute, create(type(), name, schema));
  assert_results_equal({}, run_on_all_servers(select(name)));
}

TEST_F(TableTest, Alter) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_results_equal({}, run_on_all_servers(select(name)));
  catch_on_all_servers("not found", sql_execute, select(name_1));
  EXPECT_NO_THROW(sql_execute(alter(type(), name, name_1)));
  catch_on_all_servers("not found", sql_execute, select(name));
  assert_results_equal({}, run_on_all_servers(select(name_1)));
}

TEST_F(TableTest, Drop) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  EXPECT_NO_THROW(sql_execute(drop(type(), name)));
  catch_on_all_servers("not found", sql_execute, select(name));
}

TEST_F(TableTest, DropNotExists) {
  catch_on_all_servers("not found", sql_execute, select(name));
  catch_on_all_servers("not exist", sql_execute, drop(type(), name));
}

TEST_F(TableTest, ThriftCreate) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(create_table(name));
  assert_results_equal({}, run_on_all_servers(select(name)));
}

TEST_F(TableTest, ThriftCreateAlreadyExists) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(create_table(name));
  catch_on_all_servers("already exists", create_table, name);
  assert_results_equal({}, run_on_all_servers(select(name)));
}

TEST_F(TableTest, ThriftLoad) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_results_equal({}, run_on_all_servers(select(name)));
  EXPECT_NO_THROW(load_table(name, {i(1), "1"}));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
}

TEST_F(TableTest, Select) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_results_equal({}, run_on_all_servers(select(name)));
  EXPECT_NO_THROW(sql_execute(insert(name, {{i(1), "1"}})));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
}

TEST_F(TableTest, GroupBy) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_results_equal({}, run_on_all_servers(group_by(name)));
  EXPECT_NO_THROW(sql_execute(insert(name, {{i(1), "1"}, {i(1), "2"}, {i(2), "3"}})));
  assert_results_equal({{i(1), i(2)}, {i(2), i(1)}}, run_on_all_servers(group_by(name)));
}

TEST_F(TableTest, Join) {
  catch_on_all_servers("not found", sql_execute, join({name, name_1}));
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  EXPECT_NO_THROW(sql_execute(create(type(), name_1, schema)));
  assert_results_equal({}, run_on_all_servers(join({name, name_1})));
  EXPECT_NO_THROW(sql_execute(insert(name, {{i(1), "1"}, {i(2), "2"}})));
  assert_results_equal({}, run_on_all_servers(join({name, name_1})));
  EXPECT_NO_THROW(sql_execute(insert(name_1, {{i(2), "2"}, {i(3), "3"}})));
  assert_results_equal({{i(2), "2"}}, run_on_all_servers(join({name, name_1})));
}

TEST_F(TableTest, Delete) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_results_equal({}, run_on_all_servers(group_by(name)));
  EXPECT_NO_THROW(sql_execute(insert(name, {{i(1), "1"}})));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
  EXPECT_NO_THROW(sql_execute(delete_all(name)));
  assert_results_equal({}, run_on_all_servers(select(name)));
}

class ViewTest : public TableTest {
 public:
  static constexpr char source_name[] = "source_table";
  static constexpr char properties[] = " AS SELECT * FROM source_table";

  ObjectType type() override { return ObjectType::View; }

  static void SetUpTestSuite() {
    TearDownTestSuite();
    sql_execute(create(ObjectType::Table, source_name, schema));
  }

  static void TearDownTestSuite() {
    sql_execute(drop(ObjectType::Table, source_name, true));
  }

  void TearDown() override {
    TableTest::TearDown();
    sql_execute("DELETE FROM "s + source_name);
  }
};

TEST_F(ViewTest, Create) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, properties)));
  assert_results_equal({}, run_on_all_servers(select(name)));
}

TEST_F(ViewTest, CreateAlreadyExists) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, properties)));
  catch_on_all_servers("already exists", sql_execute, create(type(), name, properties));
  assert_results_equal({}, run_on_all_servers(select(name)));
}

TEST_F(ViewTest, Drop) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, properties)));
  EXPECT_NO_THROW(sql_execute(drop(type(), name)));
  catch_on_all_servers("not exist", sql_execute, drop(type(), name));
}

TEST_F(ViewTest, DropNotExists) {
  catch_on_all_servers("not found", sql_execute, select(name));
  catch_on_all_servers("not exist", sql_execute, drop(type(), name));
}

TEST_F(ViewTest, Insert) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, properties)));
  assert_results_equal({}, run_on_all_servers(select(name)));
  EXPECT_NO_THROW(sql_execute(insert(source_name, {{i(1), "1"}})));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
}

TEST_F(ViewTest, Delete) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, properties)));
  EXPECT_NO_THROW(sql_execute(insert(source_name, {{i(1), "1"}})));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
  EXPECT_NO_THROW(sql_execute(delete_all(source_name)));
  assert_results_equal({}, run_on_all_servers(select(name)));
}

class ItasTest : public ViewTest {
 public:
  ObjectType type() override { return ObjectType::Table; }
};

TEST_F(ItasTest, Itas) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, properties)));
  assert_results_equal({}, run_on_all_servers(select(name)));
  EXPECT_NO_THROW(sql_execute(insert(source_name, {{i(1), "1"}})));
  EXPECT_NO_THROW(sql_execute(itas(name, source_name)));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
}

TEST_F(ItasTest, ItasNotExists) {
  GTEST_SKIP() << "Bug currently crashes this test";
  EXPECT_NO_THROW(sql_execute(insert(source_name, {{i(1), "1"}})));
  catch_on_all_servers("not exist", sql_execute, itas(name, source_name));
  catch_on_all_servers("not found", sql_execute, select(name));
}

class CtasTest : public ViewTest {
 public:
  ObjectType type() override { return ObjectType::Table; }
};

TEST_F(CtasTest, Ctas) {
  EXPECT_NO_THROW(sql_execute(insert(source_name, {{i(1), "1"}})));
  EXPECT_NO_THROW(sql_execute(ctas(name, source_name)));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
}

TEST_F(CtasTest, CtasAlreadyExists) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, properties)));
  catch_on_all_servers("already exists", sql_execute, ctas(name, source_name));
  assert_results_equal({}, run_on_all_servers(select(name)));
}

class DBTest : public TableTest {
 public:
  ObjectType type() override { return ObjectType::Database; }
};

TEST_F(DBTest, Create) {
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName}},
                       run_on_all_servers(get_databases));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName, name}},
                       run_on_all_servers(get_databases));
}

TEST_F(DBTest, CreateAlreadyExists) {
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName}},
                       run_on_all_servers(get_databases));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  catch_on_all_servers("already exists", sql_execute, create(type(), name));
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName, name}},
                       run_on_all_servers(get_databases));
}

TEST_F(DBTest, Alter) {
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName, name}},
                       run_on_all_servers(get_databases));
  EXPECT_NO_THROW(sql_execute(alter(type(), name, name_1)));
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName, name_1}},
                       run_on_all_servers(get_databases));
}

TEST_F(DBTest, Drop) {
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName}},
                       run_on_all_servers(get_databases));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  EXPECT_NO_THROW(sql_execute(drop(type(), name)));
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName}},
                       run_on_all_servers(get_databases));
}

TEST_F(DBTest, DropNotExists) {
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName}},
                       run_on_all_servers(get_databases));
  catch_on_all_servers("not exist", sql_execute, drop(type(), name));
  assert_results_equal({{shared::kDefaultDbName, shared::kInfoSchemaDbName}},
                       run_on_all_servers(get_databases));
}

TEST(DatabaseTest, RecreateDB) {
  sql_execute("DROP DATABASE IF EXISTS db1;");
  sql_execute("CREATE DATABASE db1;");
  g_cloud_env->servers_[0].client->connect(
      g_cloud_env->servers_[0].session, "admin", "HyperInteractive", "db1");
  sql_execute("CREATE TABLE tb1 (i int)");
  g_cloud_env->servers_[1].client->connect(
      g_cloud_env->servers_[1].session, "admin", "HyperInteractive", "db1");
  EXPECT_NO_THROW(sql_execute("SHOW CREATE TABLE tb1;", 1));
  sql_execute("DROP TABLE tb1;");
  sql_execute("DROP DATABASE db1;");
  sql_execute("CREATE DATABASE db1;");
  g_cloud_env->servers_[0].client->connect(
      g_cloud_env->servers_[0].session, "admin", "HyperInteractive", "db1");
  sql_execute("CREATE TABLE tb1 (i int)");
  g_cloud_env->servers_[1].client->connect(
      g_cloud_env->servers_[1].session, "admin", "HyperInteractive", "db1");
  EXPECT_NO_THROW(sql_execute("SHOW CREATE TABLE tb1;", 1));
  sql_execute("DROP DATABASE IF EXISTS db1;");
  g_cloud_env->servers_[0].client->connect(g_cloud_env->servers_[0].session,
                                           "admin",
                                           "HyperInteractive",
                                           shared::kDefaultDbName);
  g_cloud_env->servers_[1].client->connect(g_cloud_env->servers_[1].session,
                                           "admin",
                                           "HyperInteractive",
                                           shared::kDefaultDbName);
}

class DashboardTest : public TableTest {
 public:
  size_t dash_id_{0};
  ObjectType type() override { return ObjectType::Dashboard; }

  void TearDown() override {
    if (dash_id_ > 0) {
      dash_id_ = extract_result(delete_dashboard(dash_id_));
    }
  }
};

TEST_F(DashboardTest, Create) {
  assert_results_equal({{}}, run_on_all_servers(get_dashboards));
  EXPECT_NO_THROW(dash_id_ = extract_result(create_dashboard(name)));
  assert_results_equal({{name}}, run_on_all_servers(get_dashboards));
}

TEST_F(DashboardTest, CreateAlreadyExists) {
  assert_results_equal({{}}, run_on_all_servers(get_dashboards));
  EXPECT_NO_THROW(dash_id_ = extract_result(create_dashboard(name)));
  catch_on_all_servers("already exists", create_dashboard, name);
  assert_results_equal({{name}}, run_on_all_servers(get_dashboards));
}

TEST_F(DashboardTest, GetDashboard) {
  // Make sure get_dashboard() is updated like get_dashboards().
  assert_results_equal({{}}, run_on_all_servers(get_dashboards));
  EXPECT_NO_THROW(dash_id_ = extract_result(create_dashboard(name)));
  assert_results_equal({{name}}, run_on_all_servers(get_dashboard, dash_id_));
}

TEST_F(DashboardTest, Drop) {
  assert_results_equal({{}}, run_on_all_servers(get_dashboards));
  EXPECT_NO_THROW(dash_id_ = extract_result(create_dashboard(name)));
  assert_results_equal({{name}}, run_on_all_servers(get_dashboards));
  EXPECT_NO_THROW(dash_id_ = extract_result(delete_dashboard(dash_id_)));
  assert_results_equal({{}}, run_on_all_servers(get_dashboards));
}

TEST_F(DashboardTest, DropNotExists) {
  assert_results_equal({{}}, run_on_all_servers(get_dashboards));
  catch_on_all_servers("does not exist", delete_dashboard, dash_id_);
  assert_results_equal({{}}, run_on_all_servers(get_dashboards));
}

class UserTest : public TableTest {
 public:
  ObjectType type() override { return ObjectType::User; }
};

TEST_F(UserTest, Create) {
  assert_results_equal({{shared::kRootUsername}}, run_on_all_servers(get_users));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  assert_results_equal({{shared::kRootUsername, name}}, run_on_all_servers(get_users));
}

TEST_F(UserTest, CreateAlreadyExists) {
  assert_results_equal({{shared::kRootUsername}}, run_on_all_servers(get_users));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  catch_on_all_servers("already exists", sql_execute, create(type(), name));
  assert_results_equal({{shared::kRootUsername, name}}, run_on_all_servers(get_users));
}

TEST_F(UserTest, Alter) {
  assert_results_equal({{shared::kRootUsername}}, run_on_all_servers(get_users));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  EXPECT_NO_THROW(sql_execute(alter(type(), name, name_1)));
  assert_results_equal({{shared::kRootUsername, name_1}}, run_on_all_servers(get_users));
}

TEST_F(UserTest, Drop) {
  assert_results_equal({{shared::kRootUsername}}, run_on_all_servers(get_users));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  EXPECT_NO_THROW(sql_execute(drop(type(), name)));
  assert_results_equal({{shared::kRootUsername}}, run_on_all_servers(get_users));
}

TEST_F(UserTest, DropNotExists) {
  assert_results_equal({{shared::kRootUsername}}, run_on_all_servers(get_users));
  catch_on_all_servers("not exist", sql_execute, drop(type(), name));
  assert_results_equal({{shared::kRootUsername}}, run_on_all_servers(get_users));
}

class RoleTest : public TableTest {
 public:
  ObjectType type() override { return ObjectType::Role; }
};

TEST_F(RoleTest, Create) {
  assert_results_equal({{}}, run_on_all_servers(get_roles));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  assert_results_equal({{name}}, run_on_all_servers(get_roles));
}

TEST_F(RoleTest, CreateAlreadyExists) {
  assert_results_equal({{}}, run_on_all_servers(get_roles));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  catch_on_all_servers("already exists", sql_execute, create(type(), name));
  assert_results_equal({{name}}, run_on_all_servers(get_roles));
}

TEST_F(RoleTest, Drop) {
  assert_results_equal({{}}, run_on_all_servers(get_roles));
  EXPECT_NO_THROW(sql_execute(create(type(), name)));
  EXPECT_NO_THROW(sql_execute(drop(type(), name)));
  assert_results_equal({{}}, run_on_all_servers(get_roles));
}

TEST_F(RoleTest, DropNotExists) {
  assert_results_equal({{}}, run_on_all_servers(get_roles));
  catch_on_all_servers("not exist", sql_execute, drop(type(), name));
  assert_results_equal({{}}, run_on_all_servers(get_roles));
}

class CustomExpressionTest : public ViewTest {
 public:
  int32_t cust_id_{0};
  ObjectType type() override { return ObjectType::CustomExpression; }

  void TearDown() override {
    if (cust_id_ > 0) {
      cust_id_ = extract_result(delete_custom_expressions({cust_id_}, false));
    }
  }
};

TEST_F(CustomExpressionTest, Create) {
  assert_results_equal({{}}, run_on_all_servers(get_custom_expressions));
  EXPECT_NO_THROW(cust_id_ = extract_result(create_custom_expression(name, source_name)));
  assert_results_equal({{name}}, run_on_all_servers(get_custom_expressions));
}

TEST_F(CustomExpressionTest, CreateAlreadyExists) {
  assert_results_equal({{}}, run_on_all_servers(get_custom_expressions));
  EXPECT_NO_THROW(cust_id_ = extract_result(create_custom_expression(name, source_name)));
  catch_on_all_servers("already exists", create_custom_expression, name, source_name);
  assert_results_equal({{name}}, run_on_all_servers(get_custom_expressions));
}

TEST_F(CustomExpressionTest, Drop) {
  assert_results_equal({{}}, run_on_all_servers(get_custom_expressions));
  EXPECT_NO_THROW(cust_id_ = extract_result(create_custom_expression(name, source_name)));
  EXPECT_NO_THROW(cust_id_ = extract_result(delete_custom_expression(cust_id_)));
  assert_results_equal({{}}, run_on_all_servers(get_custom_expressions));
}

TEST_F(CustomExpressionTest, DropNotExists) {
  assert_results_equal({{}}, run_on_all_servers(get_custom_expressions));
  catch_on_all_servers("not exist", delete_custom_expression, cust_id_);
  assert_results_equal({{}}, run_on_all_servers(get_custom_expressions));
}

class ImportTableTest : public TableTest {
 public:
  static constexpr char file_name[] = "temp_file.csv";
  inline static std::string file_path_;

  static void SetUpTestSuite() {
    file_path_ = binary_path.parent_path().string() + "/" + file_name;
  }

  void SetUp() override {
    TableTest::SetUp();
    create_file(file_path_, ExpectedResultSet{{i(1), "1"}});
  }

  void TearDown() override {
    std::filesystem::remove(file_path_);
    TableTest::TearDown();
  }
};

TEST_F(ImportTableTest, ThriftImport) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_results_equal({}, run_on_all_servers(select(name)));
  EXPECT_NO_THROW(import_table(name, file_path_));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
}

TEST_F(ImportTableTest, Copy) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_results_equal({}, run_on_all_servers(group_by(name)));
  EXPECT_NO_THROW(sql_execute(copy(name, file_name, {{i(1), "1"}})));
  assert_results_equal({{i(1), "1"}}, run_on_all_servers(select(name)));
}

class ForeignTableTest : public TableTest {
 public:
  // schema needs to be initialized at runtime because the path is relative to the binary.
  std::string foreign_schema = "";
  ObjectType type() override { return ObjectType::ForeignTable; }
  inline static const ExpectedResultSet example_results{{"a", i(1), 1.1},
                                                        {"aa", i(1), 1.1},
                                                        {"aa", i(2), 2.2},
                                                        {"aaa", i(1), 1.1},
                                                        {"aaa", i(2), 2.2},
                                                        {"aaa", i(3), 3.3}};

  void SetUp() override {
    foreign_schema = "(s text, i int, d double) SERVER " +
                     shared::kDefaultDelimitedServerName + " WITH (file_path = '" +
                     binary_path.parent_path().string() +
                     "/../../Tests/FsiDataFiles/example_2.csv')";
    TableTest::SetUp();
  }
};

TEST_F(ForeignTableTest, Create) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, foreign_schema)));
  assert_results_equal(example_results,
                       run_on_all_servers(select(name, "ORDER BY s, i")));
}

TEST_F(ForeignTableTest, CreateAlreadyExists) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, foreign_schema)));
  catch_on_all_servers(
      "already exists", sql_execute, create(type(), name, foreign_schema));
  assert_results_equal(example_results,
                       run_on_all_servers(select(name, "ORDER BY s, i")));
}

TEST_F(ForeignTableTest, Alter) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, foreign_schema)));
  assert_results_equal(example_results,
                       run_on_all_servers(select(name, "ORDER BY s, i")));
  catch_on_all_servers("not found", sql_execute, select(name_1));
  EXPECT_NO_THROW(sql_execute(alter(type(), name, name_1)));
  catch_on_all_servers("not found", sql_execute, select(name));
  assert_results_equal(example_results,
                       run_on_all_servers(select(name_1, "ORDER BY s, i")));
}

TEST_F(ForeignTableTest, Drop) {
  catch_on_all_servers("not found", sql_execute, select(name));
  EXPECT_NO_THROW(sql_execute(create(type(), name, foreign_schema)));
  EXPECT_NO_THROW(sql_execute(drop(type(), name)));
  catch_on_all_servers("not found", sql_execute, select(name));
}

TEST_F(ForeignTableTest, DropNotExists) {
  catch_on_all_servers("not found", sql_execute, select(name));
  catch_on_all_servers("not exist", sql_execute, drop(type(), name));
}

class ForeignServerTest : public TableTest {
 public:
  inline static const std::string show_servers{"SHOW SERVERS"};
  inline static const std::unordered_set<std::string> default_servers{
      shared::kDefaultDelimitedServerName,
      shared::kDefaultParquetServerName,
      shared::kDefaultRegexServerName,
      shared::kDefaultRasterServerName};
  inline static const std::unordered_set<std::string> added_servers{
      shared::kDefaultDelimitedServerName,
      shared::kDefaultParquetServerName,
      shared::kDefaultRegexServerName,
      shared::kDefaultRasterServerName,
      name};
  inline static const std::string schema{
      "FOREIGN DATA WRAPPER delimited_file WITH (storage_type = 'LOCAL_FILE')"};
  ObjectType type() override { return ObjectType::Server; }
};

TEST_F(ForeignServerTest, Create) {
  assert_unordered_equal(default_servers, run_on_all_servers(show_servers));
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_unordered_equal(added_servers, run_on_all_servers(show_servers));
}

TEST_F(ForeignServerTest, CreateAlreadyExists) {
  assert_unordered_equal(default_servers, run_on_all_servers(show_servers));
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  catch_on_all_servers("already exists", sql_execute, create(type(), name, schema));
  assert_unordered_equal(added_servers, run_on_all_servers(show_servers));
}

TEST_F(ForeignServerTest, Alter) {
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  assert_unordered_equal(added_servers, run_on_all_servers(show_servers));
  EXPECT_NO_THROW(sql_execute(alter(type(), name, name_1)));
  assert_unordered_equal({shared::kDefaultDelimitedServerName,
                          shared::kDefaultParquetServerName,
                          shared::kDefaultRegexServerName,
                          shared::kDefaultRasterServerName,
                          name_1},
                         run_on_all_servers(show_servers));
}

TEST_F(ForeignServerTest, Drop) {
  assert_unordered_equal(default_servers, run_on_all_servers(show_servers));
  EXPECT_NO_THROW(sql_execute(create(type(), name, schema)));
  EXPECT_NO_THROW(sql_execute(drop(type(), name)));
  assert_unordered_equal(default_servers, run_on_all_servers(show_servers));
}

TEST_F(ForeignServerTest, DropNotExists) {
  assert_unordered_equal(default_servers, run_on_all_servers(show_servers));
  catch_on_all_servers("not exist", sql_execute, drop(type(), name));
  assert_unordered_equal(default_servers, run_on_all_servers(show_servers));
}

/*
  TODO(Misiu):
  Add tests for untested thrift API functions.
  Add tests for user-defined functions
*/

int main(int argc, char** argv) {
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
