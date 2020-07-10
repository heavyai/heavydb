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

/**
 * @file DashboardBulkCommandsTest.cpp
 * @brief Test suite for bulk dashboard commands
 */

#include <gtest/gtest.h>
#include <boost/range/combine.hpp>

#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class ShareDashboardsTest : public DBHandlerTestFixture {
 public:
  static void SetUpTestSuite() {
    createDBHandler();
    createUsersAndRoles();
  }

  static void TearDownTestSuite() {
    loginAdmin();
    dropUsersAndRoles();
  }

 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    setupDashboards();
  }
  void TearDown() override {
    teardownDashboards();
    DBHandlerTestFixture::TearDown();
  }

  std::vector<int32_t> getTestDashboardIds() const {
    std::vector<int32_t> ids;
    for (const auto& test_dashboard : test_dashboards) {
      ids.push_back(test_dashboard.dashboard_id);
    }
    return ids;
  }

  struct TestPermissions {
    TDashboardPermissions permissions;
    std::map<std::string, AccessPrivileges> privileges;

    TestPermissions(bool create_permission,
                    bool delete_permission,
                    bool edit_permission,
                    bool view_permission) {
      permissions.create_ = create_permission;
      permissions.delete_ = delete_permission;
      permissions.edit_ = edit_permission;
      permissions.view_ = view_permission;
      if (create_permission) {
        privileges.insert({"CREATE_DASHBOARD", AccessPrivileges::CREATE_DASHBOARD});
      }
      if (delete_permission) {
        privileges.insert({"DELETE_DASHBOARD", AccessPrivileges::DELETE_DASHBOARD});
      }
      if (edit_permission) {
        privileges.insert({"EDIT_DASHBOARD", AccessPrivileges::EDIT_DASHBOARD});
      }
      if (view_permission) {
        privileges.insert({"VIEW_DASHBOARD", AccessPrivileges::VIEW_DASHBOARD});
      }
    }
  };

  void shareDashBoards(const TestPermissions& test_permissions,
                       const std::vector<std::string>& groups,
                       const std::vector<int32_t>& dashboard_ids) {
    auto [handler, session_id] = DBHandlerTestFixture::getDbHandlerAndSessionId();
    handler->share_dashboards(
        session_id, dashboard_ids, groups, test_permissions.permissions);
  }

  void shareDashBoards(const TestPermissions& test_permissions,
                       const std::vector<std::string>& groups) {
    shareDashBoards(test_permissions, groups, getTestDashboardIds());
  }

  void setupNewDashboards(int count) {
    auto [handler, session_id] = DBHandlerTestFixture::getDbHandlerAndSessionId();
    for (int i = 0; i < count; ++i) {
      test_dashboards.emplace_back(i, handler, session_id);
    }
  }

  void teardownLastDashboard() { test_dashboards.pop_back(); }

  void teardownDashboards() { test_dashboards.clear(); }

  void assertExpectedDashboardPrivilegesForUsers(const TestPermissions& test_permissions,
                                                 const std::vector<std::string>& users,
                                                 bool is_shared = true) {
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    auto& catalog = getCatalog();
    for (auto const& test_dashboard : test_dashboards) {
      DBObject object(test_dashboard.dashboard_id, DBObjectType::DashboardDBObjectType);
      object.loadKey(catalog);
      for (auto const& user : users) {
        for (auto const& [privilege_name, privilege] : test_permissions.privileges) {
          object.setPrivileges(privilege);
          bool privilege_allowed = sys_catalog.checkPrivileges(user, {object});
          ASSERT_EQ(privilege_allowed, is_shared)
              << " with user " << user << ", dashboard id " << test_dashboard.dashboard_id
              << " and privilege_name " << privilege_name << " being "
              << privilege_allowed << " but expected to be " << is_shared;
        }
      }
    }
  }

  static void createUsersAndRoles() {
    sql("CREATE USER test_user_1 (password = '');");
    sql("CREATE USER test_user_2 (password = '');");
    sql("CREATE USER test_user_3 (password = '');");
    sql("CREATE ROLE test_role_1;");
    sql("GRANT test_role_1 TO test_user_2, test_user_3;");
  }

  static void dropUsersAndRoles() {
    sql("DROP USER test_user_1;");
    sql("DROP USER test_user_2;");
    sql("DROP USER test_user_3;");
    sql("DROP ROLE test_role_1;");
  }

  static void dropTestUser(const std::string& user_name) {
    try {
      sql("DROP USER " + user_name + ";");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }

 private:
  const static inline std::string dashboard_name_prefix_ = "test_dashboard";
  const static inline std::string dashboard_state_prefix_ = "test_state";
  const static inline std::string dashboard_image_hash_prefix_ = "test_image_hash";
  const static inline std::string dashboard_metadata_prefix_ = "test_metadata";
  const static inline int num_test_dasbboards_ = 3;

  struct TestDashboard {
    int dashboard_id;
    DBHandler* handler;
    const TSessionId& session;

    std::string get_suffix(const int unique_id) {
      std::stringstream suffix;
      suffix << "_" << std::setfill('0') << std::setw(2) << unique_id;
      return suffix.str();
    }

    TestDashboard(const int unique_id, DBHandler* handler, const TSessionId& session)
        : handler(handler), session(session) {
      const auto suffix = get_suffix(unique_id);
      dashboard_id = handler->create_dashboard(session,
                                               dashboard_name_prefix_ + suffix,
                                               dashboard_state_prefix_ + suffix,
                                               dashboard_image_hash_prefix_ + suffix,
                                               dashboard_metadata_prefix_ + suffix);
    }

    ~TestDashboard() { handler->delete_dashboard(session, dashboard_id); }
  };

  void setupDashboards() { setupNewDashboards(num_test_dasbboards_); }

  std::list<TestDashboard> test_dashboards;
};

struct PermissionParam {
  bool create_p, delete_p, view_p, edit_p;
};

namespace {
struct PrintToStringParamName {
  std::string operator()(const ::testing::TestParamInfo<PermissionParam>& info) const {
    std::stringstream ss;
    ss << "create_permission_" << info.param.create_p << "_delete_permission_"
       << info.param.delete_p << "_view_permission_" << info.param.view_p
       << "_edit_permission_" << info.param.edit_p;
    return ss.str();
  }
};
}  // namespace

class SharePermissionPermutationsTest
    : public ShareDashboardsTest,
      public testing::WithParamInterface<PermissionParam> {};

INSTANTIATE_TEST_SUITE_P(AllLegalPermutations,
                         SharePermissionPermutationsTest,
                         ::testing::Values(PermissionParam{true, true, true, true},
                                           PermissionParam{true, true, true, false},
                                           PermissionParam{true, true, false, false},
                                           PermissionParam{true, false, true, false},
                                           PermissionParam{false, true, true, false},
                                           PermissionParam{true, false, false, false},
                                           PermissionParam{false, true, false, false}),
                         PrintToStringParamName());

TEST_P(SharePermissionPermutationsTest, ShareAllLegalPermutations) {
  auto& param = GetParam();
  TestPermissions permissions(param.create_p, param.delete_p, param.edit_p, param.view_p);
  shareDashBoards(permissions, {"test_user_1"});
  assertExpectedDashboardPrivilegesForUsers(permissions, {"test_user_1"});
}

TEST_F(ShareDashboardsTest, NoPermissions) {
  TestPermissions no_permissions(false, false, false, false);
  executeLambdaAndAssertException(
      [&, this] { shareDashBoards(no_permissions, {"test_user_1"}); },
      "At least one privilege should be assigned for grants");
  assertExpectedDashboardPrivilegesForUsers(
      TestPermissions{true, true, true, true}, {"test_user_1"}, false);
}

TEST_F(ShareDashboardsTest, NonExistentGroupOrUser) {
  TestPermissions permissions(true, true, true, true);
  executeLambdaAndAssertException(
      [&, this] { shareDashBoards(permissions, {"non_existent_user"}); },
      "Exception: User/Role 'non_existent_user' does not exist");
}

TEST_F(ShareDashboardsTest, NonExistentDashboards) {
  TestPermissions permissions(true, true, true, true);
  auto dashboard_ids = getTestDashboardIds();
  teardownLastDashboard();
  teardownLastDashboard();
  std::stringstream error_stream, id_list;
  id_list << dashboard_ids[dashboard_ids.size() - 2] << ", "
          << dashboard_ids[dashboard_ids.size() - 1];
  error_stream << "Share dashboard(s) failed with error(s)\n"
               << "Dashboard ids " << id_list.str() << ": Dashboard id does not exist\n";
  executeLambdaAndAssertException(
      [&, this] { shareDashBoards(permissions, {"test_user_1"}, dashboard_ids); },
      error_stream.str());
  assertExpectedDashboardPrivilegesForUsers(
      TestPermissions{true, true, true, true}, {"test_user_1"}, false);
}

TEST_F(ShareDashboardsTest, GrantPermisionsToGroupWithOneUser) {
  TestPermissions permissions(true, true, true, true);
  shareDashBoards(permissions, {"test_role_1"});
  assertExpectedDashboardPrivilegesForUsers(permissions, {"test_user_2", "test_user_3"});
}

TEST_F(ShareDashboardsTest, AsUnauthorizedUser) {
  sql("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1");
  login("test_user_1", "");
  std::stringstream error_stream;
  error_stream << "Share dashboard(s) failed with error(s)\n";
  error_stream
      << "Dashboard ids " << join(getTestDashboardIds(), ", ") << ": "
      << "User should be either owner of dashboard or super user to share/unshare it\n";
  TestPermissions permissions(true, true, true, true);
  executeLambdaAndAssertException(
      [&, this] { shareDashBoards(permissions, {"test_user_1"}); }, error_stream.str());
  assertExpectedDashboardPrivilegesForUsers(
      TestPermissions{true, true, true, true}, {"test_user_1"}, false);
  loginAdmin();
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1");
}

TEST_F(ShareDashboardsTest, AsUnauthorizedUserForSomeDashboards) {
  sql("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1, test_user_2;");
  sql("GRANT CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB +
      " TO test_user_1, test_user_2;");
  teardownDashboards();
  login("test_user_1", "");
  setupNewDashboards(3);
  login("test_user_2", "");
  setupNewDashboards(1);
  auto dashboard_ids = getTestDashboardIds();
  std::stringstream error_stream, dashboard_ids_stream;
  error_stream << "Share dashboard(s) failed with error(s)\n";
  for (int i = 0; i < 3; ++i) {
    dashboard_ids_stream << (!dashboard_ids_stream.str().empty() ? ", " : "")
                         << dashboard_ids[i];
  }
  error_stream
      << "Dashboard ids " << dashboard_ids_stream.str() << ": "
      << "User should be either owner of dashboard or super user to share/unshare it\n";
  TestPermissions permissions(true, true, true, true);
  executeLambdaAndAssertException(
      [&, this] { shareDashBoards(permissions, {"test_user_3"}); }, error_stream.str());
  assertExpectedDashboardPrivilegesForUsers(
      TestPermissions{true, true, true, true}, {"test_user_3"}, false);
  loginAdmin();
  sql("REVOKE CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB +
      " FROM test_user_1, test_user_2;");
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB +
      " FROM test_user_1, test_user_2;");
}

TEST_F(ShareDashboardsTest, AsNonSuperUser) {
  sql("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1");
  sql("GRANT CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1;");
  teardownDashboards();
  login("test_user_1", "");
  setupNewDashboards(3);
  TestPermissions permissions(true, true, true, true);
  shareDashBoards(permissions, {"test_user_2"});
  assertExpectedDashboardPrivilegesForUsers(TestPermissions{true, true, true, true},
                                            {"test_user_2"});
  loginAdmin();
  sql("REVOKE CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1;");
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1");
}

TEST_F(ShareDashboardsTest, MixedErrors) {
  sql("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1");
  sql("GRANT CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1;");
  login("test_user_1", "");
  setupNewDashboards(3);
  auto dashboard_ids = getTestDashboardIds();
  teardownLastDashboard();
  teardownLastDashboard();
  std::stringstream error_stream, do_not_exist_dashboard_ids_stream,
      unauthorized_user_dashboard_ids_stream;
  for (size_t i = 0; i < dashboard_ids.size() - 3; ++i) {
    unauthorized_user_dashboard_ids_stream
        << (!unauthorized_user_dashboard_ids_stream.str().empty() ? ", " : "")
        << dashboard_ids[i];
  }
  for (size_t i = dashboard_ids.size() - 2; i < dashboard_ids.size(); ++i) {
    do_not_exist_dashboard_ids_stream
        << (!do_not_exist_dashboard_ids_stream.str().empty() ? ", " : "")
        << dashboard_ids[i];
  }
  error_stream << "Share dashboard(s) failed with error(s)\n";
  error_stream << "Dashboard ids " << do_not_exist_dashboard_ids_stream.str()
               << ": Dashboard id does not exist\n";
  error_stream
      << "Dashboard ids " << unauthorized_user_dashboard_ids_stream.str() << ": "
      << "User should be either owner of dashboard or super user to share/unshare it\n";
  TestPermissions permissions(true, true, true, true);
  executeLambdaAndAssertException(
      [&, this] { shareDashBoards(permissions, {"test_user_2"}, dashboard_ids); },
      error_stream.str());
  assertExpectedDashboardPrivilegesForUsers(
      TestPermissions{true, true, true, true}, {"test_user_2"}, false);
  loginAdmin();
  sql("REVOKE CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1;");
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1");
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
