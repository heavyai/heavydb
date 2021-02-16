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
 * @file DashboardTest.cpp
 * @brief Test suite for dashboard commands
 */

#include <gtest/gtest.h>
#include <boost/range/combine.hpp>

#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class DashboardBasicTest : public DBHandlerTestFixture {
 public:
  static void SetUpTestSuite() {
    createDBHandler();
    test_user_1_id = createTestUser("test_user", "test_pass");
  }

  static void TearDownTestSuite() {
    loginAdmin();
    dropTestUser("test_user");
  }

  size_t getNumDashboards() {
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    std::vector<TDashboard> dashboards;
    db_handler->get_dashboards(dashboards, session_id);
    return dashboards.size();
  }

 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    // Remove any dashboards that may be left by previous tests
    std::vector<TDashboard> dashboards;
    db_handler->get_dashboards(dashboards, session_id);
    if (dashboards.size()) {
      std::vector<int32_t> db_ids;
      for (const auto& dashboard : dashboards) {
        db_ids.push_back(dashboard.dashboard_id);
      }
      db_handler->delete_dashboards(session_id, db_ids);
    }
  }
  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  static int32_t createTestUser(const std::string& user_name, const std::string& pass) {
    sql("CREATE USER " + user_name + " (password = '" + pass + "');");
    sql("GRANT ACCESS ON DATABASE omnisci TO " + user_name + ";");
    Catalog_Namespace::UserMetadata user_metadata{};
    Catalog_Namespace::SysCatalog::instance().getMetadataForUser(user_name,
                                                                 user_metadata);
    return user_metadata.userId;
  }

  static void dropTestUser(const std::string& user_name) {
    try {
      sql("DROP USER " + user_name + ";");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }

  inline static int32_t test_user_1_id;
};

TEST_F(DashboardBasicTest, CreateReplaceDelete) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  auto db_id =
      db_handler->create_dashboard(session_id, "testdb", "state", "image", "metadata");
  ASSERT_EQ(getNumDashboards(), size_t(1));

  db_handler->replace_dashboard(
      session_id, db_id, "testdb", "test_user", "state", "image", "metadata");

  TDashboard db;
  db_handler->get_dashboard(db, session_id, db_id);
  CHECK_EQ(db.dashboard_owner, "test_user");

  db_handler->delete_dashboard(session_id, db_id);
  ASSERT_EQ(getNumDashboards(), size_t(0));
}

TEST_F(DashboardBasicTest, ReplaceWithTableChange) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  auto db_id = db_handler->create_dashboard(
      session_id, "testdb", "state", "image", "\"table\":\"omnisci_states\"");

  db_handler->replace_dashboard(session_id,
                                db_id,
                                "testdb",
                                "test_user",
                                "state",
                                "image",
                                "\"table\":\"omnisci_counties\"");

  db_handler->delete_dashboard(session_id, db_id);
}

struct TestPermissions {
  TDashboardPermissions permissions;
  std::map<std::string, AccessPrivileges> privileges;
  std::map<std::string, AccessPrivileges> all_privileges;

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
    all_privileges.insert({"CREATE_DASHBOARD", AccessPrivileges::CREATE_DASHBOARD});
    all_privileges.insert({"DELETE_DASHBOARD", AccessPrivileges::DELETE_DASHBOARD});
    all_privileges.insert({"EDIT_DASHBOARD", AccessPrivileges::EDIT_DASHBOARD});
    all_privileges.insert({"VIEW_DASHBOARD", AccessPrivileges::VIEW_DASHBOARD});
  }

  TestPermissions getComplement() {
    return TestPermissions{!permissions.create_,
                           !permissions.delete_,
                           !permissions.edit_,
                           !permissions.view_};
  }

  AccessPrivileges getPrivileges() {
    AccessPrivileges priv;
    if (permissions.create_) {
      priv.add(AccessPrivileges::CREATE_DASHBOARD);
    }
    if (permissions.delete_) {
      priv.add(AccessPrivileges::DELETE_DASHBOARD);
    }
    if (permissions.edit_) {
      priv.add(AccessPrivileges::EDIT_DASHBOARD);
    }
    if (permissions.view_) {
      priv.add(AccessPrivileges::VIEW_DASHBOARD);
    }
    return priv;
  }
};

class GetDashboardTest : public DBHandlerTestFixture {
 public:
  static void SetUpTestSuite() {
    createDBHandler();
    test_user_1_id = createTestUser("test_user_1", "test_pass");
  }

  static void TearDownTestSuite() {
    loginAdmin();
    dropTestUser("test_user_1");
  }

  size_t getNumDashboards() {
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    std::vector<TDashboard> dashboards;
    db_handler->get_dashboards(dashboards, session_id);
    return dashboards.size();
  }

 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    // Remove any dashboards that may be left by previous tests
    std::vector<TDashboard> dashboards;
    db_handler->get_dashboards(dashboards, session_id);
    if (dashboards.size()) {
      std::vector<int32_t> db_ids;
      for (const auto& dashboard : dashboards) {
        db_ids.push_back(dashboard.dashboard_id);
      }
      db_handler->delete_dashboards(session_id, db_ids);
    }
  }
  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  static int32_t createTestUser(const std::string& user_name, const std::string& pass) {
    sql("CREATE USER " + user_name + " (password = '" + pass + "');");
    sql("GRANT ACCESS ON DATABASE omnisci TO " + user_name + ";");
    Catalog_Namespace::UserMetadata user_metadata{};
    Catalog_Namespace::SysCatalog::instance().getMetadataForUser(user_name,
                                                                 user_metadata);
    return user_metadata.userId;
  }

  static void dropTestUser(const std::string& user_name) {
    try {
      sql("DROP USER " + user_name + ";");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }

  inline static int32_t test_user_1_id;
};

TEST_F(GetDashboardTest, AsAdmin) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  const auto test_dashboard_id = db_handler->create_dashboard(
      session_id, OMNISCI_DEFAULT_DB, "state", "image", "metadata");
  ASSERT_EQ(getNumDashboards(), size_t(1));

  TDashboard db;
  TDashboardPermissions admin_perms = TestPermissions(true, true, true, true).permissions;
  db_handler->get_dashboard(db, session_id, test_dashboard_id);
  ASSERT_EQ(admin_perms, db.dashboard_permissions);

  std::vector<TDashboard> dashboards;
  db_handler->get_dashboards(dashboards, session_id);
  ASSERT_EQ(admin_perms, dashboards[0].dashboard_permissions);
  db_handler->delete_dashboard(session_id, test_dashboard_id);
}

TEST_F(GetDashboardTest, AsUser) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();

  const auto test_dashboard_id = db_handler->create_dashboard(
      session_id, OMNISCI_DEFAULT_DB, "state", "image", "metadata");

  DBObject object(test_dashboard_id, DBObjectType::DashboardDBObjectType);
  TestPermissions usr_privs(false, false, true, true);
  TDashboardPermissions usr_perms = usr_privs.permissions;
  object.setPrivileges(usr_privs.getPrivileges());

  Catalog_Namespace::SysCatalog::instance().grantDBObjectPrivilegesBatch(
      {"test_user_1"}, {object}, getCatalog());

  TDashboard db;
  std::vector<TDashboard> dashboards;
  TSessionId usr_session;
  login("test_user_1", "test_pass", OMNISCI_DEFAULT_DB, usr_session);
  db_handler->get_dashboard(db, usr_session, test_dashboard_id);
  ASSERT_EQ(usr_perms, db.dashboard_permissions);
  db_handler->get_dashboards(dashboards, usr_session);
  ASSERT_EQ(usr_perms, dashboards[0].dashboard_permissions);
  logout(usr_session);
  db_handler->delete_dashboard(session_id, test_dashboard_id);
}

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
    for (const auto& test_dashboard : test_dashboards_) {
      ids.push_back(test_dashboard.dashboard_id);
    }
    return ids;
  }

  void shareOrUnshareDashboards(const TestPermissions& test_permissions,
                                const std::vector<std::string>& groups,
                                const std::vector<int32_t>& dashboard_ids,
                                const bool do_share = true) {
    auto [handler, session_id] = DBHandlerTestFixture::getDbHandlerAndSessionId();
    if (do_share) {
      handler->share_dashboards(
          session_id, dashboard_ids, groups, test_permissions.permissions);
    } else {
      handler->unshare_dashboards(
          session_id, dashboard_ids, groups, test_permissions.permissions);
    }
  }

  void shareOrUnshareDashboards(const TestPermissions& test_permissions,
                                const std::vector<std::string>& groups,
                                const bool do_share = true) {
    shareOrUnshareDashboards(test_permissions, groups, getTestDashboardIds(), do_share);
  }

  virtual void setupNewDashboards(int count) {
    auto [handler, session_id] = DBHandlerTestFixture::getDbHandlerAndSessionId();
    for (int i = 0; i < count; ++i) {
      test_dashboards_.emplace_back(i, handler, session_id);
    }
  }

  void teardownLastDashboard() { test_dashboards_.pop_back(); }

  void teardownDashboards() { test_dashboards_.clear(); }

  void assertExpectedDashboardPrivilegesForUsers(const TestPermissions& test_permissions,
                                                 const std::vector<std::string>& users) {
    auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
    auto& catalog = getCatalog();
    for (auto const& test_dashboard : test_dashboards_) {
      DBObject object(test_dashboard.dashboard_id, DBObjectType::DashboardDBObjectType);
      object.loadKey(catalog);
      for (auto const& user : users) {
        for (auto const& [privilege_name, privilege] : test_permissions.all_privileges) {
          object.setPrivileges(privilege);
          bool privilege_allowed = sys_catalog.checkPrivileges(user, {object});
          const auto& allowed_privileges = test_permissions.privileges;
          bool should_privilege_be_allowed =
              allowed_privileges.find(privilege_name) != allowed_privileges.end();
          ASSERT_EQ(privilege_allowed, should_privilege_be_allowed)
              << " with user " << user << ", dashboard id " << test_dashboard.dashboard_id
              << " and privilege_name " << privilege_name << " being "
              << privilege_allowed << " but expected to be "
              << should_privilege_be_allowed;
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

  const static inline std::string dashboard_name_prefix_ = "test_dashboard";
  const static inline std::string dashboard_state_prefix_ = "test_state";
  const static inline std::string dashboard_image_hash_prefix_ = "test_image_hash";
  const static inline std::string dashboard_metadata_prefix_ = "test_metadata";
  const static inline int num_test_dashboards_ = 3;

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

  std::list<TestDashboard> test_dashboards_;
  void setupDashboards() { setupNewDashboards(num_test_dashboards_); }
};

struct PermissionParam {
  bool create_p, delete_p, view_p, edit_p;
};

namespace {
struct PrintToStringParamName {
  std::string operator()(const ::testing::TestParamInfo<PermissionParam>& info) const {
    std::stringstream ss;
    const auto& perm = info.param;
    ss << "create_permission_" << perm.create_p << "_delete_permission_" << perm.delete_p
       << "_view_permission_" << perm.view_p << "_edit_permission_" << perm.edit_p;
    return ss.str();
  }

  std::string operator()(const ::testing::TestParamInfo<bool>& info) const {
    return info.param ? "share" : "unshare";
  }
};
}  // namespace

class DashboardsTestWithInitialPermissions : public ShareDashboardsTest {
 protected:
  void grantInitialPermissions() {
    for (auto const& test_dashboard : test_dashboards_) {
      DBObject object(test_dashboard.dashboard_id, DBObjectType::DashboardDBObjectType);
      object.setPrivileges(getInitialPermissions().getPrivileges());
      auto entities =
          std::vector<std::string>{"test_user_1", "test_user_2", "test_user_3"};
      if (grantInitialPermissionsToGroupOnly()) {
        entities = {"test_role_1"};
      }
      Catalog_Namespace::SysCatalog::instance().grantDBObjectPrivilegesBatch(
          entities, {object}, getCatalog());
    }
  }

  void setupNewDashboards(int count) override {
    ShareDashboardsTest::setupNewDashboards(count);
    grantInitialPermissions();
  }

  virtual TestPermissions getInitialPermissions() = 0;
  virtual bool grantInitialPermissionsToGroupOnly() { return false; }
};

class ShareDashboardsGroupTest : public ShareDashboardsTest {
 protected:
  TestPermissions getInitialPermissions() {
    return TestPermissions{false, false, false, false};
  }
};

class UnshareDashboardsGroupTest : public DashboardsTestWithInitialPermissions {
 protected:
  TestPermissions getInitialPermissions() override {
    return TestPermissions{true, true, true, true};
  }

  bool grantInitialPermissionsToGroupOnly() override { return true; }
};

class ShareDashboardsRegularTest : public ShareDashboardsTest {
 protected:
  TestPermissions getInitialPermissions() {
    return TestPermissions{false, false, false, false};
  }
};

class UnshareDashboardsRegularTest : public DashboardsTestWithInitialPermissions {
 protected:
  TestPermissions getInitialPermissions() override {
    return TestPermissions{true, true, true, true};
  }
};

class ShareAndUnshareDashboardsTest : public DashboardsTestWithInitialPermissions,
                                      public testing::WithParamInterface<bool> {
 protected:
  bool getDoShare() { return GetParam(); }

  TestPermissions getInitialPermissions() override {
    return TestPermissions{true, false, true, false};
  }
};

INSTANTIATE_TEST_SUITE_P(ShareAndUnshare,
                         ShareAndUnshareDashboardsTest,
                         ::testing::Values(true, false),
                         PrintToStringParamName());

class SharePermissionPermutationsTest
    : public ShareDashboardsTest,
      public testing::WithParamInterface<PermissionParam> {
 protected:
  const PermissionParam& getPermissionParam() { return GetParam(); }
};

class UnsharePermissionPermutationsTest
    : public DashboardsTestWithInitialPermissions,
      public testing::WithParamInterface<PermissionParam> {
 protected:
  const PermissionParam& getPermissionParam() { return GetParam(); }

  TestPermissions getInitialPermissions() override {
    return TestPermissions{true, true, true, true};
  }
};

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

INSTANTIATE_TEST_SUITE_P(AllLegalPermutations,
                         UnsharePermissionPermutationsTest,
                         ::testing::Values(PermissionParam{true, true, true, true},
                                           PermissionParam{true, true, true, false},
                                           PermissionParam{true, true, false, false},
                                           PermissionParam{true, false, true, false},
                                           PermissionParam{false, true, true, false},
                                           PermissionParam{true, false, false, false},
                                           PermissionParam{false, true, false, false}),
                         PrintToStringParamName());

TEST_P(SharePermissionPermutationsTest, ShareAllLegalPermutations) {
  auto const& param = getPermissionParam();
  TestPermissions permissions(param.create_p, param.delete_p, param.edit_p, param.view_p);
  shareOrUnshareDashboards(permissions, {"test_user_1"}, true);
  assertExpectedDashboardPrivilegesForUsers(permissions, {"test_user_1"});
}

TEST_P(UnsharePermissionPermutationsTest, UnshareAllLegalPermutations) {
  auto const& param = getPermissionParam();
  TestPermissions permissions(param.create_p, param.delete_p, param.edit_p, param.view_p);
  shareOrUnshareDashboards(permissions, {"test_user_1"}, false);
  assertExpectedDashboardPrivilegesForUsers(permissions.getComplement(), {"test_user_1"});
}

TEST_P(ShareAndUnshareDashboardsTest, NoPermissions) {
  bool do_share = getDoShare();
  TestPermissions no_permissions(false, false, false, false);
  executeLambdaAndAssertException(
      [&, this] { shareOrUnshareDashboards(no_permissions, {"test_user_1"}, do_share); },
      "At least one privilege should be assigned for " +
          std::string(do_share ? "grants" : "revokes"));
  assertExpectedDashboardPrivilegesForUsers(getInitialPermissions(), {"test_user_1"});
}

TEST_P(ShareAndUnshareDashboardsTest, NonExistentGroupOrUser) {
  bool do_share = getDoShare();
  TestPermissions permissions(true, true, true, true);
  executeLambdaAndAssertException(
      [&, this] {
        shareOrUnshareDashboards(permissions, {"non_existent_user"}, do_share);
      },
      "Exception: User/Role 'non_existent_user' does not exist");
}

TEST_P(ShareAndUnshareDashboardsTest, NonExistentDashboards) {
  bool do_share = getDoShare();
  TestPermissions permissions(true, true, true, true);
  auto dashboard_ids = getTestDashboardIds();
  teardownLastDashboard();
  teardownLastDashboard();
  std::stringstream error_stream, id_list;
  id_list << dashboard_ids[dashboard_ids.size() - 2] << ", "
          << dashboard_ids[dashboard_ids.size() - 1];
  error_stream << "Share/Unshare dashboard(s) failed with error(s)\n"
               << "Dashboard ids " << id_list.str() << ": Dashboard id does not exist\n";
  executeLambdaAndAssertException(
      [&, this] {
        shareOrUnshareDashboards(permissions, {"test_user_1"}, dashboard_ids, do_share);
      },
      error_stream.str());
  assertExpectedDashboardPrivilegesForUsers(getInitialPermissions(), {"test_user_1"});
}

TEST_F(ShareDashboardsGroupTest, Group) {
  TestPermissions permissions(true, true, true, true);
  assertExpectedDashboardPrivilegesForUsers(getInitialPermissions(),
                                            {"test_user_2", "test_user_3"});
  shareOrUnshareDashboards(permissions, {"test_role_1"}, true);
  assertExpectedDashboardPrivilegesForUsers(permissions, {"test_user_2", "test_user_3"});
}

TEST_F(UnshareDashboardsGroupTest, Group) {
  TestPermissions permissions(true, true, true, true);
  assertExpectedDashboardPrivilegesForUsers(getInitialPermissions(),
                                            {"test_user_2", "test_user_3"});
  shareOrUnshareDashboards(permissions, {"test_role_1"}, false);
  assertExpectedDashboardPrivilegesForUsers(permissions.getComplement(),
                                            {"test_user_2", "test_user_3"});
}

TEST_P(ShareAndUnshareDashboardsTest, AsUnauthorizedUser) {
  bool do_share = getDoShare();
  sql("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1");
  login("test_user_1", "");
  std::stringstream error_stream;
  error_stream << "Share/Unshare dashboard(s) failed with error(s)\n";
  error_stream
      << "Dashboard ids " << join(getTestDashboardIds(), ", ") << ": "
      << "User should be either owner of dashboard or super user to share/unshare it\n";
  TestPermissions permissions(true, true, true, true);
  executeLambdaAndAssertException(
      [&, this] { shareOrUnshareDashboards(permissions, {"test_user_1"}, do_share); },
      error_stream.str());
  assertExpectedDashboardPrivilegesForUsers(getInitialPermissions(), {"test_user_1"});
  loginAdmin();
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1");
}

TEST_P(ShareAndUnshareDashboardsTest, AsUnauthorizedUserForSomeDashboards) {
  bool do_share = getDoShare();
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
  error_stream << "Share/Unshare dashboard(s) failed with error(s)\n";
  for (int i = 0; i < 3; ++i) {
    dashboard_ids_stream << (!dashboard_ids_stream.str().empty() ? ", " : "")
                         << dashboard_ids[i];
  }
  error_stream
      << "Dashboard ids " << dashboard_ids_stream.str() << ": "
      << "User should be either owner of dashboard or super user to share/unshare it\n";
  TestPermissions permissions(true, true, true, true);
  executeLambdaAndAssertException(
      [&, this] { shareOrUnshareDashboards(permissions, {"test_user_3"}, do_share); },
      error_stream.str());
  assertExpectedDashboardPrivilegesForUsers(getInitialPermissions(), {"test_user_3"});
  loginAdmin();
  sql("REVOKE CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB +
      " FROM test_user_1, test_user_2;");
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB +
      " FROM test_user_1, test_user_2;");
}

TEST_F(ShareDashboardsRegularTest, AsNonSuperUser) {
  sql("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1");
  sql("GRANT CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1;");
  teardownDashboards();
  login("test_user_1", "");
  setupNewDashboards(3);
  TestPermissions permissions(true, true, true, true);
  shareOrUnshareDashboards(permissions, {"test_user_2"}, true);
  assertExpectedDashboardPrivilegesForUsers(permissions, {"test_user_2"});
  loginAdmin();
  sql("REVOKE CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1;");
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1");
}

TEST_F(UnshareDashboardsRegularTest, AsNonSuperUser) {
  sql("GRANT ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1");
  sql("GRANT CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " TO test_user_1;");
  teardownDashboards();
  login("test_user_1", "");
  setupNewDashboards(3);
  TestPermissions permissions(true, true, true, true);
  shareOrUnshareDashboards(permissions, {"test_user_2"}, false);
  assertExpectedDashboardPrivilegesForUsers(permissions.getComplement(), {"test_user_2"});
  loginAdmin();
  sql("REVOKE CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1;");
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1");
}

TEST_P(ShareAndUnshareDashboardsTest, MixedErrors) {
  bool do_share = getDoShare();
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
  error_stream << "Share/Unshare dashboard(s) failed with error(s)\n";
  error_stream << "Dashboard ids " << do_not_exist_dashboard_ids_stream.str()
               << ": Dashboard id does not exist\n";
  error_stream
      << "Dashboard ids " << unauthorized_user_dashboard_ids_stream.str() << ": "
      << "User should be either owner of dashboard or super user to share/unshare it\n";
  TestPermissions permissions(true, true, true, true);
  executeLambdaAndAssertException(
      [&, this] {
        shareOrUnshareDashboards(permissions, {"test_user_2"}, dashboard_ids, do_share);
      },
      error_stream.str());
  assertExpectedDashboardPrivilegesForUsers(getInitialPermissions(), {"test_user_2"});
  loginAdmin();
  sql("REVOKE CREATE DASHBOARD ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1;");
  sql("REVOKE ACCESS ON DATABASE " + OMNISCI_DEFAULT_DB + " FROM test_user_1");
}

class DashboardBulkDeleteTest : public DBHandlerTestFixture {
 public:
  static void SetUpTestSuite() {
    createDBHandler();
    test_user_1_id = createTestUser("test_user", "test_pass");
  }

  static void TearDownTestSuite() {
    loginAdmin();
    dropTestUser("test_user");
  }

  size_t getNumDashboards() {
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    std::vector<TDashboard> dashboards;
    db_handler->get_dashboards(dashboards, session_id);
    return dashboards.size();
  }

 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
    // Remove any dashboards that may be left by previous tests
    std::vector<TDashboard> dashboards;
    db_handler->get_dashboards(dashboards, session_id);
    if (dashboards.size()) {
      std::vector<int32_t> db_ids;
      for (const auto& dashboard : dashboards) {
        db_ids.push_back(dashboard.dashboard_id);
      }
      db_handler->delete_dashboards(session_id, db_ids);
    }
  }
  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  static int32_t createTestUser(const std::string& user_name, const std::string& pass) {
    sql("CREATE USER " + user_name + " (password = '" + pass + "');");
    sql("GRANT ACCESS ON DATABASE omnisci TO " + user_name + ";");
    Catalog_Namespace::UserMetadata user_metadata{};
    Catalog_Namespace::SysCatalog::instance().getMetadataForUser(user_name,
                                                                 user_metadata);
    return user_metadata.userId;
  }

  static void dropTestUser(const std::string& user_name) {
    try {
      sql("DROP USER " + user_name + ";");
    } catch (const std::exception& e) {
      // Swallow and log exceptions that may occur, since there is no "IF EXISTS" option.
      LOG(WARNING) << e.what();
    }
  }

  inline static int32_t test_user_1_id;
};

TEST_F(DashboardBulkDeleteTest, CreateDelete) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  auto db_id_1 =
      db_handler->create_dashboard(session_id, "test1", "state", "image", "metadata");
  auto db_id_2 =
      db_handler->create_dashboard(session_id, "test2", "state", "image", "metadata");
  ASSERT_EQ(getNumDashboards(), size_t(2));
  db_handler->delete_dashboards(session_id, {db_id_1, db_id_2});
  ASSERT_EQ(getNumDashboards(), size_t(0));
}

TEST_F(DashboardBulkDeleteTest, SomeInvalidIDs) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  auto db_id =
      db_handler->create_dashboard(session_id, "test1", "state", "image", "metadata");

  try {
    db_handler->delete_dashboards(session_id, {db_id, 0});
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TOmniSciException& e) {
    assertExceptionMessage(e,
                           "Exception: Delete dashboard(s) failed with "
                           "error(s):\nDashboard id: 0 - Dashboard id does not exist");
  }
  ASSERT_EQ(getNumDashboards(), size_t(1));
  db_handler->delete_dashboards(session_id, {db_id});
  ASSERT_EQ(getNumDashboards(), size_t(0));
}

TEST_F(DashboardBulkDeleteTest, NoDeleteDashboardPrivilege) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  auto db_id_1 =
      db_handler->create_dashboard(session_id, "test1", "state", "image", "metadata");
  auto db_id_2 =
      db_handler->create_dashboard(session_id, "test2", "state", "image", "metadata");
  TSessionId unprivileged_session;
  login("test_user", "test_pass", "omnisci", unprivileged_session);
  sql("GRANT DELETE ON DASHBOARD " + std::to_string(db_id_1) + " to test_user");
  try {
    db_handler->delete_dashboards(unprivileged_session, {db_id_1, db_id_2});
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TOmniSciException& e) {
    assertExceptionMessage(
        e,
        "Exception: Delete dashboard(s) failed with error(s):\nDashboard id: " +
            std::to_string(db_id_2) +
            " - User should be either owner of dashboard or super user to delete it");
  }
  ASSERT_EQ(getNumDashboards(), size_t(2));
  db_handler->delete_dashboards(session_id, {db_id_1, db_id_2});
  ASSERT_EQ(getNumDashboards(), size_t(0));
  logout(unprivileged_session);
}

TEST_F(DashboardBulkDeleteTest, NonSuperDeleteDashboardPrivilege) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  auto db_id_1 =
      db_handler->create_dashboard(session_id, "test1", "state", "image", "metadata");
  auto db_id_2 =
      db_handler->create_dashboard(session_id, "test2", "state", "image", "metadata");
  TSessionId non_super_session;
  login("test_user", "test_pass", "omnisci", non_super_session);
  sql("GRANT DELETE ON DASHBOARD " + std::to_string(db_id_1) + " to test_user");
  sql("GRANT DELETE ON DASHBOARD " + std::to_string(db_id_2) + " to test_user");
  db_handler->delete_dashboards(non_super_session, {db_id_1, db_id_2});
  ASSERT_EQ(getNumDashboards(), size_t(0));
  logout(non_super_session);
}

TEST_F(DashboardBulkDeleteTest, InvalidNoPrivilegeMix) {
  const auto& [db_handler, session_id] = getDbHandlerAndSessionId();
  auto db_id =
      db_handler->create_dashboard(session_id, "test1", "state", "image", "metadata");
  TSessionId non_super_session;
  login("test_user", "test_pass", "omnisci", non_super_session);
  try {
    db_handler->delete_dashboards(non_super_session, {0, db_id});
    FAIL() << "An exception should have been thrown for this test case.";
  } catch (const TOmniSciException& e) {
    assertExceptionMessage(
        e,
        std::string("Exception: Delete dashboard(s) failed with error(s):\n") +
            "Dashboard id: 0 - Dashboard id does not exist\n" +
            "Dashboard id: " + std::to_string(db_id) +
            " - User should be either owner of dashboard or super user to delete it");
  }
  ASSERT_EQ(getNumDashboards(), size_t(1));
  db_handler->delete_dashboards(session_id, {db_id});
  ASSERT_EQ(getNumDashboards(), size_t(0));

  logout(non_super_session);
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
