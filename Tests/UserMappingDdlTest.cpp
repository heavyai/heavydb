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
 * @file UserMappingDdlTest.cpp
 * @brief Test suite for user mapping DDL commands
 */

#include <gtest/gtest.h>

#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class UserMappingTest : public DBHandlerTestFixture {
 public:
  static void SetUpTestSuite() {
    createDBHandler();
    test_user_1_id = createTestUser("test_user_1", "test_pass");
    test_user_2_id = createTestUser("test_user_2", "test_pass");
    test_server_id = createTestServer("test_server");
  }

  static void TearDownTestSuite() {
    loginAdmin();
    dropTestUser("test_user_1");
    dropTestUser("test_user_2");
    sql("DROP SERVER IF EXISTS test_server;");
  }

 protected:
  void SetUp() override {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
  }

  void TearDown() override {
    g_enable_fsi = true;
    getCatalog().dropUserMapping(test_user_1_id, test_server_id, true);
    getCatalog().dropUserMapping(test_user_2_id, test_server_id, true);
    getCatalog().dropUserMapping(OMNISCI_ROOT_USER_ID, test_server_id, true);
    DBHandlerTestFixture::TearDown();
  }

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

  static int createTestServer(const std::string& server_name) {
    sql("CREATE SERVER " + server_name +
        " FOREIGN DATA WRAPPER omnisci_csv "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
    return getCatalog().getForeignServer(server_name)->id;
  }

  void assertExpectedUserMapping(const int32_t user_id,
                                 const int server_id = test_server_id) {
    auto& catalog = getCatalog();
    assertExpectedUserMappingAttributes(
        catalog.getUserMapping(user_id, server_id), user_id, server_id);
    auto user_mapping_from_storage =
        catalog.getUserMappingFromStorage(user_id, server_id);
    assertExpectedUserMappingAttributes(
        user_mapping_from_storage.get(), user_id, server_id);
  }

  void assertExpectedUserMappingAttributes(
      const foreign_storage::UserMapping* user_mapping,
      const int32_t user_id,
      const int server_id) {
    ASSERT_TRUE(user_mapping != nullptr);
    ASSERT_GT(user_mapping->id, 0);
    ASSERT_EQ(user_id, user_mapping->user_id);
    ASSERT_EQ(server_id, user_mapping->foreign_server_id);
    ASSERT_EQ(foreign_storage::UserMappingType::USER, user_mapping->type);

    // TODO: update when user mapping encryption is implemented
    ASSERT_TRUE(user_mapping->options.empty());
  }

  void assertExpectedPublicUserMapping() {
    auto& catalog = getCatalog();
    auto user_mapping = catalog.getUserMapping(OMNISCI_ROOT_USER_ID, test_server_id);
    assertExpectedPublicUserMappingAttributes(user_mapping);
    auto user_mapping_from_storage =
        catalog.getUserMappingFromStorage(OMNISCI_ROOT_USER_ID, test_server_id);
    assertExpectedPublicUserMappingAttributes(user_mapping_from_storage.get());
  }

  void assertExpectedPublicUserMappingAttributes(
      const foreign_storage::UserMapping* user_mapping) {
    ASSERT_NE(nullptr, user_mapping);
    ASSERT_GT(user_mapping->id, 0);
    ASSERT_EQ(OMNISCI_ROOT_USER_ID, user_mapping->user_id);
    ASSERT_EQ(test_server_id, user_mapping->foreign_server_id);
    ASSERT_EQ(foreign_storage::UserMappingType::PUBLIC, user_mapping->type);

    // TODO: update when user mapping encryption is implemented
    ASSERT_TRUE(user_mapping->options.empty());
  }

  inline static int32_t test_user_1_id;
  inline static int32_t test_user_2_id;
  inline static int test_server_id;
};

class CreateUserMappingTest : public UserMappingTest {
 protected:
  static void SetUpTestSuite() { UserMappingTest::SetUpTestSuite(); }

  static void TearDownTestSuite() { UserMappingTest::TearDownTestSuite(); }
};

TEST_F(CreateUserMappingTest, WithCurrentUser) {
  login("test_user_1", "test_pass");
  sql("CREATE USER MAPPING FOR CURRENT_USER SERVER test_server WITH (test_key = "
      "'test_value');");
  assertExpectedUserMapping(getCurrentUser().userId);
}

TEST_F(CreateUserMappingTest, WithUsername) {
  sql("CREATE USER MAPPING FOR test_user_1 SERVER test_server WITH (test_key = "
      "'test_value');");
  assertExpectedUserMapping(test_user_1_id);
}

TEST_F(CreateUserMappingTest, WithUsernameAndNonSuperUser) {
  login("test_user_1", "test_pass");
  queryAndAssertException(
      "CREATE USER MAPPING FOR test_user_2 SERVER test_server WITH (test_key = "
      "'test_value');",
      "Exception: User mappings for other users can only be created or dropped by admins "
      "or super users.");
}

TEST_F(CreateUserMappingTest, WithPublicUser) {
  sql("CREATE USER MAPPING FOR PUBLIC SERVER test_server WITH (test_key = "
      "'test_value');");
  assertExpectedPublicUserMapping();
}

TEST_F(CreateUserMappingTest, WithPublicUserAndNonSuperUser) {
  login("test_user_1", "test_pass");
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER test_server WITH (test_key = 'test_value');",
      "Exception: Public user mappings can only be created or dropped by admins or super "
      "users.");
}

TEST_F(CreateUserMappingTest, ExistingUserMappingWithIfNotExists) {
  sql("CREATE USER MAPPING FOR CURRENT_USER SERVER test_server WITH (test_key = "
      "'test_value');");
  sql("CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER SERVER test_server WITH "
      "(test_key = 'test_value');");
  assertExpectedUserMapping(getCurrentUser().userId);
}

TEST_F(CreateUserMappingTest, ExistingUserMappingWithoutIfNotExists) {
  std::string query{
      "CREATE USER MAPPING FOR CURRENT_USER SERVER test_server WITH (test_key = "
      "'test_value');"};
  sql(query);
  queryAndAssertException(
      query, "Exception: A user mapping already exists for user and foreign server.");
}

TEST_F(CreateUserMappingTest, WithPublicUserAndExistingUserSpecificMapping) {
  sql("CREATE USER MAPPING FOR test_user_1 SERVER test_server WITH (test_key = "
      "'test_value');");
  sql("CREATE USER MAPPING FOR PUBLIC SERVER test_server WITH (test_key = "
      "'test_value');");
  assertExpectedUserMapping(test_user_1_id);
  assertExpectedPublicUserMapping();
}

TEST_F(CreateUserMappingTest, WithUsernameAndExistingPublicUserMapping) {
  sql("CREATE USER MAPPING FOR PUBLIC SERVER test_server WITH (test_key = "
      "'test_value');");
  sql("CREATE USER MAPPING FOR test_user_1 SERVER test_server WITH (test_key = "
      "'test_value');");
  assertExpectedUserMapping(test_user_1_id);
  assertExpectedPublicUserMapping();
}

TEST_F(CreateUserMappingTest, NonExistentUsername) {
  queryAndAssertException(
      "CREATE USER MAPPING FOR non_existent_user SERVER test_server WITH (test_key = "
      "'test_value');",
      "Exception: User with name \"non_existent_user\" does not exist.");
}

TEST_F(CreateUserMappingTest, NonExistentServer) {
  queryAndAssertException(
      "CREATE USER MAPPING FOR test_user_1 SERVER non_existent_server WITH (test_key = "
      "'test_value');",
      "Exception: Foreign server with name \"non_existent_server\" does not exist.");
}

TEST_F(CreateUserMappingTest, FsiDisabled) {
  g_enable_fsi = false;
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER test_server WITH (test_key = 'test_value');",
      "Syntax error at: FOR");
}

class DropUserMappingTest : public UserMappingTest {
 protected:
  static void SetUpTestSuite() { UserMappingTest::SetUpTestSuite(); }

  static void TearDownTestSuite() { UserMappingTest::TearDownTestSuite(); }

  void SetUp() override {
    UserMappingTest::SetUp();
    sql("CREATE USER MAPPING FOR CURRENT_USER SERVER test_server WITH (test_key = "
        "'test_value');");
    sql("CREATE USER MAPPING FOR test_user_1 SERVER test_server WITH (test_key = "
        "'test_value');");
  }

  void assertNullUserMapping(const int32_t user_id,
                             const int server_id = test_server_id) {
    auto& catalog = getCatalog();
    ASSERT_EQ(nullptr, catalog.getUserMapping(user_id, server_id));
    ASSERT_EQ(nullptr, catalog.getUserMappingFromStorage(user_id, server_id));
  }

  void assertNullPublicUserMapping() {
    auto& catalog = getCatalog();
    ASSERT_EQ(nullptr, catalog.getUserMapping(OMNISCI_ROOT_USER_ID, test_server_id));
    ASSERT_EQ(nullptr,
              catalog.getUserMappingFromStorage(OMNISCI_ROOT_USER_ID, test_server_id));
  }
};

TEST_F(DropUserMappingTest, WithCurrentUser) {
  login("test_user_1", "test_pass");
  sql("DROP USER MAPPING FOR CURRENT_USER SERVER test_server;");
  assertNullUserMapping(getCurrentUser().userId);
}

TEST_F(DropUserMappingTest, WithUsername) {
  sql("DROP USER MAPPING FOR test_user_1 SERVER test_server;");
  assertNullUserMapping(test_user_1_id);
}

TEST_F(DropUserMappingTest, WithUsernameAndNonSuperUser) {
  login("test_user_1", "test_pass");
  queryAndAssertException("DROP USER MAPPING FOR test_user_1 SERVER test_server;",
                          "Exception: User mappings for other users can only be created "
                          "or dropped by admins or super users.");
}

TEST_F(DropUserMappingTest, WithPublicUser) {
  sql("DROP USER MAPPING FOR PUBLIC SERVER test_server;");
  assertNullPublicUserMapping();
}

TEST_F(DropUserMappingTest, WithPublicUserAndNonSuperUser) {
  login("test_user_1", "test_pass");
  queryAndAssertException("DROP USER MAPPING FOR PUBLIC SERVER test_server;",
                          "Exception: Public user mappings can only be created or "
                          "dropped by admins or super users.");
}

TEST_F(DropUserMappingTest, NonExistentUserMappingWithIfExists) {
  createTestServer("test_server_1");
  sql("DROP USER MAPPING IF EXISTS FOR CURRENT_USER SERVER test_server_1;");
  sql("DROP SERVER test_server_1;");
}

TEST_F(DropUserMappingTest, NonExistentUserMappingWithoutIfExists) {
  createTestServer("test_server_1");
  queryAndAssertException(
      "DROP USER MAPPING FOR CURRENT_USER SERVER test_server_1;",
      "Exception: A user mapping does not exist for user and foreign server.");
  sql("DROP SERVER test_server_1;");
}

TEST_F(DropUserMappingTest, NonExistentUsername) {
  queryAndAssertException(
      "DROP USER MAPPING FOR non_existent_user SERVER test_server;",
      "Exception: User with name \"non_existent_user\" does not exist.");
}

TEST_F(DropUserMappingTest, NonExistentServer) {
  queryAndAssertException(
      "DROP USER MAPPING FOR test_user_1 SERVER non_existent_server;",
      "Exception: Foreign server with name \"non_existent_server\" does not exist.");
}

TEST_F(DropUserMappingTest, DropServer) {
  auto server_id = createTestServer("test_server_1");
  sql("CREATE USER MAPPING FOR CURRENT_USER SERVER test_server_1 WITH (test_key = "
      "'test_value');");
  assertExpectedUserMapping(getCurrentUser().userId, server_id);
  sql("DROP SERVER test_server_1;");
  assertNullUserMapping(getCurrentUser().userId, server_id);
}

TEST_F(DropUserMappingTest, DropUser) {
  auto user_id = createTestUser("test_user_3", "test_pass");
  sql("CREATE USER MAPPING FOR test_user_3 SERVER test_server WITH (test_key = "
      "'test_value');");
  assertExpectedUserMapping(user_id);
  sql("DROP USER test_user_3;");
  assertNullUserMapping(user_id);
}

TEST_F(DropUserMappingTest, FsiDisabled) {
  g_enable_fsi = false;
  queryAndAssertException("DROP USER MAPPING FOR test_user_1 SERVER test_server;",
                          "Syntax error at: FOR");
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  g_enable_fsi = true;

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
