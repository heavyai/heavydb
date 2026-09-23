/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file UserMappingDdlTest.cpp
 * @brief Test suite for user mapping DDL commands
 *
 */

#include <gtest/gtest.h>

#include "Catalog/OptionsContainer.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapperFactory.h"
#include "Shared/Encryption.h"
#include "Shared/StringTransform.h"
#include "Shared/SysDefinitions.h"
#include "Tests/DBHandlerTestHelpers.h"
#include "Tests/TestHelpers.h"
#ifdef EE_FSI_ODBC
#include "DataMgr/ForeignStorage/ODBC/OdbcDataWrapper.h"
#endif  // EE_FSI_ODBC

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#ifdef HAVE_AWS_S3
extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;
#endif

class UserMappingTest : public DBHandlerTestFixture {
 protected:
#ifdef HAVE_AWS_S3
  // The testing within CreateUserMappingTest & DropUserMappingTest is not
  // exclusive to AWS_S3, however testing user mappings requires passing
  // supported_user_mapping_options_ which is unpopulated without AWS_S3/ODBC.
  inline static const std::map<std::string, std::string> S3_EXPECTED_MAP_ = {
      {"S3_ACCESS_KEY", "test_value_1"},
      {"S3_SECRET_KEY", "test_value_2"}};

  static void SetUpTestSuite() {
    createDBHandler();
    test_user_1_id = createTestUser("test_user_1", "test_pass");
    test_user_2_id = createTestUser("test_user_2", "test_pass");
    test_server_id = createTestS3Server("test_s3_server");
    test_server_id_2 = createTestLocalServer("test_local_server");
    login("test_user_1", "test_pass");
    test_server_id_user = createTestS3Server("test_s3_user_server");
    loginAdmin();
  }

  static void TearDownTestSuite() {
    loginAdmin();
    dropTestUser("test_user_1");
    dropTestUser("test_user_2");
    sql("DROP SERVER IF EXISTS test_s3_server;");
    sql("DROP SERVER IF EXISTS test_local_server;");
    sql("DROP SERVER IF EXISTS test_s3_user_server;");
  }

  void SetUp() override {
    g_enable_fsi = true;
    DBHandlerTestFixture::SetUp();
  }

  void TearDown() override {
    g_enable_fsi = true;
    getCatalog().dropUserMapping(test_user_1_id, test_server_id, true);
    getCatalog().dropUserMapping(test_user_2_id, test_server_id, true);
    getCatalog().dropUserMapping(shared::kRootUserId, test_server_id, true);
    getCatalog().dropUserMapping(shared::kRootUserId, test_server_id_2, true);
    getCatalog().dropUserMapping(shared::kRootUserId, test_server_id_user, true);
    DBHandlerTestFixture::TearDown();
  }

  static int32_t createTestUser(const std::string& user_name, const std::string& pass) {
    sql("CREATE USER " + user_name + " (password = '" + pass + "');");
    sql("GRANT ACCESS ON DATABASE " + shared::kDefaultDbName + " TO " + user_name + ";");
    sql("GRANT CREATE SERVER ON DATABASE " + shared::kDefaultDbName + " TO " + user_name +
        ";");
    Catalog_Namespace::UserMetadata user_metadata{};
    Catalog_Namespace::SysCatalog::instance().getMetadataForUser(user_name,
                                                                 user_metadata);
    return user_metadata.userId;
  }

  static void dropTestUser(const std::string& user_name) {
    sql("DROP USER IF EXISTS " + user_name + ";");
  }

  static int createTestS3Server(const std::string& server_name) {
    sql("CREATE SERVER " + server_name +
        " FOREIGN DATA WRAPPER delimited_file "
        "WITH (storage_type = 'AWS_S3', s3_bucket = 'test_bucket', aws_region = "
        "'test_region');");
    return getCatalog().getForeignServer(server_name)->id;
  }

  static int createTestLocalServer(const std::string& server_name) {
    sql("CREATE SERVER " + server_name +
        " FOREIGN DATA WRAPPER delimited_file "
        "WITH (storage_type = 'LOCAL_FILE', base_path = '/test_path/');");
    return getCatalog().getForeignServer(server_name)->id;
  }

  inline static int32_t test_user_1_id;
  inline static int32_t test_user_2_id;
  inline static int test_server_id;
  inline static int test_server_id_2;
  inline static int test_server_id_user;
#endif  // HAVE_AWS_S3

  void assertExpectedPublicUserMapping(
      const int server_id,
      const std::map<std::string, std::string>& expected_mapping) {
    auto& catalog = getCatalog();
    auto user_mapping = catalog.getUserMapping(shared::kRootUserId, server_id);
    assertExpectedPublicUserMappingAttributes(user_mapping, server_id, expected_mapping);
    auto user_mapping_from_storage =
        catalog.getUserMappingFromStorage(shared::kRootUserId, server_id);
    assertExpectedPublicUserMappingAttributes(
        user_mapping_from_storage.get(), server_id, expected_mapping);
  }

  void assertExpectedPublicUserMappingAttributes(
      const foreign_storage::UserMapping* user_mapping,
      const int server_id,
      const std::map<std::string, std::string>& expected_mapping) {
    ASSERT_NE(nullptr, user_mapping);
    ASSERT_GT(user_mapping->id, 0);
    ASSERT_EQ(shared::kRootUserId, user_mapping->user_id);
    ASSERT_EQ(server_id, user_mapping->foreign_server_id);
    ASSERT_EQ(foreign_storage::UserMappingType::PUBLIC, user_mapping->type);
    assertEncryptedOptions(user_mapping->options, expected_mapping);
  }

  void assertEncryptedOptions(const std::string& encrypted_options,
                              std::map<std::string, std::string> expected_mapping) {
    ASSERT_FALSE(encrypted_options.empty());
    for (auto const& [attr, val] : expected_mapping) {
      ASSERT_TRUE(encrypted_options.find(attr) == std::string::npos);
      ASSERT_TRUE(encrypted_options.find(val) == std::string::npos);
    }

    const auto& plain_text = PkiEncryptor::privateKeyDecrypt(encrypted_options);
    foreign_storage::OptionsContainer options_container{};
    options_container.populateOptionsMap(plain_text);

    const auto& options = options_container.options;
    for (auto const& [attr, val] : expected_mapping) {
      ASSERT_FALSE(options.find(attr) == options.end());
      ASSERT_EQ(val, options.find(attr)->second);
    }
  }
};

#ifdef HAVE_AWS_S3
class CreateUserMappingTest : public UserMappingTest {};

TEST_F(CreateUserMappingTest, NotPublic) {
  queryAndAssertException(
      "CREATE USER MAPPING FOR test_user_2 SERVER test_s3_server WITH (S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');",
      "Invalid user 'test_user_2' specified for CREATE or DROP USER MAPPING "
      "command. Only PUBLIC user mappings are supported.");
}

TEST_F(CreateUserMappingTest, WithPublicUser) {
  sql("CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server WITH (S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');");
  assertExpectedPublicUserMapping(test_server_id, S3_EXPECTED_MAP_);
}

TEST_F(CreateUserMappingTest, WithPublicUserAndNonSuperUser) {
  login("test_user_1", "test_pass");
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server WITH (S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');",
      "Public user mappings can only be created or dropped by super user or "
      "owner of the "
      "server.");
}

TEST_F(CreateUserMappingTest, WithPublicUserAndNonSuperUserOwner) {
  login("test_user_1", "test_pass");
  sql("CREATE USER MAPPING FOR PUBLIC SERVER test_s3_user_server WITH (S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');");
  assertExpectedPublicUserMapping(test_server_id_user, S3_EXPECTED_MAP_);
}

TEST_F(CreateUserMappingTest, ExistingUserMappingWithIfNotExists) {
  sql("CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server WITH (S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');");
  sql("CREATE USER MAPPING IF NOT EXISTS FOR PUBLIC SERVER test_s3_server WITH "
      "(S3_ACCESS_KEY = 'test_value_1', S3_SECRET_KEY = 'test_value_2');");
  assertExpectedPublicUserMapping(test_server_id, S3_EXPECTED_MAP_);
}

TEST_F(CreateUserMappingTest, ExistingUserMappingWithoutIfNotExists) {
  std::string query{
      "CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server WITH (S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');"};
  sql(query);
  queryAndAssertException(query,
                          "A user mapping already exists for user and foreign server.");
}

TEST_F(CreateUserMappingTest, NonExistentServer) {
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER non_existent_server WITH "
      "(S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');",
      "Foreign server with name \"non_existent_server\" does not exist.");
}

TEST_F(CreateUserMappingTest, FsiDisabled) {
  g_enable_fsi = false;
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server WITH (S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');",
      "Unsupported command: CREATE USER MAPPING");
}

TEST_F(CreateUserMappingTest, CreateOrReplaceUserMapping) {
  std::string query{
      "CREATE OR REPLACE USER MAPPING FOR PUBLIC SERVER  test_s3_server WITH "
      "(S3_ACCESS_KEY = 'test_value_1', S3_SECRET_KEY = 'test_value_2');"};
  // using a partial exception for the sake of brevity
  queryAndAssertPartialException(query,
                                 R"(SQL Error: Encountered "USER" at line 1, column 19.
Was expecting:
    "MODEL" ...)");
}

class S3CreateUserMappingTest : public UserMappingTest {};

TEST_F(S3CreateUserMappingTest, UnsupportedOption) {
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server WITH (unsupported_option = "
      "'value');",
      "Invalid user mapping option \"UNSUPPORTED_OPTION\". Option must be one "
      "of the following: S3_ACCESS_KEY, S3_SECRET_KEY, S3_SESSION_TOKEN.");
}

TEST_F(S3CreateUserMappingTest, S3StorageMissingAccessKey) {
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server WITH (S3_SECRET_KEY = "
      "'test_value_2');",
      "User mapping options must contain \"S3_ACCESS_KEY\".");
}

TEST_F(S3CreateUserMappingTest, S3StorageMissingSecretKey) {
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server WITH (S3_ACCESS_KEY = "
      "'test_value_1');",
      "User mapping options must contain \"S3_SECRET_KEY\".");
}

TEST_F(S3CreateUserMappingTest, LocalFileStorage) {
  queryAndAssertException(
      "CREATE USER MAPPING FOR PUBLIC SERVER test_local_server WITH (S3_ACCESS_KEY = "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');",
      "User mapping for the \"DELIMITED_FILE\" data wrapper can only be created "
      "for "
      "AWS S3 backed foreign servers.");
}

class DropUserMappingTest : public UserMappingTest {
 protected:
  static void SetUpTestSuite() { UserMappingTest::SetUpTestSuite(); }

  static void TearDownTestSuite() { UserMappingTest::TearDownTestSuite(); }

  void SetUp() override {
    UserMappingTest::SetUp();
    sql("CREATE USER MAPPING FOR PUBLIC SERVER test_s3_user_server WITH (S3_ACCESS_KEY "
        "= "
        "'test_value_1', S3_SECRET_KEY = 'test_value_2');");
    sql("DROP SERVER IF EXISTS test_s3_server_1");
  }

  void assertNullPublicUserMapping(const int server_id) {
    auto& catalog = getCatalog();
    ASSERT_EQ(nullptr, catalog.getUserMapping(shared::kRootUserId, server_id));
    ASSERT_EQ(nullptr, catalog.getUserMappingFromStorage(shared::kRootUserId, server_id));
  }
};

TEST_F(DropUserMappingTest, NotPublic) {
  queryAndAssertException(
      "DROP USER MAPPING FOR test_user_1 SERVER test_s3_server;",
      "Invalid user 'test_user_1' specified for CREATE or DROP USER MAPPING "
      "command. Only PUBLIC user mappings are supported.");
}

TEST_F(DropUserMappingTest, WithPublicUser) {
  sql("DROP USER MAPPING FOR PUBLIC SERVER test_s3_user_server;");
  assertNullPublicUserMapping(test_server_id_user);
}

TEST_F(DropUserMappingTest, WithPublicUserAndNonSuperUser) {
  login("test_user_1", "test_pass");
  queryAndAssertException(
      "DROP USER MAPPING FOR PUBLIC SERVER test_s3_server;",
      "Public user mappings can only be created or dropped by super user or "
      "owner of the "
      "server.");
}

TEST_F(DropUserMappingTest, WithPublicUserAndNonSuperUserOwner) {
  login("test_user_1", "test_pass");
  sql("DROP USER MAPPING FOR PUBLIC SERVER test_s3_user_server;");
  assertNullPublicUserMapping(test_server_id_user);
}

TEST_F(DropUserMappingTest, NonExistentUserMappingWithIfExists) {
  createTestS3Server("test_s3_server_1");
  sql("DROP USER MAPPING IF EXISTS FOR PUBLIC SERVER test_s3_server_1;");
  sql("DROP SERVER test_s3_server_1;");
}

TEST_F(DropUserMappingTest, NonExistentUserMappingWithoutIfExists) {
  createTestS3Server("test_s3_server_1");
  queryAndAssertException("DROP USER MAPPING FOR PUBLIC SERVER test_s3_server_1;",
                          "A user mapping does not exist for user and foreign server.");
  sql("DROP SERVER test_s3_server_1;");
}

TEST_F(DropUserMappingTest, NonExistentServer) {
  queryAndAssertException(
      "DROP USER MAPPING FOR PUBLIC SERVER non_existent_server;",
      "Foreign server with name \"non_existent_server\" does not exist.");
}

TEST_F(DropUserMappingTest, DropServer) {
  auto server_id = createTestS3Server("test_s3_server_1");
  sql("CREATE USER MAPPING FOR PUBLIC SERVER test_s3_server_1 WITH (S3_ACCESS_KEY "
      "= "
      "'test_value_1', S3_SECRET_KEY = 'test_value_2');");
  assertExpectedPublicUserMapping(server_id, S3_EXPECTED_MAP_);
  sql("DROP SERVER test_s3_server_1;");
  assertNullPublicUserMapping(server_id);
}

TEST_F(DropUserMappingTest, FsiDisabled) {
  g_enable_fsi = false;
  queryAndAssertException("DROP USER MAPPING FOR PUBLIC SERVER test_s3_server;",
                          "Unsupported command: DROP USER MAPPING");
}
#endif  // HAVE_AWS_S3

#ifdef EE_FSI_ODBC
class OdbcUserMappingTests : public UserMappingTest {
 public:
  inline static const std::map<std::string, std::string> ODBC_DSN_EXPECTED_MAP_ = {
      {"USERNAME", "odbc_user"},
      {"PASSWORD", "odbc_password"}};
  inline static const std::map<std::string, std::string> ODBC_CNSTR_EXPECTED_MAP_ = {
      {"CREDENTIAL_STRING", "odbc_credential_string"}};

  static void SetUpTestSuite() { createDBHandler(); }

  void TearDown() override { sql("DROP SERVER IF EXISTS odbc_mapping_test_server"); }

 protected:
  static int createOdbcDsnServer() {
    const auto queryString =
        "CREATE SERVER odbc_mapping_test_server FOREIGN DATA WRAPPER odbc WITH (DATA_SOURCE_NAME='sqlite');"s;
    sql(queryString);
    return getCatalog().getForeignServer("odbc_mapping_test_server")->id;
  }

  static int createOdbcCnstrServer() {
    const auto queryString =
        "CREATE SERVER odbc_mapping_test_server FOREIGN DATA WRAPPER odbc WITH (CONNECTION_STRING='DSN=sqlite');"s;
    sql(queryString);
    return getCatalog().getForeignServer("odbc_mapping_test_server")->id;
  }
};

TEST_F(OdbcUserMappingTests, DataSourceNameValidOptions) {
  const auto server_id = createOdbcDsnServer();
  std::string queryStmt =
      "CREATE USER MAPPING FOR PUBLIC SERVER odbc_mapping_test_server WITH "
      "(username='odbc_user', password='odbc_password');";
  sql(queryStmt);
  assertExpectedPublicUserMapping(server_id, ODBC_DSN_EXPECTED_MAP_);
}
TEST_F(OdbcUserMappingTests, DataSourceNameInvalidOptions) {
  createOdbcDsnServer();
  std::string queryStmt =
      "CREATE USER MAPPING FOR PUBLIC SERVER odbc_mapping_test_server WITH "
      "(credential_string='odbc_credential_str');";
  queryAndAssertException(queryStmt,
                          "The user mapping option \"CREDENTIAL_STRING\" is incompatible "
                          "with the foreign server \"odbc_mapping_test_server\" "
                          "which sets the server option \"DATA_SOURCE_NAME\".");
}
TEST_F(OdbcUserMappingTests, DataSourceNameUnrecognizedOptions) {
  createOdbcDsnServer();
  std::string queryStmt =
      "CREATE USER MAPPING FOR PUBLIC SERVER odbc_mapping_test_server WITH "
      "(invalid_option='value');";
  queryAndAssertException(
      queryStmt,
      "Invalid user mapping option \"INVALID_OPTION\". Option must be one of "
      "the following: CREDENTIAL_STRING, PASSWORD, USERNAME.");
}
TEST_F(OdbcUserMappingTests, ConnectionStringValidOptions) {
  const auto server_id = createOdbcCnstrServer();
  std::string queryStmt =
      "CREATE USER MAPPING FOR PUBLIC SERVER odbc_mapping_test_server WITH "
      "(credential_string='odbc_credential_string');";
  sql(queryStmt);
  assertExpectedPublicUserMapping(server_id, ODBC_CNSTR_EXPECTED_MAP_);
}

TEST_F(OdbcUserMappingTests, PasswordRequiresUsername) {
  createOdbcDsnServer();
  std::string queryStmt =
      "CREATE USER MAPPING FOR PUBLIC SERVER odbc_mapping_test_server WITH "
      "(password='odbc_password');";
  queryAndAssertException(
      queryStmt,
      "User mapping option \"PASSWORD\" requires a matching \"USERNAME\" option.");
}

TEST_F(OdbcUserMappingTests, OnlyUsernameOrCredentialString) {
  createOdbcCnstrServer();
  std::string queryStmt =
      "CREATE USER MAPPING FOR PUBLIC SERVER odbc_mapping_test_server WITH "
      "(username='odbc_user', credential_string='odbc_credential_string');";
  queryAndAssertException(queryStmt,
                          "User mapping must contain only one of \"USERNAME/PASSWORD\" "
                          "or \"CREDENTIAL_STRING\".");
}

TEST_F(OdbcUserMappingTests, ConnectionStringInvalidOptions) {
  createOdbcCnstrServer();
  std::string queryStmt =
      "CREATE USER MAPPING FOR PUBLIC SERVER odbc_mapping_test_server WITH "
      "(username='odbc_user', password='odbc_password');";
  queryAndAssertException(
      queryStmt,
      "The user mapping option \"USERNAME\" is incompatible with the foreign "
      "server "
      "\"odbc_mapping_test_server\" which sets the server option \"CONNECTION_STRING\".");
}
TEST_F(OdbcUserMappingTests, ConnectionStringUnrecognizedOptions) {
  createOdbcCnstrServer();
  std::string queryStmt =
      "CREATE USER MAPPING FOR PUBLIC SERVER odbc_mapping_test_server WITH "
      "(invalid_option='value');";
  queryAndAssertException(
      queryStmt,
      "Invalid user mapping option \"INVALID_OPTION\". Option must be one of "
      "the following: CREDENTIAL_STRING, PASSWORD, USERNAME.");
}
#endif  // EE_FSI_ODBC

class UserMappingOptionObfuscationTest : public testing::TestWithParam<std::string> {
 public:
  static std::set<std::string> getAllUserMappingOptions() {
    std::set<std::string> user_mapping_options;
    for (const auto& data_wrapper :
         foreign_storage::DataWrapperType::supported_data_wrapper_types) {
      auto& data_wrapper_options =
          foreign_storage::ForeignDataWrapperFactory::createForValidation(
              std::string{data_wrapper})
              ->getSupportedUserMappingOptions();
      for (const auto& option : data_wrapper_options) {
        user_mapping_options.emplace(option);
      }
    }
    return user_mapping_options;
  }
};

INSTANTIATE_TEST_SUITE_P(
    OptionObfuscationTest,
    UserMappingOptionObfuscationTest,
    testing::ValuesIn(UserMappingOptionObfuscationTest::getAllUserMappingOptions()),
    [](const auto& param_info) { return param_info.param; });

TEST_P(UserMappingOptionObfuscationTest, OptionIsObfuscated) {
  std::string expected_obfuscated_text{"{" + GetParam() + " = 'XXXXXXXX'}"};
  auto actual_obfuscated_text =
      hide_sensitive_data_from_query("{" + GetParam() + " = 'test_value'}");
  ASSERT_EQ(expected_obfuscated_text, actual_obfuscated_text);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  PkiEncryptor::setKeyStorePath("../../Tests/Encryption/ValidCert/");
  g_enable_fsi = true;
  g_enable_s3_fsi = true;

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
