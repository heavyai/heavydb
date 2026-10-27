/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file ThriftDetectServerPrivilegeTest.cpp
 * @brief S3 server-privilege tests for detect_column_types/import_table.
 *
 * Split out of LoadTableTest.cpp. These change the *test process's* AWS environment and
 * profile files to steer the in-process DBHandler, so they cannot run as thin clients
 * against a separate heavydb process and keep the in-process fixture.
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#ifdef HAVE_AWS_S3
#include "AwsHelpers.h"
#include "DataMgr/HeavyDbAwsSdk.h"
#include "Shared/ThriftTypesConvert.h"
#endif  // HAVE_AWS_S3
#include "Geospatial/GDAL.h"
#include "Tests/DBHandlerTestHelpers.h"
#include "Tests/TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#ifdef HAVE_AWS_S3
class ThriftDetectServerPrivilegeTest : public DBHandlerTestFixture {
 protected:
  inline const static std::string PUBLIC_S3_FILE =
      "s3://omnisci-fsi-test-public/FsiDataFiles/0.csv";
  inline const static std::string PRIVATE_S3_FILE =
      "s3://omnisci-fsi-test/FsiDataFiles/0.csv";
  inline const static std::string AWS_DUMMY_CREDENTIALS_DIR =
      to_string(BASE_PATH) + "/aws";
  inline static std::map<std::string, std::string> aws_environment_;

  static void SetUpTestSuite() {
    DBHandlerTestFixture::SetUpTestSuite();
    g_allow_s3_server_privileges = true;
    aws_environment_ = unset_aws_env();
    create_stub_aws_profile(AWS_DUMMY_CREDENTIALS_DIR);
  }

  static void TearDownTestSuite() {
    DBHandlerTestFixture::TearDownTestSuite();
    g_allow_s3_server_privileges = false;
    restore_aws_env(aws_environment_);
    boost::filesystem::remove_all(AWS_DUMMY_CREDENTIALS_DIR);
  }

  std::string detectTable(const std::string& file_name,
                          const std::string& s3_access_key = "",
                          const std::string& s3_secret_key = "",
                          const std::string& s3_session_token = "",
                          const std::string& s3_region = "us-west-1") {
    const auto& db_handler_and_session_id = getDbHandlerAndSessionId();
    TDetectResult thrift_result;
    TCopyParams copy_params;
    // Setting S3 credentials through copy params simulates
    // environment variables configured on the heavysql client
    copy_params.s3_access_key = s3_access_key;
    copy_params.s3_secret_key = s3_secret_key;
    copy_params.s3_session_token = s3_session_token;
    copy_params.s3_region = s3_region;
    db_handler_and_session_id.first->detect_column_types(
        thrift_result, db_handler_and_session_id.second, file_name, copy_params);
    std::stringstream oss;
    for (const auto& tct : thrift_result.row_set.row_desc) {
      oss << tct.col_name;
    }
    oss << "\n";
    for (const auto& tct : thrift_result.row_set.row_desc) {
      oss << type_info_from_thrift(tct.col_type).get_type_name();
    }
    oss << "\n";
    for (const auto& row : thrift_result.row_set.rows) {
      for (const auto& col : row.cols) {
        oss << col.val.str_val;
      }
      oss << "\n";
    }
    oss << "\nCREATE TABLE your_table_name(";
    for (size_t i = 0; i < thrift_result.row_set.row_desc.size(); ++i) {
      const auto tct = thrift_result.row_set.row_desc[i];
      oss << (i ? ", " : "") << tct.col_name << " "
          << type_info_from_thrift(tct.col_type).get_type_name();
      if (type_info_from_thrift(tct.col_type).is_string()) {
        oss << " ENCODING DICT";
      }
      if (type_info_from_thrift(tct.col_type).is_array()) {
        oss << "[";
        if (type_info_from_thrift(tct.col_type).get_size() > 0) {
          oss << type_info_from_thrift(tct.col_type).get_size();
        }
        oss << "]";
      }
    }
    oss << ");\n";

    return oss.str();
  }
};

// TODO(IAM): re-enable once anonymous access (or a CI IAM user) on the
// public S3 fixture is available. Currently the bucket no longer allows
// unsigned reads and the CI key isn't granted GetObject on it.
TEST_F(ThriftDetectServerPrivilegeTest, DISABLED_S3_Public_without_credentials) {
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  const auto result = detectTable(PUBLIC_S3_FILE);
  ASSERT_EQ(result, "i\nSMALLINT\n0\n\nCREATE TABLE your_table_name(i SMALLINT);\n");
}

TEST_F(ThriftDetectServerPrivilegeTest, S3_Private_without_credentials) {
  if (is_valid_aws_role()) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_THROW(detectTable(PRIVATE_S3_FILE), TDBException);
}

TEST_F(ThriftDetectServerPrivilegeTest, S3_Private_with_invalid_specified_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_THROW(detectTable(PRIVATE_S3_FILE, "invalid_access_key", "invalid_secret_key"),
               TDBException);
}

TEST_F(ThriftDetectServerPrivilegeTest, S3_Private_with_valid_specified_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  const auto aws_access_key_id = aws_environment_.find("AWS_ACCESS_KEY_ID")->second;
  const auto aws_secret_access_key =
      aws_environment_.find("AWS_SECRET_ACCESS_KEY")->second;
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  const auto result =
      detectTable(PRIVATE_S3_FILE, aws_access_key_id, aws_secret_access_key);
  ASSERT_EQ(result, "i\nSMALLINT\n0\n\nCREATE TABLE your_table_name(i SMALLINT);\n");
}

TEST_F(ThriftDetectServerPrivilegeTest, S3_Private_with_env_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  restore_aws_keys(aws_environment_);
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  const auto result = detectTable(PRIVATE_S3_FILE);
  ASSERT_EQ(result, "i\nSMALLINT\n0\n\nCREATE TABLE your_table_name(i SMALLINT);\n");
  unset_aws_keys();
}

TEST_F(ThriftDetectServerPrivilegeTest, S3_Private_with_profile_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, true, aws_environment_);
  const auto result = detectTable(PRIVATE_S3_FILE);
  ASSERT_EQ(result, "i\nSMALLINT\n0\n\nCREATE TABLE your_table_name(i SMALLINT);\n");
}

TEST_F(ThriftDetectServerPrivilegeTest, S3_Private_with_role_credentials) {
  if (!is_valid_aws_role()) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  const auto result = detectTable(PRIVATE_S3_FILE);
  ASSERT_EQ(result, "i\nSMALLINT\n0\n\nCREATE TABLE your_table_name(i SMALLINT);\n");
}

class ThriftImportServerPrivilegeTest : public ThriftDetectServerPrivilegeTest {
 protected:
  void SetUp() override {
    ThriftDetectServerPrivilegeTest::SetUp();
    sql("DROP TABLE IF EXISTS import_test_table;");
    sql("CREATE TABLE import_test_table(i SMALLINT);");
  }

  void TearDown() override {
    ThriftDetectServerPrivilegeTest::TearDown();
    sql("DROP TABLE IF EXISTS import_test_table;");
  }

  void importTable(const std::string& file_name,
                   const std::string& table_name,
                   const std::string& s3_access_key = "",
                   const std::string& s3_secret_key = "",
                   const std::string& s3_session_token = "",
                   const std::string& s3_region = "us-west-1") {
    const auto& db_handler_and_session_id = getDbHandlerAndSessionId();
    TCopyParams copy_params;
    // Setting S3 credentials through copy params simulates
    // environment variables configured on the heavysql client
    copy_params.s3_access_key = s3_access_key;
    copy_params.s3_secret_key = s3_secret_key;
    copy_params.s3_session_token = s3_session_token;
    copy_params.s3_region = s3_region;
    db_handler_and_session_id.first->import_table(
        db_handler_and_session_id.second, table_name, file_name, copy_params);
  }
};

// TODO(IAM): re-enable once anonymous access (or a CI IAM user) on the
// public S3 fixture is available.
TEST_F(ThriftImportServerPrivilegeTest, DISABLED_S3_Public_without_credentials) {
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_NO_THROW(importTable(PUBLIC_S3_FILE, "import_test_table"));
  sqlAndCompareResult("SELECT * FROM import_test_table", {{i(0)}});
}

TEST_F(ThriftImportServerPrivilegeTest, S3_Private_without_credentials) {
  if (is_valid_aws_role()) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_THROW(importTable(PRIVATE_S3_FILE, "import_test_table"), TDBException);
}

TEST_F(ThriftImportServerPrivilegeTest, S3_Private_with_invalid_specified_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_THROW(importTable(PRIVATE_S3_FILE,
                           "import_test_table",
                           "invalid_access_key",
                           "invalid_secret_key"),
               TDBException);
}

TEST_F(ThriftImportServerPrivilegeTest, S3_Private_with_valid_specified_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  const auto aws_access_key_id = aws_environment_.find("AWS_ACCESS_KEY_ID")->second;
  const auto aws_secret_access_key =
      aws_environment_.find("AWS_SECRET_ACCESS_KEY")->second;
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_NO_THROW(importTable(
      PRIVATE_S3_FILE, "import_test_table", aws_access_key_id, aws_secret_access_key));
  sqlAndCompareResult("SELECT * FROM import_test_table", {{i(0)}});
}

TEST_F(ThriftImportServerPrivilegeTest, S3_Private_with_env_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  restore_aws_keys(aws_environment_);
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_NO_THROW(importTable(PRIVATE_S3_FILE, "import_test_table"));
  sqlAndCompareResult("SELECT * FROM import_test_table", {{i(0)}});
  unset_aws_keys();
}

TEST_F(ThriftImportServerPrivilegeTest, S3_Private_with_profile_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, true, aws_environment_);
  EXPECT_NO_THROW(importTable(PRIVATE_S3_FILE, "import_test_table"));
  sqlAndCompareResult("SELECT * FROM import_test_table", {{i(0)}});
}

TEST_F(ThriftImportServerPrivilegeTest, S3_Private_with_role_credentials) {
  if (!is_valid_aws_role()) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_NO_THROW(importTable(PRIVATE_S3_FILE, "import_test_table"));
  sqlAndCompareResult("SELECT * FROM import_test_table", {{i(0)}});
}

#endif  // HAVE_AWS_S3

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

#ifdef HAVE_AWS_S3
  heavydb_aws_sdk::init_sdk();
#endif

  Geospatial::GDAL::init();

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

#ifdef HAVE_AWS_S3
  heavydb_aws_sdk::shutdown_sdk();
#endif

  return err;
}
