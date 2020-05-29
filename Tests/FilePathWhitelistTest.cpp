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
 * @file FilePathWhitelistTest.cpp
 * @brief Test suite for file path whitelist validation for COPY FROM, COPY TO, and CREATE
 * FOREIGN TABLE use cases
 */
#include <fstream>

#include <gtest/gtest.h>

#include "Tests/DBHandlerTestHelpers.h"
#include "Tests/TestHelpers.h"
#include "Utils/DdlUtils.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_fsi;

class FilePathWhitelistTest : public DBHandlerTestFixture,
                              public testing::WithParamInterface<std::string> {
 protected:
  static void SetUpTestSuite() {
    temp_file_path_ = "/tmp/" + boost::filesystem::unique_path().string();
    std::ofstream file{temp_file_path_};
    file.close();
  }

  static void TearDownTestSuite() {
    boost::filesystem::remove(temp_file_path_);
    boost::filesystem::remove(CONFIG_FILE_PATH);
  }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("CREATE TABLE IF NOT EXISTS test_table (col1 text);");
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    DBHandlerTestFixture::TearDown();
    ddl_utils::FilePathWhitelist::clear();
    ddl_utils::FilePathBlacklist::clear();
  }

  void setServerConfig(const std::string& file_name) {
    boost::filesystem::copy_file("../../Tests/FilePathWhitelist/" + file_name,
                                 CONFIG_FILE_PATH,
                                 boost::filesystem::copy_option::overwrite_if_exists);
  }

  std::string getQuery(const std::string& file_path) {
    std::string query;
    if (GetParam() == "CopyFrom") {
      query = "COPY test_table FROM '" + file_path + "';";
    } else if (GetParam() == "CopyTo") {
      query = "COPY (SELECT * FROM test_table) TO '" + file_path + "';";
    } else if (GetParam() == "ForeignTable") {
      query =
          "CREATE FOREIGN TABLE test_foreign_table (col1 text) "
          "SERVER omnisci_local_csv "
          "WITH (file_path = '" +
          file_path + "');";
    } else {
      UNREACHABLE();
    }
    return query;
  }

  std::string getTestCsvFilePath() {
    return boost::filesystem::canonical("../../Tests/FilePathWhitelist/example.csv")
        .string();
  }

  std::string getWhitelistedFilePath() {
    std::string file_path;
    if (GetParam() == "CopyTo") {
      file_path = temp_file_path_;
    } else {
      file_path = getTestCsvFilePath();
    }
    return file_path;
  }

  std::string getMalformedConfigErrorMessage() {
    std::string config_key;
    if (GetParam() == "CopyTo") {
      config_key = "allowed-export-paths";
    } else {
      config_key = "allowed-import-paths";
    }
    return "Configuration value for \"" + config_key +
           "\" is malformed. Value should be a list of paths with format: [ "
           "\"root-path-1\", \"root-path-2\", ... ]";
  }

  void initializeWhitelistAndAssertException(const std::string& error_message) {
    try {
      ddl_utils::FilePathWhitelist::initializeFromConfigFile(CONFIG_FILE_PATH);
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const std::runtime_error& e) {
      ASSERT_EQ(error_message, e.what());
    }
  }

  inline static const std::string CONFIG_FILE_PATH{"./file_path_whitelist_test.conf"};
  inline static std::string temp_file_path_;
};

TEST_P(FilePathWhitelistTest, WhitelistedPath) {
  setServerConfig("test_config.conf");
  ddl_utils::FilePathWhitelist::initializeFromConfigFile(CONFIG_FILE_PATH);
  std::string file_path = getWhitelistedFilePath();
  EXPECT_NO_THROW(sql(getQuery(file_path)));
}

TEST_P(FilePathWhitelistTest, NonWhitelistedPath) {
  setServerConfig("test_config.conf");
  ddl_utils::FilePathWhitelist::initializeFromConfigFile(CONFIG_FILE_PATH);
  std::string file_path =
      boost::filesystem::canonical(
          "../../Tests/Import/datafiles/with_quoted_fields_doublequotes.csv")
          .string();
  queryAndAssertException(
      getQuery(file_path),
      "Exception: File or directory path \"" + file_path + "\" is not whitelisted.");
}

TEST_P(FilePathWhitelistTest, InvalidConfigValue) {
  std::string config_path;
  if (GetParam() == "CopyTo") {
    config_path = "malformed_export_config_1.conf";
  } else {
    config_path = "malformed_import_config_1.conf";
  }
  setServerConfig(config_path);
  initializeWhitelistAndAssertException(getMalformedConfigErrorMessage());
}

TEST_P(FilePathWhitelistTest, InvalidConfigListValue) {
  std::string config_path;
  if (GetParam() == "CopyTo") {
    config_path = "malformed_export_config_2.conf";
  } else {
    config_path = "malformed_import_config_2.conf";
  }
  setServerConfig(config_path);
  initializeWhitelistAndAssertException(getMalformedConfigErrorMessage());
}

TEST_P(FilePathWhitelistTest, NonExistentFilePath) {
  setServerConfig("test_config.conf");
  queryAndAssertException(
      getQuery("/tmp/non_existent/example.csv"),
      "Exception: File or directory \"/tmp/non_existent/example.csv\" does not exist.");
}

TEST_P(FilePathWhitelistTest, NonExistentWhitelistedRootPath) {
  setServerConfig("non_existent_root_path.conf");
  initializeWhitelistAndAssertException(
      "Whitelisted root path \"./nonexistent_path\" does not exist.");
}

TEST_P(FilePathWhitelistTest, NonExistentConfigPath) {
  boost::filesystem::remove(CONFIG_FILE_PATH);
  initializeWhitelistAndAssertException("Configuration file at \"" + CONFIG_FILE_PATH +
                                        "\" does not exist.");
}

TEST_P(FilePathWhitelistTest, BlacklistedPath) {
  setServerConfig("test_config.conf");
  const auto& file_path = getWhitelistedFilePath();
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  queryAndAssertException(getQuery(file_path),
                          "Exception: Access to file or directory path \"" + file_path +
                              "\" is not allowed.");
}

INSTANTIATE_TEST_SUITE_P(FilePathWhitelistTest,
                         FilePathWhitelistTest,
                         testing::Values("CopyFrom", "CopyTo", "ForeignTable"),
                         [](const auto& param_info) { return param_info.param; });

int main(int argc, char** argv) {
  g_enable_fsi = true;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  DBHandlerTestFixture::initTestArgs(argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
