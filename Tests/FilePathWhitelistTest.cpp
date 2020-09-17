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
#include <picosha2.h>

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
    temp_file_path_ = "/tmp/" + boost::filesystem::unique_path().string() + ".csv";
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

TEST_F(FilePathWhitelistTest, DetectTypesBlacklist) {
  const auto& file_path = temp_file_path_;
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  auto db_handler_and_session_id = getDbHandlerAndSessionId();
  auto db_handler = db_handler_and_session_id.first;
  auto session_id = db_handler_and_session_id.second;
  TDetectResult temp;
  executeLambdaAndAssertException(
      [&] {
        db_handler->detect_column_types(temp, session_id, file_path, TCopyParams{});
      },
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, ImportTableBlacklist) {
  const auto& file_path = temp_file_path_;
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  auto db_handler_and_session_id = getDbHandlerAndSessionId();
  auto db_handler = db_handler_and_session_id.first;
  auto session_id = db_handler_and_session_id.second;
  executeLambdaAndAssertException(
      [&] {
        db_handler->import_table(session_id, "test_table", file_path, TCopyParams{});
      },
      "Exception: Access to file or directory path \"" + file_path +
          "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, ImportGeoTableBlacklist) {
  const auto& file_path = temp_file_path_;
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  auto db_handler_and_session_id = getDbHandlerAndSessionId();
  auto db_handler = db_handler_and_session_id.first;
  auto session_id = db_handler_and_session_id.second;
  executeLambdaAndAssertException(
      [&] {
        db_handler->import_geo_table(session_id,
                                     "test1",
                                     file_path,
                                     TCopyParams{},
                                     TRowDescriptor{},
                                     TCreateParams{});
      },
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, GetFirstGeoFileBlacklist) {
  const auto& file_path = temp_file_path_;
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  auto db_handler_and_session_id = getDbHandlerAndSessionId();
  auto db_handler = db_handler_and_session_id.first;
  auto session_id = db_handler_and_session_id.second;
  std::string temp;
  executeLambdaAndAssertException(
      [&] {
        db_handler->get_first_geo_file_in_archive(
            temp, session_id, file_path, TCopyParams());
      },
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, GetAllFilesBlacklist) {
  const auto& file_path = temp_file_path_;
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  auto db_handler_and_session_id = getDbHandlerAndSessionId();
  auto db_handler = db_handler_and_session_id.first;
  auto session_id = db_handler_and_session_id.second;
  std::vector<std::string> temp;
  executeLambdaAndAssertException(
      [&] {
        db_handler->get_all_files_in_archive(temp, session_id, file_path, TCopyParams());
      },
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, GetLayersInGeoFileBlacklist) {
  const auto& file_path = temp_file_path_;
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  auto db_handler_and_session_id = getDbHandlerAndSessionId();
  auto db_handler = db_handler_and_session_id.first;
  auto session_id = db_handler_and_session_id.second;
  std::vector<TGeoFileLayerInfo> temp;
  executeLambdaAndAssertException(
      [&] {
        db_handler->get_layers_in_geo_file(temp, session_id, file_path, TCopyParams());
      },
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, DumpTableBlacklist) {
  const auto& file_path = temp_file_path_;
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  queryAndAssertException("DUMP TABLE test_table TO '" + file_path + "';",
                          "Exception: Access to file or directory path \"" + file_path +
                              "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, RestoreTableBlacklist) {
  const auto& file_path = temp_file_path_;
  ddl_utils::FilePathBlacklist::addToBlacklist(file_path);
  queryAndAssertException("RESTORE TABLE test1 FROM '" + file_path + "';",
                          "Exception: Access to file or directory path \"" + file_path +
                              "\" is not allowed.");
}

namespace {
enum class FileLocationType {
  RELATIVE,
  ABSOLUTE,
#ifdef HAVE_AWS_S3
  S3,
#endif  // HAVE_AWS_S3
  HTTP,
  HTTPS,
  First = RELATIVE,
  Last = HTTPS
};
}  // namespace

class DBHandlerFilePathTest
    : public DBHandlerTestFixture,
      public testing::WithParamInterface<std::tuple<int, std::string>> {
 public:
  static std::string testParamsToString(const std::tuple<int, std::string>& params) {
    auto [file_location_type, suffix] = getTestParams(params);
    std::string param_str;
    if (file_location_type == FileLocationType::RELATIVE) {
      param_str = "RelativePath";
    } else if (file_location_type == FileLocationType::ABSOLUTE) {
      param_str = "AbsolutePath";
#ifdef HAVE_AWS_S3
    } else if (file_location_type == FileLocationType::S3) {
      param_str = "S3";
#endif  // HAVE_AWS_S3
    } else if (file_location_type == FileLocationType::HTTPS) {
      param_str = "Https";
    } else if (file_location_type == FileLocationType::HTTP) {
      param_str = "Http";
    } else {
      UNREACHABLE();
    }
    return param_str + suffix;
  }

 protected:
  static void SetUpTestSuite() {
    createDBHandler();
    sql("CREATE TABLE IF NOT EXISTS test_table (col1 TEXT);");
    sql("CREATE TABLE IF NOT EXISTS test_table_2 (omnisci_geo POINT);");
    boost::filesystem::create_directory(getImportPath());
  }

  static void TearDownTestSuite() {
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP TABLE IF EXISTS test_table_2;");
    boost::filesystem::remove_all(getImportPath());
  }

  static boost::filesystem::path getImportPath() {
    return boost::filesystem::canonical(BASE_PATH) / "mapd_import";
  }

  std::string getFilePath(const std::string& file_name_prefix) {
    auto [file_location_type, suffix] = getTestParams();
    std::replace(suffix.begin(), suffix.end(), '_', '.');

    std::string file_name = file_name_prefix + suffix;
    std::string path;
    if (file_location_type == FileLocationType::HTTPS) {
      path = "https://omnisci-import-test.s3-us-west-1.amazonaws.com/" + file_name;
    } else if (file_location_type == FileLocationType::HTTP) {
      path = "http://omnisci-import-test.s3-us-west-1.amazonaws.com/" + file_name;
#ifdef HAVE_AWS_S3
    } else if (file_location_type == FileLocationType::S3) {
      path = "s3://omnisci-import-test/" + file_name;
#endif  // HAVE_AWS_S3
    } else {
      auto session_id = getDbHandlerAndSessionId().second;
      auto session_directory = getImportPath() / picosha2::hash256_hex_string(session_id);
      if (!boost::filesystem::exists(session_directory)) {
        boost::filesystem::create_directory(session_directory);
      }
      auto full_file_path =
          session_directory / boost::filesystem::path(file_name).filename();
      if (!boost::filesystem::exists(full_file_path)) {
        boost::filesystem::copy_file("../../Tests/FilePathWhitelist/" + file_name,
                                     full_file_path);
      }
      if (file_location_type == FileLocationType::ABSOLUTE) {
        path = full_file_path.string();
      } else if (file_location_type == FileLocationType::RELATIVE) {
        path = file_name;
      } else {
        UNREACHABLE();
      }
    }
    return path;
  }

  static std::pair<FileLocationType, std::string> getTestParams(
      const std::tuple<int, std::string>& params = GetParam()) {
    return {static_cast<FileLocationType>(std::get<0>(params)), std::get<1>(params)};
  }
};

TEST_P(DBHandlerFilePathTest, DetectColumnTypes) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TDetectResult result;
  db_handler->detect_column_types(
      result, session_id, getFilePath("example.csv"), TCopyParams{});
}

TEST_P(DBHandlerFilePathTest, DetectColumnTypes_GeoFile) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TDetectResult result;
  db_handler->detect_column_types(
      result, session_id, getFilePath("example.geojson"), TCopyParams{});
}

TEST_P(DBHandlerFilePathTest, ImportTable) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  db_handler->import_table(
      session_id, "test_table", getFilePath("example.csv"), TCopyParams{});
}

TEST_P(DBHandlerFilePathTest, ImportGeoTable) {
// TODO: Undo test case skipping when GDAL failure is resolved
#ifdef HAVE_AWS_S3
  if (auto [file_location_type, suffix] = getTestParams();
      file_location_type == FileLocationType::S3 && suffix == "_tar_gz") {
    GTEST_SKIP();
  }
#endif  // HAVE_AWS_S3

  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  db_handler->import_geo_table(session_id,
                               "test_table_2",
                               getFilePath("example.geojson"),
                               TCopyParams{},
                               TRowDescriptor{},
                               TCreateParams{});
}

TEST_P(DBHandlerFilePathTest, GetFirstGeoFileInArchive) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  std::string result;
  db_handler->get_first_geo_file_in_archive(
      result, session_id, getFilePath("example.geojson"), TCopyParams());
}

TEST_P(DBHandlerFilePathTest, GetAllFilesInArchive) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  std::vector<std::string> result;
  db_handler->get_all_files_in_archive(
      result, session_id, getFilePath("example.geojson"), TCopyParams());
}

TEST_P(DBHandlerFilePathTest, GetLayersInGeoFile) {
// TODO: Undo test case skipping when GDAL failure is resolved
#ifdef HAVE_AWS_S3
  if (auto [file_location_type, suffix] = getTestParams();
      file_location_type == FileLocationType::S3 && suffix == "_tar_gz") {
    GTEST_SKIP();
  }
#endif  // HAVE_AWS_S3

  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  std::vector<TGeoFileLayerInfo> result;
  db_handler->get_layers_in_geo_file(
      result, session_id, getFilePath("example.geojson"), TCopyParams());
}

INSTANTIATE_TEST_SUITE_P(
    DBHandlerFilePathTest,
    DBHandlerFilePathTest,
    testing::Combine(testing::Range(static_cast<int>(FileLocationType::First),
                                    static_cast<int>(FileLocationType::Last) + 1),
                     testing::Values("", "_tar", "_gz", "_tar_gz")),
    [](const auto& param_info) {
      return DBHandlerFilePathTest::testParamsToString(param_info.param);
    });

TEST_F(FilePathWhitelistTest, ThrowOnPunctuation) {
  executeLambdaAndAssertException(
      [&] {
        validate_allowed_file_path("name_with_&", ddl_utils::DataTransferType::IMPORT);
      },
      "Punctuation \"&\" is not allowed in file path: name_with_&");
  executeLambdaAndAssertException(
      [&] {
        validate_allowed_file_path("name_with_;", ddl_utils::DataTransferType::IMPORT);
      },
      "Punctuation \";\" is not allowed in file path: name_with_;");
  executeLambdaAndAssertException(
      [&] {
        validate_allowed_file_path("name_with_\\", ddl_utils::DataTransferType::IMPORT);
      },
      "Punctuation \"\\\" is not allowed in file path: name_with_\\");
  executeLambdaAndAssertException(
      [&] {
        validate_allowed_file_path("name_with_$", ddl_utils::DataTransferType::IMPORT);
      },
      "Punctuation \"$\" is not allowed in file path: name_with_$");
  executeLambdaAndAssertException(
      [&] {
        validate_allowed_file_path("name_with_!", ddl_utils::DataTransferType::IMPORT);
      },
      "Punctuation \"!\" is not allowed in file path: name_with_!");
}

TEST_F(FilePathWhitelistTest, ThrowOnAsterisk) {
  executeLambdaAndAssertException(
      [&] {
        validate_allowed_file_path("name_with_*", ddl_utils::DataTransferType::IMPORT);
      },
      "Punctuation \"*\" is not allowed in file path: name_with_*");
}

TEST_F(FilePathWhitelistTest, AllowAsteriskForWildcard) {
  validate_allowed_file_path("/tmp/*", ddl_utils::DataTransferType::IMPORT, true);
}

TEST_F(FilePathWhitelistTest, AllowRelativePath) {
  validate_allowed_file_path("./../../Tests/FilePathWhitelist/example.csv",
                             ddl_utils::DataTransferType::IMPORT);
}

TEST_F(FilePathWhitelistTest, AllowTilde) {
  // This test case ensures that an exception is not thrown because of the
  // presence of the "~" character. Although, an exception is thrown because
  // the file path does not exist (cannot rely on home directory on test machines).
  executeLambdaAndAssertException(
      [&] {
        validate_allowed_file_path("~/test_path", ddl_utils::DataTransferType::IMPORT);
      },
      "File or directory \"~/test_path\" does not exist.");
}

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
