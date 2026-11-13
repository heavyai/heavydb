/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file FilePathWhitelistTest.cpp
 * @brief Test suite for file path whitelist validation for COPY FROM, COPY TO, and CREATE
 * FOREIGN TABLE use cases
 *
 * Thin-client test: the file path whitelist is fixed when heavydb starts, so the tests are
 * grouped by the whitelist they need and each group runs against its own server (see the
 * ctest entries in CMakeLists.txt):
 *   FilePathWhitelistTest          import path Tests/FilePathWhitelist, export /tmp/...
 *   FilePathWhitelistTest_blacklist  root path whitelisted ("/")
 *   FilePathWhitelistTest_default_paths  only the default import/export directories
 * The server blacklists /etc/passwd itself, which is what the blacklist tests use. The
 * parts that call ddl_utils directly (config parsing, validate_allowed_file_path) are in
 * FilePathValidationTest.cpp.
 */

#include <fstream>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <picosha2.h>

#include "Geospatial/ColumnNames.h"
#include "Shared/SysDefinitions.h"
#include "Tests/SqlTestFixture.h"
#include "Tests/TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class FilePathWhitelistTest : public SqlTestFixture,
                              public testing::WithParamInterface<std::string> {
 protected:
  static void SetUpTestSuite() {
    boost::filesystem::create_directories(TEMP_EXPORT_DIR);
    temp_file_path_ =
        TEMP_EXPORT_DIR + "/" + boost::filesystem::unique_path().string() + ".csv";
    std::ofstream file{temp_file_path_};
    file.close();
  }

  static void TearDownTestSuite() {
    boost::filesystem::remove(temp_file_path_);
    boost::filesystem::remove_all(TEMP_EXPORT_DIR);
    boost::filesystem::remove("symlink_test.csv");
    boost::filesystem::remove_all(DEFAULT_IMPORT_PATH);
    boost::filesystem::remove_all(DEFAULT_EXPORT_PATH);
  }

  void SetUp() override {
    SqlTestFixture::SetUp();
    sql("CREATE TABLE IF NOT EXISTS test_table (col1 text);");
  }

  void TearDown() override {
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");
    SqlTestFixture::TearDown();
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
          "SERVER default_local_delimited "
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

  std::string getTestFilePath() {
    std::string file_path;
    if (GetParam() == "CopyTo") {
      file_path = temp_file_path_;
    } else {
      file_path = getTestCsvFilePath();
    }
    return file_path;
  }

  boost::filesystem::path initDefaultImportOrExportDirectory() {
    boost::filesystem::path directory_path;
    if (GetParam() == "CopyTo") {
      directory_path = DEFAULT_EXPORT_PATH;
    } else {
      directory_path = DEFAULT_IMPORT_PATH;
    }
    if (!boost::filesystem::exists(directory_path)) {
      boost::filesystem::create_directory(directory_path);
    }
    return directory_path;
  }

  std::string setAndGetFileAtDefaultImportOrExportPath() {
    auto source_path = getTestFilePath();
    boost::filesystem::path destination_path = initDefaultImportOrExportDirectory();

    destination_path /= boost::filesystem::path(source_path).filename().string();
#if 107400 <= BOOST_VERSION
    constexpr auto options = boost::filesystem::copy_options::overwrite_existing;
#else
    constexpr auto options = boost::filesystem::copy_option::overwrite_if_exists;
#endif
    boost::filesystem::copy_file(source_path, destination_path, options);
    return boost::filesystem::canonical(destination_path).string();
  }

  std::string clearAndGetSymlinkFilePath(boost::filesystem::path destination_path) {
    destination_path = boost::filesystem::canonical(destination_path);
    destination_path /= "symlink_test.csv";
    if (boost::filesystem::exists(destination_path)) {
      boost::filesystem::remove(destination_path);
    }
    return destination_path.string();
  }

  std::string getWhitelistedSymlinkFilePath() {
    return clearAndGetSymlinkFilePath(initDefaultImportOrExportDirectory());
  }

  std::string getUnlistedSymlinkFilePath() {
    return clearAndGetSymlinkFilePath(boost::filesystem::path("."));
  }

  // A path the server blacklists on its own (see CommandLineOptions.cpp), so no
  // in-process blacklist has to be set up. The blacklist is checked before the whitelist.
  inline static const std::string BLACKLISTED_PATH{"/etc/passwd"};
  inline static const std::string TEMP_EXPORT_DIR{"/tmp/heavyai_whitelist_export_test"};
  inline static std::string temp_file_path_;
  inline static const std::string DEFAULT_IMPORT_PATH{std::string{BASE_PATH} + "/" +
                                                      shared::kDefaultImportDirName};
  inline static const std::string DEFAULT_EXPORT_PATH{std::string{BASE_PATH} + "/" +
                                                      shared::kDefaultExportDirName};
};

TEST_P(FilePathWhitelistTest, WhitelistedPath) {
  std::string file_path = getTestFilePath();
  EXPECT_NO_THROW(sql(getQuery(file_path)));
}

TEST_P(FilePathWhitelistTest, NonWhitelistedPath) {
  std::string file_path =
      boost::filesystem::canonical(
          "../../Tests/Import/datafiles/with_quoted_fields_doublequotes.csv")
          .string();
  queryAndAssertException(
      getQuery(file_path),
      "File or directory path \"" + file_path + "\" is not whitelisted.");
}

TEST_P(FilePathWhitelistTest, WhitelistedSymlinkToUnlistedPath) {
  const auto file_path =
      boost::filesystem::canonical(
          "../../Tests/Import/datafiles/with_quoted_fields_doublequotes.csv")
          .string();
  const auto symlink_path = getWhitelistedSymlinkFilePath();
  boost::filesystem::create_symlink(file_path, symlink_path);

  queryAndAssertException(getQuery(symlink_path),
                          "File or directory path \"" + symlink_path +
                              "\" (resolved to \"" + file_path +
                              "\") is not whitelisted.");
}

TEST_P(FilePathWhitelistTest, UnlistedSymlinkToWhitelistedPath) {
  const auto file_path = getTestFilePath();
  const auto symlink_path = getUnlistedSymlinkFilePath();
  boost::filesystem::create_symlink(file_path, symlink_path);

  EXPECT_NO_THROW(sql(getQuery(symlink_path)));
}

TEST_P(FilePathWhitelistTest, UnlistedSymlinkToUnlistedPath) {
  const auto file_path =
      boost::filesystem::canonical(
          "../../Tests/Import/datafiles/with_quoted_fields_doublequotes.csv")
          .string();
  const auto symlink_path = getUnlistedSymlinkFilePath();
  boost::filesystem::create_symlink(file_path, symlink_path);

  queryAndAssertException(getQuery(symlink_path),
                          "File or directory path \"" + symlink_path +
                              "\" (resolved to \"" + file_path +
                              "\") is not whitelisted.");
}

TEST_P(FilePathWhitelistTest, NonExistentFilePath) {
  queryAndAssertException(
      getQuery("/tmp/non_existent/example.csv"),
      "File or directory \"/tmp/non_existent/example.csv\" does not exist.");
}

TEST_P(FilePathWhitelistTest, BlacklistedPath) {
  const auto& file_path = BLACKLISTED_PATH;
  queryAndAssertException(
      getQuery(file_path),
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_P(FilePathWhitelistTest, UnlistedSymlinkToBlacklistedPath) {
  const auto file_path = BLACKLISTED_PATH;
  const auto symlink_path = getUnlistedSymlinkFilePath();
  boost::filesystem::create_symlink(file_path, symlink_path);
  queryAndAssertException(getQuery(symlink_path),
                          "Access to file or directory path \"" + symlink_path +
                              "\" (resolved to \"" + file_path + "\") is not allowed.");
}

TEST_P(FilePathWhitelistTest, RootPathWhitelisted) {
  std::string file_path = getTestFilePath();
  EXPECT_NO_THROW(sql(getQuery(file_path)));
  sql("DROP FOREIGN TABLE IF EXISTS test_foreign_table;");

  // Validation should fail if path is in the blacklist,
  // even when the root path is set for the whitelist.
  queryAndAssertException(
      getQuery(BLACKLISTED_PATH),
      "Access to file or directory path \"" + BLACKLISTED_PATH + "\" is not allowed.");
}

TEST_P(FilePathWhitelistTest, DefaultImportAndExportPaths) {
  auto file_path = setAndGetFileAtDefaultImportOrExportPath();
  EXPECT_NO_THROW(sql(getQuery(file_path)));
}

TEST_F(FilePathWhitelistTest, ExportRelativePath) {
  EXPECT_NO_THROW(sql("COPY (SELECT * FROM test_table) TO 'test_export_file.csv';"));
}

INSTANTIATE_TEST_SUITE_P(FilePathWhitelistTest,
                         FilePathWhitelistTest,
                         testing::Values("CopyFrom", "CopyTo", "ForeignTable"),
                         [](const auto& param_info) { return param_info.param; });

TEST_F(FilePathWhitelistTest, DetectTypesBlacklist) {
  const auto& file_path = BLACKLISTED_PATH;
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
  const auto& file_path = BLACKLISTED_PATH;
  auto db_handler_and_session_id = getDbHandlerAndSessionId();
  auto db_handler = db_handler_and_session_id.first;
  auto session_id = db_handler_and_session_id.second;
  executeLambdaAndAssertException(
      [&] {
        db_handler->import_table(session_id, "test_table", file_path, TCopyParams{});
      },
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, ImportGeoTableBlacklist) {
  const auto& file_path = BLACKLISTED_PATH;
  auto db_handler_and_session_id = getDbHandlerAndSessionId();
  auto db_handler = db_handler_and_session_id.first;
  auto session_id = db_handler_and_session_id.second;
  executeLambdaAndAssertException(
      [&] {
        db_handler->import_geo_table(
            session_id, "test1", file_path, TCopyParams{}, TRowDescriptor{});
      },
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, GetFirstGeoFileBlacklist) {
  const auto& file_path = BLACKLISTED_PATH;
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
  const auto& file_path = BLACKLISTED_PATH;
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
  const auto& file_path = BLACKLISTED_PATH;
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
  const auto& file_path = BLACKLISTED_PATH;
  queryAndAssertException(
      "DUMP TABLE test_table TO '" + file_path + "';",
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

TEST_F(FilePathWhitelistTest, RestoreTableBlacklist) {
  const auto& file_path = BLACKLISTED_PATH;
  queryAndAssertException(
      "RESTORE TABLE test1 FROM '" + file_path + "';",
      "Access to file or directory path \"" + file_path + "\" is not allowed.");
}

namespace {
enum class FileLocationType {
  RELATIVE_FT,
  ABSOLUTE_FT,
#ifdef HAVE_AWS_S3
  S3,
#endif  // HAVE_AWS_S3
  HTTP,
  HTTPS,
  First = RELATIVE_FT,
  Last = HTTPS
};

constexpr char kRemoteFixtureHostAndPath[] =
    "raw.githubusercontent.com/heavyai/heavydb/master/Tests/FilePathWhitelist/";

using FileLocationTypeValue = std::underlying_type_t<FileLocationType>;

constexpr FileLocationTypeValue fileLocationTypeValue(
    const FileLocationType file_location_type) {
  return static_cast<FileLocationTypeValue>(file_location_type);
}

std::vector<FileLocationTypeValue> getEnabledFileLocationTypes() {
  std::vector<FileLocationTypeValue> enabled_file_location_types{
      fileLocationTypeValue(FileLocationType::RELATIVE_FT),
      fileLocationTypeValue(FileLocationType::ABSOLUTE_FT),
      fileLocationTypeValue(FileLocationType::HTTP),
      fileLocationTypeValue(FileLocationType::HTTPS)};

  return enabled_file_location_types;
}
}  // namespace

class DBHandlerFilePathTest
    : public SqlTestFixture,
      public testing::WithParamInterface<std::tuple<FileLocationTypeValue, std::string>> {
 public:
  static std::string testParamsToString(
      const std::tuple<FileLocationTypeValue, std::string>& params) {
    auto [file_location_type, suffix] = getTestParams(params);
    std::string param_str;
    if (file_location_type == FileLocationType::RELATIVE_FT) {
      param_str = "RelativePath";
    } else if (file_location_type == FileLocationType::ABSOLUTE_FT) {
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
    sql("CREATE TABLE IF NOT EXISTS test_table (col1 TEXT);");
    sql("CREATE TABLE IF NOT EXISTS test_table_2 (" + Geospatial::kGeoColumnName +
        " POINT);");
    boost::filesystem::create_directory(getImportPath());
  }

  static void TearDownTestSuite() {
    sql("DROP TABLE IF EXISTS test_table;");
    sql("DROP TABLE IF EXISTS test_table_2;");
    boost::filesystem::remove_all(getImportPath());
  }

  static boost::filesystem::path getImportPath() {
    return boost::filesystem::canonical(BASE_PATH) / shared::kDefaultImportDirName;
  }

  std::string getFilePath(const std::string& file_name_prefix) {
    auto [file_location_type, suffix] = getTestParams();
    std::replace(suffix.begin(), suffix.end(), '_', '.');

    std::string file_name = file_name_prefix + suffix;
    std::string path;
    if (file_location_type == FileLocationType::HTTPS) {
      path = std::string("https://") + kRemoteFixtureHostAndPath + file_name;
    } else if (file_location_type == FileLocationType::HTTP) {
      path = std::string("http://") + kRemoteFixtureHostAndPath + file_name;
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
      if (file_location_type == FileLocationType::ABSOLUTE_FT) {
        path = full_file_path.string();
      } else if (file_location_type == FileLocationType::RELATIVE_FT) {
        path = file_name;
      } else {
        UNREACHABLE();
      }
    }
    return path;
  }

  TCopyParams getCopyParams() {
    TCopyParams copy_params;
    auto [file_location_type, _] = getTestParams();
#ifdef HAVE_AWS_S3
    if (file_location_type == FileLocationType::S3) {
      copy_params.__set_s3_region("US-WEST-1");
    }
#endif
    return copy_params;
  }

  static std::pair<FileLocationType, std::string> getTestParams(
      const std::tuple<FileLocationTypeValue, std::string>& params = GetParam()) {
    return {static_cast<FileLocationType>(std::get<0>(params)), std::get<1>(params)};
  }
};

TEST_P(DBHandlerFilePathTest, DetectColumnTypes) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TDetectResult result;
  db_handler->detect_column_types(
      result, session_id, getFilePath("example.csv"), getCopyParams());
}

TEST_P(DBHandlerFilePathTest, DetectColumnTypes_GeoFile) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  TDetectResult result;
  db_handler->detect_column_types(
      result, session_id, getFilePath("example.geojson"), getCopyParams());
}

TEST_P(DBHandlerFilePathTest, ImportTable) {
  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  db_handler->import_table(
      session_id, "test_table", getFilePath("example.csv"), getCopyParams());
}

TEST_P(DBHandlerFilePathTest, ImportGeoTable) {
// TODO: Undo test case skipping when GDAL failure is resolved
#ifdef HAVE_AWS_S3
  if (auto [file_location_type, suffix] = getTestParams();
      file_location_type == FileLocationType::S3 && suffix == "_tar_gz") {
    GTEST_SKIP();
  }
#endif  // HAVE_AWS_S3

  TCopyParams copy_params;
  copy_params.source_type = TSourceType::GEO_FILE;

  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  db_handler->import_geo_table(session_id,
                               "test_table_2",
                               getFilePath("example.geojson"),
                               copy_params,
                               TRowDescriptor{});
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

  TCopyParams copy_params;
  copy_params.source_type = TSourceType::GEO_FILE;

  auto [db_handler, session_id] = getDbHandlerAndSessionId();
  std::vector<TGeoFileLayerInfo> result;
  db_handler->get_layers_in_geo_file(
      result, session_id, getFilePath("example.geojson"), copy_params);
}

INSTANTIATE_TEST_SUITE_P(
    DBHandlerFilePathTest,
    DBHandlerFilePathTest,
    testing::Combine(testing::ValuesIn(getEnabledFileLocationTypes()),
                     testing::Values("", "_tar", "_gz", "_tar_gz")),
    [](const auto& param_info) {
      return DBHandlerFilePathTest::testParamsToString(param_info.param);
    });

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    SqlTestFixture::initConnection(argc, argv);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = 1;
  }
  return err;
}
