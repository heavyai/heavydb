/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file FilePathValidationTest.cpp
 * @brief Unit tests for the file path whitelist configuration and path validation in
 * ddl_utils. These call ddl_utils directly (no server), so they only link Utils. The
 * tests that run COPY FROM, COPY TO and CREATE FOREIGN TABLE against a server are in
 * FilePathWhitelistTest.cpp.
 */

#include <string>

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "Shared/SysDefinitions.h"
#include "Tests/TestHelpers.h"
#include "Utils/DdlUtils.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

// Parameterized by the kind of statement whose whitelist the config error is about
// (the same parameters as the server tests in FilePathWhitelistTest).
class FilePathWhitelistTest : public ::testing::Test,
                              public testing::WithParamInterface<std::string> {
 protected:
  static void TearDownTestSuite() { boost::filesystem::remove(CONFIG_FILE_PATH); }

  void SetUp() override { ddl_utils::FilePathWhitelist::clear(); }

  void TearDown() override { ddl_utils::FilePathWhitelist::clear(); }

  void setServerConfig(const std::string& file_name) {
    boost::filesystem::copy_file("../../Tests/FilePathWhitelist/" + file_name,
                                 CONFIG_FILE_PATH,
#if 107400 <= BOOST_VERSION
                                 boost::filesystem::copy_options::overwrite_existing
#else
                                 boost::filesystem::copy_option::overwrite_if_exists
#endif
    );
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

  void initializeWhitelist() {
    std::string allowed_import_paths, allowed_export_paths;
    namespace po = boost::program_options;
    po::options_description desc{};
    desc.add_options()("allowed-import-paths",
                       po::value<std::string>(&allowed_import_paths));
    desc.add_options()("allowed-export-paths",
                       po::value<std::string>(&allowed_export_paths));
    po::variables_map vm;
    po::store(po::parse_config_file<char>(CONFIG_FILE_PATH.c_str(), desc, true), vm);
    po::notify(vm);
    ddl_utils::FilePathWhitelist::initialize(
        BASE_PATH, allowed_import_paths, allowed_export_paths);
  }

  void initializeWhitelistAndAssertException(const std::string& error_message) {
    try {
      initializeWhitelist();
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const std::runtime_error& e) {
      ASSERT_EQ(error_message, e.what());
    }
  }

  void whitelistRootPath() {
    setServerConfig("root_path_config.conf");
    initializeWhitelist();
  }

  template <typename Lambda>
  static void executeLambdaAndAssertException(Lambda lambda,
                                              const std::string& error_message) {
    try {
      lambda();
      FAIL() << "Expected exception with message: " << error_message;
    } catch (const std::runtime_error& e) {
      ASSERT_EQ(error_message, e.what());
    }
  }

  inline static const std::string CONFIG_FILE_PATH{"./file_path_whitelist_test.conf"};
};

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

TEST_P(FilePathWhitelistTest, NonExistentWhitelistedRootPath) {
  setServerConfig("non_existent_root_path.conf");
  initializeWhitelistAndAssertException(
      "Whitelisted root path \"./nonexistent_path\" does not exist.");
}

INSTANTIATE_TEST_SUITE_P(FilePathWhitelistTest,
                         FilePathWhitelistTest,
                         testing::Values("CopyFrom", "CopyTo", "ForeignTable"),
                         [](const auto& param_info) { return param_info.param; });

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
  whitelistRootPath();
  auto test_path =
      boost::filesystem::canonical("../../Tests/FilePathWhitelist").string() + "/*";
  validate_allowed_file_path(test_path, ddl_utils::DataTransferType::IMPORT, true);
}

TEST_F(FilePathWhitelistTest, AllowRelativePath) {
  whitelistRootPath();
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
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = 1;
  }
  return err;
}
