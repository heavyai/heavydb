/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file SelectClientTest.cpp
 * @brief Test suite for S3 Select client class for FSI S3 data wrapper
 *
 */

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "Tests/TestHelpers.h"

#include "Archive/S3Archive.h"
#include "Catalog/ForeignServer.h"
#include "DataMgr/ForeignStorage/AbstractFileStorageDataWrapper.h"
#include "DataMgr/ForeignStorage/S3SelectClient.h"
#include "DataMgr/HeavyDbAwsSdk.h"
#include "ImportExport/CopyParams.h"
#include "Tests/AwsHelpers.h"

extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;

namespace {
foreign_storage::ForeignServer create_foreign_server() {
  foreign_storage::OptionsMap options;
  options[foreign_storage::AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY] =
      foreign_storage::AbstractFileStorageDataWrapper::S3_STORAGE_TYPE;
  options[foreign_storage::AbstractFileStorageDataWrapper::S3_BUCKET_KEY] =
      "fsi-s3-select-experiments";
  options[foreign_storage::AbstractFileStorageDataWrapper::AWS_REGION_KEY] = "us-west-1";
  foreign_storage::ForeignServer foreign_server{"default", "CSV", options, 0};

  return foreign_server;
}

// Create vector of pairs partitioning file into num_partitions
std::vector<foreign_storage::S3ScanRange>
get_partition_ranges(size_t size, int num_partitions, const std::string& filename) {
  std::vector<foreign_storage::S3ScanRange> ranges;
  int partition_size = size / num_partitions;
  size_t start = 0;
  while (start < size) {
    size_t end = start + partition_size - 1;
    if (end > size - 1) {
      end = size - 1;
    }
    ranges.emplace_back(filename, start, end);
    start = end + 1;
  }
  return ranges;
}

size_t get_file_size(foreign_storage::S3SelectClient& test_client) {
  auto file_infos = test_client.getFileInfos({});
  CHECK(file_infos.size() == 1);
  return file_infos.begin()->second;
}

std::string get_columns(foreign_storage::S3SelectClient& test_client,
                        const std::vector<int>& columns_to_fetch,
                        int num_partitions,
                        const std::string& filename) {
  std::string result;
  for (const auto& range :
       get_partition_ranges(get_file_size(test_client), num_partitions, filename)) {
    result += test_client.getColumnsAsCsv(columns_to_fetch, range);
  }
  return result;
}

std::shared_ptr<Aws::Auth::AWSCredentialsProvider> get_credentials() {
  auto key = get_aws_keys_from_env();
  return std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(key.first, key.second);
}

std::string get_reference_result(const std::vector<int>& columns_to_fetch) {
  std::string reference = "";
  // clang-format off
  std::vector<std::vector<std::string>> string_rows = {{"true","100","30000","2000000000","9000000000000000000","10.1","100.1234","00:00:10","1/1/2000 00:00:59","1/1/2000","text_1","\"quoted text\""},
    {"false","110","30500","2000500000","9000000050000000000","100.12","2.1234","00:10:00","6/15/2020 00:59:59","6/15/2020","text_2","\"quoted text 2\""},
    {"true","120","31000","2100000000","9100000000000000000","1000.123","100.1","10:00:00","12/31/2500 23:59:59","12/31/2500","text_3","\"quoted text 3\""}};
  // clang-format on
  for (const auto& row : string_rows) {
    for (auto iter = columns_to_fetch.begin(); iter != columns_to_fetch.end(); ++iter) {
      if (iter != columns_to_fetch.begin()) {
        reference += ",";
      }
      reference += row[*iter];
    }
    reference += "\n";
  }
  return reference;
}

std::vector<std::pair<int, std::string>> get_stat_column_id_types() {
  // Stats can only be generated for numeric types
  return {{1, "int"},
          {2, "int"},
          {3, "int"},
          {4, "int"},
          {5, "float"},
          {6, "DECIMAL(10, 5)"}};
}

std::string get_column_stats(const std::vector<int> column_indexes) {
  const std::vector<std::vector<std::string>> column_stats = {
      {},
      {"100", "120", "1"},  // Column has nulls
      {"30000", "31000", "0"},
      {"2000000000", "2100000000", "0"},
      {"9000000000000000000", "9100000000000000000", "1"},
      {"10.1", "1000.123", "0"},
      {"2.1234", "100.1234", "0"}};
  std::string stats;

  for (auto iter = column_indexes.begin(); iter != column_indexes.end(); ++iter) {
    if (iter != column_indexes.begin()) {
      stats += ",";
    }
    auto index = *iter;
    CHECK(index < 7);
    stats += column_stats[index][0] + "," + column_stats[index][1] + "," +
             column_stats[index][2];
  }
  stats += "\n";
  return stats;
}
}  // namespace

class S3SelectTest : public testing::Test {};

class S3PartitionedSelectTest : public S3SelectTest,
                                public testing::WithParamInterface<int> {};

INSTANTIATE_TEST_SUITE_P(S3PartitionedSelect,
                         S3PartitionedSelectTest,
                         // Single request vs many
                         ::testing::Values(1, 8));

TEST_P(S3PartitionedSelectTest, GetColumn) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient(
      "scalar_types.csv", &server, get_credentials(), import_export::CopyParams());

  std::vector<int> columns_to_fetch = {1};
  ASSERT_EQ(get_columns(test_client, columns_to_fetch, GetParam(), "scalar_types.csv"),
            get_reference_result(columns_to_fetch));
}

TEST_P(S3PartitionedSelectTest, GetColumns) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient(
      "scalar_types.csv", &server, get_credentials(), import_export::CopyParams());

  std::vector<int> columns_to_fetch = {1, 2, 4};
  ASSERT_EQ(get_columns(test_client, columns_to_fetch, GetParam(), "scalar_types.csv"),
            get_reference_result(columns_to_fetch));
}

TEST_P(S3PartitionedSelectTest, GetColumns_NoHeader) {
  auto server = create_foreign_server();
  auto copy_params = import_export::CopyParams();
  copy_params.has_header = import_export::ImportHeaderRow::kNoHeader;
  auto test_client = foreign_storage::S3SelectClient(
      "scalar_types_noheader.csv", &server, get_credentials(), copy_params);

  std::vector<int> columns_to_fetch = {1, 2, 4};
  ASSERT_EQ(
      get_columns(test_client, columns_to_fetch, GetParam(), "scalar_types_noheader.csv"),
      get_reference_result(columns_to_fetch));
}

// File with 1-3 character row to exercise line boundary conditions
TEST_P(S3PartitionedSelectTest, GetColumn_LineBoundaries) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient(
      "0_255_1col.csv", &server, get_credentials(), import_export::CopyParams());

  std::vector<int> columns_to_fetch = {0};

  std::string reference;
  for (int i = 0; i < 256; i++) {
    reference += std::to_string(i);
    reference += "\n";
  }
  ASSERT_EQ(get_columns(test_client, columns_to_fetch, GetParam(), "0_255_1col.csv"),
            reference);
}

TEST_P(S3PartitionedSelectTest, CountRows) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient(
      "scalar_types.csv", &server, get_credentials(), import_export::CopyParams());

  auto ranges =
      get_partition_ranges(get_file_size(test_client), GetParam(), "scalar_types.csv");

  int total_rows = 0;
  for (int rows : test_client.getNumRows(ranges)) {
    total_rows += rows;
  }

  ASSERT_EQ(total_rows, 3);
}

TEST_P(S3PartitionedSelectTest, CountRowsDirectory) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient(
      "example_1_dir", &server, get_credentials(), import_export::CopyParams());
  ASSERT_EQ(test_client.getNumRows(
                {foreign_storage::S3ScanRange("example_1_dir/example_1a.csv"),
                 foreign_storage::S3ScanRange("example_1_dir/example_1b.csv"),
                 foreign_storage::S3ScanRange("example_1_dir/example_1c.csv")}),
            std::vector<int>({1, 1, 1}));
}

TEST_F(S3SelectTest, GetFileInfos) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient(
      "scalar_types.csv", &server, get_credentials(), import_export::CopyParams());
  auto file_infos = test_client.getFileInfos({});
  ASSERT_EQ(file_infos.size(), 1u);
  ASSERT_EQ(file_infos.begin()->first, "scalar_types.csv");
  ASSERT_EQ(file_infos.begin()->second, 453u);
}

TEST_F(S3SelectTest, GetFileInfosDirectory) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient(
      "example_1_dir", &server, get_credentials(), import_export::CopyParams());
  auto file_infos = test_client.getFileInfos({});
  ASSERT_EQ(file_infos.size(), 3u);
  ASSERT_NE(
      std::find(file_infos.begin(),
                file_infos.end(),
                std::pair<std::string, size_t>("example_1_dir/example_1a.csv", 23u)),
      file_infos.end());
  ASSERT_NE(
      std::find(file_infos.begin(),
                file_infos.end(),
                std::pair<std::string, size_t>("example_1_dir/example_1b.csv", 27u)),
      file_infos.end());
  ASSERT_NE(
      std::find(file_infos.begin(),
                file_infos.end(),
                std::pair<std::string, size_t>("example_1_dir/example_1c.csv", 28u)),
      file_infos.end());
}

TEST_F(S3SelectTest, GetStats) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient("scalar_types_w_nulls.csv",
                                                     &server,
                                                     get_credentials(),
                                                     import_export::CopyParams());

  const auto column_id_types = get_stat_column_id_types();

  for (const auto& type : column_id_types) {
    ASSERT_EQ(test_client.getStatsAsCsv(
                  {type}, {foreign_storage::S3ScanRange("scalar_types_w_nulls.csv")})[0],
              get_column_stats({type.first}));
  }
}

TEST_F(S3SelectTest, GetStatsMulti) {
  auto server = create_foreign_server();
  auto test_client = foreign_storage::S3SelectClient("scalar_types_w_nulls.csv",
                                                     &server,
                                                     get_credentials(),
                                                     import_export::CopyParams());

  ASSERT_EQ(test_client.getStatsAsCsv(
                get_stat_column_id_types(),
                {foreign_storage::S3ScanRange("scalar_types_w_nulls.csv")})[0],
            get_column_stats({1, 2, 3, 4, 5, 6}));
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  g_enable_fsi = true;
  g_enable_s3_fsi = true;

  heavydb_aws_sdk::init_sdk();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  heavydb_aws_sdk::shutdown_sdk();

  return err;
}
