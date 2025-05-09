/*
 * Copyright 2025 HEAVY.AI, Inc.
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
 * @file ForeignDataWrapperUnitTest.cpp
 * @brief Test suite for unit tests on foreign data wrappers
 */

#include "DBHandlerTestHelpers.h"
#include "DataMgr/ForeignStorage/CsvDataWrapper.h"
#include "DataMgr/ForeignStorage/ParquetDataWrapper.h"
#include "Tests/ForeignTableTestHelpers.h"
#include "Tests/TestHelpers.h"

using namespace foreign_storage;

class DoubleToPointCompressionUnitTest
    : public ForeignDataWrapperUnitTest,
      public ::testing::WithParamInterface<std::string> {
 public:
  std::string getServerName() const override {
    if (GetParam() == "csv") {
      return shared::kDefaultDelimitedServerName;
    } else if (GetParam() == "parquet") {
      return shared::kDefaultParquetServerName;
    } else {
      UNREACHABLE();
      return "";
    }
  }

  std::unique_ptr<ForeignDataWrapper> createWrapperPtr(int32_t db_id,
                                                       ForeignTable* ft,
                                                       UserMapping* um) const override {
    if (GetParam() == "csv") {
      return std::make_unique<CsvDataWrapper>(
          db_id_, foreign_table_.get(), user_mapping_.get());
    } else if (GetParam() == "parquet") {
      return std::make_unique<ParquetDataWrapper>(
          db_id_, foreign_table_.get(), user_mapping_.get());
    } else {
      UNREACHABLE();
      return nullptr;
    }
  }
};

TEST_P(DoubleToPointCompressionUnitTest, SingleColumn) {
  createWrapper("../../Tests/FsiDataFiles/GeoTypes/lon_lat_separate." + GetParam(),
                createPointIndexSchema());
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(3U, meta_vec.size());

  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);

  auto decomp_coords = buffer_wrappers.getDecompressedCoordsAt(
      point_t, {db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(decomp_coords.size(), 2U);
  // The compression process has some lossy-ness that is hard to capture with exact value
  // comparison.  Compare with an epsilon to account for this.
  EXPECT_NEAR(decomp_coords[0], 0.0, 0.0001);
  EXPECT_NEAR(decomp_coords[1], 1.0, 0.0001);
}

TEST_P(DoubleToPointCompressionUnitTest, ReverseLonLat) {
  createWrapper("../../Tests/FsiDataFiles/GeoTypes/lon_lat_reversed." + GetParam(),
                createPointIndexSchema(),
                {{AbstractFileStorageDataWrapper::LONLAT_KEY, "false"}});
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(3U, meta_vec.size());

  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);

  auto decomp_coords = buffer_wrappers.getDecompressedCoordsAt(
      point_t, {db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(decomp_coords.size(), 2U);
  EXPECT_NEAR(decomp_coords[0], 0.0, 0.0001);
  EXPECT_NEAR(decomp_coords[1], 1.0, 0.0001);
}

TEST_P(DoubleToPointCompressionUnitTest, HasNulls) {
  if (GetParam() == "csv") {
    // TODO(Misiu): This test should work with CSV, but it currently breaks with a parsing
    // error if the lon value is null.
    GTEST_SKIP() << "Skipped due to CSV parsing bug";
  }

  createWrapper("../../Tests/FsiDataFiles/GeoTypes/lon_lat_nulls." + GetParam(),
                createPointIndexSchema());
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(3U, meta_vec.size());

  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(
      buffer_wrappers.buffers, {}, buffer_wrappers.delete_buffer.get());

  // Column 3 is the array values for the point column (the actual data for both lat/lon).
  // The data is organized by alternating lat/lon elements, i.e.:
  // buffer[0] == lat[0], buffer[1] == lon[0]
  auto decomp_coords = buffer_wrappers.getDecompressedCoordsAt(
      point_t, {db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(decomp_coords.size(), 6U);
  // The compression process has some lossy-ness that is hard to capture with exact value
  // comparison.  Compare with an epsilon to account for this.
  // Null Lat values show  up as "-180", null Lon values show up as "-90".
  EXPECT_NEAR(decomp_coords[0], 0.0, 0.0001);
  EXPECT_NEAR(decomp_coords[2], -180.0, 0.0001);
  EXPECT_NEAR(decomp_coords[4], -180.0, 0.0001);  // paired lon is null, so this is null.
  EXPECT_NEAR(decomp_coords[1], 1.0, 0.0001);
  EXPECT_NEAR(decomp_coords[3], -90.0, 0.0001);  // paired lat is null, so this is null.
  EXPECT_NEAR(decomp_coords[5], -90.0, 0.0001);
}

// Make sure we handle colums after a compressed point, including a second compressed
// point.
TEST_P(DoubleToPointCompressionUnitTest, MultiColumn) {
  createWrapper(
      "../../Tests/FsiDataFiles/GeoTypes/idx_lon_lat_space_lon_lat." + GetParam(),
      createIdxPointSpacePointSchema());
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(6U, meta_vec.size());

  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);

  auto& buffer1 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 1, 0});
  ASSERT_EQ(buffer1.size(), 12U);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[0], 0);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[1], 1);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[2], 2);

  // The 'point' column (2) is purely logical and contains no interesting data.  Skip to
  // look at the array column (3).
  auto decomp_coords1 = buffer_wrappers.getDecompressedCoordsAt(
      point_t, {db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(decomp_coords1.size(), 6U);
  EXPECT_NEAR(decomp_coords1[0], 0.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[2], 1.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[4], 2.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[1], 3.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[3], 4.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[5], 5.0, 0.0001);

  auto& buffer3 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 4, 0});
  ASSERT_EQ(buffer3.size(), 12U);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[0], 0);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[1], 1);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[2], 2);

  auto decomp_coords2 = buffer_wrappers.getDecompressedCoordsAt(
      point_t, {db_id_, foreign_table_->tableId, 6, 0});
  ASSERT_EQ(decomp_coords2.size(), 6U);
  EXPECT_NEAR(decomp_coords2[0], 6.0, 0.0001);
  EXPECT_NEAR(decomp_coords2[2], 7.0, 0.0001);
  EXPECT_NEAR(decomp_coords2[4], 8.0, 0.0001);
  EXPECT_NEAR(decomp_coords2[1], 9.0, 0.0001);
  EXPECT_NEAR(decomp_coords2[3], 10.0, 0.0001);
  EXPECT_NEAR(decomp_coords2[5], 11.0, 0.0001);
}

INSTANTIATE_TEST_SUITE_P(DoubleToPointCompressionUnitTest,
                         DoubleToPointCompressionUnitTest,
                         testing::Values("csv", "parquet"),
                         [](const auto& info) { return info.param; });

class ParquetUnitTest : public ForeignDataWrapperUnitTest {
 public:
  std::string getServerName() const override { return shared::kDefaultParquetServerName; }

  std::unique_ptr<ForeignDataWrapper> createWrapperPtr(int32_t db_id,
                                                       ForeignTable* ft,
                                                       UserMapping* um) const override {
    return std::make_unique<ParquetDataWrapper>(
        db_id_, foreign_table_.get(), user_mapping_.get());
  }
};

TEST_F(ParquetUnitTest, Preview) {
  createWrapper("../../Tests/FsiDataFiles/GeoTypes/lon_lat_separate.parquet",
                createDoubleSchema());
  auto preview = static_cast<ParquetDataWrapper*>(wrapper_.get())->getDataPreview(1);
  DataPreview expected{{"index", "lon", "lat"},
                       {kBIGINT, kDOUBLE, kDOUBLE},
                       {{"0", "0.000000", "1.000000"}},
                       0};
  EXPECT_EQ(preview, expected);
}

TEST_F(ParquetUnitTest, PointCompressionFloatToPoint) {
  createWrapper("../../Tests/FsiDataFiles/GeoTypes/idx_lon_lat_float.parquet",
                createPointIndexSchema());
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(3U, meta_vec.size());

  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);

  auto& buffer1 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 1, 0});
  ASSERT_EQ(buffer1.size(), 12U);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[0], 0);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[1], 1);
  EXPECT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[2], 2);

  auto decomp_coords1 = buffer_wrappers.getDecompressedCoordsAt(
      point_t, {db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(decomp_coords1.size(), 6U);
  EXPECT_NEAR(decomp_coords1[0], 0.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[2], 1.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[4], 2.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[1], 3.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[3], 4.0, 0.0001);
  EXPECT_NEAR(decomp_coords1[5], 5.0, 0.0001);
}

TEST_F(ParquetUnitTest, PointCompressionFloatDoubleMismatch) {
  createWrapper("../../Tests/FsiDataFiles/GeoTypes/idx_lon_lat_float_double.parquet",
                createPointIndexSchema());
  DBHandlerTestFixture::executeLambdaAndAssertPartialException(
      [this] { populateChunkMetadata(); }, "Mismatched number of logical columns");
}

TEST_F(ParquetUnitTest, PointCompressionIntToPoint) {
  createWrapper("../../Tests/FsiDataFiles/GeoTypes/lon_lat_separate_int.parquet",
                createPointIndexSchema());
  DBHandlerTestFixture::executeLambdaAndAssertPartialException(
      [this] { populateChunkMetadata(); }, "Mismatched number of logical columns");
}

int main(int argc, char** argv) {
  g_enable_fsi = true;
  g_enable_s3_fsi = true;
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    testing::AddGlobalTestEnvironment(new DBHandlerTestEnvironment);
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_fsi = false;
  return err;
}
