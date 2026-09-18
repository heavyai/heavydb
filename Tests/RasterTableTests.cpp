/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <filesystem>
#include "Catalog/Catalog.h"
#include "Catalog/ForeignTable.h"
#include "Catalog/SysCatalog.h"
#include "Catalog/UserMapping.h"
#include "DataMgr/ChunkMetadata.h"
#include "DataMgr/ForeignStorage/RasterDataWrapper.h"
#include "Fragmenter/RasterFragmenter.h"
#include "Geospatial/GDAL.h"
#include "Logger/Logger.h"
#include "Shared/Encryption.h"
#include "Tests/DBHandlerTestHelpers.h"
#include "Tests/ForeignTableTestHelpers.h"
#if defined(HAVE_AWS_S3)
#include "DataMgr/HeavyDbAwsSdk.h"
#include "Tests/AwsHelpers.h"
#endif  // defined(HAVE_AWS_S3)

using namespace foreign_storage;
using ADW = AbstractFileStorageDataWrapper;
using RDW = RasterDataWrapper;
using FT = ForeignTable;

#define WIDTH RDW::RASTER_WIDTH_KEY
#define HEIGHT RDW::RASTER_HEIGHT_KEY

extern bool g_enable_legacy_raster_import;
extern bool g_raster_logging;
extern bool g_enable_s3_fsi;

// TODO(Misiu): Create a suite of raster files that have the same data buf use different
// file types so that we can perform the same tests with different file types.

const std::string raster_prefix{"../../Tests/Import/datafiles/raster/"};

// These are set at runtime based on the executable's path.
std::string raster_data_dir, small_tiff_file_name, png_file_name, geo_tiff_file_name,
    hdf5_file_name, geo_tiff_dir, grip_file_name, zarr_archive, zarr_file_name,
    simple_tiff_file_name, geo_tiff_null_file_name, remote_tiff_file_name, tmp_dir,
    binary_path;

namespace {
template <class T>
ChunkMetadata create_meta(const RasterTileInfo& raster_tile, T, T, bool = false) {
  UNREACHABLE();
  return ChunkMetadata();
}

template <>
ChunkMetadata create_meta(const RasterTileInfo& raster_tile,
                          double min,
                          double max,
                          bool has_nulls) {
  auto num_elems = raster_tile.width * raster_tile.height;
  return ChunkMetadata(kDOUBLE,
                       num_elems * 8,
                       num_elems,
                       ChunkStats{{.doubleval = min}, {.doubleval = max}, has_nulls},
                       raster_tile);
}

template <>
ChunkMetadata create_meta(const RasterTileInfo& raster_tile,
                          float min,
                          float max,
                          bool has_nulls) {
  auto num_elems = raster_tile.width * raster_tile.height;
  return ChunkMetadata(kFLOAT,
                       num_elems * 4,
                       num_elems,
                       ChunkStats{{.floatval = min}, {.floatval = max}, has_nulls},
                       raster_tile);
}

template <>
ChunkMetadata create_meta(const RasterTileInfo& raster_tile,
                          int32_t min,
                          int32_t max,
                          bool has_nulls) {
  auto num_elems = raster_tile.width * raster_tile.height;
  return ChunkMetadata(kINT,
                       num_elems * 4,
                       num_elems,
                       ChunkStats{{.intval = min}, {.intval = max}, has_nulls},
                       raster_tile);
}

template <>
ChunkMetadata create_meta(const RasterTileInfo& raster_tile,
                          int16_t min,
                          int16_t max,
                          bool has_nulls) {
  auto num_elems = raster_tile.width * raster_tile.height;
  return ChunkMetadata(kSMALLINT,
                       num_elems * 2,
                       num_elems,
                       ChunkStats{{.smallintval = min}, {.smallintval = max}, has_nulls},
                       raster_tile);
}

template <class T>
ChunkMetadata create_default_meta(const RasterTileInfo&) {
  return ChunkMetadata();
}

template <>
ChunkMetadata create_default_meta<float_t>(const RasterTileInfo& raster_tile) {
  auto num_elems = raster_tile.width * raster_tile.height;
  return ChunkMetadata(kFLOAT,
                       num_elems * 4,
                       num_elems,
                       ChunkStats{{.floatval = std::numeric_limits<float_t>::max()},
                                  {.floatval = std::numeric_limits<float_t>::lowest()},
                                  true},
                       raster_tile);
}

template <>
ChunkMetadata create_default_meta<int32_t>(const RasterTileInfo& raster_tile) {
  auto num_elems = raster_tile.width * raster_tile.height;
  return ChunkMetadata(kINT,
                       num_elems * 4,
                       num_elems,
                       ChunkStats{{.intval = std::numeric_limits<int32_t>::max()},
                                  {.intval = std::numeric_limits<int32_t>::lowest()},
                                  true},
                       raster_tile);
}

ChunkMetadata create_placeholder_meta_point(const RasterTileInfo& raster_tile) {
  auto num_elems = raster_tile.width * raster_tile.height;
  return ChunkMetadata(SQLTypeInfo(kPOINT, 0, 0, false, kENCODING_GEOINT, 32, kNULLT),
                       0,
                       num_elems,
                       ChunkStats{{.stringval = nullptr}, {.stringval = nullptr}, false},
                       raster_tile);
}

ChunkMetadata create_placeholder_meta_compressed_array(
    const RasterTileInfo& raster_tile) {
  auto num_elems = raster_tile.width * raster_tile.height;
  SQLTypeInfo coords_type = SQLTypeInfo(kARRAY, false);
  coords_type.set_subtype(kTINYINT);
  coords_type.set_size(8);  // compressed
  return ChunkMetadata(coords_type,
                       num_elems * 8,
                       num_elems,
                       ChunkStats{{.tinyintval = std::numeric_limits<int8_t>::max()},
                                  {.tinyintval = std::numeric_limits<int8_t>::lowest()},
                                  false},
                       raster_tile);
}

std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> create_simple_placeholder_meta(
    int32_t db_id,
    int32_t tb_id) {
  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> simple_meta;
  for (auto x = 0; x < 4; ++x) {
    for (auto y = 0; y < 4; ++y) {
      auto frag = x + (y * 4);
      auto lat = x * 16, lon = y * 16;
      RasterTileInfo tileInfo{16, 16, {0, x, y}};
      simple_meta.emplace(ChunkKey{db_id, tb_id, 1, frag},
                          std::make_shared<ChunkMetadata>(
                              create_meta<double_t>(tileInfo, lat, lat + 15)));
      simple_meta.emplace(ChunkKey{db_id, tb_id, 2, frag},
                          std::make_shared<ChunkMetadata>(
                              create_meta<double_t>(tileInfo, lon, lon + 15)));
      simple_meta.emplace(
          ChunkKey{db_id, tb_id, 3, frag},
          std::make_shared<ChunkMetadata>(create_default_meta<int32_t>(tileInfo)));
    }
  }
  return simple_meta;
}

std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> create_simple_meta(int32_t db_id,
                                                                      int32_t tb_id) {
  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> simple_meta;
  for (auto x = 0; x < 4; ++x) {
    for (auto y = 0; y < 4; ++y) {
      auto frag = x + (y * 4);
      auto lat = x * 16, lon = y * 16;
      RasterTileInfo tileInfo{16, 16, {0, x, y}};
      simple_meta.emplace(ChunkKey{db_id, tb_id, 1, frag},
                          std::make_shared<ChunkMetadata>(
                              create_meta<double_t>(tileInfo, lat, lat + 15)));
      simple_meta.emplace(ChunkKey{db_id, tb_id, 2, frag},
                          std::make_shared<ChunkMetadata>(
                              create_meta<double_t>(tileInfo, lon, lon + 15)));
      simple_meta.emplace(
          ChunkKey{db_id, tb_id, 3, frag},
          std::make_shared<ChunkMetadata>(create_meta<int32_t>(tileInfo, frag, frag)));
    }
  }
  return simple_meta;
}

std::string create_wrapper_file(const std::string& wrapper_string) {
  std::string serial_wrapper_file{"tmp_serial_wrapper.txt"};
  std::ofstream ofs(serial_wrapper_file);
  if (!ofs) {
    throw std::runtime_error{"Error trying to create file"};
  }
  ofs << wrapper_string;
  return serial_wrapper_file;
}

template <typename Lambda>
inline void assert_throw_contains(Lambda lambda, const std::string& text) {
  try {
    lambda();
    FAIL() << "Testcase expected exception containing text: '" << text << "'";
  } catch (const std::exception& e) {
    ASSERT_TRUE(std::string(e.what()).find(text) != std::string::npos)
        << "Error: '" << e.what() << "' should contain '" << text << "'";
  }
}
}  // namespace

class RasterTableUnitTest : public ForeignDataWrapperUnitTest {
 public:
  static std::list<ColumnDescriptor> createIntSchema() {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kINT, db_id_));
    return columns;
  }

  static std::list<ColumnDescriptor> createFloatSchema() {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    return columns;
  }

  static std::list<ColumnDescriptor> createPointSchema() {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(
        ColumnDescriptor(0,
                         0,
                         "p",
                         SQLTypeInfo(kPOINT, 0, 0, false, kENCODING_GEOINT, 32, kNULLT),
                         db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    return columns;
  }

  static std::list<ColumnDescriptor> createUncompressedPointSchema() {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "p", SQLTypeInfo(kPOINT), db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    return columns;
  }

  static std::list<ColumnDescriptor> createPointIntSchema() {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    return columns;
  }

  static std::list<ColumnDescriptor> createPointSmallIntSchema() {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    return columns;
  }

  virtual std::list<ColumnDescriptor> createDefaultSchema() const {
    return createIntSchema();
  }

  std::string getServerName() const override { return shared::kDefaultRasterServerName; }

  std::unique_ptr<ForeignDataWrapper> createWrapperPtr(int32_t db_id,
                                                       ForeignTable* ft,
                                                       UserMapping* um) const override {
    return std::make_unique<RasterDataWrapper>(
        db_id_, foreign_table_.get(), user_mapping_.get());
  }

  void createWrapper(const std::string& file_name,
                     const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& param_columns = {},
                     const OptionsMap& extra_options = {{}}) {
    OptionsMap options;
    options[RDW::RASTER_WIDTH_KEY] = std::to_string(width);
    options[RDW::RASTER_HEIGHT_KEY] = std::to_string(height);

    for (auto& [key, val] : extra_options) {
      options[key] = val;
    }

    const auto columns =
        (param_columns.size() > 0) ? param_columns : createDefaultSchema();
    ForeignDataWrapperUnitTest::createWrapper(file_name, columns, options);
  }

  virtual void createWrapper(const size_t width,
                             const size_t height,
                             const std::list<ColumnDescriptor>& columns,
                             const OptionsMap& map) = 0;
};

class SmallTiffTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& columns = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(small_tiff_file_name, width, height, columns, map);
  }

  void createWrapper() {
    OptionsMap options{
        {ForeignTable::REFRESH_TIMING_TYPE_KEY, ForeignTable::MANUAL_REFRESH_TIMING_TYPE},
        {ForeignTable::REFRESH_UPDATE_TYPE_KEY, ForeignTable::ALL_REFRESH_UPDATE_TYPE}};
    ForeignDataWrapperUnitTest::createWrapper(
        small_tiff_file_name, createDefaultSchema(), options);
  }
};

TEST_F(SmallTiffTest, NumFragments20x40) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(150U, meta_vec.size());
}

TEST_F(SmallTiffTest, NumFragments40x20) {
  createWrapper(40, 20);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(150U, meta_vec.size());
}

TEST_F(SmallTiffTest, NumFragments20x20) {
  createWrapper(20, 20);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(300U, meta_vec.size());
}

TEST_F(SmallTiffTest, NumFragments1x1) {
  createWrapper(1, 1);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(120000U, meta_vec.size());
}

// Table's raster dimensions do not fit nicely in raster file's dimensions.
TEST_F(SmallTiffTest, NumFragments20x80) {
  createWrapper(20, 80);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(90U, meta_vec.size());
}

TEST_F(SmallTiffTest, NumFragments80x20) {
  createWrapper(80, 20);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(90U, meta_vec.size());
}

TEST_F(SmallTiffTest, NumFragments80x80) {
  createWrapper(80, 80);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(27U, meta_vec.size());
}

TEST_F(SmallTiffTest, NumFragments300x30) {
  createWrapper(300, 30);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(21U, meta_vec.size());
}

// Is this a legal configuration?
TEST_F(SmallTiffTest, NumFragments300x300) {
  createWrapper(300, 300);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(3U, meta_vec.size());
}

// Lat/Lon should be calculable during metadata scan.
TEST_F(SmallTiffTest, Metadata20x40) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
      create_meta<double>({20, 40, {0, 0, 0}}, 45.02797208719269, 45.033360807800015));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 10}),
      create_meta<double>({20, 40, {0, 0, 1}}, 45.02614666604606, 45.03153472205765));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 35}),
      create_meta<double>({20, 40, {0, 5, 3}}, 45.003510798799581, 45.008895840009515));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 2, 0}),
      create_meta<double>({20, 40, {0, 0, 0}}, 62.644479319411481, 62.648331509145756));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 2, 10}),
      create_meta<double>({20, 40, {0, 0, 1}}, 62.640926319007448, 62.644778502771807));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 2, 35}),
      create_meta<double>({20, 40, {0, 5, 3}}, 62.635861135132984, 62.639712912772751));
  EXPECT_CHUNK_METADATA_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 0}),
                           create_default_meta<int32_t>({20, 40, {0, 0, 0}}));
}

// Widening fragments by 3x allows us to pre-calculate expected min/max of fragments.
TEST_F(SmallTiffTest, Metadata60x40) {
  createWrapper(60, 40);
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
      create_meta<double>({60, 40, {0, 0, 0}}, 45.020375436206358, 45.033360807800015));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 3}),
      create_meta<double>({20, 40, {0, 3, 0}}, 44.993784339306437, 44.999170025815438));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 10}),
      create_meta<double>({60, 40, {0, 2, 2}}, 44.993939680500745, 45.006921042079085));
}

// Heightening fragment size lets us pre-calculate metadata as well.
TEST_F(SmallTiffTest, Metadata20x80) {
  createWrapper(20, 80);
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
      create_meta<double>({20, 80, {0, 0, 0}}, 45.02614666604606, 45.033360807800015));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 10}),
      create_meta<double>({20, 80, {0, 0, 1}}, 45.022496603126839, 45.029708896106634));
  // This fragment is only 20x40
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 20}),
      create_meta<double>({20, 40, {0, 0, 2}}, 45.020671961354246, 45.026058023578621));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 2, 0}),
      create_meta<double>({20, 80, {0, 0, 0}}, 62.640926319007448, 62.648331509145756));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 2, 10}),
      create_meta<double>({20, 80, {0, 0, 1}}, 62.633820283707379, 62.641225484900517));
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 2, 20}),
      create_meta<double>({20, 40, {0, 0, 2}}, 62.630267248811336, 62.634119414665939));
}

// If all data is populated then we should have updated metadata.
TEST_F(SmallTiffTest, Data20x40) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);
  const auto meta_map = buffer_wrappers.getMetadata();
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 0}),
            create_meta<int32_t>({20, 40, {0, 0, 0}}, 0, 0, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 9}),
            create_meta<int32_t>({20, 40, {0, 9, 0}}, 0, 365, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 10}),
            create_meta<int32_t>({20, 40, {0, 0, 1}}, 0, 0, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 39}),
            create_meta<int32_t>({20, 40, {0, 9, 3}}, 0, 309, false));
}

// Only the populated buffers should have updated metadata.
TEST_F(SmallTiffTest, PartialData20x40) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  auto meta_map = create_meta_map(meta_vec);
  FragmentBuffers buffer_wrappers(
      {{db_id_, foreign_table_->tableId, 3, 0}, {db_id_, foreign_table_->tableId, 3, 9}});
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);

  // Populated buffers will have fully populated metadata.
  EXPECT_EQ(buffer_wrappers.getMetadata({db_id_, foreign_table_->tableId, 3, 0}),
            create_meta<int32_t>({20, 40, {0, 0, 0}}, 0, 0, false));
  EXPECT_EQ(buffer_wrappers.getMetadata({db_id_, foreign_table_->tableId, 3, 9}),
            create_meta<int32_t>({20, 40, {0, 9, 0}}, 0, 365, false));

  // Unpopulated buffers will have default metadata.
  EXPECT_EQ(*meta_map.at({db_id_, foreign_table_->tableId, 3, 10}),
            create_default_meta<int32_t>({20, 40, {0, 0, 1}}));
  EXPECT_EQ(*meta_map.at({db_id_, foreign_table_->tableId, 3, 39}),
            create_default_meta<int32_t>({20, 40, {0, 9, 3}}));

  // Confirm expected data in buffers.
  auto& buffer1 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(buffer1.size(), 3200U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[799], 0);

  auto& buffer2 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 9});
  ASSERT_EQ(buffer2.size(), 3200U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer2.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer2.getMemoryPtr())[799], 275);
}

TEST_F(SmallTiffTest, Data60x40) {
  createWrapper(60, 40);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers({{db_id_, foreign_table_->tableId, 3, 0},
                                   {db_id_, foreign_table_->tableId, 3, 3},
                                   {db_id_, foreign_table_->tableId, 3, 10}});
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);
  auto& buffer1 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(buffer1.size(), 9600U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[2399], 0);
  auto& buffer2 =
      buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 3});  // 20x40 fragment.
  ASSERT_EQ(buffer2.size(), 3200U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer2.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer2.getMemoryPtr())[799], 275);
  auto& buffer3 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 10});
  ASSERT_EQ(buffer3.size(), 9600U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[2399], 0);
}

TEST_F(SmallTiffTest, Data20x80) {
  createWrapper(20, 80);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers({{db_id_, foreign_table_->tableId, 3, 0},
                                   {db_id_, foreign_table_->tableId, 3, 9},
                                   {db_id_, foreign_table_->tableId, 3, 29}});
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);
  auto& buffer1 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(buffer1.size(), 6400U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[1599], 0);
  auto& buffer2 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 9});
  ASSERT_EQ(buffer2.size(), 6400U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer2.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer2.getMemoryPtr())[1599], 287);
  auto& buffer3 =
      buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 29});  // 20x40 fragment.
  ASSERT_EQ(buffer3.size(), 3200U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[799], 201);
}

TEST_F(SmallTiffTest, Auto) {
  createWrapper();
  auto meta_vec = populateChunkMetadata();
  // 10 fragments because block size for this file is 200x20 (file size is 200x200).
  ASSERT_EQ(30U, meta_vec.size());
}

TEST_F(SmallTiffTest, RestoreWrapper) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);
  auto wrapper_string = wrapper_->getSerializedDataWrapper();
  auto wrapper_file = create_wrapper_file(wrapper_string);
  ScopeGuard file_guard = [wrapper_file]() { std::filesystem::remove(wrapper_file); };

  wrapper_->restoreDataWrapperInternals(wrapper_file, meta_vec);
  FragmentBuffers buffer_wrappers2(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers2.buffers, {}, nullptr);
  EXPECT_EQ(buffer_wrappers, buffer_wrappers2);
}

#ifdef HAVE_AWS_S3
class S3TiffTest : public RasterTableUnitTest {
 public:
  inline static const std::string server_name{"s3_raster_server"};

  static void SetUpTestSuite() {
    g_allow_s3_server_privileges = true;
    RasterTableUnitTest::SetUpTestSuite();
  }

  static void TearDownTestSuite() {
    RasterTableUnitTest::TearDownTestSuite();
    g_allow_s3_server_privileges = false;
  }

  void TearDown() override {
    RasterTableUnitTest::TearDown();
    cat_ptr_->dropForeignServer(server_name);
  }

  OptionsMap getS3Credentials() const {
    const auto& env_key = get_aws_keys_from_env();
    return {{"S3_ACCESS_KEY", env_key.first}, {"S3_SECRET_KEY", env_key.second}};
  }

  OptionsMap getS3CredentialsSts(const ForeignServer* server) const {
    CHECK(server);
    const auto& env_key = get_aws_keys_from_env();
    const auto& sts_credentials = generate_sts_credentials(env_key, server);
    return {{"S3_ACCESS_KEY", sts_credentials.GetAccessKeyId()},
            {"S3_SECRET_KEY", sts_credentials.GetSecretAccessKey()},
            {"S3_SESSION_TOKEN", sts_credentials.GetSessionToken()}};
  }

  const ForeignServer* createServer(const std::string& bucket) const {
    auto server = std::make_unique<foreign_storage::ForeignServer>(
        server_name,
        foreign_storage::DataWrapperType::RASTER,
        std::map<std::string, std::string, std::less<>>{{"STORAGE_TYPE", "AWS_S3"},
                                                        {"S3_BUCKET", bucket},
                                                        {"AWS_REGION", "us-west-1"}},
        shared::kRootUserId);
    server->validate();
    cat_ptr_->createForeignServer(std::move(server), false);
    return cat_ptr_->getForeignServer(server_name);
  }

  std::unique_ptr<UserMapping> createUserMapping(const ForeignServer* server,
                                                 const OptionsMap& options) const {
    CHECK(server);
    auto user_mapping = std::make_unique<UserMapping>();
    user_mapping->setOptions(options);
    user_mapping->foreign_server_id = server->id;
    user_mapping->user_id = shared::kRootUserId;
    user_mapping->type = UserMappingType::PUBLIC;
    user_mapping->validate(server);
    return user_mapping;
  }

  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& columns = {},
                     const OptionsMap& extra_options = {{}}) override {
    auto server = createServer("omnisci-fsi-test-public/FsiDataFiles");
    createS3Wrapper(width, height, server, nullptr, columns, extra_options);
  }

  void createS3Wrapper(const size_t width,
                       const size_t height,
                       const ForeignServer* server,
                       const UserMapping* user_mapping,
                       const std::list<ColumnDescriptor>& columns = {},
                       const OptionsMap& table_options = {{}}) {
    std::string file_name{"s1b_small.tiff"};
    OptionsMap options{};
    options[AbstractFileStorageDataWrapper::FILE_PATH_KEY] = file_name;
    options[RasterDataWrapper::RASTER_WIDTH_KEY] = std::to_string(width);
    options[RasterDataWrapper::RASTER_HEIGHT_KEY] = std::to_string(height);
    // Refresh options are not used in testing, but required for a valid foreign table.
    options[ForeignTable::REFRESH_TIMING_TYPE_KEY] =
        ForeignTable::MANUAL_REFRESH_TIMING_TYPE;
    options[ForeignTable::REFRESH_UPDATE_TYPE_KEY] =
        ForeignTable::ALL_REFRESH_UPDATE_TYPE;

    for (auto& [key, val] : table_options) {
      options[key] = val;
    }

    foreign_table_ = std::make_unique<ForeignTable>();
    foreign_table_->populateOptionsMap(json_from_map(options));
    foreign_table_->foreign_server = server;

    cat_ptr_->createTable(*foreign_table_,
                          (columns.size() < 1) ? createDefaultSchema() : columns,
                          {},
                          true);

    wrapper_ =
        std::make_unique<RasterDataWrapper>(db_id_, foreign_table_.get(), user_mapping);
  }
};

// TODO: Re-enable the two tests below when intermittent crash issue is resolved.
TEST_F(S3TiffTest, DISABLED_Public) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(150U, meta_vec.size());
}

TEST_F(S3TiffTest, DISABLED_Data20x80) {
  createWrapper(20, 80);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers({{db_id_, foreign_table_->tableId, 3, 0},
                                   {db_id_, foreign_table_->tableId, 3, 9},
                                   {db_id_, foreign_table_->tableId, 3, 29}});
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);
  auto& buffer1 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(buffer1.size(), 6400U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer1.getMemoryPtr())[1599], 0);
  auto& buffer2 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 9});
  ASSERT_EQ(buffer2.size(), 6400U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer2.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer2.getMemoryPtr())[1599], 287);
  auto& buffer3 =
      buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 29});  // 20x40 fragment.
  ASSERT_EQ(buffer3.size(), 3200U);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[0], 0);
  ASSERT_EQ(reinterpret_cast<int32_t*>(buffer3.getMemoryPtr())[799], 201);
}
#endif  // HAVE_AWS_S3

class PngTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(png_file_name, width, height, schema, map);
  }

  std::list<ColumnDescriptor> createDefaultSchema() const override {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_3", kSMALLINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createSingleFilteredSchema() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kSMALLINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createDoubleFilteredSchema() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_3", kSMALLINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createNoRenameFilteredSchema() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1_2", kSMALLINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createInvalidPointColumnTypeSchema() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_3", kSMALLINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createInvalidBandColumnTypeSchema() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_3", kINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createInvalidBandColumnCountSchema1() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_3", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_4", kINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createInvalidBandColumnCountSchema2() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createPackedColorSchema() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "packed_color", kINT, db_id_));
    return columns;
  }

  std::list<ColumnDescriptor> createMultiplePackedColorSchema() const {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kSMALLINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "packed_color1", kINT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "packed_color2", kINT, db_id_));
    return columns;
  }
};

TEST_F(PngTest, NumFragments20x40) {  // 16x6
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(480U, meta_vec.size());
}

TEST_F(PngTest, NumFragments40x20) {
  createWrapper(40, 20);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(480U, meta_vec.size());
}

TEST_F(PngTest, FilterBand) {
  createWrapper(20,
                40,
                createSingleFilteredSchema(),
                {{RDW::RASTER_FILTER_BANDS_KEY, "band_2=band_1_2"}});
  auto meta_vec = populateChunkMetadata();
  // Only one band.
  ASSERT_EQ(288U, meta_vec.size());
}

TEST_F(PngTest, FilterBandsFail) {
  ASSERT_THROW(
      createWrapper(20,
                    40,
                    createSingleFilteredSchema(),
                    {{RDW::RASTER_FILTER_BANDS_KEY, "band_2=band_1_2, band_3=band_1_3"}}),
      ColumnTypeMismatchException);
}

TEST_F(PngTest, FilterBands) {
  createWrapper(20,
                40,
                createDoubleFilteredSchema(),
                {{RDW::RASTER_FILTER_BANDS_KEY, "band_2=band_1_2, band_3=band_1_3"}});
  auto meta_vec = populateChunkMetadata();
  // Two bands.
  ASSERT_EQ(384U, meta_vec.size());
}

TEST_F(PngTest, FilterBandsRepeatedFail) {
  createWrapper(20,
                40,
                createDoubleFilteredSchema(),
                {{RDW::RASTER_FILTER_BANDS_KEY, "band_2=band_1_2, band_3=band_1_2"}});
  assert_throw_contains([&] { populateChunkMetadata(); },
                        "Found repeated specified band name 'band_1_2'");
}

TEST_F(PngTest, FilterBandNoRename) {
  createWrapper(20,
                40,
                createNoRenameFilteredSchema(),
                {{RDW::RASTER_FILTER_BANDS_KEY, "band_1_2"}});
  auto meta_vec = populateChunkMetadata();
  // One band.
  ASSERT_EQ(288U, meta_vec.size());
}

TEST_F(PngTest, PackedColor) {
  createWrapper(
      20,
      40,
      createPackedColorSchema(),
      {{RDW::RASTER_FILTER_BANDS_KEY, "packed_color=band_1_1/band_1_2/band_1_3/sRGB"}});
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(288U, meta_vec.size());
}

TEST_F(PngTest, PackedColorMultipleFail) {
  createWrapper(20,
                40,
                createMultiplePackedColorSchema(),
                {{RDW::RASTER_FILTER_BANDS_KEY,
                  "packed_color1=band_1_1/band_1_2/band_1_3/sRGB,packed_color2=band_1_1/"
                  "band_1_2/band_1_3/sRGB"}});
  assert_throw_contains([&] { populateChunkMetadata(); },
                        "found multiple packed-color expressions");
}

TEST_F(PngTest, PackedColorBandDoesNotExistFail) {
  createWrapper(
      20,
      40,
      createPackedColorSchema(),
      {{RDW::RASTER_FILTER_BANDS_KEY, "packed_color=band_1_1/band_1_2/band_1_4/sRGB"}});
  assert_throw_contains(
      [&] { populateChunkMetadata(); },
      "Specified import band name 'band_1_4' was not found in the input raster file");
}

TEST_F(PngTest, InvalidPointColumnTypeFail) {
  createWrapper(20, 40, createInvalidPointColumnTypeSchema());
  assert_throw_contains([&] { populateChunkMetadata(); },
                        "Must do World/File Transform with DOUBLE/FLOAT Point type");
}

TEST_F(PngTest, InvalidBandColumnTypeFail) {
  createWrapper(20, 40, createInvalidBandColumnTypeSchema());
  assert_throw_contains(
      [&] { populateChunkMetadata(); },
      "column 'band_1', column is type INT, file band is type SMALLINT");
}

TEST_F(PngTest, InvalidBandColumnCountTooMany) {
  createWrapper(20, 40, createInvalidBandColumnCountSchema1());
  assert_throw_contains([&] { populateChunkMetadata(); },
                        "file contains 3 bands, table has 4 non-coord columns");
}

TEST_F(PngTest, InvalidBandColumnCountTooFew) {
  createWrapper(20, 40, createInvalidBandColumnCountSchema2());
  assert_throw_contains([&] { populateChunkMetadata(); },
                        "file contains 3 bands, table has 2 non-coord columns");
}

class GeoTiffTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(geo_tiff_file_name, width, height, schema, map);
  }

  std::list<ColumnDescriptor> createDefaultSchema() const override {
    return createFloatSchema();
  }
};

TEST_F(GeoTiffTest, NumFragments20x40) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(150U, meta_vec.size());
}

TEST_F(GeoTiffTest, Default) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(150U, meta_vec.size());
  const auto& [key1, meta1] = meta_vec[0];
  EXPECT_EQ(key1, (ChunkKey{db_id_, foreign_table_->tableId, 1, 0}));
  EXPECT_EQ(meta1->sqlType, kDOUBLE);  // point x
  const auto& [key2, meta2] = meta_vec[50];
  EXPECT_EQ(key2, (ChunkKey{db_id_, foreign_table_->tableId, 2, 0}));
  EXPECT_EQ(meta2->sqlType, kDOUBLE);  // point y
  const auto& [key3, meta3] = meta_vec[100];
  EXPECT_EQ(key3, (ChunkKey{db_id_, foreign_table_->tableId, 3, 0}));
  EXPECT_EQ(meta3->sqlType, kFLOAT);  // band 1
}

TEST_F(GeoTiffTest, EncodedPoint) {
  createWrapper(20, 40, createPointSchema());
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(150U, meta_vec.size());
  const auto& [key1, meta1] = meta_vec[0];  // point x/y
  EXPECT_EQ(key1, (ChunkKey{db_id_, foreign_table_->tableId, 1, 0, 1}));
  EXPECT_EQ(meta1->sqlType, point_t);
  const auto& [key2, meta2] = meta_vec[50];  // coords
  EXPECT_EQ(key2, (ChunkKey{db_id_, foreign_table_->tableId, 2, 0}));
  EXPECT_EQ(meta2->sqlType, SQLTypeInfo(kARRAY, kENCODING_NONE, 0, kTINYINT));
  const auto& [key3, meta3] = meta_vec[100];  // coords
  EXPECT_EQ(key3, (ChunkKey{db_id_, foreign_table_->tableId, 3, 0}));
  EXPECT_EQ(meta3->sqlType, kFLOAT);
}

TEST_F(GeoTiffTest, UnencodedPoint) {
  createWrapper(20, 40, createUncompressedPointSchema());
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(150U, meta_vec.size());
  const auto& [key1, meta1] = meta_vec[0];  // point x/y
  EXPECT_EQ(key1, (ChunkKey{db_id_, foreign_table_->tableId, 1, 0, 1}));
  EXPECT_EQ(meta1->sqlType, kPOINT);
  const auto& [key2, meta2] = meta_vec[50];  // coords
  EXPECT_EQ(key2, (ChunkKey{db_id_, foreign_table_->tableId, 2, 0}));
  EXPECT_EQ(meta2->sqlType, SQLTypeInfo(kARRAY, kENCODING_NONE, 0, kTINYINT));
  const auto& [key3, meta3] = meta_vec[100];  // band 1
  EXPECT_EQ(key3, (ChunkKey{db_id_, foreign_table_->tableId, 3, 0}));
  EXPECT_EQ(meta3->sqlType, kFLOAT);
}

TEST_F(GeoTiffTest, ValidateLatLonMismatch) {
  std::list<ColumnDescriptor> columns{ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_),
                                      ColumnDescriptor(0, 0, "y", kFLOAT, db_id_)};
  ASSERT_THROW(createWrapper(20, 40, columns), ColumnTypeMismatchException);
}

TEST_F(GeoTiffTest, Metadata20x40) {
  createWrapper(20, 40, createPointSchema());
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0, 1}),
            create_placeholder_meta_point({20, 40, {0, 0, 0}}));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 10, 1}),
            create_placeholder_meta_point({20, 40, {0, 0, 1}}));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 2, 0}),
            create_placeholder_meta_compressed_array({20, 40, {0, 0, 0}}));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 2, 10}),
            create_placeholder_meta_compressed_array({20, 40, {0, 0, 1}}));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 0}),
            create_default_meta<float_t>({20, 40, {0, 0, 0}}));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 10}),
            create_default_meta<float_t>({20, 40, {0, 0, 1}}));
}

TEST_F(GeoTiffTest, PointAndBandData20x40) {
  createWrapper(20, 40, createPointSchema());
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);

  // Point coords columns should have actual data in them, point logical columns should
  // have no data in them and it makes no sense to fetch their buffers.
  {
    auto& buffer = buffer_wrappers.at({db_id_, foreign_table_->tableId, 2, 0});
    ASSERT_EQ(buffer.size(), 6400U);
    EXPECT_EQ(buffer.getMemoryPtr()[0], -3);     // First array.
    EXPECT_EQ(buffer.getMemoryPtr()[16], 20);    // Second array.
    EXPECT_EQ(buffer.getMemoryPtr()[6384], 79);  // Last array.
  }
  {
    // Fragment #10 should wrap around to the second band width.
    auto& buffer = buffer_wrappers.at({db_id_, foreign_table_->tableId, 2, 10});
    ASSERT_EQ(buffer.size(), 6400U);
    EXPECT_EQ(buffer.getMemoryPtr()[0], -120);
    EXPECT_EQ(buffer.getMemoryPtr()[6384], -38);
  }
  {
    // Last fragment.
    auto& buffer = buffer_wrappers.at({db_id_, foreign_table_->tableId, 2, 49});
    ASSERT_EQ(buffer.size(), 6400U);
    EXPECT_EQ(buffer.getMemoryPtr()[0], 24);
    EXPECT_EQ(buffer.getMemoryPtr()[6384], 106);
  }

  // Point columns should have actual data in them.
  auto& buffer1 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(buffer1.size(), 3200U);
  EXPECT_EQ(reinterpret_cast<float_t*>(buffer1.getMemoryPtr())[0], 286.871826171875);
  EXPECT_EQ(reinterpret_cast<float_t*>(buffer1.getMemoryPtr())[799], 286.804931640625);
  auto& buffer2 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 9});
  ASSERT_EQ(buffer1.size(), 3200U);
  EXPECT_EQ(reinterpret_cast<float_t*>(buffer2.getMemoryPtr())[0], 286.35272216796875);
  EXPECT_EQ(reinterpret_cast<float_t*>(buffer2.getMemoryPtr())[799], 286.22567749023438);
}

TEST_F(GeoTiffTest, PopulatedMetadata20x40) {
  createWrapper(20, 40, createPointSchema());
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);

  const auto meta_map = buffer_wrappers.getMetadata();
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 0}),
            create_meta<float_t>({20, 40, {0, 0, 0}}, 286.759430, 286.887238, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 9}),
            create_meta<float_t>({20, 40, {0, 9, 0}}, 286.204224, 286.363831, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 10}),
            create_meta<float_t>({20, 40, {0, 0, 1}}, 286.729340, 286.987122, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 39}),
            create_meta<float_t>({20, 40, {0, 9, 3}}, 285.876984, 285.962524, false));

  auto& buffer1 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 0});
  ASSERT_EQ(buffer1.size(), 3200U);
  EXPECT_FLOAT_EQ(reinterpret_cast<float_t*>(buffer1.getMemoryPtr())[0], 286.871826);
  EXPECT_FLOAT_EQ(reinterpret_cast<float_t*>(buffer1.getMemoryPtr())[799], 286.804932);
  auto& buffer2 = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 9});
  ASSERT_EQ(buffer1.size(), 3200U);
  EXPECT_FLOAT_EQ(reinterpret_cast<float_t*>(buffer2.getMemoryPtr())[0], 286.352722);
  EXPECT_FLOAT_EQ(reinterpret_cast<float_t*>(buffer2.getMemoryPtr())[799], 286.225677);
}

TEST_F(GeoTiffTest, PointTransformInvalid) {
  ASSERT_THROW(createWrapper(20, 40, {}, {{"RASTER_POINT_TRANSFORM", "nonesense"}}),
               InvalidOptionException);
}

TEST_F(GeoTiffTest, PointTransformNoneFail) {
  ASSERT_THROW(createWrapper(20, 40, {}, {{"RASTER_POINT_TRANSFORM", "none"}}),
               ColumnTypeMismatchException);
}

TEST_F(GeoTiffTest, PointTransformNone) {
  createWrapper(20, 40, createPointIntSchema(), {{"RASTER_POINT_TRANSFORM", "none"}});
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
            create_meta<int32_t>({20, 40, {0, 0, 0}}, 0, 19, false));
}

TEST_F(GeoTiffTest, PointTransformSmall) {
  createWrapper(
      20, 40, createPointSmallIntSchema(), {{"RASTER_POINT_TRANSFORM", "none"}});
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
            create_meta<int16_t>({20, 40, {0, 0, 0}}, 0, 19, false));
}

TEST_F(GeoTiffTest, PointTransformFile) {
  createWrapper(20, 40, {}, {{"RASTER_POINT_TRANSFORM", "file"}});
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
            create_meta<double>({20, 40, {0, 0, 0}}, 309552, 309571, false));
}

TEST_F(GeoTiffTest, PointTransformWorld) {
  createWrapper(20, 40, {}, {{"RASTER_POINT_TRANSFORM", "world"}});
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
            create_meta<double>(
                {20, 40, {0, 0, 0}}, -83.225148059368721, -83.224914897099296, false));
}

TEST_F(GeoTiffTest, PointTransformWorldFail) {
  ASSERT_THROW(createWrapper(
                   20, 40, createPointIntSchema(), {{"RASTER_POINT_TRANSFORM", "world"}}),
               ColumnTypeMismatchException);
}

TEST_F(GeoTiffTest, PointTransformAuto) {
  createWrapper(20, 40, {}, {{"RASTER_POINT_TRANSFORM", "auto"}});
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
            create_meta<double>(
                {20, 40, {0, 0, 0}}, -83.225148059368721, -83.224914897099296, false));
}

// This test is checking that the correct fragments are preserved/removed via the bounding
// box clipping option.  The full file would be 10x5 fragments, but the bounding box cuts
// off the fragments where x >= 5, leaving a 5x5 box.
TEST_F(GeoTiffTest, BoundingBoxClipRight) {
  createWrapper(
      20,
      40,
      {},
      {{"RASTER_POINT_TRANSFORM", "world"}, {"BOUNDING_BOX_CLIP", "-84,0,-83.224,80"}});
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(meta_vec.size(), 75u);  // would be 150 without clipping

  const auto meta_map = create_meta_map(meta_vec);
  // First fragment was not clipped, but it's down-neighbour was remapped.
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at({db_id_, foreign_table_->tableId, 1, 0}),
      create_meta<double>({20, 40, {0, 0, 0}}, -83.225148059368721, -83.224914897099296));
  // The fragment to the right of this one should have been clipped.
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at({db_id_, foreign_table_->tableId, 1, 4}),
      create_meta<double>({20, 40, {0, 4, 0}}, -83.224214022135712, -83.223980864414102));
  // Fragment #5 was clipped, the new #5 should start the next row.
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at({db_id_, foreign_table_->tableId, 1, 5}),
      create_meta<double>({20, 40, {0, 0, 1}}, -83.225136439238554, -83.224903278328924));
  // 25 should now be the last fragment.
  EXPECT_CHUNK_METADATA_EQ(
      *meta_map.at({db_id_, foreign_table_->tableId, 1, 24}),
      create_meta<double>({20, 40, {0, 4, 4}}, -83.224167562339588, -83.223934410056657));
}

class GeoTiffNullTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(
        geo_tiff_null_file_name, width, height, schema, map);
  }

  std::list<ColumnDescriptor> createDefaultSchema() const override {
    return createFloatSchema();
  }
};

TEST_F(GeoTiffNullTest, Nulls) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);

  auto& buffer = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 49});
  ASSERT_EQ(buffer.size(), 3200U);
  EXPECT_EQ(reinterpret_cast<float_t*>(buffer.getMemoryPtr())[799], FLT_MIN);
}

class Hdf5Test : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(hdf5_file_name, width, height, schema, map);
  }

  std::list<ColumnDescriptor> createDefaultSchema() const override {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kTINYINT, db_id_));
    return columns;
  }
};

TEST_F(Hdf5Test, NumFragments20x40) {
  createWrapper(20, 40);
  assert_throw_contains([&] { populateChunkMetadata(); },
                        "datasource/band dimensions are inconsistent");
}

class GeoTiffDirTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(geo_tiff_dir, width, height, schema, map);
  }

  std::list<ColumnDescriptor> createDefaultSchema() const override {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    return columns;
  }
};

TEST_F(GeoTiffDirTest, NumFragments) {
  createWrapper(216, 216);
  auto meta_vec = populateChunkMetadata();
  // 3 columns, and two fragments (one for each file).
  ASSERT_EQ(6U, meta_vec.size());
}

TEST_F(GeoTiffDirTest, NumFragmentsPartial) {
  createWrapper(100, 100);
  auto meta_vec = populateChunkMetadata();
  // 3 columns, and 18 fragments (9 for each file).
  ASSERT_EQ(54U, meta_vec.size());
}

TEST_F(GeoTiffDirTest, OneFileNumFragments) {
  createWrapper(216,
                216,
                {},
                {{AbstractFileStorageDataWrapper::FILE_PATH_KEY,
                  raster_prefix + "geotif/USGS_13_n33*.tif"}});
  auto meta_vec = populateChunkMetadata();
  // 3 columns (only one file).
  ASSERT_EQ(3U, meta_vec.size());
}

TEST_F(GeoTiffDirTest, PathFilter) {
  createWrapper(216, 216, {}, {{"REGEX_PATH_FILTER", ".*n33.*"}});
  auto meta_vec = populateChunkMetadata();
  // One file
  ASSERT_EQ(3U, meta_vec.size());
}

TEST_F(GeoTiffDirTest, Metadata) {
  createWrapper(216, 216);
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
            create_meta<double>(
                {216, 216, {0, 0, 0}}, -116.00055555555799, -115.00407921372494, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 1}),
            create_meta<double>(
                {216, 216, {1, 0, 0}}, -125.00055555629299, -124.00407921777249, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 0}),
            create_default_meta<float>({216, 216, {0, 0, 0}}));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 3, 1}),
            create_default_meta<float>({216, 216, {1, 0, 0}}));
}

TEST_F(GeoTiffDirTest, FileSortOrder) {
  createWrapper(
      216,
      216,
      {},
      {{"FILE_SORT_ORDER_BY", "REGEX"}, {"FILE_SORT_REGEX", ".*w1[0-9]([0-9])"}});
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  // The regex should reverse the sort order.
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 1}),
            create_meta<double>(
                {216, 216, {1, 0, 0}}, -116.00055555555799, -115.00407921372494, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
            create_meta<double>(
                {216, 216, {0, 0, 0}}, -125.00055555629299, -124.00407921777249, false));
}

TEST_F(GeoTiffDirTest, MetadataPartial) {
  createWrapper(100, 100);
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  // First file
  // TODO(Misiu): Right now neighbours don't see across file boundaries.
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 0}),
            create_meta<double>(
                {100, 100, {0, 0, 0}}, -116.00055555555799, -115.5417129609465, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 8}),
            create_meta<double>(
                {16, 16, {0, 2, 2}}, -115.07360081896911, -115.00407921372494, false));
  // Second file
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 9}),
            create_meta<double>(
                {100, 100, {1, 0, 0}}, -125.00055555629299, -124.54171296320681, false));
  EXPECT_EQ(*meta_map.at(ChunkKey{db_id_, foreign_table_->tableId, 1, 17}),
            create_meta<double>(
                {16, 16, {1, 2, 2}}, -124.07360082278554, -124.00407921777249, false));
}

TEST_F(GeoTiffDirTest, Data) {
  createWrapper(216, 216);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);
  {
    auto& buffer = buffer_wrappers.at({db_id_, foreign_table_->tableId, 1, 0});
    ASSERT_EQ(buffer.size(), 46656 * sizeof(double));
    EXPECT_DOUBLE_EQ(reinterpret_cast<double*>(buffer.getMemoryPtr())[0],
                     -116.00055555555799);
    EXPECT_DOUBLE_EQ(reinterpret_cast<double*>(buffer.getMemoryPtr())[46655],
                     -115.00407921372494);
  }
  {
    auto& buffer = buffer_wrappers.at({db_id_, foreign_table_->tableId, 1, 1});
    ASSERT_EQ(buffer.size(), 46656 * sizeof(double));
    EXPECT_DOUBLE_EQ(reinterpret_cast<double*>(buffer.getMemoryPtr())[0],
                     -125.00055555629299);
    EXPECT_DOUBLE_EQ(reinterpret_cast<double*>(buffer.getMemoryPtr())[46655],
                     -124.00407921777249);
  }
  {
    auto& buffer = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 0});
    ASSERT_EQ(buffer.size(), 46656 * sizeof(float));
    EXPECT_FLOAT_EQ(reinterpret_cast<float*>(buffer.getMemoryPtr())[0], 318.570709);
    EXPECT_FLOAT_EQ(reinterpret_cast<float*>(buffer.getMemoryPtr())[46655],
                    1.17549435e-38);
  }
  {
    auto& buffer = buffer_wrappers.at({db_id_, foreign_table_->tableId, 3, 1});
    ASSERT_EQ(buffer.size(), 46656 * sizeof(float));
    EXPECT_FLOAT_EQ(reinterpret_cast<float*>(buffer.getMemoryPtr())[0], 0);
    EXPECT_FLOAT_EQ(reinterpret_cast<float*>(buffer.getMemoryPtr())[46655], 657.200684);
  }
}

class AppendUnitTest : public GeoTiffDirTest {
 public:
  void SetUp() override {
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directory(tmp_dir);
    std::filesystem::copy(raster_prefix + "geotif/USGS_13_n33w116.tif",
                          tmp_dir + "/file_1.tif");
    GeoTiffDirTest::SetUp();
  }

  void TearDown() override {
    GeoTiffDirTest::TearDown();
    std::filesystem::remove_all(tmp_dir);
  }

  class MockAppendRDW : public RasterDataWrapper {
   public:
    MockAppendRDW(const int db_id,
                  const ForeignTable* foreign_table,
                  const UserMapping* user_mapping)
        : RasterDataWrapper(db_id, foreign_table, user_mapping){};

    void clearWrapperData() override {
      if (throw_on_clear_) {
        throw std::runtime_error{"Wrapper clear unexpected."};
      } else {
        RasterDataWrapper::clearWrapperData();
      }
    }

    bool throw_on_clear_{false};
  };

  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    OptionsMap options;
    options[ADW::FILE_PATH_KEY] = tmp_dir;
    options[RDW::RASTER_WIDTH_KEY] = std::to_string(width);
    options[RDW::RASTER_HEIGHT_KEY] = std::to_string(height);
    // Refresh options are not used intesting, but required for a valid foreign table.
    options[FT::REFRESH_TIMING_TYPE_KEY] = FT::MANUAL_REFRESH_TIMING_TYPE;
    options[FT::REFRESH_UPDATE_TYPE_KEY] = FT::ALL_REFRESH_UPDATE_TYPE;

    for (auto& [key, val] : map) {
      options[key] = val;
    }

    const auto columns = createDefaultSchema();

    foreign_table_ = create_foreign_table();
    foreign_table_->populateOptionsMap(json_from_map(options));
    foreign_table_->foreign_server =
        cat_ptr_->getForeignServer(shared::kDefaultRasterServerName);

    cat_ptr_->createTable(*foreign_table_, columns, {}, true);

    user_mapping_ = nullptr;
    wrapper_ = std::make_unique<MockAppendRDW>(
        db_id_, foreign_table_.get(), user_mapping_.get());

    // These validation steps would usually happen in the catalog during table creation.
    wrapper_->validateServerOptions(foreign_table_->foreign_server);
    wrapper_->validateTableOptions(foreign_table_.get());
    wrapper_->validateSchema(columns, foreign_table_.get());
  }
};

TEST_F(AppendUnitTest, Default) {
  createWrapper(20, 40);
  auto meta_vec = populateChunkMetadata();
  EXPECT_EQ(meta_vec.size(), 198U);

  dynamic_cast<MockAppendRDW*>(wrapper_.get())->throw_on_clear_ = true;

  std::filesystem::copy(raster_prefix + "geotif/USGS_13_n41w125.tif",
                        tmp_dir + "/file_2.tif");
  ChunkMetadataVector meta_vec_after;
  EXPECT_THROW(wrapper_->populateChunkMetadata(meta_vec_after), std::runtime_error);
}

TEST_F(AppendUnitTest, Append) {
  createWrapper(20, 40, {}, {{"REFRESH_UPDATE_TYPE", "APPEND"}});
  auto meta_vec = populateChunkMetadata();
  EXPECT_EQ(meta_vec.size(), 198U);

  dynamic_cast<MockAppendRDW*>(wrapper_.get())->throw_on_clear_ = true;

  std::filesystem::copy(raster_prefix + "geotif/USGS_13_n41w125.tif",
                        tmp_dir + "/file_2.tif");
  auto meta_vec_after = populateChunkMetadata();

  EXPECT_EQ(meta_vec_after.size(), 396U);
}

TEST_F(AppendUnitTest, AppendNothing) {
  createWrapper(20, 40, {}, {{"REFRESH_UPDATE_TYPE", "APPEND"}});
  auto meta_vec = populateChunkMetadata();
  EXPECT_EQ(meta_vec.size(), 198U);

  dynamic_cast<MockAppendRDW*>(wrapper_.get())->throw_on_clear_ = true;

  auto meta_vec_after = populateChunkMetadata();
  EXPECT_EQ(meta_vec_after.size(), 198U);
}

class GripTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(grip_file_name, width, height, schema, map);
  }

  std::list<ColumnDescriptor> createDefaultSchema() const override {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kDOUBLE, db_id_));
    for (size_t i = 0; i < 49; ++i) {
      columns.emplace_back(
          ColumnDescriptor(0, 0, "band_" + std::to_string(i), kDOUBLE, db_id_));
    }
    return columns;
  }
};

TEST_F(GripTest, NumFragments20x40) {
  createWrapper(20, 20);
  auto meta_vec = populateChunkMetadata();
  ASSERT_EQ(51U, meta_vec.size());
}

TEST_F(GripTest, Data20x40) {
  createWrapper(20, 20);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);
  auto [size, first, last] =
      buffer_wrappers.getSizeFirstLast<double>({db_id_, foreign_table_->tableId, 10, 0});
  EXPECT_EQ(size, 3200U);
  EXPECT_DOUBLE_EQ(first, 4.0832653045654297);
  EXPECT_DOUBLE_EQ(last, 3.5207650661468506);
  auto [size_1, first_1, last_1] =
      buffer_wrappers.getSizeFirstLast<double>({db_id_, foreign_table_->tableId, 51, 0});
  EXPECT_EQ(size_1, 3200U);
  EXPECT_DOUBLE_EQ(first_1, 300.60000610351562);
  EXPECT_DOUBLE_EQ(last_1, 302.20001220703125);
}

class ZarrArchiveTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(zarr_archive, width, height, schema, map);
  }

  std::list<ColumnDescriptor> createDefaultSchema() const override {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kTINYINT, db_id_));
    return columns;
  }
};

class ZarrTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& schema = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(zarr_file_name, width, height, schema, map);
  }

  std::list<ColumnDescriptor> createDefaultSchema() const override {
    std::list<ColumnDescriptor> columns{};
    columns.emplace_back(ColumnDescriptor(0, 0, "x", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "y", kDOUBLE, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_1", kFLOAT, db_id_));
    columns.emplace_back(ColumnDescriptor(0, 0, "band_2", kTINYINT, db_id_));
    return columns;
  }
};

class SimpleTiffTest : public RasterTableUnitTest {
 public:
  void createWrapper(const size_t width,
                     const size_t height,
                     const std::list<ColumnDescriptor>& columns = {},
                     const OptionsMap& map = {{}}) override {
    RasterTableUnitTest::createWrapper(
        simple_tiff_file_name, width, height, columns, map);
  }
};

TEST_F(SimpleTiffTest, Metadata16x16) {
  createWrapper(16, 16);
  auto meta_vec = populateChunkMetadata();
  const auto meta_map = create_meta_map(meta_vec);
  ASSERT_EQ(48U, meta_map.size());  // 16 chunks per band, 3 bands.
  auto placeholder_meta = create_simple_placeholder_meta(db_id_, foreign_table_->tableId);
  for (const auto& [key, val] : meta_map) {
    EXPECT_EQ(*meta_map.at(key), *placeholder_meta.at(key)) << show_chunk(key);
  }
}

TEST_F(SimpleTiffTest, Data16x16) {
  createWrapper(16, 16);
  auto meta_vec = populateChunkMetadata();
  FragmentBuffers buffer_wrappers(meta_vec);
  wrapper_->populateChunkBuffers(buffer_wrappers.buffers, {}, nullptr);
  const auto meta_map = buffer_wrappers.getMetadata();
  ASSERT_EQ(48U, meta_map.size());  // 16 chunks per band, 3 bands.
  auto simple_meta = create_simple_meta(db_id_, foreign_table_->tableId);
  for (const auto& [key, val] : meta_map) {
    EXPECT_EQ(*meta_map.at(key), *simple_meta.at(key)) << show_chunk(key);
  }
}

// Base class that contains helper functions.
class RasterIntegrationTest : public DBHandlerTestFixture {
 public:
  inline static const std::string raster_table{"raster_table"};

  std::string createRasterForeignTable(
      const std::map<std::string, std::string>& override_options) const {
    // Default table settings
    std::map<std::string, std::string> opts = {
        {"table_name", raster_table},
        {WIDTH, "20"},
        {HEIGHT, "40"},
        {"schema", "(x DOUBLE, y DOUBLE, band_1 INTEGER)"},
        {"file_name", small_tiff_file_name},
        {"server", "default_local_raster"},
        {"extra_options", ""}};

    for (const auto& [key, val] : override_options) {
      opts[key] = val;
    }

    std::stringstream ss;
    ss << "CREATE FOREIGN TABLE " << opts["table_name"] << " " << opts["schema"]
       << " SERVER " << opts["server"] << " WITH ("
       << "file_path = '" << opts["file_name"] << "'"
       << (opts[WIDTH].size() > 0 ? (", " + WIDTH + " = " + opts[WIDTH]) : "")
       << (opts[HEIGHT].size() > 0 ? (", " + HEIGHT + " = " + opts[HEIGHT]) : "")
       << (opts["extra_options"].size() > 0 ? (", " + opts["extra_options"]) : "") << ")";
    return ss.str();
  }

  std::string createRasterTable(
      const std::map<std::string, std::string>& override_options) const {
    std::map<std::string, std::string> opts = {
        {"table_name", raster_table}, {"schema", "(x DOUBLE, y DOUBLE, band_1 INTEGER)"}};

    for (const auto& [key, val] : override_options) {
      opts[key] = val;
    }

    std::stringstream ss;
    ss << "CREATE TABLE " << opts["table_name"] << " " << opts["schema"];
    return ss.str();
  }

  std::string populateRasterTable(
      const std::map<std::string, std::string>& override_options) const {
    // Default table settings
    std::map<std::string, std::string> opts = {{"table_name", raster_table},
                                               {WIDTH, "20"},
                                               {HEIGHT, "40"},
                                               {"file_name", small_tiff_file_name},
                                               {"server", "default_local_raster"},
                                               {"extra_options", ""}};

    for (const auto& [key, val] : override_options) {
      opts[key] = val;
    }

    std::stringstream ss;
    ss << "COPY " + opts["table_name"] + " FROM '" + opts["file_name"] +
              "' WITH (source_type='raster_file'"
       << (opts[WIDTH].size() > 0 ? (", " + WIDTH + " = " + opts[WIDTH]) : "")
       << (opts[HEIGHT].size() > 0 ? (", " + HEIGHT + " = " + opts[HEIGHT]) : "")
       << (opts["extra_options"].size() > 0 ? (", " + opts["extra_options"]) : "") << ")";
    return ss.str();
  }
};

// Parameterized test that will test Raster Foreign Table and Foreign Table for same
// results.
class ImportHCCompareTest : public RasterIntegrationTest,
                            public ::testing::WithParamInterface<std::string> {
 public:
  void TearDown() override {
    DBHandlerTestFixture::SetUp();
    dropParamTypeTable();
  }

  void createParamTypeTable(const std::map<std::string, std::string>& opts = {}) const {
    if (const auto& test_type = GetParam(); test_type == "HeavyConnect") {
      sql(createRasterForeignTable(opts));
    } else if (test_type == "Import") {
      sql(createRasterTable(opts));
      sql(populateRasterTable(opts));
    } else {
      UNREACHABLE();
    }
  }

  void dropParamTypeTable(const std::string& table_name = raster_table) const {
    if (const auto& test_type = GetParam(); test_type == "HeavyConnect") {
      sql("DROP FOREIGN TABLE IF EXISTS " + table_name);
    } else if (test_type == "Import") {
      sql("DROP TABLE IF EXISTS " + table_name);
    } else {
      UNREACHABLE();
    }
  }
};

TEST_P(ImportHCCompareTest, InvalidHeight) {
  executeLambdaAndAssertException(
      [this] {
        createParamTypeTable({{WIDTH, "20"}, {HEIGHT, "0"}});
      },
      "Table '" + raster_table + "' with " + HEIGHT + "='0': " + HEIGHT +
          " must be an integer value greater than zero");
}

TEST_P(ImportHCCompareTest, InvalidWidth) {
  executeLambdaAndAssertException(
      [this] {
        createParamTypeTable({{WIDTH, "0"}, {HEIGHT, "40"}});
      },
      "Table '" + raster_table + "' with " + WIDTH + "='0': " + WIDTH +
          " must be an integer value greater than zero");
}

TEST_P(ImportHCCompareTest, Auto) {
  createParamTypeTable({{WIDTH, ""}, {HEIGHT, ""}});
  sqlAndCompareResult("select count(*) from " + raster_table, {{i(40000)}});
}

// TODO(Misiu): Look into what the correct error handling for these test are for import.
TEST_P(ImportHCCompareTest, FailWithBadFile) {
  if (GetParam() == "Import") {
    GTEST_SKIP() << "Currently unsupported for import";
  }

  createParamTypeTable(
      {{"file_name",
        raster_prefix + "USGS_1m_x30y441_OH_Columbus_2019_small_truncated.tif"},
       {"schema", "(x DOUBLE, y DOUBLE, band_1 FLOAT)"}});
  queryAndAssertPartialException("select avg(band_1) from " + raster_table,
                                 "Failed to read raster pixels");
}

TEST_P(ImportHCCompareTest, FailDropWithAllNull) {
  if (GetParam() == "Import") {
    GTEST_SKIP() << "Currently unsupported for import";
  }

  executeLambdaAndAssertException(
      [this] {
        createParamTypeTable(
            {{"file_name",
              raster_prefix + "USGS_1m_x30y441_OH_Columbus_2019_small_truncated.tif"},
             {"schema", "(x DOUBLE, y DOUBLE, band_1 FLOAT)"},
             {"extra_options", "raster_drop_if_all_null=true"}});
      },
      "Invalid foreign table option \"RASTER_DROP_IF_ALL_NULL\".");
}

TEST_P(ImportHCCompareTest, FailMaxReject) {
  if (GetParam() == "Import") {
    GTEST_SKIP() << "Currently unsupported for import";
  }

  executeLambdaAndAssertException(
      [this] {
        createParamTypeTable(
            {{WIDTH, "20"},
             {HEIGHT, "0"},
             {"file_name",
              raster_prefix + "USGS_1m_x30y441_OH_Columbus_2019_small_truncated.tif"},
             {"schema", "(x DOUBLE, y DOUBLE, band_1 FLOAT)"},
             {"extra_options", "max_reject=1"}});
      },
      "Invalid foreign table option \"MAX_REJECT\".");
}

TEST_P(ImportHCCompareTest, OnlyHeightSet) {
  executeLambdaAndAssertException(
      [this] {
        createParamTypeTable({{WIDTH, ""}, {HEIGHT, "10"}});
      },
      "RASTER_TILE_WIDTH and RASTER_TILE_HEIGHT must both be set or unset.");
}

TEST_P(ImportHCCompareTest, OnlyWidthSet) {
  executeLambdaAndAssertException(
      [this] {
        createParamTypeTable({{WIDTH, "10"}, {HEIGHT, ""}});
      },
      "RASTER_TILE_WIDTH and RASTER_TILE_HEIGHT must both be set or unset.");
}

TEST_P(ImportHCCompareTest, FragmentSize) {
  createParamTypeTable({{WIDTH, ""}, {HEIGHT, ""}});
  sqlAndCompareResult("select count(*) from " + raster_table, {{i(40000)}});
  auto& cat = getCatalog();
  auto td = cat.getMetadataForTable(raster_table, false);
  // File block size is 200x20, so 4,000 pixels.  This leaves 10 fragments for 40,000
  // pixels.
  EXPECT_EQ(td->fragmenter->getNumFragments(), 10U);
}

TEST_P(ImportHCCompareTest, RasterTileMetadata) {
  createParamTypeTable({{WIDTH, "150"}, {HEIGHT, "170"}});
  // Three columns, with one full and 3 edge tiles each.
  std::vector<Fragmenter_Namespace::RasterMeshRenderingMetadata> expected{
      {150, 170, 0, {-1, 1, -1, 2}},
      {50, 170, 1, {0, -1, -1, 3}},
      {150, 30, 2, {-1, 3, 0, -1}},
      {50, 30, 3, {2, -1, 1, -1}}};

  auto cat = &getCatalog();
  auto td = cat->getMetadataForTable(raster_table, true);
  auto tile_vec =
      dynamic_cast<Fragmenter_Namespace::RasterFragmenter*>(td->fragmenter.get())
          ->computeRasterMeshRenderingMetadata();
  EXPECT_EQ(tile_vec, expected);
}

TEST_P(ImportHCCompareTest, RasterTileMetadataMultiFile) {
  createParamTypeTable({{WIDTH, "150"}, {HEIGHT, "170"}});
  // Three columns, with one full and 3 edge tiles each.
  std::vector<Fragmenter_Namespace::RasterMeshRenderingMetadata> expected{
      {150, 170, 0, {-1, 1, -1, 2}},
      {50, 170, 1, {0, -1, -1, 3}},
      {150, 30, 2, {-1, 3, 0, -1}},
      {50, 30, 3, {2, -1, 1, -1}}};

  auto cat = &getCatalog();
  auto td = cat->getMetadataForTable(raster_table, true);
  auto tile_vec =
      dynamic_cast<Fragmenter_Namespace::RasterFragmenter*>(td->fragmenter.get())
          ->computeRasterMeshRenderingMetadata();
  EXPECT_EQ(tile_vec, expected);
}

// This test verifies that we have no way of identifying a raster table in a dumped
// format (it gets restored as a regular table with an InsertOrderFragmenter).  This is
// not the behaviour we would like, and the behaviour should be updated when we implement
// raster table as a proper table type.
TEST_P(ImportHCCompareTest, DISABLED_LegacyMetadataDumpRestore) {
  if (GetParam() == "HeavyConnect") {
    GTEST_SKIP() << "Test only applies to import";
  }
  sql("RESTORE TABLE " + raster_table + " FROM '" + binary_path +
      "/../../Tests/Export/TableDump/legacy_raster.gz'");
  sqlAndCompareResult("select count(*) from " + raster_table, {{i(40000)}});

  std::vector<Fragmenter_Namespace::RasterMeshRenderingMetadata> expected{
      {150, 170, 0, {-1, 1, -1, 2}},
      {50, 170, 1, {0, -1, -1, 3}},
      {150, 30, 2, {-1, 3, 0, -1}},
      {50, 30, 3, {2, -1, 1, -1}}};

  auto cat = &getCatalog();
  auto td = cat->getMetadataForTable(raster_table, true);
  auto raster_fragmenter =
      dynamic_cast<Fragmenter_Namespace::RasterFragmenter*>(td->fragmenter.get());
  ASSERT_NE(raster_fragmenter, nullptr);
}

INSTANTIATE_TEST_SUITE_P(ImportHCCompareTest,
                         ImportHCCompareTest,
                         ::testing::Values("HeavyConnect", "Import"),
                         [](const auto& info) { return info.param; });

// Adds additional work to create/initialize a table.
class InitializedImportHCCompareTest : public ImportHCCompareTest {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    createParamTypeTable();
  }
};

TEST_P(InitializedImportHCCompareTest, Metadata) {
  sqlAndCompareResult("select count(*) from " + raster_table, {{i(40000)}});
  sqlAndCompareResult("select count(*) from " + raster_table + " where band_1 != 0",
                      {{i(1000)}});
  sqlAndCompareResult(
      "select count(*) from " + raster_table + " where x > 45.015 and x < 45.02",
      {{i(5267)}});
}

TEST_P(InitializedImportHCCompareTest, Data) {
  sqlAndCompareResult(
      "select count(*) from " + raster_table + " where x > 45.01999 and x < 45.02",
      {{i(10)}});
  sqlAndCompareResult("select SUM(x), SUM(y), SUM(band_1) from " + raster_table +
                          " where x > 45.01999 and x < 45.02",
                      {{450.19994820719717, 626.38816935383261, 0.0}});
}

TEST_P(InitializedImportHCCompareTest, Import) {
  // Check to make sure import and foreign table fragments are in the same order.
  sqlAndCompareResult("select avg(x) from " + raster_table + " where rowid < 800",
                      {{45.030666333564547}});
  sqlAndCompareResult(
      "select avg(x) from " + raster_table + " where rowid > 799 AND rowid < 1600",
      {{45.026867884247686}});
}

INSTANTIATE_TEST_SUITE_P(InitializedImportHCCompareTest,
                         InitializedImportHCCompareTest,
                         ::testing::Values("HeavyConnect", "Import"),
                         [](const auto& info) { return info.param; });

#ifdef HAVE_AWS_S3
class S3IntegrationTest : public DBHandlerTestFixture {
 public:
  inline static const std::string private_server{"private_s3_raster_server"};
  inline static const std::string public_server{"public_s3_raster_server"};

  static void SetUpTestSuite() {
    g_allow_s3_server_privileges = true;
    DBHandlerTestFixture::SetUpTestSuite();
    sql("DROP SERVER IF EXISTS " + public_server);
    sql("DROP SERVER IF EXISTS " + private_server);
    sql(createServer(public_server, "omnisci-fsi-test-public/FsiDataFiles/"));
    sql(createServer(private_server, "omnisci-fsi-test/FsiDataFiles/"));
  }

  static void TearDownTestSuite() {
    sql("DROP SERVER IF EXISTS " + public_server);
    sql("DROP SERVER IF EXISTS " + private_server);
    g_allow_s3_server_privileges = false;
    DBHandlerTestFixture::TearDownTestSuite();
  }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("DROP FOREIGN TABLE IF EXISTS raster_ft");
    sql("DROP TABLE IF EXISTS raster_table");
    sql("DROP USER MAPPING IF EXISTS FOR PUBLIC SERVER " + private_server + ";");
  }

  void TearDown() override {
    sql("DROP FOREIGN TABLE IF EXISTS raster_ft");
    sql("DROP TABLE IF EXISTS raster_table");
    sql("DROP USER MAPPING IF EXISTS FOR PUBLIC SERVER " + private_server + ";");
    DBHandlerTestFixture::TearDown();
  }

  static bool insufficientPrivateCredentials() {
    return !is_valid_aws_key(get_aws_keys_from_env());
  }

  static std::string createServer(const std::string& name, const std::string& bucket) {
    std::stringstream ss;
    ss << "CREATE SERVER " << name << " FOREIGN DATA WRAPPER raster_file "
       << "WITH (storage_type = 'AWS_S3', s3_bucket = '" << bucket
       << "', AWS_REGION = 'us-west-1');";
    return ss.str();
  }

  static void createForeignTable(const std::string& server,
                                 const std::string& file_name = "s1b_small.tiff",
                                 const std::string& band_type = "INT",
                                 const std::string& options = "") {
    std::stringstream ss;
    ss << "CREATE FOREIGN TABLE raster_ft (x DOUBLE, y DOUBLE, band_1 " << band_type
       << ") SERVER " << server << " WITH (file_path = '" << file_name << "'";
    if (options != "") {
      ss << ", " << options;
    }
    ss << ")";
    sql(ss.str());
  }

  static void createUserMappingForS3(const std::string& server_name,
                                     const std::string& access_key,
                                     const std::string& secret_key,
                                     const std::string session_token = "") {
    sql("CREATE USER MAPPING FOR PUBLIC SERVER " + server_name +
        " WITH (s3_access_key='" + access_key + "', s3_secret_key='" + secret_key +
        (session_token.empty() ? "" : "', s3_session_token='" + session_token) + "');");
  }

  static void createTable() {
    sql("create table raster_table (x DOUBLE, y DOUBLE, band_1 INTEGER)");
  }

  static void importTableS3() {
    sql("COPY raster_table FROM "
        "'s3://omnisci-fsi-test-public/FsiDataFiles/s1b_small.tiff' WITH "
        "(source_type='raster_file', s3_region='us-west-1');");
  }
};

// TODO(IAM): re-enable once a CI IAM user with read on the raster
// fixtures in the public bucket is provisioned. Currently AccessDenied.
TEST_F(S3IntegrationTest, DISABLED_Public) {
  createForeignTable(public_server);
  sqlAndCompareResult("select count(*) from raster_ft;", {{i(40000)}});
}

TEST_F(S3IntegrationTest, Private) {
  if (insufficientPrivateCredentials()) {
    GTEST_SKIP() << "Insufficient private credentials to run test";
  }
  const auto& env_key = get_aws_keys_from_env();
  createUserMappingForS3(private_server, env_key.first, env_key.second);
  createForeignTable(private_server);
  sqlAndCompareResult("select count(*) from raster_ft;", {{i(40000)}});
}

TEST_F(S3IntegrationTest, STS) {
  if (insufficientPrivateCredentials()) {
    GTEST_SKIP() << "Insufficient private credentials to run test";
  }
  const auto& env_key = get_aws_keys_from_env();
  const auto& server = getCatalog().getForeignServer(private_server);
  const auto& sts_credentials = generate_sts_credentials(env_key, server);
  createUserMappingForS3(private_server,
                         sts_credentials.GetAccessKeyId(),
                         sts_credentials.GetSecretAccessKey(),
                         sts_credentials.GetSessionToken());
  createForeignTable(private_server);
  sqlAndCompareResult("select count(*) from raster_ft;", {{i(40000)}});
}

TEST_F(S3IntegrationTest, Import) {
  createTable();
  importTableS3();
  sqlAndCompareResult("select count(*) from raster_table", {{i(40000)}});
}

// TODO(IAM): re-enable once a CI IAM user with read on the raster
// fixtures in the public bucket is provisioned. Currently AccessDenied.
TEST_F(S3IntegrationTest, DISABLED_MultipleFiles) {
  createForeignTable(public_server, "geotif", "FLOAT");
  sqlAndCompareResult("select count(*) from raster_ft;", {{i(93312)}});
}

// TODO(IAM): re-enable once a CI IAM user with read on the raster
// fixtures in the public bucket is provisioned. Currently AccessDenied.
TEST_F(S3IntegrationTest, DISABLED_PathFilter) {
  createForeignTable(public_server, "geotif", "FLOAT", "REGEX_PATH_FILTER='.*n33.*'");
  sqlAndCompareResult("select count(*) from raster_ft;", {{i(46656)}});
}

// TODO(IAM): re-enable once a CI IAM user with read on the raster
// fixtures in the public bucket is provisioned. Currently AccessDenied.
TEST_F(S3IntegrationTest, DISABLED_FileSortOrder) {
  createForeignTable(public_server,
                     "geotif",
                     "FLOAT",
                     "FILE_SORT_ORDER_BY='REGEX', FILE_SORT_REGEX='.*w1[0-9]([0-9])'");
  // This isnt' really testing anything beyond that the option does not error out.
  sqlAndCompareResult("select count(*) from raster_ft;", {{i(93312)}});
}
#endif  // HAVE_AWS_S3

class RefreshTest : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    std::filesystem::remove_all(tmp_dir);
    std::filesystem::create_directory(tmp_dir);
    std::filesystem::copy(geo_tiff_dir + "/USGS_13_n33w116.tif", tmp_dir);
    sql("drop foreign table if exists raster_ft");
  }

  void TearDown() override {
    sql("drop foreign table if exists raster_ft");
    std::filesystem::remove_all(tmp_dir);
    DBHandlerTestFixture::TearDown();
  }
};

TEST_F(RefreshTest, Append) {
  sql("create foreign table raster_ft (x double, y double, band_1 float) server "
      "default_local_raster with (file_path='" +
      tmp_dir + "', " + RDW::RASTER_WIDTH_KEY + "=216, " + RDW::RASTER_HEIGHT_KEY +
      "=216, "
      "refresh_update_type='append', refresh_timing_type='manual')");
  sqlAndCompareResult("select count(*) from raster_ft", {{i(46656)}});
  std::filesystem::copy(geo_tiff_dir + "/USGS_13_n41w125.tif", tmp_dir);
  sql("refresh foreign tables raster_ft");
  sqlAndCompareResult("select count(*) from raster_ft", {{i(93312)}});
}

TEST_F(RefreshTest, AppendNoReread) {
  sql("create foreign table raster_ft (x double, y double, band_1 float) server "
      "default_local_raster with (file_path='" +
      tmp_dir + "', " + RDW::RASTER_WIDTH_KEY + "=216, " + RDW::RASTER_HEIGHT_KEY +
      "=216, "
      "refresh_update_type='append', refresh_timing_type='manual')");
  sqlAndCompareResult("select count(*) from raster_ft", {{i(46656)}});
  std::filesystem::copy(geo_tiff_dir + "/USGS_13_n41w125.tif", tmp_dir);
  std::filesystem::remove_all(tmp_dir + "/USGS_13_n33w116.tif");
  sql("refresh foreign tables raster_ft");
  // Since we are doing an append refresh, the deleted file should stay in the cache
  // unchanged.
  sqlAndCompareResult("select count(*) from raster_ft", {{i(93312)}});
}

TEST_F(RefreshTest, RefreshDir) {
  sql("create foreign table raster_ft (x double, y double, band_1 float) server "
      "default_local_raster with (file_path='" +
      tmp_dir + "', " + RDW::RASTER_WIDTH_KEY + "=216, " + RDW::RASTER_HEIGHT_KEY +
      "=216, "
      "refresh_update_type='all', refresh_timing_type='manual')");
  std::filesystem::copy(geo_tiff_dir + "/USGS_13_n33w116.tif",
                        tmp_dir + "/USGS_13_n33w117.tif");
  sqlAndCompareResult("select count(*) from raster_ft", {{i(93312)}});

  std::filesystem::copy(geo_tiff_dir + "/USGS_13_n41w125.tif",
                        tmp_dir + "/USGS_13_n41w125.tif");
  std::filesystem::copy(geo_tiff_dir + "/USGS_13_n41w125.tif",
                        tmp_dir + "/USGS_13_n41w126.tif");

  // Count should not have changed pre-refresh
  sqlAndCompareResult("select count(*) from raster_ft", {{i(93312)}});
  sql("refresh foreign tables raster_ft with (evict='true')");
  // Count has changed post-refresh.
  sqlAndCompareResult("select count(*) from raster_ft", {{i(93312 * 2)}});
}

TEST_F(RefreshTest, RefreshDirBoundingBox) {
  sql("create foreign table raster_ft (x double, y double, band_1 float) server "
      "default_local_raster with (file_path='" +
      tmp_dir + "', " + RDW::RASTER_WIDTH_KEY + "=216, " + RDW::RASTER_HEIGHT_KEY +
      "=216, bounding_box_clip='-116,32,-115,33', "
      "refresh_update_type='all', refresh_timing_type='manual')");
  std::filesystem::copy(geo_tiff_dir + "/USGS_13_n41w125.tif",
                        tmp_dir + "/USGS_13_n41w125.tif");
  sqlAndCompareResult("select count(*) from raster_ft", {{i(46656)}});

  std::filesystem::copy(geo_tiff_dir + "/USGS_13_n33w116.tif",
                        tmp_dir + "/USGS_13_n33w117.tif");
  std::filesystem::copy(geo_tiff_dir + "/USGS_13_n41w125.tif",
                        tmp_dir + "/USGS_13_n41w126.tif");
  sql("refresh foreign tables raster_ft with (evict='true')");
  sqlAndCompareResult("select count(*) from raster_ft", {{i(93312)}});
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);

  g_raster_logging = true;

  testing::InitGoogleTest(&argc, argv);
  PkiEncryptor::setKeyStorePath("../../Tests/Encryption/ValidCert/");

  binary_path = std::filesystem::canonical(argv[0]).parent_path().string();
  tmp_dir = binary_path + "/tmp_dir";
  raster_data_dir = binary_path + "/" + raster_prefix;
  // 200x200, 1 band
  small_tiff_file_name = raster_data_dir + "s1b_small.tiff";

  // 320x225, 3 bands
  png_file_name = raster_prefix + "beach.png";

  // 200x200, 1 band.
  geo_tiff_file_name = raster_prefix + "USGS_1m_x30y441_OH_Columbus_2019_small.tif";

  // 200x200, 1 band.  Last pixel is null.
  geo_tiff_null_file_name =
      raster_prefix + "USGS_1m_x30y441_OH_Columbus_2019_small_last_pixel_null.tif";

  // 2 bands:
  // band_1_1: size: 360x180, block size: 360x1, type: float32_t
  // band_2_1: size: 256x3,   block size: 256x1, type: uint8_t
  // Notes: This file uses inconsistent band dimensions and can't be imported into a
  // single file.
  hdf5_file_name = raster_prefix + "Q2012034.L3m_DAY_SCI_V5.0_SSS_1deg.hdf5";

  // 2 files:
  // Found Band 'Layer_1', with dimensions 216x216, block size 216x9, float
  // Found Band 'band_1_1', with dimensions 216x216, block size 216x9, float
  // Notes: These files are interleaved, so we are reading both files into one set of
  // columns.
  geo_tiff_dir = raster_prefix + "geotif";

  // 49 bands:
  // 'Maximum___Composite_radar_reflectivity__dB_', 20x20, block size 20x1, type: double
  // 'Echo_Top__m_', 20x20, block size 20x1, type: double
  // '_prodType_0__cat_16__subcat_201_____', 20x20, block size 20x1, type: double
  // 'Vertically_integrated_liquid__kg_m_', 20x20, block size 20x1, type: double
  // 'Visibility__m_', 20x20, block size 20x1, type: double
  // 'Derived_radar_reflectivity__dB_', 20x20, block size 20x1, type: double
  // 'Derived_radar_reflectivity__dB__2', 20x20, block size 20x1, type: double
  // 'Wind_speed__gust___m_s_', 20x20, block size 20x1, type: double
  // 'Updraft_Helicity__m_2_s_2_', 20x20, block size 20x1, type: double
  // 'u_component_of_wind__m_s_', 20x20, block size 20x1, type: double
  // 'v_component_of_wind__m_s_', 20x20, block size 20x1, type: double
  // 'Pressure__Pa_', 20x20, block size 20x1, type: double
  // 'Geopotential_height__gpm_', 20x20, block size 20x1, type: double
  // 'Temperature__C_', 20x20, block size 20x1, type: double
  // 'Specific_humidity__kg_kg_', 20x20, block size 20x1, type: double
  // 'Dew_point_temperature__C_', 20x20, block size 20x1, type: double
  // 'u_component_of_wind__m_s__2', 20x20, block size 20x1, type: double
  // 'v_component_of_wind__m_s__2', 20x20, block size 20x1, type: double
  // 'Wind_speed__m_s_', 20x20, block size 20x1, type: double
  // 'u_component_of_wind__m_s__3', 20x20, block size 20x1, type: double
  // 'v_component_of_wind__m_s__3', 20x20, block size 20x1, type: double
  // 'Downward_short_wave_radiation_flux__W__m_2__', 20x20, block size 20x1, type: double
  // 'Visible_Beam_Downward_Solar_Flux__W__m_2__', 20x20, block size 20x1, type: double
  // 'Percent_frozen_precipitation____', 20x20, block size 20x1, type: double
  // 'Precipitation_rate__kg__m_2_s__', 20x20, block size 20x1, type: double
  // 'Total_precipitation__kg__m_2__', 20x20, block size 20x1, type: double
  // 'Water_equivalent_of_accumulated_snow_depth__kg__m_2__', 20x20, block size 20x1,
  // type: double 'Frozen_Rain__kg__m_2__', 20x20, block size 20x1, type: double
  // 'Categorical_snow__0_no__1_yes_', 20x20, block size 20x1, type: double
  // 'Categorical_ice_pellets__0_no__1_yes_', 20x20, block size 20x1, type: double
  // 'Categorical_freezing_rain__0_no__1_yes_', 20x20, block size 20x1, type: double
  // 'Categorical_rain__0_no__1_yes_', 20x20, block size 20x1, type: double
  // 'Total_column_integrated_cloud_water__kg__m_2__', 20x20, block size 20x1, type:
  // double 'Total_column_integrated_cloud_ice__kg__m_2__', 20x20, block size 20x1, type:
  // double 'Geopotential_height__gpm__2', 20x20, block size 20x1, type: double
  // 'Geopotential_height__gpm__3', 20x20, block size 20x1, type: double
  // 'Geopotential_height__gpm__4', 20x20, block size 20x1, type: double
  // 'Upward_long_wave_radiation_flux__W__m_2__', 20x20, block size 20x1, type: double
  // 'Downward_short_wave_radiation_flux__W__m_2___2', 20x20, block size 20x1, type:
  // double 'Downward_long_wave_radiation_flux__W__m_2__', 20x20, block size 20x1, type:
  // double 'Upward_short_wave_radiation_flux__W__m_2__', 20x20, block size 20x1, type:
  // double 'Upward_long_wave_radiation_flux__W__m_2___2', 20x20, block size 20x1, type:
  // double 'Visible_Beam_Downward_Solar_Flux__W__m_2___2', 20x20, block size 20x1, type:
  // double 'Visible_Diffuse_Downward_Solar_Flux__W__m_2__', 20x20, block size 20x1, type:
  // double 'Upward_short_wave_radiation_flux__W__m_2___2', 20x20, block size 20x1, type:
  // double 'Simulated_Brightness_Temperature_for_GOES_12__Channel_3__C_', 20x20, block
  // size 20x1, type: double
  // 'Simulated_Brightness_Temperature_for_GOES_12__Channel_4__C_', 20x20, block size
  // 20x1, type: double 'Simulated_Brightness_Temperature_for_GOES_11__Channel_3__C_',
  // 20x20, block size 20x1, type: double
  // 'Simulated_Brightness_Temperature_for_GOES_11__Channel_4__C_', 20x20, block size
  // 20x1, type: double
  grip_file_name = raster_prefix + "hrrr.t00z.wrfsubhf00_small.grib2";

  // Not sure about this archive.  Seems to be commented out in other tests?
  zarr_archive = raster_prefix + "small.zarr.tgz";
  // File does not exist.
  zarr_file_name = raster_prefix + "small.zarr";

  // 64x64, 16x16 blocksize, 1 band.  Each block is composed of an artificial value (block
  // 0 is all '0's, block 1 is all '1's, etc...)
  simple_tiff_file_name = raster_data_dir + "Simple/simple.tif";

  remote_tiff_file_name = "omnisci-fsi-test-public/FsiDataFiles/s1b_small.tiff";

  g_enable_legacy_raster_import = false;  // Use new import for comparison with HC.
  g_raster_logging = true;
  g_enable_s3_fsi = true;

#ifdef HAVE_AWS_S3
  heavydb_aws_sdk::init_sdk();
#endif

  Geospatial::GDAL::init();

  int err = 0;
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
