/*
 * Copyright 2020, OmniSci, Inc.
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

#include "QueryRunner/QueryRunner.h"

#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <string>

#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/Execute.h"
#include "Shared/scope.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

bool skip_tests_on_gpu(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests_on_gpu(dt)) {                               \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

template <typename TEST_BODY>
void executeAllScenarios(TEST_BODY fn) {
  for (const auto overlaps_state : {true, false}) {
    const auto enable_overlaps_hashjoin_state = g_enable_overlaps_hashjoin;
    const auto enable_hashjoin_many_to_many_state = g_enable_hashjoin_many_to_many;

    g_enable_overlaps_hashjoin = overlaps_state;
    g_enable_hashjoin_many_to_many = overlaps_state;
    g_trivial_loop_join_threshold = overlaps_state ? 1 : 1000;

    ScopeGuard reset_overlaps_state = [&enable_overlaps_hashjoin_state,
                                       &enable_hashjoin_many_to_many_state] {
      g_enable_overlaps_hashjoin = enable_overlaps_hashjoin_state;
      g_enable_overlaps_hashjoin = enable_hashjoin_many_to_many_state;
      g_trivial_loop_join_threshold = 1000;
    };

    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      fn(dt);
    }
  }
}

// clang-format off
const auto cleanup_stmts = {
    R"(drop table if exists does_intersect_a;)",
    R"(drop table if exists does_intersect_b;)",
    R"(drop table if exists does_not_intersect_a;)",
    R"(drop table if exists does_not_intersect_b;)",
    R"(drop table if exists empty_table;)"
};

// clang-format off
const auto init_stmts_ddl = {
    R"(create table does_intersect_a (id int,
                                      poly geometry(polygon, 4326),
                                      mpoly geometry(multipolygon, 4326),
                                      pt geometry(point, 4326));
    )",
    R"(create table does_intersect_b (id int,
                                      poly geometry(polygon, 4326),
                                      mpoly geometry(multipolygon, 4326),
                                      pt geometry(point, 4326));
    )",
    R"(create table does_not_intersect_a (id int,
                                        poly geometry(polygon, 4326),
                                        mpoly geometry(multipolygon, 4326),
                                        pt geometry(point, 4326));
    )",
    R"(create table does_not_intersect_b (id int,
                                        poly geometry(polygon, 4326),
                                        mpoly geometry(multipolygon, 4326),
                                        pt geometry(point, 4326));
    )",
    R"(create table empty_table (id int,
                           poly geometry(polygon, 4326),
                           mpoly geometry(multipolygon, 4326),
                           pt geometry(point, 4326));
    )"
};

const auto init_stmts_dml = {
    R"(insert into does_intersect_a
       values (0,
              'polygon((25 25,30 25,30 30,25 30,25 25))',
              'multipolygon(((25 25,30 25,30 30,25 30,25 25)))',
              'point(22 22)');
    )",
    R"(insert into does_intersect_a 
       values (1,
              'polygon((2 2,10 2,10 10,2 10,2 2))',
              'multipolygon(((2 2,10 2,10 10,2 10,2 2)))',
              'point(8 8)');
    )",
    R"(insert into does_intersect_a
       values (2,
              'polygon((2 2,10 2,10 10,2 10,2 2))',
              'multipolygon(((2 2,10 2,10 10,2 10,2 2)))',
              'point(8 8)');
    )",
    R"(insert into does_intersect_b
       values (0,
              'polygon((0 0,30 0,30 0,30 30,0 0))',
              'multipolygon(((0 0,30 0,30 0,30 30,0 0)))',
              'point(8 8)');
    )",
    R"(insert into does_intersect_b
       values (1,
              'polygon((25 25,30 25,30 30,25 30,25 25))',
              'multipolygon(((25 25,30 25,30 30,25 30,25 25)))',
              'point(28 28)');
    )",
    R"(insert into does_not_intersect_a
       values (1,
              'polygon((0 0,0 1,1 0,1 1,0 0))',
              'multipolygon(((0 0,0 1,1 0,1 1,0 0)))',
              'point(0 0)');
    )",
    R"(insert into does_not_intersect_a
       values (1,
              'polygon((0 0,0 1,1 0,1 1,0 0))',
              'multipolygon(((0 0,0 1,1 0,1 1,0 0)))',
              'point(0 0)');
    )",
    R"(insert into does_not_intersect_a
       values (1,
              'polygon((0 0,0 1,1 0,1 1,0 0))',
              'multipolygon(((0 0,0 1,1 0,1 1,0 0)))',
              'point(0 0)');
    )",
    R"(insert into does_not_intersect_b
       values (1,
              'polygon((2 2,2 4,4 2,4 4,2 2))',
              'multipolygon(((2 2,2 4,4 2,4 4,2 2)))',
              'point(2 2)');
    )"
};
// clang-format on

class OverlapsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    for (const auto& stmt : cleanup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : init_stmts_ddl) {
      QR::get()->runDDLStatement(stmt);
    }

    for (const auto& stmt : init_stmts_dml) {
      QR::get()->runSQL(stmt, ExecutorDeviceType::CPU);
    }
  }

  static void TearDownTestSuite() {
    for (const auto& stmt : cleanup_stmts) {
      QR::get()->runDDLStatement(stmt);
    }
  }
};

TargetValue execSQL(const std::string& stmt,
                    const ExecutorDeviceType dt,
                    const bool geo_return_geo_tv = true) {
  auto rows = QR::get()->runSQL(stmt, dt, true, false);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << stmt;
  return crt_row[0];
}

TEST_F(OverlapsTest, SimplePointInPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    const auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_a WHERE ST_Intersects(poly, pt);";
    ASSERT_EQ(static_cast<int64_t>(2), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, InnerJoinPointInPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON ST_Intersects(a.poly, "
        "b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));

    sql =
        "SELECT count(*) from does_intersect_b as b JOIN "
        "does_intersect_a as a ON ST_Intersects(a.poly, b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));
  });
}

// TODO(jclay): This should succeed without failure.
// For now, we test against the (incorrect) failure.
TEST_F(OverlapsTest, InnerJoinPolyInPointIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    const auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON ST_Intersects(a.pt, "
        "b.poly);";
    if (g_enable_hashjoin_many_to_many) {
      EXPECT_ANY_THROW(execSQL(sql, dt));
    } else {
      // Note(jclay): We return 0, postgis returns 4
      // Note(adb): Now we return 3. Progress?
      ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));
    }
  });
}

TEST_F(OverlapsTest, InnerJoinPolyPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Intersects(a.poly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, InnerJoinMPolyPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Intersects(a.mpoly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, InnerJoinMPolyMPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Intersects(a.mpoly, b.mpoly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, LeftJoinMPolyPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  LEFT JOIN does_intersect_b as b
                  ON ST_Intersects(a.mpoly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, LeftJoinMPolyMPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  LEFT JOIN does_intersect_b as b
                  ON ST_Intersects(a.mpoly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, InnerJoinPolyPolyIntersectsTranspose) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    const auto sql = R"(SELECT count(*) from does_intersect_a as a
                        JOIN does_intersect_b as b
                        ON ST_Intersects(b.poly, a.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, LeftJoinPolyPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_b as b
                      LEFT JOIN does_intersect_a as a
                      ON ST_Intersects(a.poly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(4), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, LeftJoinPointInPolyIntersects) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                      LEFT JOIN does_intersect_b as b
                      ON ST_Intersects(b.poly, a.pt);)";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));
  });
}

// TODO(jclay): This should succeed without failure.
// Look into rewriting this in overlaps rewrite.
// For now, we test against the (incorrect) failure.
// It should return 3.
TEST_F(OverlapsTest, LeftJoinPointInPolyIntersectsWrongLHS) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                      LEFT JOIN does_intersect_b as b
                      ON ST_Intersects(a.poly, b.pt);)";
    if (g_enable_hashjoin_many_to_many) {
      EXPECT_ANY_THROW(execSQL(sql, dt));
    } else {
      ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));
    }
  });
}

TEST_F(OverlapsTest, InnerJoinPolyPolyContains) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_b as b
                  JOIN does_intersect_a as a
                  ON ST_Contains(a.poly, b.poly);)";
    ASSERT_EQ(static_cast<int64_t>(0), v<int64_t>(execSQL(sql, dt)));
  });
}

// TODO(jclay): The following runtime functions are not implemented:
// - ST_Contains_MultiPolygon_MultiPolygon
// - ST_Contains_MultiPolygon_Polygon
// As a result, the following should succeed rather than throw error.
TEST_F(OverlapsTest, InnerJoinMPolyPolyContains) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Contains(a.mpoly, b.poly);)";
    EXPECT_ANY_THROW(execSQL(sql, dt));
  });
}

TEST_F(OverlapsTest, InnerJoinMPolyMPolyContains) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_a as a
                  JOIN does_intersect_b as b
                  ON ST_Contains(a.mpoly, b.poly);)";
    // should return 4
    EXPECT_ANY_THROW(execSQL(sql, dt));
  });
}

// NOTE(jclay): We don't support multipoly / poly ST_Contains
TEST_F(OverlapsTest, LeftJoinMPolyPolyContains) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_b as b
                  LEFT JOIN does_intersect_a as a
                  ON ST_Contains(a.mpoly, b.poly);)";
    // should return 4
    EXPECT_ANY_THROW(execSQL(sql, dt));
  });
}

TEST_F(OverlapsTest, LeftJoinMPolyMPolyContains) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql = R"(SELECT count(*) from does_intersect_b as b
                  LEFT JOIN does_intersect_a as a
                  ON ST_Contains(a.mpoly, b.mpoly);)";
    // should return 4
    EXPECT_ANY_THROW(execSQL(sql, dt));
  });
}

TEST_F(OverlapsTest, JoinPolyPointContains) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    auto sql =
        "SELECT "
        "count(*) from "
        "does_intersect_b as b JOIN does_intersect_a as a ON ST_Contains(a.poly, b.pt);";
    ASSERT_EQ(static_cast<int64_t>(3), v<int64_t>(execSQL(sql, dt)));

    // sql =
    //     "SELECT "
    //     "count(*) from "
    //     "does_intersect_b as b JOIN does_intersect_a as a ON ST_Contains(a.pt,
    //     b.poly);";
    // ASSERT_EQ(static_cast<int64_t>(0), v<int64_t>(execSQL(sql, dt)));
  });
}

TEST_F(OverlapsTest, PolyPolyDoesNotIntersect) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(execSQL("SELECT count(*) FROM does_not_intersect_b as b "
                                 "JOIN does_not_intersect_a as a "
                                 "ON ST_Intersects(a.poly, b.poly);",
                                 dt)));
  });
}

TEST_F(OverlapsTest, EmptyPolyPolyJoin) {
  executeAllScenarios([](ExecutorDeviceType dt) -> void {
    const auto sql =
        "SELECT count(*) FROM does_not_intersect_a as a "
        "JOIN empty_table as b "
        "ON ST_Intersects(a.poly, b.poly);";
    ASSERT_EQ(static_cast<int64_t>(0), v<int64_t>(execSQL(sql, dt)));
  });
}

class OverlapsJoinHashTableMock : public OverlapsJoinHashTable {
 public:
  struct ExpectedValues {
    size_t entry_count;
    size_t emitted_keys_count;
  };

  static std::shared_ptr<OverlapsJoinHashTableMock> getInstance(
      const std::shared_ptr<Analyzer::BinOper> condition,
      const std::vector<InputTableInfo>& query_infos,
      const Data_Namespace::MemoryLevel memory_level,
      ColumnCacheMap& column_cache,
      Executor* executor,
      const int device_count,
      const QueryHint& query_hint,
      const std::vector<OverlapsJoinHashTableMock::ExpectedValues>& expected_values) {
    auto hash_join = std::make_shared<OverlapsJoinHashTableMock>(condition,
                                                                 query_infos,
                                                                 memory_level,
                                                                 column_cache,
                                                                 executor,
                                                                 device_count,
                                                                 expected_values);
    hash_join->registerQueryHint(query_hint);
    hash_join->reifyWithLayout(HashType::OneToMany);
    return hash_join;
  }

  OverlapsJoinHashTableMock(const std::shared_ptr<Analyzer::BinOper> condition,
                            const std::vector<InputTableInfo>& query_infos,
                            const Data_Namespace::MemoryLevel memory_level,
                            ColumnCacheMap& column_cache,
                            Executor* executor,
                            const int device_count,
                            const std::vector<ExpectedValues>& expected_values)
      : OverlapsJoinHashTable(condition,
                              query_infos,
                              memory_level,
                              column_cache,
                              executor,
                              normalize_column_pairs(condition.get(),
                                                     *executor->getCatalog(),
                                                     executor->getTemporaryTables()),
                              device_count)
      , expected_values_per_step_(expected_values) {}

 protected:
  void reifyImpl(std::vector<ColumnsForDevice>& columns_per_device,
                 const Fragmenter_Namespace::TableInfo& query_info,
                 const HashType layout,
                 const size_t shard_count,
                 const size_t entry_count,
                 const size_t emitted_keys_count) final {
    EXPECT_EQ(step_, expected_values_per_step_.size());
    EXPECT_LT(step_ - 1, expected_values_per_step_.size());
    auto& expected_values = expected_values_per_step_[step_ - 1];
    EXPECT_EQ(entry_count, expected_values.entry_count);
    EXPECT_EQ(emitted_keys_count, expected_values.emitted_keys_count);
    return;
  }

  // returns entry_count, emitted_keys_count
  std::pair<size_t, size_t> approximateTupleCount(
      const std::vector<double>& bucket_sizes_for_dimension,
      std::vector<ColumnsForDevice>& columns_per_device) final {
    auto [entry_count, emitted_keys_count] = OverlapsJoinHashTable::approximateTupleCount(
        bucket_sizes_for_dimension, columns_per_device);
    return std::make_pair(entry_count, emitted_keys_count);
  }

  // returns entry_count, emitted_keys_count
  std::pair<size_t, size_t> computeHashTableCounts(
      const size_t shard_count,
      const std::vector<double>& bucket_sizes_for_dimension,
      std::vector<ColumnsForDevice>& columns_per_device) final {
    auto [entry_count, emitted_keys_count] =
        OverlapsJoinHashTable::computeHashTableCounts(
            shard_count, bucket_sizes_for_dimension, columns_per_device);
    EXPECT_LT(step_, expected_values_per_step_.size());
    auto& expected_values = expected_values_per_step_[step_];
    EXPECT_EQ(entry_count, expected_values.entry_count);
    EXPECT_EQ(emitted_keys_count, expected_values.emitted_keys_count);
    step_++;
    return std::make_pair(entry_count, emitted_keys_count);
  }

  std::vector<ExpectedValues> expected_values_per_step_;
  size_t step_{0};
};

class BucketSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS bucket_size_poly;");
    QR::get()->runDDLStatement("CREATE TABLE bucket_size_poly (poly MULTIPOLYGON);");
    QR::get()->runSQL(
        R"(INSERT INTO bucket_size_poly VALUES ('MULTIPOLYGON(((0 0, 0 2, 2 0, 2 2)))');)",
        ExecutorDeviceType::CPU);
    QR::get()->runSQL(
        R"(INSERT INTO bucket_size_poly VALUES ('MULTIPOLYGON(((0 0, 0 2, 2 0, 2 2)))');)",
        ExecutorDeviceType::CPU);
    QR::get()->runSQL(
        R"(INSERT INTO bucket_size_poly VALUES ('MULTIPOLYGON(((2 2, 2 4, 4 2, 4 4)))');)",
        ExecutorDeviceType::CPU);
    QR::get()->runSQL(
        R"(INSERT INTO bucket_size_poly VALUES ('MULTIPOLYGON(((0 0, 0 50, 50 0, 50 50)))');)",
        ExecutorDeviceType::CPU);

    QR::get()->runDDLStatement("DROP TABLE IF EXISTS bucket_size_pt;");
    QR::get()->runDDLStatement("CREATE TABLE bucket_size_pt (pt POINT);");
  }

  void TearDown() override {
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS bucket_size_poly;");
    QR::get()->runDDLStatement("DROP TABLE IF EXISTS bucket_size_pt;");
  }

 public:
  static std::pair<std::shared_ptr<Analyzer::BinOper>, std::vector<InputTableInfo>>
  getOverlapsBuildInfo() {
    auto catalog = QR::get()->getCatalog();
    CHECK(catalog);

    std::vector<InputTableInfo> query_infos;

    const auto pts_td = catalog->getMetadataForTable("bucket_size_pt");
    CHECK(pts_td);
    const auto pts_cd = catalog->getMetadataForColumn(pts_td->tableId, "pt");
    CHECK(pts_cd);
    auto pt_col_var = std::make_shared<Analyzer::ColumnVar>(
        pts_cd->columnType, pts_cd->tableId, pts_cd->columnId, 0);
    query_infos.emplace_back(InputTableInfo{pts_td->tableId, build_table_info({pts_td})});

    const auto poly_td = catalog->getMetadataForTable("bucket_size_poly");
    CHECK(poly_td);
    const auto poly_cd = catalog->getMetadataForColumn(poly_td->tableId, "poly");
    CHECK(poly_cd);
    const auto bounds_cd =
        catalog->getMetadataForColumn(poly_td->tableId, poly_cd->columnId + 4);
    CHECK(bounds_cd && bounds_cd->columnType.is_array());
    auto poly_col_var = std::make_shared<Analyzer::ColumnVar>(
        bounds_cd->columnType, poly_cd->tableId, bounds_cd->columnId, 1);
    query_infos.emplace_back(
        InputTableInfo{poly_td->tableId, build_table_info({poly_td})});

    auto condition = std::make_shared<Analyzer::BinOper>(
        kBOOLEAN, kOVERLAPS, kANY, pt_col_var, poly_col_var);
    return std::make_pair(condition, query_infos);
  }
};

TEST_F(BucketSizeTest, OverlapsTunerEarlyOut) {
  // 3 steps, early out due to increasing keys per bin
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);
  auto executor = QR::get()->getExecutor();
  executor->setCatalog(catalog.get());

  auto [condition, query_infos] = BucketSizeTest::getOverlapsBuildInfo();

  ColumnCacheMap column_cache;
  std::vector<OverlapsJoinHashTableMock::ExpectedValues> expected_values;
  expected_values.emplace_back(OverlapsJoinHashTableMock::ExpectedValues{1340, 688});
  expected_values.emplace_back(OverlapsJoinHashTableMock::ExpectedValues{1340, 688});

  auto hash_table =
      OverlapsJoinHashTableMock::getInstance(condition,
                                             query_infos,
                                             Data_Namespace::MemoryLevel::CPU_LEVEL,
                                             column_cache,
                                             executor.get(),
                                             /*device_count=*/1,
                                             QueryHint::defaults(),
                                             expected_values);
  CHECK(hash_table);
}

TEST_F(BucketSizeTest, OverlapsTooBig) {
  auto catalog = QR::get()->getCatalog();
  CHECK(catalog);
  auto executor = QR::get()->getExecutor();
  executor->setCatalog(catalog.get());

  auto [condition, query_infos] = BucketSizeTest::getOverlapsBuildInfo();

  ColumnCacheMap column_cache;
  std::vector<OverlapsJoinHashTableMock::ExpectedValues> expected_values;
  expected_values.emplace_back(OverlapsJoinHashTableMock::ExpectedValues{1340, 688});
  expected_values.emplace_back(OverlapsJoinHashTableMock::ExpectedValues{1340, 688});

  QueryHint hint;
  hint.overlaps_max_size = 2;
  auto hash_table =
      OverlapsJoinHashTableMock::getInstance(condition,
                                             query_infos,
                                             Data_Namespace::MemoryLevel::CPU_LEVEL,
                                             column_cache,
                                             executor.get(),
                                             /*device_count=*/1,
                                             hint,
                                             expected_values);
  CHECK(hash_table);
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  QR::reset();
  return err;
}
