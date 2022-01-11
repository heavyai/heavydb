/*
 * Copyright 2019 OmniSci, Inc.
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

#include "TestHelpers.h"

#include "Catalog/Catalog.h"
#include "DBHandlerTestHelpers.h"
#include "QueryEngine/TableOptimizer.h"

#include <gtest/gtest.h>
#include <string>
#include <utility>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern float g_vacuum_min_selectivity;

namespace {

#define ASSERT_METADATA(type, tag)                                   \
  template <typename T, bool enabled = std::is_same<T, type>::value> \
  void assert_metadata(const ChunkStats& chunkStats,                 \
                       const T min,                                  \
                       const T max,                                  \
                       const bool has_nulls,                         \
                       const std::enable_if_t<enabled, type>* = 0) { \
    ASSERT_EQ(chunkStats.min.tag##val, min);                         \
    ASSERT_EQ(chunkStats.max.tag##val, max);                         \
    ASSERT_EQ(chunkStats.has_nulls, has_nulls);                      \
  }

ASSERT_METADATA(bool, bool)
ASSERT_METADATA(int8_t, tinyint)
ASSERT_METADATA(int16_t, smallint)
ASSERT_METADATA(int32_t, int)
ASSERT_METADATA(int64_t, bigint)
ASSERT_METADATA(float, float)
ASSERT_METADATA(double, double)

template <typename T, typename... Args>
void check_column_metadata_impl(const ChunkMetadataMap& metadata_map,
                                const int column_idx,  // -1 is $deleted
                                const T min,
                                const T max,
                                const bool has_nulls) {
  auto chunk_metadata_itr = metadata_map.find(column_idx);
  if (column_idx < 0) {
    chunk_metadata_itr--;
  }
  CHECK(chunk_metadata_itr != metadata_map.end());
  const auto& chunk_metadata = chunk_metadata_itr->second;
  assert_metadata<T>(chunk_metadata->chunkStats, min, max, has_nulls);
}

template <typename T, typename... Args>
void check_column_metadata_impl(const ChunkMetadataMap& metadata_map,
                                const int column_idx,  // -1 is $deleted
                                const T min,
                                const T max,
                                const bool has_nulls,
                                Args&&... args) {
  check_column_metadata_impl(metadata_map, column_idx, min, max, has_nulls);
  using T1 = typename std::tuple_element<1, std::tuple<Args...>>::type;
  check_column_metadata_impl<T1>(metadata_map, std::forward<Args>(args)...);
}

template <typename... Args>
auto check_column_metadata =
    [](const Fragmenter_Namespace::FragmentInfo& fragment, Args&&... args) {
      const auto metadata_map = fragment.getChunkMetadataMapPhysical();
      using T = typename std::tuple_element<1, std::tuple<Args...>>::type;
      check_column_metadata_impl<T>(metadata_map, std::forward<Args>(args)...);
    };

template <typename... Args>
auto check_fragment_metadata(Args&&... args) -> auto {
  static_assert(sizeof...(Args) % 4 == 0,
                "check_fragment_metadata expects arguments to be a multiple of 4");
  return std::make_tuple(check_column_metadata<Args...>,
                         std::make_tuple<Args...>(std::move(args)...));
}

template <typename FUNC, typename... Args>
void run_op_per_fragment(const Catalog_Namespace::Catalog& catalog,
                         const TableDescriptor* td,
                         FUNC f,
                         Args&&... args) {
  auto* fragmenter = td->fragmenter.get();
  CHECK(fragmenter);
  const auto table_info = fragmenter->getFragmentsForQuery();
  for (const auto& fragment : table_info.fragments) {
    f(fragment, std::forward<Args>(args)...);
  }
}

template <typename FUNC, typename... Args, std::size_t... Is>
void run_op_per_fragment(const Catalog_Namespace::Catalog& catalog,
                         const TableDescriptor* td,
                         FUNC f,
                         std::tuple<Args...> tuple,
                         std::index_sequence<Is...>) {
  run_op_per_fragment(catalog, td, f, std::forward<Args>(std::get<Is>(tuple))...);
}

template <typename FUNC, typename... Args>
void run_op_per_fragment(const Catalog_Namespace::Catalog& catalog,
                         const TableDescriptor* td,
                         std::tuple<FUNC, std::tuple<Args...>> tuple) {
  run_op_per_fragment(catalog,
                      td,
                      std::get<0>(tuple),
                      std::get<1>(tuple),
                      std::index_sequence_for<Args...>{});
}

void recompute_metadata(const TableDescriptor* td,
                        const Catalog_Namespace::Catalog& cat) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  TableOptimizer optimizer(td, executor.get(), cat);
  EXPECT_NO_THROW(optimizer.recomputeMetadata());
}

void vacuum_and_recompute_metadata(const TableDescriptor* td,
                                   const Catalog_Namespace::Catalog& cat) {
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  TableOptimizer optimizer(td, executor.get(), cat);
  EXPECT_NO_THROW(optimizer.vacuumDeletedRows());
  EXPECT_NO_THROW(optimizer.recomputeMetadata());
}

static const std::string g_table_name{"metadata_test"};

}  // namespace

class MultiFragMetadataUpdate : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    EXPECT_NO_THROW(sql("DROP TABLE IF EXISTS " + g_table_name + ";"));
    EXPECT_NO_THROW(
        sql("CREATE TABLE " + g_table_name +
            " (x INT, y INT NOT NULL, z INT "
            "ENCODING FIXED(8), a DOUBLE, b FLOAT, d DATE, dd DATE "
            "ENCODING FIXED(16), c TEXT ENCODING DICT(32)) WITH (FRAGMENT_SIZE=4);"));

    TestHelpers::ValuesGenerator gen(g_table_name);

    for (int i = 0; i < 5; i++) {
      std::string date_str = i % 2 == 0 ? "'1/1/2019'" : "'2/2/2020'";
      const auto insert_query =
          gen(i, i, i, i * 1.1, i * 1.2, date_str, date_str, "'foo'");
      sql(insert_query);
    }

    for (int i = 0; i < 5; i++) {
      std::string date_str = i % 2 == 0 ? "'5/30/2021'" : "'6/30/2022'";
      const int multiplier = i % 2 == 0 ? -1 : 1;
      const auto insert_query = gen(multiplier * i,
                                    multiplier * i,
                                    multiplier * i,
                                    std::to_string(multiplier * i * 1.1),
                                    multiplier * i * 1.2,
                                    date_str,
                                    date_str,
                                    "'bar'");
      sql(insert_query);
    }

    for (size_t i = 6; i < 11; i++) {
      std::string insert_query;
      if (i % 2 == 0) {
        insert_query = gen(i, i, i, i * 1.1, i * 1.2, "null", "null", "'hello'");
      } else {
        insert_query = gen("null",
                           std::numeric_limits<int32_t>::min(),
                           "null",
                           "null",
                           "null",
                           "'10/11/1981'",
                           "'10/11/1981'",
                           "'world'");
      }

      sql(insert_query);
    }

    for (int i = 0; i < 5; i++) {
      const auto insert_query = gen("null",
                                    std::numeric_limits<int32_t>::max(),
                                    "null",
                                    "null",
                                    "null",
                                    "null",
                                    "null",
                                    "null");
      sql(insert_query);
    }
  }

  void TearDown() override {
    EXPECT_NO_THROW(sql("DROP TABLE IF EXISTS " + g_table_name + ";"));
    DBHandlerTestFixture::TearDown();
  }
};

TEST_F(MultiFragMetadataUpdate, NoChanges) {
  std::vector<ChunkMetadataMap> metadata_for_fragments;
  {
    const auto& cat = getCatalog();
    const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

    // Get chunk metadata before recomputing
    auto store_original_metadata =
        [&metadata_for_fragments](const Fragmenter_Namespace::FragmentInfo& fragment) {
          metadata_for_fragments.push_back(fragment.getChunkMetadataMapPhysical());
        };

    run_op_per_fragment(cat, td, store_original_metadata);
    recompute_metadata(td, cat);
  }

  // Make sure metadata matches after recomputing
  {
    const auto& cat = getCatalog();
    const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

    auto* fragmenter = td->fragmenter.get();
    CHECK(fragmenter);
    const auto table_info = fragmenter->getFragmentsForQuery();

    size_t ctr = 0;
    auto check_metadata_equality =
        [&ctr,
         &metadata_for_fragments](const Fragmenter_Namespace::FragmentInfo& fragment) {
          ASSERT_LT(ctr, metadata_for_fragments.size());
          ASSERT_TRUE(metadata_for_fragments[ctr++] ==
                      fragment.getChunkMetadataMapPhysical());
        };
    run_op_per_fragment(cat, td, check_metadata_equality);
  }
}

class MetadataUpdate : public DBHandlerTestFixture,
                       public testing::WithParamInterface<bool> {
 protected:
  static void SetUpTestSuite() {
    g_enable_auto_metadata_update = false;
    g_vacuum_min_selectivity = 1.1;
  }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    auto is_sharded = GetParam();
    CHECK(!is_sharded);
    int shard_count{1};
    EXPECT_NO_THROW(sql("DROP TABLE IF EXISTS " + g_table_name + ";"));
    EXPECT_NO_THROW(sql("CREATE TABLE " + g_table_name +
                        " (x INT, y INT NOT NULL, z INT "
                        "ENCODING FIXED(8), a DOUBLE, b FLOAT, d DATE, dd DATE "
                        "ENCODING FIXED(16), c TEXT ENCODING DICT(32), skey int"
                        ") WITH (FRAGMENT_SIZE=5, max_rollback_epochs = 25);"));

    TestHelpers::ValuesGenerator gen(g_table_name);
    for (int sh = 0; sh < shard_count; ++sh) {
      sql(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "'foo'", sh));
      sql(gen(2, 2, 2, 2, 2, "'12/31/2012'", "'12/31/2012'", "'foo'", sh));
      sql(gen("null", 2, "null", "null", "null", "null", "'1/1/1940'", "'foo'", sh));
    }
  }

  void TearDown() override {
    EXPECT_NO_THROW(sql("DROP TABLE IF EXISTS " + g_table_name + ";"));
    DBHandlerTestFixture::TearDown();
  }
};

TEST_P(MetadataUpdate, InitialMetadata) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_op_per_fragment(
      cat,
      td,
      check_fragment_metadata(
          // Check int col: expected range 1,2 nulls
          /* id = */ 1,
          /* min = */ 1,
          /* max = 2 */ 2,
          /* has_nulls = */ true,

          // Check int not null col: expected range 1,2 no nulls
          2,
          1,
          2,
          false,

          // Check int encoded call: expected range 1,2 nulls
          3,
          1,
          2,
          true,

          // Check double col: expected range 1.0,2.0 nulls
          4,
          (double)1.0,
          2.0,
          true,

          // Check float col: expected range 1.0,2.0 nulls
          5,
          (float)1.0,
          2.0,
          true,

          // Check date in days 32 col: expected range 1262304000,1356912000 nulls
          6,
          1262304000,
          1356912000,
          true,

          // Check date in days 16 col: expected range -946771200,1356912000 nulls
          7,
          -946771200,
          1356912000,
          false,

          // Check col c TEXT ENCODING DICT(32): expected range [0, 0]
          8,
          0,
          0,
          false));
}

TEST_P(MetadataUpdate, IntUpdate) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  sql("UPDATE " + g_table_name + " SET x = 3 WHERE x = 1;");

  // Check int col: expected range 1,3 nulls
  run_op_per_fragment(cat, td, check_fragment_metadata(1, (int32_t)1, 3, true));

  sql("UPDATE " + g_table_name + " SET x = 0 WHERE x = 3;");

  recompute_metadata(td, cat);
  // Check int col: expected range 1,2 nulls
  run_op_per_fragment(cat, td, check_fragment_metadata(1, (int32_t)0, 2, true));
}

TEST_P(MetadataUpdate, IntRemoveNull) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  sql("UPDATE " + g_table_name + " SET x = 3;");

  recompute_metadata(td, cat);
  // Check int col: expected range 1,2 nulls
  run_op_per_fragment(cat, td, check_fragment_metadata(1, (int32_t)3, 3, false));
}

TEST_P(MetadataUpdate, NotNullInt) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  sql("UPDATE " + g_table_name + " SET y = " +
      std::to_string(std::numeric_limits<int32_t>::lowest() + 1) + " WHERE y = 1;");
  // Check int col: expected range 1,3 nulls
  run_op_per_fragment(
      cat,
      td,
      check_fragment_metadata(2, std::numeric_limits<int32_t>::lowest() + 1, 2, false));

  sql("UPDATE " + g_table_name + " SET y = 1;");

  recompute_metadata(td, cat);
  run_op_per_fragment(cat, td, check_fragment_metadata(2, (int32_t)1, 1, false));
}

TEST_P(MetadataUpdate, DateNarrowRange) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  sql("UPDATE " + g_table_name + " SET d = '1/1/2010';");

  recompute_metadata(td, cat);
  // Check date in days 32 col: expected range 1262304000,1262304000 nulls
  run_op_per_fragment(
      cat, td, check_fragment_metadata(6, (int64_t)1262304000, 1262304000, false));
}

TEST_P(MetadataUpdate, SmallDateNarrowMin) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  sql("UPDATE " + g_table_name + " SET dd = '1/1/2010' WHERE dd = '1/1/1940';");

  recompute_metadata(td, cat);
  run_op_per_fragment(
      cat, td, check_fragment_metadata(7, (int64_t)1262304000, 1356912000, false));
}

TEST_P(MetadataUpdate, SmallDateNarrowMax) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  sql("UPDATE " + g_table_name + " SET dd = '1/1/2010' WHERE dd = '12/31/2012';");

  recompute_metadata(td, cat);
  run_op_per_fragment(
      cat, td, check_fragment_metadata(7, (int64_t)-946771200, 1262304000, false));
}

TEST_P(MetadataUpdate, DeleteReset) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  sql("DELETE FROM  " + g_table_name + " WHERE dd = '12/31/2012';");
  run_op_per_fragment(cat, td, check_fragment_metadata(-1, false, true, false));

  vacuum_and_recompute_metadata(td, cat);
  run_op_per_fragment(cat, td, check_fragment_metadata(-1, false, false, false));
}

TEST_P(MetadataUpdate, EncodedStringNull) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  TestHelpers::ValuesGenerator gen(g_table_name);
  sql(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "'abc'", 0));
  vacuum_and_recompute_metadata(td, cat);
  run_op_per_fragment(cat, td, check_fragment_metadata(8, 0, 1, false));

  sql(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "null", 0));
  vacuum_and_recompute_metadata(td, cat);
  run_op_per_fragment(cat, td, check_fragment_metadata(8, 0, 1, true));
}

TEST_P(MetadataUpdate, AlterAfterOptimize) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);
  run_op_per_fragment(cat, td, check_fragment_metadata(1, 1, 2, true));
  sql("DELETE FROM  " + g_table_name + " WHERE x IS NULL;");
  vacuum_and_recompute_metadata(td, cat);
  run_op_per_fragment(cat, td, check_fragment_metadata(1, 1, 2, false));
  // test ADD one column
  EXPECT_NO_THROW(sql("ALTER TABLE " + g_table_name + " ADD (c99 int default 99);"));
  run_op_per_fragment(cat, td, check_fragment_metadata(12, 99, 99, false));
  // test ADD multiple columns
  EXPECT_NO_THROW(
      sql("ALTER TABLE " + g_table_name + " ADD (c88 int default 88, cnn int);"));
  run_op_per_fragment(cat, td, check_fragment_metadata(13, 88, 88, false));
  run_op_per_fragment(cat,
                      td,
                      check_fragment_metadata(14,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              true));
}

TEST_P(MetadataUpdate, AlterAfterEmptied) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);
  sql("DELETE FROM  " + g_table_name + ";");
  vacuum_and_recompute_metadata(td, cat);
  run_op_per_fragment(cat,
                      td,
                      check_fragment_metadata(1,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              false));
  // test ADD one column to make sure column is added even if no row exists
  EXPECT_NO_THROW(sql("ALTER TABLE " + g_table_name + " ADD (c99 int default 99);"));
  run_op_per_fragment(cat,
                      td,
                      check_fragment_metadata(12,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              false));
  // test ADD multiple columns
  EXPECT_NO_THROW(
      sql("ALTER TABLE " + g_table_name + " ADD (c88 int default 88, cnn int);"));
  run_op_per_fragment(cat,
                      td,
                      check_fragment_metadata(13,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              false));
  run_op_per_fragment(cat,
                      td,
                      check_fragment_metadata(14,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              false));
}

INSTANTIATE_TEST_SUITE_P(ShardedAndNonShardedTable,
                         MetadataUpdate,
                         testing::Values(false),
                         [](const auto& param_info) {
                           return (param_info.param ? "ShardedTable" : "NonShardedTable");
                         });

class DeletedRowsMetadataUpdateTest : public DBHandlerTestFixture {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
    sql("create table test_table (i int);");
  }

  void TearDown() override {
    sql("drop table test_table;");
    DBHandlerTestFixture::TearDown();
  }
};

TEST_F(DeletedRowsMetadataUpdateTest, ComputeMetadataAfterDelete) {
  sql("insert into test_table values (1);");
  sql("insert into test_table values (2);");
  sql("insert into test_table values (3);");
  sqlAndCompareResult("select * from test_table;", {{i(1)}, {i(2)}, {i(3)}});

  sql("delete from test_table where i <= 2;");
  sqlAndCompareResult("select * from test_table;", {{i(3)}});

  const auto& catalog = getCatalog();
  const auto td = catalog.getMetadataForTable("test_table");
  recompute_metadata(td, catalog);
  sqlAndCompareResult("select * from test_table;", {{i(3)}});
}

class OptimizeTableVacuumTest : public DBHandlerTestFixture {
 protected:
  static void SetUpTestSuite() { g_vacuum_min_selectivity = 1.1; }

  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
    g_enable_auto_metadata_update = true;
  }

  void TearDown() override {
    sql("drop table test_table;");
    File_Namespace::FileMgr::setNumPagesPerDataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE);
    File_Namespace::FileMgr::setNumPagesPerMetadataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE);
    g_enable_auto_metadata_update = false;
    DBHandlerTestFixture::TearDown();
  }

  void assertUsedPageCount(int64_t used_data_page_count) {
    TQueryResult result;
    sql(result, "show table details test_table;");

    ASSERT_EQ("total_data_page_count", result.row_set.row_desc[18].col_name);
    auto total_data_page_count = result.row_set.columns[18].data.int_col[0];

    ASSERT_EQ("total_free_data_page_count", result.row_set.row_desc[19].col_name);
    auto total_free_data_page_count = result.row_set.columns[19].data.int_col[0];

    ASSERT_EQ(used_data_page_count, total_data_page_count - total_free_data_page_count);
  }

  void insertRange(int start, int end) {
    for (int value = start; value <= end; value++) {
      sql("insert into test_table values (" + std::to_string(value) + ");");
    }
  }

  void assertFileAndFreePageCount(int64_t metadata_file_count,
                                  int64_t free_metadata_page_count,
                                  int64_t data_file_count,
                                  int64_t free_data_page_count) {
    TQueryResult result;
    sql(result, "show table details test_table;");
    EXPECT_EQ("metadata_file_count", result.row_set.row_desc[12].col_name);
    EXPECT_EQ(metadata_file_count, result.row_set.columns[12].data.int_col[0]);

    EXPECT_EQ("total_free_metadata_page_count", result.row_set.row_desc[15].col_name);
    EXPECT_EQ(free_metadata_page_count, result.row_set.columns[15].data.int_col[0]);

    EXPECT_EQ("data_file_count", result.row_set.row_desc[16].col_name);
    EXPECT_EQ(data_file_count, result.row_set.columns[16].data.int_col[0]);

    EXPECT_EQ("total_free_data_page_count", result.row_set.row_desc[19].col_name);
    EXPECT_EQ(free_data_page_count, result.row_set.columns[19].data.int_col[0]);
  }

  void insertRange(int start, int end, int column_count) {
    for (int value = start; value <= end; value++) {
      std::string query{"insert into test_table values ("};
      for (int i = 0; i < column_count; i++) {
        if (i > 0) {
          query += ", ";
        }
        query += std::to_string(value);
      }
      query += ");";
      sql(query);
    }
  }
};

TEST_F(OptimizeTableVacuumTest, TableWithDeletedRows) {
  sql("create table test_table (i int);");
  sql("insert into test_table values (10);");
  sql("insert into test_table values (20);");
  sql("insert into test_table values (30);");
  sql("delete from test_table where i <= 20;");
  sql("optimize table test_table with (vacuum = 'true');");
}

TEST_F(OptimizeTableVacuumTest, SingleChunkVersionAndDeletedFragment) {
  sql("create table test_table (i int) with (fragment_size = 2, max_rollback_epochs = "
      "0);");
  insertRange(1, 3);

  // 4 chunks (includes "$deleted" column chunks), each using one page
  assertUsedPageCount(4);

  sql("delete from test_table where i <= 2;");
  assertUsedPageCount(4);

  sql("optimize table test_table with (vacuum = 'true');");
  // 2 pages for the first fragment chunks should be rolled-off
  assertUsedPageCount(2);

  sqlAndCompareResult("select * from test_table;", {{i(3)}});
}

TEST_F(OptimizeTableVacuumTest, MultipleChunkVersionsAndDeletedFragment) {
  // Create table with a page size that allows for a maximum of 2 integers per page
  sql("create table test_table (i int) with (fragment_size = 2, max_rollback_epochs = "
      "5);");
  insertRange(1, 3);

  // 4 chunks (includes "$deleted" column chunks), each using one page
  assertUsedPageCount(4);

  sql("delete from test_table where i <= 2;");
  // Additional page for new "$deleted" column chunk that marks rows as deleted
  assertUsedPageCount(5);

  sql("optimize table test_table with (vacuum = 'true');");
  // All chunks/chunk pages are still kept
  assertUsedPageCount(5);

  sqlAndCompareResult("select * from test_table;", {{i(3)}});
}

TEST_F(OptimizeTableVacuumTest, UpdateAndCompactTableData) {
  // Each page write creates a new file
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);

  sql("create table test_table (i int) with (max_rollback_epochs = 25);");
  sql("insert into test_table values (10);");
  // 2 chunk page writes and 2 metadata page writes. One for
  // the "i" column and a second for the "$deleted" column
  assertFileAndFreePageCount(2, 0, 2, 0);

  // 2 additional pages/files for the "i" chunk
  sql("update test_table set i = i + 10;");
  assertFileAndFreePageCount(3, 0, 3, 0);

  // 2 additional pages/files for the "i" chunk
  sql("update test_table set i = i + 10;");
  assertFileAndFreePageCount(4, 0, 4, 0);

  // Rolls off/frees oldest 2 "i" chunk/metadata pages
  sql("alter table test_table set max_rollback_epochs = 0;");
  assertFileAndFreePageCount(4, 2, 4, 2);

  // Compaction deletes the 4 free pages from above.
  sql("optimize table test_table with (vacuum = 'true');");
  assertFileAndFreePageCount(2, 0, 2, 0);

  // Verify that subsequent queries work as expected
  sqlAndCompareResult("select * from test_table;", {{i(30)}});
  sql("update test_table set i = i - 5;");
  sqlAndCompareResult("select * from test_table;", {{i(25)}});
}

TEST_F(OptimizeTableVacuumTest, InsertAndCompactTableData) {
  // Each page write creates a new file
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);

  sql("create table test_table (i int) with (fragment_size = 2, max_rollback_epochs = "
      "25);");
  insertRange(1, 3, 1);
  // 4 chunk page writes. 2 for the "i" column and "$deleted" column each.
  // 6 metadata page writes for each insert (3 inserts for 2 columns).
  assertFileAndFreePageCount(6, 0, 4, 0);

  // 1 chunk page write and 1 metadata page write for the updated
  // "$deleted" chunk.
  sql("delete from test_table where i <= 2;");
  assertFileAndFreePageCount(7, 0, 5, 0);

  // Rolls off/frees oldest "$deleted" chunk page and 3 metadata
  // pages (2 from initial insert and 1 from "$deleted" chunk update).
  sql("alter table test_table set max_rollback_epochs = 0;");
  assertFileAndFreePageCount(7, 3, 5, 1);

  // Optimize frees up pages for the deleted 2 chunks.
  // Compaction deletes the 4 free pages from above in
  // addition to the 2 freed pages.
  sql("optimize table test_table with (vacuum = 'true');");
  assertFileAndFreePageCount(4, 0, 2, 0);

  // Verify that subsequent queries work as expected
  sqlAndCompareResult("select * from test_table;", {{i(3)}});
  sql("insert into test_table values(5);");
  sqlAndCompareResult("select * from test_table;", {{i(3)}, {i(5)}});
}

TEST_F(OptimizeTableVacuumTest, MultiplePagesPerFile) {
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(4);
  File_Namespace::FileMgr::setNumPagesPerDataFile(2);

  sql("create table test_table (i int) with (max_rollback_epochs = 25);");
  sql("insert into test_table values (10);");
  // 2 chunk page writes and 2 metadata page writes. One for
  // the "i" column and a second for the "$deleted" column
  assertFileAndFreePageCount(1, 2, 1, 0);

  // 2 additional pages for the "i" chunk
  sql("update test_table set i = i + 10;");
  assertFileAndFreePageCount(1, 1, 2, 1);

  // 2 additional pages for the "i" chunk
  sql("update test_table set i = i + 10;");
  assertFileAndFreePageCount(1, 0, 2, 0);

  // Rolls off/frees oldest 2 "i" chunk/metadata pages
  sql("alter table test_table set max_rollback_epochs = 0;");
  assertFileAndFreePageCount(1, 2, 2, 2);

  // Compaction deletes empty data file.
  sql("optimize table test_table with (vacuum = 'true');");
  assertFileAndFreePageCount(1, 2, 1, 0);

  // Verify that subsequent queries work as expected
  sqlAndCompareResult("select * from test_table;", {{i(30)}});
  sql("update test_table set i = i - 5;");
  sqlAndCompareResult("select * from test_table;", {{i(25)}});
}

// This test case covers a use case where compaction deletes files containing some of
// the chunk pages for a previously deleted chunk. This would previously result in a crash
// due to a bug where an attempt is made to restore old chunk pages for the deleted chunk,
// which results in an assertion failure because of incomplete chunk pages for the chunk.
TEST_F(OptimizeTableVacuumTest, PartialOldChunkPagesRemainAfterCompaction) {
  File_Namespace::FileMgr::setNumPagesPerDataFile(3);

  // The following page size (derived from `reserved header size (32) + big int size (8)`)
  // results in one big int entry/row per page.
  constexpr size_t ONE_BIG_INT_PAGE_SIZE = 40;
  sql("create table test_table (i bigint) with (max_rollback_epochs = 0, page_size = " +
      std::to_string(ONE_BIG_INT_PAGE_SIZE) + ", fragment_size = 4);");

  // Fill up fragment. This should occupy 5 data pages (4 pages for the big int entries
  // + 1 page for the $deleted$ chunk) across 2 files (since there are 3 pages per file,
  // per above setting).
  // sql("insert into test_table select * from (values (1), (2), (3), (4));");
  sql("insert into test_table select * from (values (1), (2), (3), (4));");
  assertFileAndFreePageCount(1, 4094, 2, 1);

  // Insert into second fragment uses 2 additional pages (1 for the big int entry + 1
  // for the $deleted$ chunk).
  sql("insert into test_table values (5);");
  assertFileAndFreePageCount(1, 4092, 3, 2);

  // Delete uses a new page for the $deleted$ chunk and
  // frees the old $deleted$ chunk page.
  sql("delete from test_table where i <= 4;");
  assertFileAndFreePageCount(1, 4092, 3, 2);

  // Inserts re-use the 2 free pages
  sql("insert into test_table values (6);");
  sql("insert into test_table values (7);");
  assertFileAndFreePageCount(1, 4092, 3, 0);

  // Optimize should delete the first data file, which contains the freed first 3 pages of
  // the deleted first fragment's "i" chunk. The second file should contain the freed 4th
  // page for the deleted first fragment's "i" chunk and data for chunks in the second
  // fragment. The third file should contain the freed page for the first fragment's
  // $deleted$ chunk and data for chunks in the second fragment.
  //
  // A crash would previously occur here due to an attempt to restore rolled-off pages for
  // the deleted first fragment's "i" chunk, starting with the 4th page for the chunk.
  sql("optimize table test_table with (vacuum = 'true');");
  assertFileAndFreePageCount(1, 4092, 2, 2);

  // Verify that subsequent queries work as expected
  sqlAndCompareResult("select * from test_table;", {{i(5)}, {i(6)}, {i(7)}});
}

TEST_F(OptimizeTableVacuumTest, UpdateAfterVacuumedDeletedFragment) {
  sql("create table test_table (i int) with (fragment_size = 2);");
  insertRange(1, 6);

  // Delete second fragment
  sql("delete from test_table where i = 3 or i = 4;");
  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{i(1)}, {i(2)}, {i(5)}, {i(6)}});

  // Do an update that changes metadata of third fragment
  sql("update test_table set i = 6 where i = 5;");
  sqlAndCompareResult("select * from test_table;", {{i(1)}, {i(2)}, {i(6)}, {i(6)}});
}

TEST_F(OptimizeTableVacuumTest, VarLengthArrayColumnWithFirstValueNull) {
  sql("create table test_table (i integer[]);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values ({1, 2, 3});");
  sql("insert into test_table values ({4, 5, 6});");
  sqlAndCompareResult("select * from test_table;",
                      {{Null}, {array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});
}

TEST_F(OptimizeTableVacuumTest, VarLengthArrayColumnWithLastValueNull) {
  sql("create table test_table (i integer[]);");
  sql("insert into test_table values ({1, 2, 3});");
  sql("insert into test_table values ({4, 5, 6});");
  sql("insert into test_table values (null);");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}, {Null}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{array({i(4), i(5), i(6)})}, {Null}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{array({i(4), i(5), i(6)})}, {Null}});
}

TEST_F(OptimizeTableVacuumTest, VarLengthArrayColumnWithFirstAndSubsequentNullValue) {
  sql("create table test_table (i integer[]);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values ({1, 2, 3});");
  sql("insert into test_table values ({4, 5, 6});");
  sqlAndCompareResult(
      "select * from test_table;",
      {{Null}, {Null}, {array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;",
                      {{Null}, {array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;",
                      {{Null}, {array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});
}

TEST_F(OptimizeTableVacuumTest, VarLengthArrayColumnWithAllNullValues) {
  sql("create table test_table (i integer[]);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values (null);");
  sqlAndCompareResult("select * from test_table;", {{Null}, {Null}, {Null}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{Null}, {Null}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{Null}, {Null}});
}

TEST_F(OptimizeTableVacuumTest, VarLengthArrayColumnWithNullInBetween) {
  sql("create table test_table (i integer[]);");
  sql("insert into test_table values ({1, 2, 3});");
  sql("insert into test_table values (null);");
  sql("insert into test_table values ({4, 5, 6});");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1), i(2), i(3)})}, {Null}, {array({i(4), i(5), i(6)})}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{Null}, {array({i(4), i(5), i(6)})}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{Null}, {array({i(4), i(5), i(6)})}});
}

TEST_F(OptimizeTableVacuumTest, VarLengthArrayColumnWithNullValueAndNullDeleted) {
  sql("create table test_table (i integer[]);");
  sql("insert into test_table values ({1, 2, 3});");
  sql("insert into test_table values (null);");
  sql("insert into test_table values ({4, 5, 6});");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1), i(2), i(3)})}, {Null}, {array({i(4), i(5), i(6)})}});

  sql("delete from test_table where rowid = 1;");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});
}

TEST_F(OptimizeTableVacuumTest,
       VarLengthArrayColumnWithNullAfterOffsetLessThanDefaultNullPadding) {
  sql("create table test_table (i integer[]);");
  // First array has a size that is less than ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE
  sql("insert into test_table values ({1});");
  sql("insert into test_table values (null);");
  sql("insert into test_table values ({4, 5, 6});");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1)})}, {Null}, {array({i(4), i(5), i(6)})}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{Null}, {array({i(4), i(5), i(6)})}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{Null}, {array({i(4), i(5), i(6)})}});
}

TEST_F(OptimizeTableVacuumTest, VarLengthArrayColumnWithAllRowsDeleted) {
  sql("create table test_table (i integer[]);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values ({1, 2, 3});");
  sql("insert into test_table values ({4, 5, 6});");
  sqlAndCompareResult("select * from test_table;",
                      {{Null}, {array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});

  sql("delete from test_table;");
  sqlAndCompareResult("select * from test_table;", {});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {});
}

TEST_F(OptimizeTableVacuumTest, VarLengthArrayColumnWithNotNullConstraint) {
  sql("create table test_table (i integer[] not null);");
  sql("insert into test_table values ({1, 2, 3});");
  sql("insert into test_table values ({4, 5, 6});");
  sqlAndCompareResult("select * from test_table;",
                      {{array({i(1), i(2), i(3)})}, {array({i(4), i(5), i(6)})}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{array({i(4), i(5), i(6)})}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{array({i(4), i(5), i(6)})}});
}

TEST_F(OptimizeTableVacuumTest, VarLengthTextArrayColumnWithNullValues) {
  sql("create table test_table (i integer, t text[]);");
  sql("insert into test_table values (1, null);");
  sql("insert into test_table values (2, null);");
  sqlAndCompareResult("select * from test_table;", {{i(1), Null}, {i(2), Null}});

  sql("delete from test_table where i = 1;");
  sqlAndCompareResult("select * from test_table;", {{i(2), Null}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{i(2), Null}});
}

TEST_F(OptimizeTableVacuumTest, NoneEncodedStringColumnWithFirstValueNull) {
  sql("create table test_table (t text encoding none);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values ('a');");
  sql("insert into test_table values ('b');");
  sqlAndCompareResult("select * from test_table;", {{Null}, {"a"}, {"b"}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{"a"}, {"b"}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{"a"}, {"b"}});
}

TEST_F(OptimizeTableVacuumTest, NoneEncodedStringColumnWithNullInBetween) {
  sql("create table test_table (t text encoding none);");
  sql("insert into test_table values ('a');");
  sql("insert into test_table values (null);");
  sql("insert into test_table values ('b');");
  sqlAndCompareResult("select * from test_table;", {{"a"}, {Null}, {"b"}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{Null}, {"b"}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{Null}, {"b"}});
}

TEST_F(OptimizeTableVacuumTest, NoneEncodedStringColumnWithLastValueNull) {
  sql("create table test_table (t text encoding none);");
  sql("insert into test_table values ('a');");
  sql("insert into test_table values ('b');");
  sql("insert into test_table values (null);");
  sqlAndCompareResult("select * from test_table;", {{"a"}, {"b"}, {Null}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{"b"}, {Null}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{"b"}, {Null}});
}

TEST_F(OptimizeTableVacuumTest, NoneEncodedStringColumnWithAllNullValues) {
  sql("create table test_table (t text encoding none);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values (null);");
  sql("insert into test_table values (null);");
  sqlAndCompareResult("select * from test_table;", {{Null}, {Null}, {Null}});

  sql("delete from test_table where rowid = 0;");
  sqlAndCompareResult("select * from test_table;", {{Null}, {Null}});

  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{Null}, {Null}});
}

class VarLenColumnUpdateTest : public DBHandlerTestFixture {
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
  }

  void TearDown() override {
    sql("drop table if exists test_table;");
    DBHandlerTestFixture::TearDown();
  }
};

// This covers a use case that would previously result in row deletion when reusing a data
// page containing non-zero values
TEST_F(VarLenColumnUpdateTest, ChunkUpdateReusesDataPage) {
  sql("create table test_table (i integer, t text encoding none) with "
      "(max_rollback_epochs = 0);");
  sql("insert into test_table values (1, 'a');");
  sql("insert into test_table values (2, 'b');");
  sql("insert into test_table values (3, 'c');");
  sqlAndCompareResult("select * from test_table;",
                      {{i(1), "a"}, {i(2), "b"}, {i(3), "c"}});

  // Rolls off page containing old version of chunk
  sql("update test_table set i = i + 1;");
  sqlAndCompareResult("select * from test_table;",
                      {{i(2), "a"}, {i(3), "b"}, {i(4), "c"}});

  // Update table to contain 2 rows
  sql("delete from test_table where i >= 4;");
  sqlAndCompareResult("select * from test_table;", {{i(2), "a"}, {i(3), "b"}});
  sql("optimize table test_table with (vacuum = 'true');");
  sqlAndCompareResult("select * from test_table;", {{i(2), "a"}, {i(3), "b"}});

  // Variable length update results in $deleted$ column with 3 rows. If inserted 0 value
  // for new row is not copied over to the new page, this would result in deletion of the
  // new row.
  sql("update test_table set t = 'e' where t = 'a';");
  sqlAndCompareResult("select * from test_table;", {{i(3), "b"}, {i(2), "e"}});
}

class OpportunisticVacuumingTest : public OptimizeTableVacuumTest {
 protected:
  void SetUp() override {
    OptimizeTableVacuumTest::SetUp();
    sql("drop table if exists test_table;");
    g_vacuum_min_selectivity = 0.1;
  }

  void TearDown() override {
    sql("drop table if exists test_table;");
    DBHandlerTestFixture::TearDown();
  }

  void insertRange(int start, int end, const std::string& str_value) {
    for (int value = start; value <= end; value++) {
      auto number_str = std::to_string(value);
      sql("insert into test_table values (" + number_str + ", '" + str_value +
          number_str + "');");
    }
  }

  void assertChunkContentAndMetadata(int32_t fragment_id,
                                     const std::vector<int32_t>& values,
                                     bool has_nulls = false,
                                     const std::string& table_name = "test_table") {
    auto [chunk, chunk_metadata] = getChunkAndMetadata("i", fragment_id, table_name);
    auto buffer = chunk.getBuffer();
    ASSERT_EQ(values.size() * sizeof(int32_t), buffer->size());
    assertCommonChunkMetadata(buffer, values.size(), has_nulls);

    if (!values.empty()) {
      auto min = *std::min_element(values.begin(), values.end());
      auto max = *std::max_element(values.begin(), values.end());
      assertMinAndMax(min, max, chunk_metadata);

      std::vector<int32_t> actual_values(values.size());
      buffer->read(reinterpret_cast<int8_t*>(actual_values.data()), buffer->size());
      EXPECT_EQ(values, actual_values);
    }
  }

  void assertTextChunkContentAndMetadata(int32_t fragment_id,
                                         const std::vector<std::string>& values,
                                         bool has_nulls = false,
                                         const std::string& table_name = "test_table") {
    auto [chunk, chunk_metadata] = getChunkAndMetadata("t", fragment_id, table_name);
    auto data_buffer = chunk.getBuffer();
    assertCommonChunkMetadata(data_buffer, values.size(), has_nulls);

    auto index_buffer = chunk.getIndexBuf();
    if (values.empty()) {
      EXPECT_EQ(static_cast<size_t>(0), data_buffer->size());
      EXPECT_EQ(static_cast<size_t>(0), index_buffer->size());
    } else {
      ASSERT_EQ(index_buffer->size() % sizeof(int32_t), static_cast<size_t>(0));
      std::vector<int32_t> index_values(index_buffer->size() / sizeof(int32_t));
      ASSERT_EQ(values.size() + 1, index_values.size());
      index_buffer->read(reinterpret_cast<int8_t*>(index_values.data()),
                         index_buffer->size());

      std::string data_values(data_buffer->size(), '\0');
      data_buffer->read(reinterpret_cast<int8_t*>(data_values.data()),
                        data_buffer->size());

      int32_t cumulative_length{0};
      for (size_t i = 0; i < values.size(); i++) {
        EXPECT_EQ(cumulative_length, index_values[i]);
        cumulative_length += values[i].size();
        EXPECT_EQ(
            values[i],
            data_values.substr(index_values[i], index_values[i + 1] - index_values[i]));
      }
      EXPECT_EQ(cumulative_length, index_values[values.size()]);
    }
  }

  void assertCommonChunkMetadata(AbstractBuffer* buffer,
                                 size_t num_elements,
                                 bool has_nulls) {
    ASSERT_TRUE(buffer->hasEncoder());
    std::shared_ptr<ChunkMetadata> chunk_metadata = std::make_shared<ChunkMetadata>();
    buffer->getEncoder()->getMetadata(chunk_metadata);
    EXPECT_EQ(buffer->size(), chunk_metadata->numBytes);
    EXPECT_EQ(num_elements, chunk_metadata->numElements);
    EXPECT_EQ(has_nulls, chunk_metadata->chunkStats.has_nulls);
  }

  void assertFragmentRowCount(size_t row_count) {
    auto td = getCatalog().getMetadataForTable("test_table");
    ASSERT_TRUE(td->fragmenter != nullptr);
    ASSERT_EQ(row_count, td->fragmenter->getNumRows());
  }

  void assertMinAndMax(int32_t min,
                       int32_t max,
                       std::shared_ptr<ChunkMetadata> chunk_metadata) {
    EXPECT_EQ(min, chunk_metadata->chunkStats.min.intval);
    EXPECT_EQ(max, chunk_metadata->chunkStats.max.intval);
  }

  void assertMinAndMax(int64_t min,
                       int64_t max,
                       std::shared_ptr<ChunkMetadata> chunk_metadata) {
    EXPECT_EQ(min, chunk_metadata->chunkStats.min.bigintval);
    EXPECT_EQ(max, chunk_metadata->chunkStats.max.bigintval);
  }

  void assertMinAndMax(float min,
                       float max,
                       std::shared_ptr<ChunkMetadata> chunk_metadata) {
    EXPECT_EQ(min, chunk_metadata->chunkStats.min.floatval);
    EXPECT_EQ(max, chunk_metadata->chunkStats.max.floatval);
  }

  void assertMinAndMax(double min,
                       double max,
                       std::shared_ptr<ChunkMetadata> chunk_metadata) {
    EXPECT_EQ(min, chunk_metadata->chunkStats.min.doubleval);
    EXPECT_EQ(max, chunk_metadata->chunkStats.max.doubleval);
  }

  std::pair<Chunk_NS::Chunk, std::shared_ptr<ChunkMetadata>> getChunkAndMetadata(
      const std::string& column_name,
      int32_t fragment_id,
      const std::string& table_name = "test_table") {
    auto& catalog = getCatalog();
    auto& data_mgr = catalog.getDataMgr();
    auto td = catalog.getMetadataForTable(table_name);
    auto cd = catalog.getMetadataForColumn(td->tableId, column_name);
    Chunk_NS::Chunk chunk;
    ChunkKey chunk_key{catalog.getDatabaseId(), td->tableId, cd->columnId, fragment_id};
    if (cd->columnType.is_varlen_indeed()) {
      chunk_key.emplace_back(2);
      chunk.setIndexBuffer(data_mgr.getChunkBuffer(chunk_key, MemoryLevel::DISK_LEVEL));
      chunk_key.back() = 1;
    }
    chunk.setBuffer(data_mgr.getChunkBuffer(chunk_key, MemoryLevel::DISK_LEVEL));
    CHECK(chunk.getBuffer()->hasEncoder());
    std::shared_ptr<ChunkMetadata> chunk_metadata = std::make_shared<ChunkMetadata>();
    chunk.getBuffer()->getEncoder()->getMetadata(chunk_metadata);
    return {chunk, chunk_metadata};
  }

  std::shared_ptr<StringDictionary> getStringDictionary(const std::string& column_name) {
    auto& catalog = getCatalog();
    auto td = catalog.getMetadataForTable("test_table");
    auto cd = catalog.getMetadataForColumn(td->tableId, column_name);
    CHECK(cd->columnType.is_dict_encoded_string());
    auto dict_metadata = catalog.getMetadataForDict(cd->columnType.get_comp_param());
    CHECK(dict_metadata);
    return dict_metadata->stringDict;
  }

  using DataTypesTestRow = std::
      tuple<int16_t, std::string, std::string, float, double, std::string, std::string>;

  void assertFragmentMetadataForDataTypesTest(const std::vector<DataTypesTestRow>& rows,
                                              int32_t fragment_id,
                                              bool has_nulls) {
    // Assert metadata for "i" chunk
    assertCommonChunkMetadata(
        rows, "i", fragment_id, has_nulls, sizeof(int16_t) * rows.size());
    assertMinMaxMetadata<int16_t, 0>(rows, "i", fragment_id, NULL_SMALLINT);

    // Assert metadata for "t" chunk
    assertCommonChunkMetadata(
        rows, "t", fragment_id, has_nulls, sizeof(int32_t) * rows.size());
    auto string_dictionary = getStringDictionary("t");
    assertMinMaxMetadata<int32_t, 1, true>(
        rows,
        "t",
        fragment_id,
        NULL_INT,
        [string_dictionary](const std::string& str_value) {
          return string_dictionary->getIdOfString(str_value);
        });

    // Assert metadata for "t_none" chunk
    size_t chunk_size{0};
    for (const auto& row : rows) {
      chunk_size += std::get<2>(row).size();
    }
    assertCommonChunkMetadata(rows, "t_none", fragment_id, has_nulls, chunk_size);
    // Skip min/max metadata check for none encoded string column, since this metadata is
    // not updated

    // Assert metadata for "f" chunk
    assertCommonChunkMetadata(
        rows, "f", fragment_id, has_nulls, sizeof(float) * rows.size());
    assertMinMaxMetadata<float, 3>(rows, "f", fragment_id, NULL_FLOAT);

    // Assert metadata for "d_arr" chunk
    assertCommonChunkMetadata(
        rows, "d_arr", fragment_id, has_nulls, sizeof(double) * rows.size());
    assertMinMaxMetadata<double, 4>(rows, "d_arr", fragment_id, NULL_DOUBLE);

    // Assert metadata for "tm_arr" chunk
    assertCommonChunkMetadata(
        rows, "tm_arr", fragment_id, has_nulls, sizeof(int64_t) * rows.size());
    // Skip min/max metadata check for variable length array column, since this metadata
    // is not updated

    // Assert metadata for "dt" chunk
    assertCommonChunkMetadata(
        rows, "dt", fragment_id, has_nulls, sizeof(int32_t) * rows.size());
    assertMinMaxMetadata<int64_t, 6, true>(
        rows, "dt", fragment_id, NULL_INT, [](const std::string& str_value) {
          SQLTypeInfo type{kDATE, false};
          return StringToDatum(str_value, type).bigintval;
        });
  }

  void assertCommonChunkMetadata(const std::vector<DataTypesTestRow>& rows,
                                 const std::string& column_name,
                                 int32_t fragment_id,
                                 bool has_nulls,
                                 size_t expected_chunk_size) {
    auto [chunk, chunk_metadata] = getChunkAndMetadata(column_name, fragment_id);
    ASSERT_EQ(expected_chunk_size, chunk.getBuffer()->size());
    assertCommonChunkMetadata(chunk.getBuffer(), rows.size(), has_nulls);
  }

  template <typename EncodedType, int32_t column_index, bool convert_input = false>
  void assertMinMaxMetadata(
      const std::vector<DataTypesTestRow>& rows,
      const std::string& column_name,
      int32_t fragment_id,
      EncodedType null_sentinel,
      std::function<EncodedType(const std::string&)> type_converter = nullptr) {
    auto chunk_metadata = getChunkAndMetadata(column_name, fragment_id).second;
    auto [min, max] = getMinAndMax<EncodedType, column_index, convert_input>(
        rows, null_sentinel, type_converter);
    assertMinAndMax(min, max, chunk_metadata);
  }

  template <typename EncodedType, int32_t column_index, bool convert_input>
  std::pair<EncodedType, EncodedType> getMinAndMax(
      const std::vector<DataTypesTestRow>& rows,
      EncodedType null_sentinel,
      std::function<EncodedType(const std::string&)> type_converter) {
    EncodedType min{std::numeric_limits<EncodedType>::max()},
        max{std::numeric_limits<EncodedType>::lowest()};
    for (const auto& row : rows) {
      if constexpr (convert_input) {
        auto str_value = std::get<column_index>(row);
        if (!str_value.empty()) {
          auto value = type_converter(str_value);
          min = std::min(min, value);
          max = std::max(max, value);
        }
      } else {
        auto value = std::get<column_index>(row);
        if (value != null_sentinel) {
          min = std::min(min, value);
          max = std::max(max, value);
        }
      }
    }
    return {min, max};
  }
};

TEST_F(OpportunisticVacuumingTest, DeletedFragment) {
  sql("create table test_table (i int) with (fragment_size = 3, "
      "max_rollback_epochs = 25);");
  OptimizeTableVacuumTest::insertRange(1, 5);

  assertChunkContentAndMetadata(0, {1, 2, 3});
  assertChunkContentAndMetadata(1, {4, 5});

  sql("delete from test_table where i <= 3;");

  assertChunkContentAndMetadata(0, {});
  assertChunkContentAndMetadata(1, {4, 5});
  assertFragmentRowCount(2);
  sqlAndCompareResult("select * from test_table;", {{i(4)}, {i(5)}});
}

TEST_F(OpportunisticVacuumingTest,
       DeleteQueryAndPercentDeletedRowsBelowSelectivityThreshold) {
  sql("create table test_table (i int) with (fragment_size = 5, "
      "max_rollback_epochs = 25);");
  OptimizeTableVacuumTest::insertRange(1, 10);

  assertChunkContentAndMetadata(0, {1, 2, 3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8, 9, 10});

  g_vacuum_min_selectivity = 0.45;
  sql("delete from test_table where i <= 2 or i >= 9;");

  assertChunkContentAndMetadata(0, {1, 2, 3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8, 9, 10});
  assertFragmentRowCount(10);
  sqlAndCompareResult("select * from test_table;",
                      {{i(3)}, {i(4)}, {i(5)}, {i(6)}, {i(7)}, {i(8)}});
}

TEST_F(OpportunisticVacuumingTest,
       DeleteQueryAndPercentDeletedRowsAboveSelectivityThreshold) {
  sql("create table test_table (i int) with (fragment_size = 5, "
      "max_rollback_epochs = 25);");
  OptimizeTableVacuumTest::insertRange(1, 10);

  assertChunkContentAndMetadata(0, {1, 2, 3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8, 9, 10});

  g_vacuum_min_selectivity = 0.35;
  sql("delete from test_table where i <= 2 or i >= 9;");

  assertChunkContentAndMetadata(0, {3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8});
  assertFragmentRowCount(6);
  sqlAndCompareResult("select * from test_table;",
                      {{i(3)}, {i(4)}, {i(5)}, {i(6)}, {i(7)}, {i(8)}});
}

TEST_F(OpportunisticVacuumingTest,
       DeleteQueryAndPercentDeletedRowsAboveSelectivityThresholdAndUncappedEpoch) {
  sql("create table test_table (i int) with (fragment_size = 5);");
  getCatalog().setUncappedTableEpoch("test_table");
  OptimizeTableVacuumTest::insertRange(1, 10);

  assertChunkContentAndMetadata(0, {1, 2, 3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8, 9, 10});

  g_vacuum_min_selectivity = 0.35;
  sql("delete from test_table where i <= 2 or i >= 9;");

  assertChunkContentAndMetadata(0, {1, 2, 3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8, 9, 10});
  assertFragmentRowCount(10);
  sqlAndCompareResult("select * from test_table;",
                      {{i(3)}, {i(4)}, {i(5)}, {i(6)}, {i(7)}, {i(8)}});
}

TEST_F(OpportunisticVacuumingTest, VarLenColumnUpdateAndEntireFragmentUpdated) {
  sql("create table test_table (i int, t text encoding none) with (fragment_size = 3, "
      "max_rollback_epochs = 25);");
  insertRange(1, 5, "abc");

  assertChunkContentAndMetadata(0, {1, 2, 3});
  assertChunkContentAndMetadata(1, {4, 5});

  assertTextChunkContentAndMetadata(0, {"abc1", "abc2", "abc3"});
  assertTextChunkContentAndMetadata(1, {"abc4", "abc5"});

  sql("update test_table set t = 'test_val' where i <= 3;");

  // When a variable length column is updated, the entire row is marked as deleted and a
  // new row with updated values is appended to the end of the table.
  assertChunkContentAndMetadata(0, {});
  assertChunkContentAndMetadata(1, {4, 5, 1});
  assertChunkContentAndMetadata(2, {2, 3});

  assertTextChunkContentAndMetadata(0, {});
  assertTextChunkContentAndMetadata(1, {"abc4", "abc5", "test_val"});
  assertTextChunkContentAndMetadata(2, {"test_val", "test_val"});

  assertFragmentRowCount(5);
  sqlAndCompareResult("select * from test_table;",
                      {{i(4), "abc4"},
                       {i(5), "abc5"},
                       {i(1), "test_val"},
                       {i(2), "test_val"},
                       {i(3), "test_val"}});
}

TEST_F(OpportunisticVacuumingTest,
       VarLenColumnUpdateAndPercentDeletedRowsBelowSelectivityThreshold) {
  sql("create table test_table (i int, t text encoding none) with (fragment_size = 5, "
      "max_rollback_epochs = 25);");
  insertRange(1, 10, "abc");

  assertChunkContentAndMetadata(0, {1, 2, 3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8, 9, 10});

  assertTextChunkContentAndMetadata(0, {"abc1", "abc2", "abc3", "abc4", "abc5"});
  assertTextChunkContentAndMetadata(1, {"abc6", "abc7", "abc8", "abc9", "abc10"});

  g_vacuum_min_selectivity = 0.45;
  sql("update test_table set t = 'test_val' where i <= 2 or i >= 9;");

  // When a variable length column is updated, the entire row is marked as deleted and a
  // new row with updated values is appended to the end of the table.
  assertChunkContentAndMetadata(0, {1, 2, 3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8, 9, 10});
  assertChunkContentAndMetadata(2, {1, 2, 9, 10});

  assertTextChunkContentAndMetadata(0, {"abc1", "abc2", "abc3", "abc4", "abc5"});
  assertTextChunkContentAndMetadata(1, {"abc6", "abc7", "abc8", "abc9", "abc10"});
  assertTextChunkContentAndMetadata(2, {"test_val", "test_val", "test_val", "test_val"});

  assertFragmentRowCount(14);
  sqlAndCompareResult("select * from test_table;",
                      {{i(3), "abc3"},
                       {i(4), "abc4"},
                       {i(5), "abc5"},
                       {i(6), "abc6"},
                       {i(7), "abc7"},
                       {i(8), "abc8"},
                       {i(1), "test_val"},
                       {i(2), "test_val"},
                       {i(9), "test_val"},
                       {i(10), "test_val"}});
}

TEST_F(OpportunisticVacuumingTest,
       VarLenColumnUpdateAndPercentDeletedRowsAboveSelectivityThreshold) {
  sql("create table test_table (i int, t text encoding none) with (fragment_size = 5, "
      "max_rollback_epochs = 25);");
  insertRange(1, 10, "abc");

  assertChunkContentAndMetadata(0, {1, 2, 3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8, 9, 10});

  assertTextChunkContentAndMetadata(0, {"abc1", "abc2", "abc3", "abc4", "abc5"});
  assertTextChunkContentAndMetadata(1, {"abc6", "abc7", "abc8", "abc9", "abc10"});

  g_vacuum_min_selectivity = 0.35;
  sql("update test_table set t = 'test_val' where i <= 2 or i >= 9;");

  // When a variable length column is updated, the entire row is marked as deleted and a
  // new row with updated values is appended to the end of the table.
  assertChunkContentAndMetadata(0, {3, 4, 5});
  assertChunkContentAndMetadata(1, {6, 7, 8});
  assertChunkContentAndMetadata(2, {1, 2, 9, 10});

  assertTextChunkContentAndMetadata(0, {"abc3", "abc4", "abc5"});
  assertTextChunkContentAndMetadata(1, {"abc6", "abc7", "abc8"});
  assertTextChunkContentAndMetadata(2, {"test_val", "test_val", "test_val", "test_val"});

  assertFragmentRowCount(10);
  sqlAndCompareResult("select * from test_table;",
                      {{i(3), "abc3"},
                       {i(4), "abc4"},
                       {i(5), "abc5"},
                       {i(6), "abc6"},
                       {i(7), "abc7"},
                       {i(8), "abc8"},
                       {i(1), "test_val"},
                       {i(2), "test_val"},
                       {i(9), "test_val"},
                       {i(10), "test_val"}});
}

TEST_F(OpportunisticVacuumingTest, DifferentDataTypesMetadataUpdate) {
  sql("create table test_table (i integer encoding fixed(16), t text, "
      "t_none text encoding none, f float, d_arr double[1], tm_arr timestamp[], "
      "dt date) with (fragment_size = 2, max_rollback_epochs = 25);");
  sql("insert into test_table values (1, 'test_1', 'test_1', 1.5, {10.5}, "
      "{'2021-01-01 00:10:00'}, '2021-01-01');");
  sql("insert into test_table values (2, 'test_2', 'test_2', 2.5, {20.5}, "
      "{'2021-02-01 00:10:00'}, '2021-02-01');");
  sql("insert into test_table values (3, 'test_3', 'test_3', 3.5, {30.5}, "
      "{'2021-03-01 00:10:00'}, '2021-03-01');");
  sql("insert into test_table values (4, 'test_4', 'test_4', 4.5, {40.5}, "
      "{'2021-04-01 00:10:00'}, '2021-04-01');");
  sql("insert into test_table values (5, 'test_5', 'test_5', 5.5, {50.5}, "
      "{'2021-05-01 00:10:00'}, '2021-05-01');");

  assertFragmentMetadataForDataTypesTest(
      {{1, "test_1", "test_1", 1.5f, 10.5, "2021-01-01 00:10:00", "2021-01-01"},
       {2, "test_2", "test_2", 2.5f, 20.5, "2021-02-01 00:10:00", "2021-02-01"}},
      0,
      false);
  assertFragmentMetadataForDataTypesTest(
      {{3, "test_3", "test_3", 3.5f, 30.5, "2021-03-01 00:10:00", "2021-03-01"},
       {4, "test_4", "test_4", 4.5f, 40.5, "2021-04-01 00:10:00", "2021-04-01"}},
      1,
      false);
  assertFragmentMetadataForDataTypesTest(
      {{5, "test_5", "test_5", 5.5f, 50.5, "2021-05-01 00:10:00", "2021-05-01"}},
      2,
      false);

  // Increase values
  sql("update test_table set i = 10, t = 'test_10', t_none = 'test_10', f = 100.5, "
      "d_arr = ARRAY[1000.5], tm_arr = ARRAY['2021-10-10 00:10:00'], "
      "dt = '2021-10-10' where i = 2;");

  // Set values to null
  sql("update test_table set i = null, t = null, t_none = null, f = null, "
      "d_arr = ARRAY[null], tm_arr = null, dt = null where i = 3;");

  // Decrease values
  sql("update test_table set i = 0, t = 'test', t_none = 'test', f = 0.5, "
      "d_arr = ARRAY[1.5], tm_arr = ARRAY['2020-01-01 00:10:00'], "
      "dt = '2020-01-01' where i = 5;");

  // When a variable length column is updated, the entire row is marked as deleted and a
  // new row with updated values is appended to the end of the table.
  assertFragmentMetadataForDataTypesTest(
      {{1, "test_1", "test_1", 1.5f, 10.5, "2021-01-01 00:10:00", "2021-01-01"}},
      0,
      false);
  assertFragmentMetadataForDataTypesTest(
      {{4, "test_4", "test_4", 4.5f, 40.5, "2021-04-01 00:10:00", "2021-04-01"}},
      1,
      false);
  assertFragmentMetadataForDataTypesTest(
      {{10, "test_10", "test_10", 100.5f, 1000.5, "2021-10-10 00:10:00", "2021-10-10"}},
      2,
      false);
  assertFragmentMetadataForDataTypesTest(
      {{NULL_SMALLINT, "", "", NULL_FLOAT, NULL_DOUBLE, "", ""},
       {0, "test", "test", 0.5f, 1.5, "2020-01-01 00:10:00", "2020-01-01"}},
      3,
      true);

  // clang-format off
  sqlAndCompareResult(
      "select * from test_table order by i;",
      {{i(0), "test", "test", 0.5f, array({1.5}), array({"2020-01-01 00:10:00"}), "2020-01-01"},
       {i(1), "test_1", "test_1", 1.5f, array({10.5}), array({"2021-01-01 00:10:00"}), "2021-01-01"},
       {i(4), "test_4", "test_4", 4.5f, array({40.5}), array({"2021-04-01 00:10:00"}), "2021-04-01"},
       {i(10), "test_10", "test_10", 100.5f, array({1000.5}), array({"2021-10-10 00:10:00"}), "2021-10-10"},
       {Null_i, Null, Null, NULL_FLOAT, array({NULL_DOUBLE}), Null, Null}});
  // clang-format on
}

TEST_F(OpportunisticVacuumingTest, VarLenColumnUpdateOfConsecutiveInnerFragments) {
  sql("create table test_table (i int, t text encoding none) with (fragment_size = 3, "
      "max_rollback_epochs = 25);");
  insertRange(1, 15, "abc");

  assertChunkContentAndMetadata(0, {1, 2, 3});
  assertChunkContentAndMetadata(1, {4, 5, 6});
  assertChunkContentAndMetadata(2, {7, 8, 9});
  assertChunkContentAndMetadata(3, {10, 11, 12});
  assertChunkContentAndMetadata(4, {13, 14, 15});

  assertTextChunkContentAndMetadata(0, {"abc1", "abc2", "abc3"});
  assertTextChunkContentAndMetadata(1, {"abc4", "abc5", "abc6"});
  assertTextChunkContentAndMetadata(2, {"abc7", "abc8", "abc9"});
  assertTextChunkContentAndMetadata(3, {"abc10", "abc11", "abc12"});
  assertTextChunkContentAndMetadata(4, {"abc13", "abc14", "abc15"});

  // Do an update that touches only the inner fragments
  sql("update test_table set t = 'test_val' where i = 6 or i = 7 or i = 11;");

  // When a variable length column is updated, the entire row is marked as deleted and a
  // new row with updated values is appended to the end of the table.
  assertChunkContentAndMetadata(0, {1, 2, 3});
  assertChunkContentAndMetadata(1, {4, 5});
  assertChunkContentAndMetadata(2, {8, 9});
  assertChunkContentAndMetadata(3, {10, 12});
  assertChunkContentAndMetadata(4, {13, 14, 15});
  assertChunkContentAndMetadata(5, {6, 7, 11});

  assertTextChunkContentAndMetadata(0, {"abc1", "abc2", "abc3"});
  assertTextChunkContentAndMetadata(1, {"abc4", "abc5"});
  assertTextChunkContentAndMetadata(2, {"abc8", "abc9"});
  assertTextChunkContentAndMetadata(3, {"abc10", "abc12"});
  assertTextChunkContentAndMetadata(4, {"abc13", "abc14", "abc15"});
  assertTextChunkContentAndMetadata(5, {"test_val", "test_val", "test_val"});

  assertFragmentRowCount(15);
  // clang-format off
  sqlAndCompareResult("select * from test_table;",
                      {{i(1), "abc1"}, {i(2), "abc2"}, {i(3), "abc3"},
                       {i(4), "abc4"}, {i(5), "abc5"},
                       {i(8), "abc8"}, {i(9), "abc9"},
                       {i(10), "abc10"},{i(12), "abc12"},
                       {i(13), "abc13"}, {i(14), "abc14"}, {i(15), "abc15"},
                       {i(6), "test_val"}, {i(7), "test_val"}, {i(11), "test_val"}});
  // clang-format on
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  // Disable automatic metadata update in order to ensure
  // that metadata is not automatically updated for other
  // tests that do and assert metadata updates.
  g_enable_auto_metadata_update = false;

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
