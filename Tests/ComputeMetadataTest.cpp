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
  const auto shards = catalog.getPhysicalTablesDescriptors(td);
  for (const auto shard : shards) {
    auto* fragmenter = shard->fragmenter.get();
    CHECK(fragmenter);
    const auto table_info = fragmenter->getFragmentsForQuery();
    for (const auto& fragment : table_info.fragments) {
      f(fragment, std::forward<Args>(args)...);
    }
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
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    auto is_sharded = GetParam();
    int shard_count{1};
    if (is_sharded) {
      shard_count = 4;
    }
    std::string phrase_shard_key = (is_sharded ? ", SHARD KEY (skey)" : "");
    std::string phrase_shard_count =
        (is_sharded ? ", SHARD_COUNT = " + std::to_string(shard_count) : "");
    EXPECT_NO_THROW(sql("DROP TABLE IF EXISTS " + g_table_name + ";"));
    EXPECT_NO_THROW(sql("CREATE TABLE " + g_table_name +
                        " (x INT, y INT NOT NULL, z INT "
                        "ENCODING FIXED(8), a DOUBLE, b FLOAT, d DATE, dd DATE "
                        "ENCODING FIXED(16), c TEXT ENCODING DICT(32), skey int" +
                        phrase_shard_key +
                        ") WITH (FRAGMENT_SIZE=5, max_rollback_epochs = 25" +
                        phrase_shard_count + ");"));

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
  for (int sh = 0; sh < std::max(1, td->nShards); ++sh) {
    sql(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "'abc'", sh));
  }
  vacuum_and_recompute_metadata(td, cat);
  run_op_per_fragment(cat, td, check_fragment_metadata(8, 0, 1, false));

  for (int sh = 0; sh < std::max(1, td->nShards); ++sh) {
    sql(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "null", sh));
  }
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
                                              true));
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

INSTANTIATE_TEST_SUITE_P(ShardedAndNonShardedTable,
                         MetadataUpdate,
                         testing::Values(true, false),
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
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists test_table;");
  }

  void TearDown() override {
    sql("drop table test_table;");
    File_Namespace::FileMgr::setNumPagesPerDataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_DATA_FILE);
    File_Namespace::FileMgr::setNumPagesPerMetadataFile(
        File_Namespace::FileMgr::DEFAULT_NUM_PAGES_PER_METADATA_FILE);
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

  sql("create table test_table (i int);");
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

  // Compaction deletes the 4 free pages from above. Metadata
  // re-computation (in the optimize command) creates a new
  // metadata page/file.
  sql("optimize table test_table with (vacuum = 'true');");
  assertFileAndFreePageCount(3, 1, 2, 0);

  // Verify that subsequent queries work as expected
  sqlAndCompareResult("select * from test_table;", {{i(30)}});
  sql("update test_table set i = i - 5;");
  sqlAndCompareResult("select * from test_table;", {{i(25)}});
}

TEST_F(OptimizeTableVacuumTest, InsertAndCompactTableData) {
  // Each page write creates a new file
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);

  sql("create table test_table (i int) with (fragment_size = 2);");
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
  assertFileAndFreePageCount(2, 0, 2, 0);

  // Verify that subsequent queries work as expected
  sqlAndCompareResult("select * from test_table;", {{i(3)}});
  sql("insert into test_table values(5);");
  sqlAndCompareResult("select * from test_table;", {{i(3)}, {i(5)}});
}

TEST_F(OptimizeTableVacuumTest, UpdateAndCompactShardedTableData) {
  // Each page write creates a new file
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);

  sql("create table test_table (i int, f float, shard key(i)) with (shard_count = 4);");
  insertRange(1, 4, 2);
  // 12 chunk page writes and 12 metadata page writes. Each shard with
  // 3 metadata/data page writes for columns "i", "f", and "$deleted".
  assertFileAndFreePageCount(12, 0, 12, 0);

  // 2 additional pages/files for the "i" chunk per shard
  sql("update test_table set f = f + 10;");
  assertFileAndFreePageCount(16, 0, 16, 0);

  // 2 additional pages/files for the "i" chunk per shard
  sql("update test_table set f = f + 10;");
  assertFileAndFreePageCount(20, 0, 20, 0);

  // Rolls off/frees oldest 2 "f" chunk/metadata pages per shard
  sql("alter table test_table set max_rollback_epochs = 0;");
  assertFileAndFreePageCount(20, 8, 20, 8);

  // Compaction deletes the 16 free pages from above. Metadata
  // re-computation (in the optimize command) creates a new
  // metadata page/file per shard.
  sql("optimize table test_table with (vacuum = 'true');");
  assertFileAndFreePageCount(16, 4, 12, 0);

  // Verify that subsequent queries work as expected
  sqlAndCompareResult("select * from test_table order by i;",
                      {{i(1), 21.0f}, {i(2), 22.0f}, {i(3), 23.0f}, {i(4), 24.0f}});
  sql("update test_table set f = f - 5;");
  sqlAndCompareResult("select * from test_table order by i;",
                      {{i(1), 16.0f}, {i(2), 17.0f}, {i(3), 18.0f}, {i(4), 19.0f}});
}

TEST_F(OptimizeTableVacuumTest, InsertAndCompactShardedTableData) {
  // Each page write creates a new file
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(1);
  File_Namespace::FileMgr::setNumPagesPerDataFile(1);

  sql("create table test_table (i int, shard key(i)) with (fragment_size = 2, "
      "shard_count = 4, max_rollback_epochs = 25);");
  insertRange(1, 12, 1);
  // 4 chunk page writes per shard. 2 for the "i" column and "$deleted"
  // column each. 6 metadata page writes per shard for each insert.
  assertFileAndFreePageCount(24, 0, 16, 0);

  // 1 chunk page write per shard and 1 metadata page write per shard
  // for the updated "$deleted" chunk.
  sql("delete from test_table where i <= 8;");
  assertFileAndFreePageCount(28, 0, 20, 0);

  // Rolls off/frees oldest "$deleted" chunk page per shard and 3
  // metadata pages per shard (2 from initial insert and 1 from
  // "$deleted" chunk update).
  sql("alter table test_table set max_rollback_epochs = 0;");
  assertFileAndFreePageCount(28, 12, 20, 4);

  // Optimize frees up pages for the deleted 2 chunks per shard
  // (8 total). Compaction deletes the 16 free pages from above
  // in addition to the 8 freed pages.
  sql("optimize table test_table with (vacuum = 'true');");
  assertFileAndFreePageCount(8, 0, 8, 0);

  // Verify that subsequent queries work as expected
  sqlAndCompareResult("select * from test_table order by i;",
                      {{i(9)}, {i(10)}, {i(11)}, {i(12)}});
  sql("insert into test_table values(15);");
  sqlAndCompareResult("select * from test_table order by i;",
                      {{i(9)}, {i(10)}, {i(11)}, {i(12)}, {i(15)}});
}

TEST_F(OptimizeTableVacuumTest, MultiplePagesPerFile) {
  File_Namespace::FileMgr::setNumPagesPerMetadataFile(4);
  File_Namespace::FileMgr::setNumPagesPerDataFile(2);

  sql("create table test_table (i int);");
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
