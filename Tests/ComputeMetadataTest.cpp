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

#include "../Catalog/Catalog.h"
#include "../QueryEngine/Execute.h"
#include "../QueryEngine/TableOptimizer.h"
#include "../QueryRunner/QueryRunner.h"

#include <gtest/gtest.h>
#include <string>
#include <utility>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;

namespace {

inline void run_ddl_statement(const std::string& stmt) {
  QR::get()->runDDLStatement(stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, false, false);
}

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
void run_op_per_fragment(const TableDescriptor* td, FUNC f, Args&&... args) {
  const auto shards = QR::get()->getCatalog()->getPhysicalTablesDescriptors(td);
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
void run_op_per_fragment(const TableDescriptor* td,
                         FUNC f,
                         std::tuple<Args...> tuple,
                         std::index_sequence<Is...>) {
  run_op_per_fragment(td, f, std::forward<Args>(std::get<Is>(tuple))...);
}

template <typename FUNC, typename... Args>
void run_op_per_fragment(const TableDescriptor* td,
                         std::tuple<FUNC, std::tuple<Args...>> tuple) {
  run_op_per_fragment(
      td, std::get<0>(tuple), std::get<1>(tuple), std::index_sequence_for<Args...>{});
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

class MultiFragMetadataUpdate : public ::testing::Test {
  void SetUp() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS " + g_table_name + ";"));
    EXPECT_NO_THROW(run_ddl_statement(
        "CREATE TABLE " + g_table_name +
        " (x INT, y INT NOT NULL, z INT "
        "ENCODING FIXED(8), a DOUBLE, b FLOAT, d DATE, dd DATE "
        "ENCODING FIXED(16), c TEXT ENCODING DICT(32)) WITH (FRAGMENT_SIZE=4);"));

    TestHelpers::ValuesGenerator gen(g_table_name);

    for (int i = 0; i < 5; i++) {
      std::string date_str = i % 2 == 0 ? "'1/1/2019'" : "'2/2/2020'";
      const auto insert_query =
          gen(i, i, i, i * 1.1, i * 1.2, date_str, date_str, "'foo'");
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
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
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
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

      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
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
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    }
  }

  void TearDown() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS " + g_table_name + ";"));
  }
};

TEST_F(MultiFragMetadataUpdate, NoChanges) {
  std::vector<ChunkMetadataMap> metadata_for_fragments;
  {
    const auto cat = QR::get()->getCatalog();
    const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

    // Get chunk metadata before recomputing
    auto store_original_metadata =
        [&metadata_for_fragments](const Fragmenter_Namespace::FragmentInfo& fragment) {
          metadata_for_fragments.push_back(fragment.getChunkMetadataMapPhysical());
        };

    run_op_per_fragment(td, store_original_metadata);
    recompute_metadata(td, *cat);
  }

  // Make sure metadata matches after recomputing
  {
    const auto cat = QR::get()->getCatalog();
    const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

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
    run_op_per_fragment(td, check_metadata_equality);
  }
}

template <int NSHARDS>
class MetadataUpdate : public ::testing::Test {
  void SetUp() override {
    std::string phrase_shard_key = NSHARDS > 1 ? ", SHARD KEY (skey)" : "";
    std::string phrase_shard_count =
        NSHARDS > 1 ? ", SHARD_COUNT = " + std::to_string(NSHARDS) : "";
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS " + g_table_name + ";"));
    EXPECT_NO_THROW(run_ddl_statement(
        "CREATE TABLE " + g_table_name +
        " (x INT, y INT NOT NULL, z INT "
        "ENCODING FIXED(8), a DOUBLE, b FLOAT, d DATE, dd DATE "
        "ENCODING FIXED(16), c TEXT ENCODING DICT(32), skey int" +
        phrase_shard_key + ") WITH (FRAGMENT_SIZE=5" + phrase_shard_count + ");"));

    TestHelpers::ValuesGenerator gen(g_table_name);
    for (int sh = 0; sh < NSHARDS; ++sh) {
      run_multiple_agg(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "'foo'", sh),
                       ExecutorDeviceType::CPU);
      run_multiple_agg(gen(2, 2, 2, 2, 2, "'12/31/2012'", "'12/31/2012'", "'foo'", sh),
                       ExecutorDeviceType::CPU);
      run_multiple_agg(
          gen("null", 2, "null", "null", "null", "null", "'1/1/1940'", "'foo'", sh),
          ExecutorDeviceType::CPU);
    }
  }

  void TearDown() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS " + g_table_name + ";"));
  }
};

using MetadataUpdate_Unsharded = MetadataUpdate<1>;
using MetadataUpdate_Sharded = MetadataUpdate<4>;

#define BODY_F(test_class, test_name) test_class##_##test_name##_body()
#define TEST_F1(test_class, test_name, sharded_or_not) \
  TEST_F(test_class##_##sharded_or_not, test_name) { BODY_F(test_class, test_name); }
#define TEST_UNSHARDED_AND_SHARDED(test_class, test_name) \
  TEST_F1(test_class, test_name, Unsharded)               \
  TEST_F1(test_class, test_name, Sharded)

void BODY_F(MetadataUpdate, InitialMetadata) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_op_per_fragment(
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

void BODY_F(MetadataUpdate, IntUpdate) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("UPDATE " + g_table_name + " SET x = 3 WHERE x = 1;",
                   ExecutorDeviceType::CPU);

  // Check int col: expected range 1,3 nulls
  run_op_per_fragment(td, check_fragment_metadata(1, (int32_t)1, 3, true));

  run_multiple_agg("UPDATE " + g_table_name + " SET x = 0 WHERE x = 3;",
                   ExecutorDeviceType::CPU);

  recompute_metadata(td, *cat);
  // Check int col: expected range 1,2 nulls
  run_op_per_fragment(td, check_fragment_metadata(1, (int32_t)0, 2, true));
}

void BODY_F(MetadataUpdate, IntRemoveNull) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("UPDATE " + g_table_name + " SET x = 3;", ExecutorDeviceType::CPU);

  recompute_metadata(td, *cat);
  // Check int col: expected range 1,2 nulls
  run_op_per_fragment(td, check_fragment_metadata(1, (int32_t)3, 3, false));
}

void BODY_F(MetadataUpdate, NotNullInt) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("UPDATE " + g_table_name + " SET y = " +
                       std::to_string(std::numeric_limits<int32_t>::lowest() + 1) +
                       " WHERE y = 1;",
                   ExecutorDeviceType::CPU);
  // Check int col: expected range 1,3 nulls
  run_op_per_fragment(
      td,
      check_fragment_metadata(2, std::numeric_limits<int32_t>::lowest() + 1, 2, false));

  run_multiple_agg("UPDATE " + g_table_name + " SET y = 1;", ExecutorDeviceType::CPU);

  recompute_metadata(td, *cat);
  run_op_per_fragment(td, check_fragment_metadata(2, (int32_t)1, 1, false));
}

void BODY_F(MetadataUpdate, DateNarrowRange) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("UPDATE " + g_table_name + " SET d = '1/1/2010';",
                   ExecutorDeviceType::CPU);

  recompute_metadata(td, *cat);
  // Check date in days 32 col: expected range 1262304000,1262304000 nulls
  run_op_per_fragment(td,
                      check_fragment_metadata(6, (int64_t)1262304000, 1262304000, false));
}

void BODY_F(MetadataUpdate, SmallDateNarrowMin) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg(
      "UPDATE " + g_table_name + " SET dd = '1/1/2010' WHERE dd = '1/1/1940';",
      ExecutorDeviceType::CPU);

  recompute_metadata(td, *cat);
  run_op_per_fragment(td,
                      check_fragment_metadata(7, (int64_t)1262304000, 1356912000, false));
}

void BODY_F(MetadataUpdate, SmallDateNarrowMax) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg(
      "UPDATE " + g_table_name + " SET dd = '1/1/2010' WHERE dd = '12/31/2012';",
      ExecutorDeviceType::CPU);

  recompute_metadata(td, *cat);
  run_op_per_fragment(td,
                      check_fragment_metadata(7, (int64_t)-946771200, 1262304000, false));
}

void BODY_F(MetadataUpdate, DeleteReset) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("DELETE FROM  " + g_table_name + " WHERE dd = '12/31/2012';",
                   ExecutorDeviceType::CPU);
  run_op_per_fragment(td, check_fragment_metadata(-1, false, true, false));

  vacuum_and_recompute_metadata(td, *cat);
  run_op_per_fragment(td, check_fragment_metadata(-1, false, false, false));
}

void BODY_F(MetadataUpdate, EncodedStringNull) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  TestHelpers::ValuesGenerator gen(g_table_name);
  for (int sh = 0; sh < std::max(1, td->nShards); ++sh) {
    run_multiple_agg(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "'abc'", sh),
                     ExecutorDeviceType::CPU);
  }
  vacuum_and_recompute_metadata(td, *cat);
  run_op_per_fragment(td, check_fragment_metadata(8, 0, 1, false));

  for (int sh = 0; sh < std::max(1, td->nShards); ++sh) {
    run_multiple_agg(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "null", sh),
                     ExecutorDeviceType::CPU);
  }
  vacuum_and_recompute_metadata(td, *cat);
  run_op_per_fragment(td, check_fragment_metadata(8, 0, 1, true));
}

void BODY_F(MetadataUpdate, AlterAfterOptimize) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);
  run_op_per_fragment(td, check_fragment_metadata(1, 1, 2, true));
  run_multiple_agg("DELETE FROM  " + g_table_name + " WHERE x IS NULL;",
                   ExecutorDeviceType::CPU);
  vacuum_and_recompute_metadata(td, *cat);
  run_op_per_fragment(td, check_fragment_metadata(1, 1, 2, false));
  // test ADD one column
  EXPECT_NO_THROW(
      run_ddl_statement("ALTER TABLE " + g_table_name + " ADD (c99 int default 99);"));
  run_op_per_fragment(td, check_fragment_metadata(12, 99, 99, false));
  // test ADD multiple columns
  EXPECT_NO_THROW(run_ddl_statement("ALTER TABLE " + g_table_name +
                                    " ADD (c88 int default 88, cnn int);"));
  run_op_per_fragment(td, check_fragment_metadata(13, 88, 88, false));
  run_op_per_fragment(td,
                      check_fragment_metadata(14,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              true));
}

void BODY_F(MetadataUpdate, AlterAfterEmptied) {
  const auto cat = QR::get()->getCatalog();
  const auto td = cat->getMetadataForTable(g_table_name, /*populateFragmenter=*/true);
  run_multiple_agg("DELETE FROM  " + g_table_name + ";", ExecutorDeviceType::CPU);
  vacuum_and_recompute_metadata(td, *cat);
  run_op_per_fragment(td,
                      check_fragment_metadata(1,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              false));
  // test ADD one column to make sure column is added even if no row exists
  EXPECT_NO_THROW(
      run_ddl_statement("ALTER TABLE " + g_table_name + " ADD (c99 int default 99);"));
  run_op_per_fragment(td,
                      check_fragment_metadata(12,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              true));
  // test ADD multiple columns
  EXPECT_NO_THROW(run_ddl_statement("ALTER TABLE " + g_table_name +
                                    " ADD (c88 int default 88, cnn int);"));
  run_op_per_fragment(td, check_fragment_metadata(13, 88, 88, false));
  run_op_per_fragment(td,
                      check_fragment_metadata(14,
                                              std::numeric_limits<int32_t>::max(),
                                              std::numeric_limits<int32_t>::lowest(),
                                              true));
}
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, AlterAfterEmptied)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, AlterAfterOptimize)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, InitialMetadata)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, IntUpdate)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, IntRemoveNull)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, NotNullInt)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, DateNarrowRange)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, SmallDateNarrowMin)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, SmallDateNarrowMax)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, DeleteReset)
TEST_UNSHARDED_AND_SHARDED(MetadataUpdate, EncodedStringNull)

int main(int argc, char** argv) {
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
