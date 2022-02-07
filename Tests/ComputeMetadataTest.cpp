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
#include "Catalog/CatalogSchemaProvider.h"
#include "DBHandlerTestHelpers.h"
#include "QueryEngine/TableOptimizer.h"

#include <gtest/gtest.h>
#include <string>
#include <utility>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_auto_metadata_update;
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
  executor->setSchemaProvider(
      std::make_shared<Catalog_Namespace::CatalogSchemaProvider>(&cat));
  TableOptimizer optimizer(td, executor.get(), cat);
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
  static void SetUpTestSuite() { g_enable_auto_metadata_update = false; }

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

TEST_P(MetadataUpdate, EncodedStringNull) {
  const auto& cat = getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  TestHelpers::ValuesGenerator gen(g_table_name);
  sql(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "'abc'", 0));
  recompute_metadata(td, cat);
  run_op_per_fragment(cat, td, check_fragment_metadata(8, 0, 1, false));

  sql(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "null", 0));
  recompute_metadata(td, cat);
  run_op_per_fragment(cat, td, check_fragment_metadata(8, 0, 1, true));
}

INSTANTIATE_TEST_SUITE_P(ShardedAndNonShardedTable,
                         MetadataUpdate,
                         testing::Values(false),
                         [](const auto& param_info) {
                           return (param_info.param ? "ShardedTable" : "NonShardedTable");
                         });

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
