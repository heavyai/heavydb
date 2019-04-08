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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

namespace {

std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QueryRunner::run_ddl_statement(create_table_stmt, g_session);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QueryRunner::run_multiple_agg(
      query_str, g_session, device_type, false, false, nullptr);
}

template <typename FUNC>
void run_op_per_fragment(const TableDescriptor* td, FUNC f) {
  auto* fragmenter = td->fragmenter;
  CHECK(fragmenter);
  const auto table_info = fragmenter->getFragmentsForQuery();
  for (const auto& fragment : table_info.fragments) {
    f(fragment);
  }
}

void recompute_metadata(const TableDescriptor* td,
                        const Catalog_Namespace::Catalog& cat) {
  auto executor = Executor::getExecutor(cat.getCurrentDB().dbId);
  TableOptimizer optimizer(td, executor.get(), cat);
  EXPECT_NO_THROW(optimizer.recomputeMetadata());
}

static const std::string g_table_name{"metadata_test"};

}  // namespace

class MultiFragMetadataUpdate : public ::testing::Test {
  virtual void SetUp() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS " + g_table_name + ";"));
    EXPECT_NO_THROW(run_ddl_statement(
        "CREATE TABLE " + g_table_name +
        " (x INT, y INT NOT NULL, z INT "
        "ENCODING FIXED(8), a DOUBLE, b FLOAT, d DATE, dd DATE "
        "ENCODING FIXED(16), c TEXT ENCODING DICT(32)) WITH (FRAGMENT_SIZE=4);"));

    TestHelpers::ValuesGenerator gen(g_table_name);

    for (size_t i = 0; i < 5; i++) {
      std::string date_str = i % 2 == 0 ? "'1/1/2019'" : "'2/2/2020'";
      const auto insert_query =
          gen(i, i, i, i * 1.1, i * 1.2, date_str, date_str, "'foo'");
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    }

    for (size_t i = 0; i < 5; i++) {
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

    for (size_t i = 0; i < 5; i++) {
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

  virtual void TearDown() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS " + g_table_name + ";"));
  }
};

TEST_F(MultiFragMetadataUpdate, NoChanges) {
  std::vector<std::map<int, ChunkMetadata>> metadata_for_fragments;
  {
    const auto& cat = g_session->getCatalog();
    const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

    // Get chunk metadata before recomputing
    auto store_original_metadata =
        [&metadata_for_fragments](const Fragmenter_Namespace::FragmentInfo& fragment) {
          metadata_for_fragments.push_back(fragment.getChunkMetadataMapPhysical());
        };

    run_op_per_fragment(td, store_original_metadata);
    recompute_metadata(td, cat);
  }

  // Make sure metadata matches after recomputing
  {
    const auto& cat = g_session->getCatalog();
    const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

    auto* fragmenter = td->fragmenter;
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

class MetadataUpdate : public ::testing::Test {
  virtual void SetUp() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS " + g_table_name + ";"));
    EXPECT_NO_THROW(
        run_ddl_statement("CREATE TABLE " + g_table_name +
                          " (x INT, y INT NOT NULL, z INT "
                          "ENCODING FIXED(8), a DOUBLE, b FLOAT, d DATE, dd DATE "
                          "ENCODING FIXED(16), c TEXT ENCODING DICT(32)) WITH "
                          "(FRAGMENT_SIZE=5);"));

    TestHelpers::ValuesGenerator gen(g_table_name);
    run_multiple_agg(gen(1, 1, 1, 1, 1, "'1/1/2010'", "'1/1/2010'", "'foo'"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(gen(2, 2, 2, 2, 2, "'12/31/2012'", "'12/31/2012'", "'foo'"),
                     ExecutorDeviceType::CPU);
    run_multiple_agg(
        gen("null", 2, "null", "null", "null", "null", "'1/1/1940'", "'foo'"),
        ExecutorDeviceType::CPU);
  }

  virtual void TearDown() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS " + g_table_name + ";"));
  }
};

TEST_F(MetadataUpdate, InitialMetadata) {
  const auto& cat = g_session->getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  auto check_initial_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        // Check int col: expected range 1,2 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(1);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.intval, 1);
          ASSERT_EQ(chunk_metadata.chunkStats.max.intval, 2);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, true);
        }

        // Check int not null col: expected range 1,2 no nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(2);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.intval, 1);
          ASSERT_EQ(chunk_metadata.chunkStats.max.intval, 2);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, false);
        }

        // Check int encoded call: expected range 1,2 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(3);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.intval, 1);
          ASSERT_EQ(chunk_metadata.chunkStats.max.intval, 2);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, true);
        }

        // Check double col: expected range 1.0,2.0 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(4);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.doubleval, static_cast<double>(1.0));
          ASSERT_EQ(chunk_metadata.chunkStats.max.doubleval, static_cast<double>(2.0));
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, true);
        }

        // Check float col: expected range 1.0,2.0 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(5);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.floatval, static_cast<float>(1.0));
          ASSERT_EQ(chunk_metadata.chunkStats.max.floatval, static_cast<float>(2.0));
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, true);
        }

        // Check date in days 32 col: expected range 1262304000,1356912000 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(6);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.timeval, 1262304000);
          ASSERT_EQ(chunk_metadata.chunkStats.max.timeval, 1356912000);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, true);
        }

        // Check date in days 16 col: expected range -946771200,1356912000 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(7);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.timeval, -946771200);
          ASSERT_EQ(chunk_metadata.chunkStats.max.timeval, 1356912000);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, false);
        }
      };

  run_op_per_fragment(td, check_initial_metadata_values);
}

TEST_F(MetadataUpdate, IntUpdate) {
  const auto& cat = g_session->getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("UPDATE " + g_table_name + " SET x = 3 WHERE x = 1;",
                   ExecutorDeviceType::CPU);

  auto check_initial_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        // Check int col: expected range 1,3 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(1);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.intval, 1);
          ASSERT_EQ(chunk_metadata.chunkStats.max.intval, 3);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, true);
        }
      };

  run_op_per_fragment(td, check_initial_metadata_values);

  run_multiple_agg("UPDATE " + g_table_name + " SET x = 0 WHERE x = 3;",
                   ExecutorDeviceType::CPU);

  recompute_metadata(td, cat);

  auto check_recomputed_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        // Check int col: expected range 1,2 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(1);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.intval, 0);
          ASSERT_EQ(chunk_metadata.chunkStats.max.intval, 2);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, true);
        }
      };

  run_op_per_fragment(td, check_recomputed_metadata_values);
}

TEST_F(MetadataUpdate, IntRemoveNull) {
  const auto& cat = g_session->getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("UPDATE " + g_table_name + " SET x = 3;", ExecutorDeviceType::CPU);

  recompute_metadata(td, cat);

  auto check_recomputed_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        // Check int col: expected range 1,2 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(1);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.intval, 3);
          ASSERT_EQ(chunk_metadata.chunkStats.max.intval, 3);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, false);
        }
      };

  run_op_per_fragment(td, check_recomputed_metadata_values);
}

TEST_F(MetadataUpdate, NotNullInt) {
  const auto& cat = g_session->getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("UPDATE " + g_table_name + " SET y = " +
                       std::to_string(std::numeric_limits<int32_t>::lowest() + 1) +
                       " WHERE y = 1;",
                   ExecutorDeviceType::CPU);

  auto check_initial_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        // Check int col: expected range 1,3 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(2);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.intval,
                    std::numeric_limits<int32_t>::lowest() + 1);
          ASSERT_EQ(chunk_metadata.chunkStats.max.intval, 2);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, false);
        }
      };

  run_op_per_fragment(td, check_initial_metadata_values);

  run_multiple_agg("UPDATE " + g_table_name + " SET y = 1;", ExecutorDeviceType::CPU);

  recompute_metadata(td, cat);

  auto check_recomputed_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        // Check int col: expected range 1,2 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(2);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.intval, 1);
          ASSERT_EQ(chunk_metadata.chunkStats.max.intval, 1);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, false);
        }
      };

  run_op_per_fragment(td, check_recomputed_metadata_values);
}

TEST_F(MetadataUpdate, DateNarrowRange) {
  const auto& cat = g_session->getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg("UPDATE " + g_table_name + " SET d = '1/1/2010';",
                   ExecutorDeviceType::CPU);

  recompute_metadata(td, cat);

  auto check_recomputed_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        // Check date in days 32 col: expected range 1262304000,1262304000 nulls
        {
          const auto chunk_metadata_itr = metadata_map.find(6);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.timeval, 1262304000);
          ASSERT_EQ(chunk_metadata.chunkStats.max.timeval, 1262304000);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, false);
        }
      };

  run_op_per_fragment(td, check_recomputed_metadata_values);
}

TEST_F(MetadataUpdate, SmallDateNarrowMin) {
  const auto& cat = g_session->getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg(
      "UPDATE " + g_table_name + " SET dd = '1/1/2010' WHERE dd = '1/1/1940';",
      ExecutorDeviceType::CPU);

  recompute_metadata(td, cat);

  auto check_recomputed_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        {
          const auto chunk_metadata_itr = metadata_map.find(7);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.timeval, 1262304000);
          ASSERT_EQ(chunk_metadata.chunkStats.max.timeval, 1356912000);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, false);
        }
      };

  run_op_per_fragment(td, check_recomputed_metadata_values);
}

TEST_F(MetadataUpdate, SmallDateNarrowMax) {
  const auto& cat = g_session->getCatalog();
  const auto td = cat.getMetadataForTable(g_table_name, /*populateFragmenter=*/true);

  run_multiple_agg(
      "UPDATE " + g_table_name + " SET dd = '1/1/2010' WHERE dd = '12/31/2012';",
      ExecutorDeviceType::CPU);

  recompute_metadata(td, cat);

  auto check_recomputed_metadata_values =
      [](const Fragmenter_Namespace::FragmentInfo& fragment) {
        const auto metadata_map = fragment.getChunkMetadataMapPhysical();
        {
          const auto chunk_metadata_itr = metadata_map.find(7);
          CHECK(chunk_metadata_itr != metadata_map.end());
          const auto& chunk_metadata = chunk_metadata_itr->second;
          ASSERT_EQ(chunk_metadata.chunkStats.min.timeval, -946771200);
          ASSERT_EQ(chunk_metadata.chunkStats.max.timeval, 1262304000);
          ASSERT_EQ(chunk_metadata.chunkStats.has_nulls, false);
        }
      };

  run_op_per_fragment(td, check_recomputed_metadata_values);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  g_session.reset(QueryRunner::get_session(BASE_PATH));

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  g_session.reset(nullptr);
  return err;
}