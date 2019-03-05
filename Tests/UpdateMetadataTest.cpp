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

#include <boost/program_options.hpp>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std::literals::string_literals;

namespace {

bool g_keep_test_data{false};
std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QueryRunner::run_ddl_statement(create_table_stmt, g_session);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QueryRunner::run_multiple_agg(
      query_str, g_session, device_type, false, false, nullptr);
}

}  // namespace

class MetadataUpdate : public ::testing::Test {
  void SetUp() override {}

  void TearDown() override {}
};

namespace {

class TableCycler {
 public:
  TableCycler(std::string const& drop_if_statement,
              std::string const& create_statement,
              std::string const& drop_statement)
      : drop_if_statement_(drop_if_statement)
      , create_statement_(create_statement)
      , drop_statement_(drop_statement) {
    run_ddl_statement(drop_if_statement_);
    run_ddl_statement(create_statement_);
  };

  TableCycler(TableCycler&& rhs)
      : drop_if_statement_(std::move(rhs.drop_if_statement_))
      , create_statement_(std::move(rhs.create_statement_))
      , drop_statement_(std::move(rhs.drop_statement_)) {}

  TableCycler(TableCycler const&) = delete;
  TableCycler& operator=(TableCycler const&) = delete;

  template <typename TEST_FUNCTOR>
  auto operator()(TEST_FUNCTOR f) const {
    f();
  }

  ~TableCycler() {
    if (!drop_statement_.empty()) {
      if (!g_keep_test_data) {
        run_ddl_statement(drop_statement_);
      }
    }
  }

 private:
  std::string drop_if_statement_;
  std::string create_statement_;
  std::string drop_statement_;
};

auto make_table_cycler(char const* table_name, char const* column_type) {
  auto drop_if_stmt = "drop table if exists "s + std::string(table_name) + ";"s;
  auto create_stmt = "create table "s + std::string(table_name) + "( x "s +
                     std::string(column_type) + " );"s;
  auto drop_stmt = "drop table "s + std::string(table_name) + ";"s;

  return TableCycler(drop_if_stmt, create_stmt, drop_stmt);
}

auto query = [](std::string query_str) {
  run_multiple_agg(query_str.c_str(), ExecutorDeviceType::CPU);
};

auto get_metadata_vec =
    [](std::string table_name, std::string column_name = "x"s) -> auto {
  auto& cat = g_session->getCatalog();
  auto& data_manager = cat.getDataMgr();

  auto table_desc = cat.getMetadataForTable(table_name);
  auto column_desc = cat.getMetadataForColumn(table_desc->tableId, column_name);
  ChunkKey timestamp_ck{
      cat.getCurrentDB().dbId, table_desc->tableId, column_desc->columnId};

  std::vector<std::pair<ChunkKey, ChunkMetadata>> chunkMetadataVec;
  data_manager.getChunkMetadataVecForKeyPrefix(chunkMetadataVec, timestamp_ck);

  return chunkMetadataVec;
};

}  // namespace

TEST_F(MetadataUpdate, MetadataTimestampNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  // For timestamp column x:
  // 1 - Check that the MIN metadata value is unperturbed
  // 2 - Check that the presence of null is properly signaled
  make_table_cycler("timestamp_null_table", "timestamp")([&] {
    query("insert into timestamp_null_table values ('1946-06-14 10:54:00');");
    query("insert into timestamp_null_table values ('2017-01-20 12:00:00');");

    auto pre_metadata = get_metadata_vec("timestamp_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update timestamp_null_table set x = NULL where x < '2000-01-01 00:00:00';");

    auto post_metadata = get_metadata_vec("timestamp_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // Check that a null update doesn't change the before and after min-max
    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.bigintval,
              post_metadata_chunk.chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval,
              post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update timestamp_null_table set x='1000-01-01 00:00:00' where x is NULL;");
    post_metadata = get_metadata_vec("timestamp_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // At this point, we prove that -INT_MIN is not set because
    // of a null update, and that the min is constrained to
    // the integer value corresponding to the minimum timestamp
    //
    // Addendum:  There is no range checking on timestamps, however.
    auto timestamp_minval = post_metadata_chunk.chunkStats.min.bigintval;
    ASSERT_EQ(timestamp_minval, -30610224000);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    // Set the max, then verify that the min/max are the max possible for timestamp ranges
    // and also verify that the has null is still true
    query(
        "update timestamp_null_table set x='12/31/2900 23:59:59.999' where x='1000-01-01 "
        "00:00:00';");
    post_metadata = get_metadata_vec("timestamp_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval, 29379542399);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.bigintval, -30610224000);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.bigintval,
              pre_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.bigintval,
              pre_metadata_chunk.chunkStats.min.bigintval);
  });
}

TEST_F(MetadataUpdate, MetadataTimestampNotNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("timestamp_not_null_table", "timestamp not null")([&] {
    query("insert into timestamp_not_null_table values ('1946-06-14 10:54:00');");
    query("insert into timestamp_not_null_table values ('2017-01-20 12:00:00');");

    auto pre_metadata = get_metadata_vec("timestamp_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    // Should throw
    EXPECT_THROW(query("update timestamp_not_null_table set x = NULL where x < "
                       "'2000-01-01 00:00:00';"),
                 std::runtime_error);

    // This query currently flips has_null to true; may need correcting
    // query( "insert into timestamp_not_null_table values ('-9223372036854775808');" );

    // This throws because it complains about NULL into a NOT NULL column, but it should
    // accept the non-null token.
    EXPECT_THROW(query("update timestamp_not_null_table set x='-9223372036854775808' "
                       "where x < '2000-01-01 00:00:00';"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("timestamp_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // This should never flip to true if it's a not-null table
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    query(
        "update timestamp_not_null_table set x=cast('-9223372036854775807' as timestamp) "
        "where x < '2000-01-01 00:00:00';");
    query(
        "update timestamp_not_null_table set x=cast('29379542399' as timestamp) where x "
        "> '2000-01-01 00:00:00';");

    post_metadata = get_metadata_vec("timestamp_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_timestamp_minval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.min.bigintval);
    int64_t post_timestamp_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_timestamp_minval, -9223372036854775807);
    ASSERT_EQ(post_timestamp_maxval, 29379542399);

    // Check out-of-range max timestamp update
    query(
        "update timestamp_not_null_table set x=cast('9223372036854775807' as timestamp) "
        "where x < '2000-01-01 00:00:00';");
    post_metadata = get_metadata_vec("timestamp_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;
    post_timestamp_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.bigintval);

    ASSERT_EQ(post_timestamp_maxval, 9223372036854775807);
  });
}

TEST_F(MetadataUpdate, MetadataTimeNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("time_null_table", "time")([&] {
    query("insert into time_null_table values ('10:54:00');");
    query("insert into time_null_table values ('12:00:00');");

    auto pre_metadata = get_metadata_vec("time_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update time_null_table set x = NULL where x < '12:00:00';");

    auto post_metadata = get_metadata_vec("time_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // Check that a null update doesn't change the before and after min-max
    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.bigintval,
              post_metadata_chunk.chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval,
              post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update time_null_table set x='00:00:00' where x is NULL;");
    post_metadata = get_metadata_vec("time_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto time_minval = post_metadata_chunk.chunkStats.min.bigintval;
    ASSERT_EQ(time_minval, 0);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    // Set the max, then verify that the min/max are the max possible for time ranges
    // and also verify that the has null is still true
    query("update time_null_table set x='23:59:59.999' where x='12:00:00';");
    post_metadata = get_metadata_vec("time_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval, 86399);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.bigintval, 0);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.bigintval,
              pre_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.bigintval,
              pre_metadata_chunk.chunkStats.min.bigintval);

    EXPECT_THROW(query("update time_null_table set x=cast('-9223372036854775807' as "
                       "time) where x = '00:00:00';"),
                 std::runtime_error);
    EXPECT_THROW(query("update time_null_table set x=cast('9223372036854775807' as time) "
                       "where x = '23:59:59';"),
                 std::runtime_error);
  });
}

TEST_F(MetadataUpdate, MetadataTimeNotNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("time_not_null_table", "time not null")([&] {
    query("insert into time_not_null_table values ('10:54:00');");
    query("insert into time_not_null_table values ('12:00:00');");

    auto pre_metadata = get_metadata_vec("time_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    // Should throw
    EXPECT_THROW(query("update time_not_null_table set x = NULL where x < "
                       "'12:00:00';"),
                 std::runtime_error);

    EXPECT_THROW(query("update time_not_null_table set x=cast('-9223372036854775808' as "
                       "time) where x < '12:00:00';"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("time_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // This should never flip to true if it's a not-null table
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update time_not_null_table set x=cast('-9223372036854775807' as "
                       "time) where x < '12:00:00';"),
                 std::runtime_error);
    EXPECT_THROW(query("update time_not_null_table set x=cast('9223372036854775807' as "
                       "time) where x > '2000-01-01 00:00:00';"),
                 std::runtime_error);

    query("update time_not_null_table set x='23:59:59.999' where x='12:00:00';");
    query("update time_not_null_table set x='00:00:00' where x='10:54:00';");

    post_metadata = get_metadata_vec("time_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval, 86399);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.bigintval, 0);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.bigintval,
              pre_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.bigintval,
              pre_metadata_chunk.chunkStats.min.bigintval);
  });
}

TEST_F(MetadataUpdate, MetadataDateNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("date_null_table", "date")([&] {
    query("insert into date_null_table values ('1946-06-14');");
    query("insert into date_null_table values ('1974-02-11');");

    auto pre_metadata = get_metadata_vec("date_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update date_null_table set x = NULL where x < '1950-01-01'");

    auto post_metadata = get_metadata_vec("date_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // Check that a null update doesn't change the before and after min-max
    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.bigintval,
              post_metadata_chunk.chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval,
              post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update date_null_table set x='-185542587187199' where x is NULL;");
    post_metadata = get_metadata_vec("date_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto date_minval = post_metadata_chunk.chunkStats.min.bigintval;
    EXPECT_EQ(date_minval, -185542587187200);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update date_null_table set x='185542587187199' where x>'1950-01-01';");
    post_metadata = get_metadata_vec("date_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    EXPECT_EQ(post_metadata_chunk.chunkStats.max.bigintval, 185542587100800);
    EXPECT_EQ(post_metadata_chunk.chunkStats.min.bigintval, -185542587187200);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.bigintval,
              pre_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.bigintval,
              pre_metadata_chunk.chunkStats.min.bigintval);
  });
}

TEST_F(MetadataUpdate, MetadataDecimalNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("decimal_null_table", "decimal(18,1)")([&] {
    query("insert into decimal_null_table values ( 10.1 );");
    query("insert into decimal_null_table values ( 20.1 );");

    auto pre_metadata = get_metadata_vec("decimal_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update decimal_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("decimal_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.bigintval,
              post_metadata_chunk.chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval,
              post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    EXPECT_THROW(query("update decimal_null_table set x = -922337203685477580.7;"),
                 std::runtime_error);
    EXPECT_THROW(query("update decimal_null_table set x = 922337203685477580.7;"),
                 std::runtime_error);

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.bigintval,
              post_metadata_chunk.chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval,
              post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update decimal_null_table set x = 10.0 where x is NULL;");
    query("update decimal_null_table set x = 20.2 where x > 15;");

    post_metadata = get_metadata_vec("decimal_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.min.bigintval, 100);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval, 202);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
  });
}

TEST_F(MetadataUpdate, MetadataDecimalNotNull) {
  make_table_cycler("decimal_not_null_table", "decimal(18,1) not null")([&] {
    query("insert into decimal_not_null_table values ( 10.1 );");
    query("insert into decimal_not_null_table values ( 20.1 );");

    auto pre_metadata = get_metadata_vec("decimal_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update decimal_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);

    auto post_metadata = get_metadata_vec("decimal_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.bigintval,
              post_metadata_chunk.chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval,
              post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update decimal_not_null_table set x = -922337203685477580.7;"),
                 std::runtime_error);
    EXPECT_THROW(query("update decimal_not_null_table set x = 922337203685477580.7;"),
                 std::runtime_error);

    query("update decimal_not_null_table set x = 10.0 where x < 15;");
    query("update decimal_not_null_table set x = 20.2 where x > 15;");

    post_metadata = get_metadata_vec("decimal_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.min.bigintval, 100);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval, 202);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);
  });
}

TEST_F(MetadataUpdate, MetadataIntegerNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("integer_null_table", "integer")([&] {
    query("insert into integer_null_table values (10);");
    query("insert into integer_null_table values (20);");

    auto pre_metadata = get_metadata_vec("integer_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update integer_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("integer_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.intval,
              post_metadata_chunk.chunkStats.min.intval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.intval,
              post_metadata_chunk.chunkStats.max.intval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update integer_null_table set x=-2147483647 where x is NULL;");
    post_metadata = get_metadata_vec("integer_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto int_minval = post_metadata_chunk.chunkStats.min.intval;
    ASSERT_EQ(int_minval, -2147483647);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update integer_null_table set x=2147483647 where x > 15;");
    post_metadata = get_metadata_vec("integer_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.max.intval, 2147483647);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.intval, -2147483647);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.intval,
              pre_metadata_chunk.chunkStats.max.intval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.intval,
              pre_metadata_chunk.chunkStats.min.intval);
  });
}

TEST_F(MetadataUpdate, IntegerNotNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("integer_not_null_table", "integer not null")([&] {
    query("insert into integer_not_null_table values (10);");
    query("insert into integer_not_null_table values (20);");

    auto pre_metadata = get_metadata_vec("integer_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update integer_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);
    EXPECT_THROW(query("update integer_not_null_table set x = -2147483648 where x < 15;"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("integer_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    query("update integer_not_null_table set x=-2147483647 where x < 15;");
    query("update integer_not_null_table set x=2147483647 where x > 15;");

    post_metadata = get_metadata_vec("integer_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_integer_minval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.min.intval);
    int64_t post_integer_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.intval);
    ASSERT_EQ(post_integer_minval, -2147483647);
    ASSERT_EQ(post_integer_maxval, 2147483647);

    // Check out-of-range max
    EXPECT_THROW(query("update integer_not_null_table set x=2147483647+12 where x < 15;"),
                 std::runtime_error);

    post_metadata = get_metadata_vec("integer_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;
    post_integer_maxval = static_cast<int64_t>(post_metadata_chunk.chunkStats.max.intval);

    ASSERT_EQ(post_integer_maxval, 2147483647);
  });
}

TEST_F(MetadataUpdate, MetadataTinyIntNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("tinyint_null_table", "tinyint")([&] {
    query("insert into tinyint_null_table values (10);");
    query("insert into tinyint_null_table values (20);");

    auto pre_metadata = get_metadata_vec("tinyint_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update tinyint_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("tinyint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.tinyintval,
              post_metadata_chunk.chunkStats.min.tinyintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.tinyintval,
              post_metadata_chunk.chunkStats.max.tinyintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update tinyint_null_table set x=-127 where x is NULL;");
    post_metadata = get_metadata_vec("tinyint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto tinyint_minval = post_metadata_chunk.chunkStats.min.tinyintval;
    ASSERT_EQ(tinyint_minval, -127);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update tinyint_null_table set x=127 where x > 15;");
    post_metadata = get_metadata_vec("tinyint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.max.tinyintval, 127);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.tinyintval, -127);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.tinyintval,
              pre_metadata_chunk.chunkStats.max.tinyintval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.tinyintval,
              pre_metadata_chunk.chunkStats.min.tinyintval);
  });
}

TEST_F(MetadataUpdate, MetadataTinyIntNotNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("tinyint_not_null_table", "tinyint not null")([&] {
    query("insert into tinyint_not_null_table values (10);");
    query("insert into tinyint_not_null_table values (20);");

    auto pre_metadata = get_metadata_vec("tinyint_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update tinyint_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);
    EXPECT_THROW(query("update tinyint_not_null_table set x = -128 where x < 15;"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("tinyint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    query("update tinyint_not_null_table set x=-127 where x < 15;");
    query("update tinyint_not_null_table set x=127 where x > 15;");

    post_metadata = get_metadata_vec("tinyint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_tinyint_minval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.min.tinyintval);
    int64_t post_tinyint_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.tinyintval);
    ASSERT_EQ(post_tinyint_minval, -127);
    ASSERT_EQ(post_tinyint_maxval, 127);

    // Check out-of-range max -- should wrap around to -117 here, then not widen the
    // metadata
    query("update tinyint_not_null_table set x=127+12 where x < 15;");

    post_metadata = get_metadata_vec("tinyint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;
    post_tinyint_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.tinyintval);
    post_tinyint_minval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.min.tinyintval);

    ASSERT_EQ(post_tinyint_maxval, 127);
    ASSERT_EQ(post_tinyint_minval, -127);
  });
}

TEST_F(MetadataUpdate, MetadataSmallIntNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("smallint_null_table", "smallint")([&] {
    query("insert into smallint_null_table values (10);");
    query("insert into smallint_null_table values (20);");

    auto pre_metadata = get_metadata_vec("smallint_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update smallint_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("smallint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.smallintval,
              post_metadata_chunk.chunkStats.min.smallintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.smallintval,
              post_metadata_chunk.chunkStats.max.smallintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update smallint_null_table set x=-32767 where x is NULL;");
    post_metadata = get_metadata_vec("smallint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto smallint_minval = post_metadata_chunk.chunkStats.min.smallintval;
    ASSERT_EQ(smallint_minval, -32767);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update smallint_null_table set x=32767 where x > 15;");
    post_metadata = get_metadata_vec("smallint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.max.smallintval, 32767);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.smallintval, -32767);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.smallintval,
              pre_metadata_chunk.chunkStats.max.smallintval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.smallintval,
              pre_metadata_chunk.chunkStats.min.smallintval);
  });
}

TEST_F(MetadataUpdate, MetadataSmallIntNotNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("smallint_not_null_table", "smallint not null")([&] {
    query("insert into smallint_not_null_table values (10);");
    query("insert into smallint_not_null_table values (20);");

    auto pre_metadata = get_metadata_vec("smallint_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update smallint_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);
    EXPECT_THROW(query("update smallint_not_null_table set x = -32768 where x < 15;"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("smallint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    query("update smallint_not_null_table set x=-32767 where x < 15;");
    query("update smallint_not_null_table set x=32767 where x > 15;");

    post_metadata = get_metadata_vec("smallint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_smallint_minval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.min.smallintval);
    int64_t post_smallint_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.smallintval);
    ASSERT_EQ(post_smallint_minval, -32767);
    ASSERT_EQ(post_smallint_maxval, 32767);

    // Check out-of-range max -- should wrap around to -117 here, then not widen the
    // metadata
    EXPECT_THROW(query("update smallint_not_null_table set x=32767+12 where x < 15;"),
                 std::runtime_error);

    post_metadata = get_metadata_vec("smallint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;
    post_smallint_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.smallintval);
    post_smallint_minval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.min.smallintval);

    ASSERT_EQ(post_smallint_maxval, 32767);
    ASSERT_EQ(post_smallint_minval, -32767);
  });
}

TEST_F(MetadataUpdate, MetadataBigIntNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("bigint_null_table", "bigint")([&] {
    query("insert into bigint_null_table values (10);");
    query("insert into bigint_null_table values (20);");

    auto pre_metadata = get_metadata_vec("bigint_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update bigint_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("bigint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.bigintval,
              post_metadata_chunk.chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval,
              post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update bigint_null_table set x=-9223372036854775807 where x is NULL;");
    post_metadata = get_metadata_vec("bigint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto bigint_minval = post_metadata_chunk.chunkStats.min.bigintval;
    ASSERT_EQ(bigint_minval, -9223372036854775807);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update bigint_null_table set x=9223372036854775807 where x > 15;");
    post_metadata = get_metadata_vec("bigint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.max.bigintval, 9223372036854775807);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.bigintval, -9223372036854775807);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.bigintval,
              pre_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.bigintval,
              pre_metadata_chunk.chunkStats.min.bigintval);
  });
}

TEST_F(MetadataUpdate, MetadataBigIntNotNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("bigint_not_null_table", "bigint not null")([&] {
    query("insert into bigint_not_null_table values (10);");
    query("insert into bigint_not_null_table values (20);");

    auto pre_metadata = get_metadata_vec("bigint_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update bigint_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);
    EXPECT_THROW(
        query("update bigint_not_null_table set x = -9223372036854775808 where x < 15;"),
        std::runtime_error);
    auto post_metadata = get_metadata_vec("bigint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    query("update bigint_not_null_table set x=-9223372036854775807 where x < 15;");
    query("update bigint_not_null_table set x=9223372036854775807 where x > 15;");

    post_metadata = get_metadata_vec("bigint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_bigint_minval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.min.bigintval);
    int64_t post_bigint_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.bigintval);
    ASSERT_EQ(post_bigint_minval, -9223372036854775807);
    ASSERT_EQ(post_bigint_maxval, 9223372036854775807);

    // Check out-of-range max -- should wrap around to -117 here, then not widen the
    // metadata
    EXPECT_THROW(
        query("update bigint_not_null_table set x=9223372036854775807+12 where x < 15;"),
        std::runtime_error);

    post_metadata = get_metadata_vec("bigint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;
    post_bigint_maxval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.max.bigintval);
    post_bigint_minval =
        static_cast<int64_t>(post_metadata_chunk.chunkStats.min.bigintval);

    ASSERT_EQ(post_bigint_maxval, 9223372036854775807);
    ASSERT_EQ(post_bigint_minval, -9223372036854775807);
  });
}

TEST_F(MetadataUpdate, MetadataBooleanNull) {
  return;  // Boolean updates need a fix; currently broken.

  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("boolean_null_table", "boolean")([&] {
    query("insert into boolean_null_table values ('t');");
    query("insert into boolean_null_table values ('f');");

    auto pre_metadata = get_metadata_vec("boolean_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    // Currently a bug; bool column can't be updated to null
    query("update boolean_null_table set x = NULL where x = true;");

    auto post_metadata = get_metadata_vec("boolean_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.tinyintval,
              post_metadata_chunk.chunkStats.min.tinyintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.tinyintval,
              post_metadata_chunk.chunkStats.max.tinyintval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update boolean_null_table set x='f' where x is NULL;");
    post_metadata = get_metadata_vec("boolean_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto tinyint_minval = post_metadata_chunk.chunkStats.min.tinyintval;
    ASSERT_EQ(tinyint_minval, 0);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update boolean_null_table set x=True where x = false;");
    post_metadata = get_metadata_vec("boolean_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.max.tinyintval, 1);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.tinyintval, 0);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.tinyintval,
              pre_metadata_chunk.chunkStats.max.tinyintval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.tinyintval,
              pre_metadata_chunk.chunkStats.min.tinyintval);
  });
}

TEST_F(MetadataUpdate, MetadataBooleanNotNull) {
  return;

  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("boolean_not_null_table", "boolean not null")([&] {
    query("insert into boolean_not_null_table values ('t');");
    query("insert into boolean_not_null_table values ('f');");

    auto pre_metadata = get_metadata_vec("boolean_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update boolean_not_null_table set x = NULL where x = false"),
                 std::runtime_error);

    auto post_metadata = get_metadata_vec("boolean_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);
  });
}

TEST_F(MetadataUpdate, MetadataFloatNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("float_null_table", "float")([&] {
    query("insert into float_null_table values (10.1234);");
    query("insert into float_null_table values (20.4321);");

    auto pre_metadata = get_metadata_vec("float_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update float_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("float_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.floatval,
              post_metadata_chunk.chunkStats.min.floatval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.floatval,
              post_metadata_chunk.chunkStats.max.floatval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update float_null_table set x=-3.40282E38 where x is NULL;");
    query("update float_null_table set x=3.40282E38 where x > 15;");

    post_metadata = get_metadata_vec("float_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_FLOAT_EQ(post_metadata_chunk.chunkStats.max.floatval, 3.40282E38);
    ASSERT_FLOAT_EQ(post_metadata_chunk.chunkStats.min.floatval, -3.40282E38);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.floatval,
              pre_metadata_chunk.chunkStats.max.floatval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.floatval,
              pre_metadata_chunk.chunkStats.min.floatval);
  });
}

TEST_F(MetadataUpdate, MetadataFloatNotNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("float_not_null_table", "float not null")([&] {
    query("insert into float_not_null_table values (10.1234);");
    query("insert into float_not_null_table values (20.4321);");

    auto pre_metadata = get_metadata_vec("float_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update float_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);

    auto post_metadata = get_metadata_vec("float_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.floatval,
              post_metadata_chunk.chunkStats.min.floatval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.floatval,
              post_metadata_chunk.chunkStats.max.floatval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    // Doesn't work on not null columns; still seen as null.
    // query( "update float_not_null_table set x = 1.175494351E-38 where x < 15;" );
    query("update float_not_null_table set x=3.40282E38 where x > 15;");

    post_metadata = get_metadata_vec("float_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_FLOAT_EQ(post_metadata_chunk.chunkStats.max.floatval, 3.40282E38);
    // Removed; see comment just above.
    // ASSERT_FLOAT_EQ(post_metadata_chunk.chunkStats.min.floatval, 1.175494351E-38);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.floatval,
              pre_metadata_chunk.chunkStats.max.floatval);
    ASSERT_FLOAT_EQ(post_metadata_chunk.chunkStats.min.floatval,
                    pre_metadata_chunk.chunkStats.min.floatval);
  });
}

TEST_F(MetadataUpdate, MetadataDoubleNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("double_null_table", "double")([&] {
    query("insert into double_null_table values (10.1234);");
    query("insert into double_null_table values (20.4321);");

    auto pre_metadata = get_metadata_vec("double_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    query("update double_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("double_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.doubleval,
              post_metadata_chunk.chunkStats.min.doubleval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.doubleval,
              post_metadata_chunk.chunkStats.max.doubleval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);

    query("update double_null_table set x=-1.79769313486231571E308 where x is NULL;");
    query("update double_null_table set x=1.79769313486231571e+308 where x > 15;");

    post_metadata = get_metadata_vec("double_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_DOUBLE_EQ(post_metadata_chunk.chunkStats.max.doubleval,
                     1.79769313486231571e+308);
    ASSERT_DOUBLE_EQ(post_metadata_chunk.chunkStats.min.doubleval,
                     -1.79769313486231571e+308);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.doubleval,
              pre_metadata_chunk.chunkStats.max.doubleval);
    ASSERT_LT(post_metadata_chunk.chunkStats.min.doubleval,
              pre_metadata_chunk.chunkStats.min.doubleval);
  });
}

TEST_F(MetadataUpdate, MetadataDoubleNotNull) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("double_not_null_table", "double not null")([&] {
    query("insert into double_not_null_table values (10.1234);");
    query("insert into double_not_null_table values (20.4321);");

    auto pre_metadata = get_metadata_vec("double_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, false);

    EXPECT_THROW(query("update double_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);

    auto post_metadata = get_metadata_vec("double_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk.chunkStats.min.doubleval,
              post_metadata_chunk.chunkStats.min.doubleval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.doubleval,
              post_metadata_chunk.chunkStats.max.doubleval);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);

    // Doesn't work on not null columns; still seen as null.
    // query( "update double_not_null_table set x = 1.175494351E-38 where x < 15;" );
    query("update double_not_null_table set x=1.79769313486231571e+308 where x > 15;");

    post_metadata = get_metadata_vec("double_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_DOUBLE_EQ(post_metadata_chunk.chunkStats.max.doubleval,
                     1.79769313486231571e+308);
    // Removed; see comment just above.
    // ASSERT_DOUBLE_EQ(post_metadata_chunk.chunkStats.min.doubleval, 1.175494351E-38);
    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, false);
    ASSERT_GT(post_metadata_chunk.chunkStats.max.doubleval,
              pre_metadata_chunk.chunkStats.max.doubleval);
    ASSERT_DOUBLE_EQ(post_metadata_chunk.chunkStats.min.doubleval,
                     pre_metadata_chunk.chunkStats.min.doubleval);
  });
}

TEST_F(MetadataUpdate, MetadataStringDict8Null) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("presidents", "text encoding dict(8)")([&] {
    query("insert into presidents values ('Ronald Reagan');");
    query("insert into presidents values ('Donald Trump');");
    query("insert into presidents values ('Dwight Eisenhower');");
    query("insert into presidents values ('Teddy Roosevelt');");
    query("insert into presidents values (NULL);");

    run_ddl_statement(
        "alter table presidents add column presidents_copy text encoding dict(8);");

    auto pre_metadata = get_metadata_vec("presidents", "presidents_copy"s);
    auto pre_metadata_chunk = pre_metadata[0].second;
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, true);

    query("update presidents set presidents_copy=x;");
    auto post_metadata = get_metadata_vec("presidents", "presidents_copy"s);
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.intval, 0);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.intval, 3);
  });
}

TEST_F(MetadataUpdate, MetadataStringDict16Null) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("safe_cities", "text encoding dict(16)")([&] {
    query("insert into safe_cities values ('El Paso');");
    query("insert into safe_cities values ('Pingyao');");
    query("insert into safe_cities values ('Veliky Novgorod');");
    query("insert into safe_cities values (NULL);");

    run_ddl_statement(
        "alter table safe_cities add column safe_cities_copy text encoding dict(16);");

    auto pre_metadata = get_metadata_vec("safe_cities", "safe_cities_copy"s);
    auto pre_metadata_chunk = pre_metadata[0].second;
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, true);

    query("update safe_cities set safe_cities_copy=x;");
    auto post_metadata = get_metadata_vec("safe_cities", "safe_cities_copy"s);
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.intval, 0);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.intval, 2);
  });
}

TEST_F(MetadataUpdate, MetadataStringDict32Null) {
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  make_table_cycler("candidates", "text encoding dict(32)")([&] {
    query("insert into candidates values ('Lightfoot');");
    query("insert into candidates values ('Wilson');");
    query("insert into candidates values ('Chico');");
    query("insert into candidates values ('Preckwinkle');");
    query("insert into candidates values ('Mendoza');");
    query("insert into candidates values (NULL);");

    run_ddl_statement(
        "alter table candidates add column candidates_copy text encoding dict(16);");

    auto pre_metadata = get_metadata_vec("candidates", "candidates_copy"s);
    auto pre_metadata_chunk = pre_metadata[0].second;
    ASSERT_EQ(pre_metadata_chunk.chunkStats.has_nulls, true);

    query("update candidates set candidates_copy=x;");
    auto post_metadata = get_metadata_vec("candidates", "candidates_copy"s);
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk.chunkStats.has_nulls, true);
    ASSERT_EQ(post_metadata_chunk.chunkStats.min.intval, 0);
    ASSERT_EQ(post_metadata_chunk.chunkStats.max.intval, 4);
  });
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("keep-data")) {
    g_keep_test_data = true;
  }

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
