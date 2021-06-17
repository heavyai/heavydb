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

#include <regex>
#include <string>

#include "Catalog/Catalog.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/TableOptimizer.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/DateTimeParser.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <boost/program_options.hpp>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std::literals::string_literals;
using QR = QueryRunner::QueryRunner;
namespace {

bool g_keep_test_data{false};

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, false, false);
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
  auto cat = QR::get()->getCatalog();
  auto& data_manager = cat->getDataMgr();

  auto table_desc = cat->getMetadataForTable(table_name);
  auto column_desc = cat->getMetadataForColumn(table_desc->tableId, column_name);
  ChunkKey timestamp_ck{
      cat->getCurrentDB().dbId, table_desc->tableId, column_desc->columnId};

  ChunkMetadataVector chunkMetadataVec;
  data_manager.getChunkMetadataVecForKeyPrefix(chunkMetadataVec, timestamp_ck);

  return chunkMetadataVec;
};

}  // namespace

TEST_F(MetadataUpdate, MetadataTimestampNull) {
  // For timestamp column x:
  // 1 - Check that the MIN metadata value is unperturbed
  // 2 - Check that the presence of null is properly signaled
  make_table_cycler("timestamp_null_table", "timestamp")([&] {
    query("insert into timestamp_null_table values ('1946-06-14 10:54:00');");
    query("insert into timestamp_null_table values ('2017-01-20 12:00:00');");

    auto pre_metadata = get_metadata_vec("timestamp_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update timestamp_null_table set x = NULL where x < '2000-01-01 00:00:00';");

    auto post_metadata = get_metadata_vec("timestamp_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // Check that a null update doesn't change the before and after min-max
    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.bigintval,
              post_metadata_chunk->chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval,
              post_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update timestamp_null_table set x='1000-01-01 00:00:00' where x is NULL;");
    post_metadata = get_metadata_vec("timestamp_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // At this point, we prove that -INT_MIN is not set because
    // of a null update, and that the min is constrained to
    // the integer value corresponding to the minimum timestamp
    //
    // Addendum:  There is no range checking on timestamps, however.
    auto timestamp_minval = post_metadata_chunk->chunkStats.min.bigintval;
    ASSERT_EQ(timestamp_minval, -30610224000);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    // Set the max, then verify that the min/max are the max possible for timestamp ranges
    // and also verify that the has null is still true
    query(
        "update timestamp_null_table set x='12/31/2900 23:59:59.999' where x='1000-01-01 "
        "00:00:00';");
    post_metadata = get_metadata_vec("timestamp_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval, 29379542399);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.bigintval, -30610224000);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.bigintval,
              pre_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.bigintval,
              pre_metadata_chunk->chunkStats.min.bigintval);
  });
}

TEST_F(MetadataUpdate, MetadataTimestampNotNull) {
  make_table_cycler("timestamp_not_null_table", "timestamp not null")([&] {
    query("insert into timestamp_not_null_table values ('1946-06-14 10:54:00');");
    query("insert into timestamp_not_null_table values ('2017-01-20 12:00:00');");

    auto pre_metadata = get_metadata_vec("timestamp_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

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
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

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
        static_cast<int64_t>(post_metadata_chunk->chunkStats.min.bigintval);
    int64_t post_timestamp_maxval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.bigintval);
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
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.bigintval);

    ASSERT_EQ(post_timestamp_maxval, 9223372036854775807);
  });
}

TEST_F(MetadataUpdate, MetadataTimeNull) {
  make_table_cycler("time_null_table", "time")([&] {
    query("insert into time_null_table values ('10:54:00');");
    query("insert into time_null_table values ('12:00:00');");

    auto pre_metadata = get_metadata_vec("time_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update time_null_table set x = NULL where x < '12:00:00';");

    auto post_metadata = get_metadata_vec("time_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // Check that a null update doesn't change the before and after min-max
    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.bigintval,
              post_metadata_chunk->chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval,
              post_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update time_null_table set x='00:00:00' where x is NULL;");
    post_metadata = get_metadata_vec("time_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto time_minval = post_metadata_chunk->chunkStats.min.bigintval;
    ASSERT_EQ(time_minval, 0);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    // Set the max, then verify that the min/max are the max possible for time ranges
    // and also verify that the has null is still true
    query("update time_null_table set x='23:59:59.999' where x='12:00:00';");
    post_metadata = get_metadata_vec("time_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval, 86399);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.bigintval, 0);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.bigintval,
              pre_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.bigintval,
              pre_metadata_chunk->chunkStats.min.bigintval);

    EXPECT_THROW(query("update time_null_table set x=cast('-9223372036854775807' as "
                       "time) where x = '00:00:00';"),
                 std::runtime_error);
  });
}

TEST_F(MetadataUpdate, MetadataTimeNotNull) {
  make_table_cycler("time_not_null_table", "time not null")([&] {
    query("insert into time_not_null_table values ('10:54:00');");
    query("insert into time_not_null_table values ('12:00:00');");

    auto pre_metadata = get_metadata_vec("time_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

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
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

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

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval, 86399);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.bigintval, 0);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.bigintval,
              pre_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.bigintval,
              pre_metadata_chunk->chunkStats.min.bigintval);
  });
}

TEST_F(MetadataUpdate, MetadataDateNull) {
  make_table_cycler("date_null_table", "date")([&] {
    query("insert into date_null_table values ('1946-06-14');");
    query("insert into date_null_table values ('1974-02-11');");

    auto pre_metadata = get_metadata_vec("date_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update date_null_table set x = NULL where x < '1950-01-01'");

    auto post_metadata = get_metadata_vec("date_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // Check that a null update doesn't change the before and after min-max
    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.bigintval,
              post_metadata_chunk->chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval,
              post_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update date_null_table set x='-185542587187199' where x is NULL;");
    post_metadata = get_metadata_vec("date_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto date_minval = post_metadata_chunk->chunkStats.min.bigintval;
    EXPECT_EQ(date_minval, -185542587187200);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update date_null_table set x='185542587187199' where x>'1950-01-01';");
    post_metadata = get_metadata_vec("date_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    EXPECT_EQ(post_metadata_chunk->chunkStats.max.bigintval, 185542587100800);
    EXPECT_EQ(post_metadata_chunk->chunkStats.min.bigintval, -185542587187200);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.bigintval,
              pre_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.bigintval,
              pre_metadata_chunk->chunkStats.min.bigintval);
  });
}

TEST_F(MetadataUpdate, MetadataDecimalNull) {
  make_table_cycler("decimal_null_table", "decimal(18,1)")([&] {
    query("insert into decimal_null_table values ( 10.1 );");
    query("insert into decimal_null_table values ( 20.1 );");

    auto pre_metadata = get_metadata_vec("decimal_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update decimal_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("decimal_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.bigintval,
              post_metadata_chunk->chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval,
              post_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    EXPECT_THROW(query("update decimal_null_table set x = -922337203685477580.7;"),
                 std::runtime_error);
    EXPECT_THROW(query("update decimal_null_table set x = 922337203685477580.7;"),
                 std::runtime_error);

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.bigintval,
              post_metadata_chunk->chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval,
              post_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update decimal_null_table set x = 10.0 where x is NULL;");
    query("update decimal_null_table set x = 20.2 where x > 15;");

    post_metadata = get_metadata_vec("decimal_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.min.bigintval, 100);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval, 202);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
  });
}

TEST_F(MetadataUpdate, MetadataDecimalNotNull) {
  make_table_cycler("decimal_not_null_table", "decimal(18,1) not null")([&] {
    query("insert into decimal_not_null_table values ( 10.1 );");
    query("insert into decimal_not_null_table values ( 20.1 );");

    auto pre_metadata = get_metadata_vec("decimal_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update decimal_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);

    auto post_metadata = get_metadata_vec("decimal_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.bigintval,
              post_metadata_chunk->chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval,
              post_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update decimal_not_null_table set x = -922337203685477580.7;"),
                 std::runtime_error);
    EXPECT_THROW(query("update decimal_not_null_table set x = 922337203685477580.7;"),
                 std::runtime_error);

    query("update decimal_not_null_table set x = 10.0 where x < 15;");
    query("update decimal_not_null_table set x = 20.2 where x > 15;");

    post_metadata = get_metadata_vec("decimal_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.min.bigintval, 100);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval, 202);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);
  });
}

TEST_F(MetadataUpdate, MetadataIntegerNull) {
  make_table_cycler("integer_null_table", "integer")([&] {
    query("insert into integer_null_table values (10);");
    query("insert into integer_null_table values (20);");

    auto pre_metadata = get_metadata_vec("integer_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update integer_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("integer_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.intval,
              post_metadata_chunk->chunkStats.min.intval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.intval,
              post_metadata_chunk->chunkStats.max.intval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update integer_null_table set x=-2147483647 where x is NULL;");
    post_metadata = get_metadata_vec("integer_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto int_minval = post_metadata_chunk->chunkStats.min.intval;
    ASSERT_EQ(int_minval, -2147483647);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update integer_null_table set x=2147483647 where x > 15;");
    post_metadata = get_metadata_vec("integer_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.intval, 2147483647);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.intval, -2147483647);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.intval,
              pre_metadata_chunk->chunkStats.max.intval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.intval,
              pre_metadata_chunk->chunkStats.min.intval);
  });
}

TEST_F(MetadataUpdate, IntegerNotNull) {
  make_table_cycler("integer_not_null_table", "integer not null")([&] {
    query("insert into integer_not_null_table values (10);");
    query("insert into integer_not_null_table values (20);");

    auto pre_metadata = get_metadata_vec("integer_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update integer_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);
    EXPECT_THROW(query("update integer_not_null_table set x = -2147483648 where x < 15;"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("integer_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

    query("update integer_not_null_table set x=-2147483647 where x < 15;");
    query("update integer_not_null_table set x=2147483647 where x > 15;");

    post_metadata = get_metadata_vec("integer_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_integer_minval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.min.intval);
    int64_t post_integer_maxval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.intval);
    ASSERT_EQ(post_integer_minval, -2147483647);
    ASSERT_EQ(post_integer_maxval, 2147483647);

    // Check out-of-range max
    EXPECT_THROW(query("update integer_not_null_table set x=2147483647+12 where x < 15;"),
                 std::runtime_error);

    post_metadata = get_metadata_vec("integer_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;
    post_integer_maxval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.intval);

    ASSERT_EQ(post_integer_maxval, 2147483647);
  });
}

TEST_F(MetadataUpdate, MetadataTinyIntNull) {
  make_table_cycler("tinyint_null_table", "tinyint")([&] {
    query("insert into tinyint_null_table values (10);");
    query("insert into tinyint_null_table values (20);");

    auto pre_metadata = get_metadata_vec("tinyint_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update tinyint_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("tinyint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.tinyintval,
              post_metadata_chunk->chunkStats.min.tinyintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.tinyintval,
              post_metadata_chunk->chunkStats.max.tinyintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update tinyint_null_table set x=-127 where x is NULL;");
    post_metadata = get_metadata_vec("tinyint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto tinyint_minval = post_metadata_chunk->chunkStats.min.tinyintval;
    ASSERT_EQ(tinyint_minval, -127);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update tinyint_null_table set x=127 where x > 15;");
    post_metadata = get_metadata_vec("tinyint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.tinyintval, 127);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.tinyintval, -127);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.tinyintval,
              pre_metadata_chunk->chunkStats.max.tinyintval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.tinyintval,
              pre_metadata_chunk->chunkStats.min.tinyintval);
  });
}

TEST_F(MetadataUpdate, MetadataTinyIntNotNull) {
  make_table_cycler("tinyint_not_null_table", "tinyint not null")([&] {
    query("insert into tinyint_not_null_table values (10);");
    query("insert into tinyint_not_null_table values (20);");

    auto pre_metadata = get_metadata_vec("tinyint_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update tinyint_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);
    EXPECT_THROW(query("update tinyint_not_null_table set x = -128 where x < 15;"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("tinyint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

    query("update tinyint_not_null_table set x=-127 where x < 15;");
    query("update tinyint_not_null_table set x=127 where x > 15;");

    post_metadata = get_metadata_vec("tinyint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_tinyint_minval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.min.tinyintval);
    int64_t post_tinyint_maxval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.tinyintval);
    ASSERT_EQ(post_tinyint_minval, -127);
    ASSERT_EQ(post_tinyint_maxval, 127);

    // Check out-of-range max -- should wrap around to -117 here, then not widen the
    // metadata
    EXPECT_THROW(query("update tinyint_not_null_table set x=127+12 where x < 15;"),
                 std::runtime_error);

    post_metadata = get_metadata_vec("tinyint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;
    post_tinyint_maxval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.tinyintval);
    post_tinyint_minval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.min.tinyintval);

    ASSERT_EQ(post_tinyint_maxval, 127);
    ASSERT_EQ(post_tinyint_minval, -127);
  });
}

TEST_F(MetadataUpdate, MetadataAddColumnWithDeletes) {
  make_table_cycler("numbers", "int")([&] {
    query("insert into numbers values (1);");
    query("insert into numbers values (2);");
    query("insert into numbers values (3);");
    query("insert into numbers values (4);");

    {
      auto result =
          run_multiple_agg("select count(*) from numbers", ExecutorDeviceType::CPU);
      const auto row = result->getNextRow(false, false);
      ASSERT_EQ(row.size(), size_t(1));
      ASSERT_EQ(TestHelpers::v<int64_t>(row[0]), int64_t(4));
    }

    query("delete from numbers where x > 2;");

    {
      auto result =
          run_multiple_agg("select count(*) from numbers", ExecutorDeviceType::CPU);
      const auto row = result->getNextRow(false, false);
      ASSERT_EQ(row.size(), size_t(1));
      ASSERT_EQ(TestHelpers::v<int64_t>(row[0]), int64_t(2));
    }

    auto pre_metadata = get_metadata_vec("numbers", "$deleted$");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 4U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.tinyintval, int8_t(0));
    ASSERT_EQ(pre_metadata_chunk->chunkStats.max.tinyintval, int8_t(1));
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    run_ddl_statement("alter table numbers add column zebra int;");

    auto post_metadata = get_metadata_vec("numbers", "$deleted$");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->numElements, 4U);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.tinyintval, int8_t(0));
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.tinyintval, int8_t(1));
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);
  });
}

TEST_F(MetadataUpdate, MetadataSmallIntNull) {
  make_table_cycler("smallint_null_table", "smallint")([&] {
    query("insert into smallint_null_table values (10);");
    query("insert into smallint_null_table values (20);");

    auto pre_metadata = get_metadata_vec("smallint_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update smallint_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("smallint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.smallintval,
              post_metadata_chunk->chunkStats.min.smallintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.smallintval,
              post_metadata_chunk->chunkStats.max.smallintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update smallint_null_table set x=-32767 where x is NULL;");
    post_metadata = get_metadata_vec("smallint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto smallint_minval = post_metadata_chunk->chunkStats.min.smallintval;
    ASSERT_EQ(smallint_minval, -32767);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update smallint_null_table set x=32767 where x > 15;");
    post_metadata = get_metadata_vec("smallint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.smallintval, 32767);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.smallintval, -32767);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.smallintval,
              pre_metadata_chunk->chunkStats.max.smallintval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.smallintval,
              pre_metadata_chunk->chunkStats.min.smallintval);
  });
}

TEST_F(MetadataUpdate, MetadataSmallIntNotNull) {
  make_table_cycler("smallint_not_null_table", "smallint not null")([&] {
    query("insert into smallint_not_null_table values (10);");
    query("insert into smallint_not_null_table values (20);");

    auto pre_metadata = get_metadata_vec("smallint_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update smallint_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);
    EXPECT_THROW(query("update smallint_not_null_table set x = -32768 where x < 15;"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("smallint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

    query("update smallint_not_null_table set x=-32767 where x < 15;");
    query("update smallint_not_null_table set x=32767 where x > 15;");

    post_metadata = get_metadata_vec("smallint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_smallint_minval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.min.smallintval);
    int64_t post_smallint_maxval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.smallintval);
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
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.smallintval);
    post_smallint_minval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.min.smallintval);

    ASSERT_EQ(post_smallint_maxval, 32767);
    ASSERT_EQ(post_smallint_minval, -32767);
  });
}

TEST_F(MetadataUpdate, MetadataBigIntNull) {
  make_table_cycler("bigint_null_table", "bigint")([&] {
    query("insert into bigint_null_table values (10);");
    query("insert into bigint_null_table values (20);");

    auto pre_metadata = get_metadata_vec("bigint_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update bigint_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("bigint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.bigintval,
              post_metadata_chunk->chunkStats.min.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval,
              post_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update bigint_null_table set x=-9223372036854775807 where x is NULL;");
    post_metadata = get_metadata_vec("bigint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto bigint_minval = post_metadata_chunk->chunkStats.min.bigintval;
    ASSERT_EQ(bigint_minval, -9223372036854775807);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update bigint_null_table set x=9223372036854775807 where x > 15;");
    post_metadata = get_metadata_vec("bigint_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.bigintval, 9223372036854775807);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.bigintval, -9223372036854775807);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.bigintval,
              pre_metadata_chunk->chunkStats.max.bigintval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.bigintval,
              pre_metadata_chunk->chunkStats.min.bigintval);
  });
}

TEST_F(MetadataUpdate, MetadataBigIntNotNull) {
  make_table_cycler("bigint_not_null_table", "bigint not null")([&] {
    query("insert into bigint_not_null_table values (10);");
    query("insert into bigint_not_null_table values (20);");

    auto pre_metadata = get_metadata_vec("bigint_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update bigint_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);
    EXPECT_THROW(
        query("update bigint_not_null_table set x = -9223372036854775808 where x < 15;"),
        std::runtime_error);
    auto post_metadata = get_metadata_vec("bigint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

    query("update bigint_not_null_table set x=-9223372036854775807 where x < 15;");
    query("update bigint_not_null_table set x=9223372036854775807 where x > 15;");

    post_metadata = get_metadata_vec("bigint_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    // Because there is no range checking, we have to just check
    // that this unwieldy value ended up making it to the minval anyway
    int64_t post_bigint_minval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.min.bigintval);
    int64_t post_bigint_maxval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.bigintval);
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
        static_cast<int64_t>(post_metadata_chunk->chunkStats.max.bigintval);
    post_bigint_minval =
        static_cast<int64_t>(post_metadata_chunk->chunkStats.min.bigintval);

    ASSERT_EQ(post_bigint_maxval, 9223372036854775807);
    ASSERT_EQ(post_bigint_minval, -9223372036854775807);
  });
}

TEST_F(MetadataUpdate, MetadataBooleanNull) {
  make_table_cycler("boolean_null_table", "boolean")([&] {
    query("insert into boolean_null_table values ('f');");

    auto pre_metadata = get_metadata_vec("boolean_null_table");
    auto pre_metadata_chunk = pre_metadata[0].second;

    query("insert into boolean_null_table values ('t');");

    auto post_metadata = get_metadata_vec("boolean_null_table");
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.tinyintval,
              post_metadata_chunk->chunkStats.min.tinyintval);
    ASSERT_LT(pre_metadata_chunk->chunkStats.max.tinyintval,
              post_metadata_chunk->chunkStats.max.tinyintval);
  });

  make_table_cycler("boolean_null_table", "boolean")([&] {
    query("insert into boolean_null_table values ('t');");

    auto pre_metadata = get_metadata_vec("boolean_null_table");
    auto pre_metadata_chunk = pre_metadata[0].second;

    query("insert into boolean_null_table values ('f');");

    auto post_metadata = get_metadata_vec("boolean_null_table");
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_GT(pre_metadata_chunk->chunkStats.min.tinyintval,
              post_metadata_chunk->chunkStats.min.tinyintval);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.max.tinyintval,
              post_metadata_chunk->chunkStats.max.tinyintval);
  });

  make_table_cycler("boolean_null_table", "boolean")([&] {
    query("insert into boolean_null_table values ('t');");

    auto pre_metadata = get_metadata_vec("boolean_null_table");
    auto pre_metadata_chunk = pre_metadata[0].second;

    query("insert into boolean_null_table values (NULL);");

    auto post_metadata = get_metadata_vec("boolean_null_table");
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.tinyintval,
              post_metadata_chunk->chunkStats.min.tinyintval);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.max.tinyintval,
              post_metadata_chunk->chunkStats.max.tinyintval);
  });

  make_table_cycler("boolean_null_table", "boolean")([&] {
    query("insert into boolean_null_table values ('f');");

    auto pre_metadata = get_metadata_vec("boolean_null_table");
    auto pre_metadata_chunk = pre_metadata[0].second;

    query("insert into boolean_null_table values (NULL);");

    auto post_metadata = get_metadata_vec("boolean_null_table");
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.tinyintval,
              post_metadata_chunk->chunkStats.min.tinyintval);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.max.tinyintval,
              post_metadata_chunk->chunkStats.max.tinyintval);
  });

  make_table_cycler("boolean_null_table", "boolean")([&] {
    query("insert into boolean_null_table values ('t');");
    query("insert into boolean_null_table values ('f');");

    auto pre_metadata = get_metadata_vec("boolean_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update boolean_null_table set x = NULL where x = true;");

    auto post_metadata = get_metadata_vec("boolean_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.tinyintval,
              post_metadata_chunk->chunkStats.min.tinyintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.tinyintval,
              post_metadata_chunk->chunkStats.max.tinyintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update boolean_null_table set x='f' where x is NULL;");
    post_metadata = get_metadata_vec("boolean_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto tinyint_minval = post_metadata_chunk->chunkStats.min.tinyintval;
    ASSERT_EQ(tinyint_minval, 0);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update boolean_null_table set x=True where x = false;");
    post_metadata = get_metadata_vec("boolean_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.tinyintval, 1);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.tinyintval, 0);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
  });
}

TEST_F(MetadataUpdate, MetadataBooleanNotNull) {
  make_table_cycler("boolean_not_null_table", "boolean not null")([&] {
    query("insert into boolean_not_null_table values ('t');");
    query("insert into boolean_not_null_table values ('f');");

    auto pre_metadata = get_metadata_vec("boolean_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update boolean_not_null_table set x = NULL where x = false"),
                 std::runtime_error);

    auto post_metadata = get_metadata_vec("boolean_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);
  });
}

TEST_F(MetadataUpdate, MetadataFloatNull) {
  make_table_cycler("float_null_table", "float")([&] {
    query("insert into float_null_table values (10.1234);");
    query("insert into float_null_table values (20.4321);");

    auto pre_metadata = get_metadata_vec("float_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update float_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("float_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.floatval,
              post_metadata_chunk->chunkStats.min.floatval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.floatval,
              post_metadata_chunk->chunkStats.max.floatval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update float_null_table set x=-3.40282E38 where x is NULL;");
    query("update float_null_table set x=3.40282E38 where x > 15;");

    post_metadata = get_metadata_vec("float_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_FLOAT_EQ(post_metadata_chunk->chunkStats.max.floatval, 3.40282E38);
    ASSERT_FLOAT_EQ(post_metadata_chunk->chunkStats.min.floatval, -3.40282E38);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.floatval,
              pre_metadata_chunk->chunkStats.max.floatval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.floatval,
              pre_metadata_chunk->chunkStats.min.floatval);
  });
}

TEST_F(MetadataUpdate, MetadataFloatNotNull) {
  make_table_cycler("float_not_null_table", "float not null")([&] {
    query("insert into float_not_null_table values (10.1234);");
    query("insert into float_not_null_table values (20.4321);");

    auto pre_metadata = get_metadata_vec("float_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update float_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);

    auto post_metadata = get_metadata_vec("float_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.floatval,
              post_metadata_chunk->chunkStats.min.floatval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.floatval,
              post_metadata_chunk->chunkStats.max.floatval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

    // Doesn't work on not null columns; still seen as null.
    // query( "update float_not_null_table set x = 1.175494351E-38 where x < 15;" );
    query("update float_not_null_table set x=3.40282E38 where x > 15;");

    post_metadata = get_metadata_vec("float_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_FLOAT_EQ(post_metadata_chunk->chunkStats.max.floatval, 3.40282E38);
    // Removed; see comment just above.
    // ASSERT_FLOAT_EQ(post_metadata_chunk->chunkStats.min.floatval, 1.175494351E-38);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.floatval,
              pre_metadata_chunk->chunkStats.max.floatval);
    ASSERT_FLOAT_EQ(post_metadata_chunk->chunkStats.min.floatval,
                    pre_metadata_chunk->chunkStats.min.floatval);
  });
}

TEST_F(MetadataUpdate, MetadataDoubleNull) {
  make_table_cycler("double_null_table", "double")([&] {
    query("insert into double_null_table values (10.1234);");
    query("insert into double_null_table values (20.4321);");

    auto pre_metadata = get_metadata_vec("double_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update double_null_table set x = NULL where x < 15;");

    auto post_metadata = get_metadata_vec("double_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.doubleval,
              post_metadata_chunk->chunkStats.min.doubleval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.doubleval,
              post_metadata_chunk->chunkStats.max.doubleval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update double_null_table set x=-1.79769313486231571E308 where x is NULL;");
    query("update double_null_table set x=1.79769313486231571e+308 where x > 15;");

    post_metadata = get_metadata_vec("double_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_DOUBLE_EQ(post_metadata_chunk->chunkStats.max.doubleval,
                     1.79769313486231571e+308);
    ASSERT_DOUBLE_EQ(post_metadata_chunk->chunkStats.min.doubleval,
                     -1.79769313486231571e+308);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.doubleval,
              pre_metadata_chunk->chunkStats.max.doubleval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.doubleval,
              pre_metadata_chunk->chunkStats.min.doubleval);
  });
}

TEST_F(MetadataUpdate, MetadataDoubleNotNull) {
  make_table_cycler("double_not_null_table", "double not null")([&] {
    query("insert into double_not_null_table values (10.1234);");
    query("insert into double_not_null_table values (20.4321);");

    auto pre_metadata = get_metadata_vec("double_not_null_table");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update double_not_null_table set x = NULL where x < 15;"),
                 std::runtime_error);

    auto post_metadata = get_metadata_vec("double_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.doubleval,
              post_metadata_chunk->chunkStats.min.doubleval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.doubleval,
              post_metadata_chunk->chunkStats.max.doubleval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

    // Doesn't work on not null columns; still seen as null.
    // query( "update double_not_null_table set x = 1.175494351E-38 where x < 15;" );
    query("update double_not_null_table set x=1.79769313486231571e+308 where x > 15;");

    post_metadata = get_metadata_vec("double_not_null_table");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_DOUBLE_EQ(post_metadata_chunk->chunkStats.max.doubleval,
                     1.79769313486231571e+308);
    // Removed; see comment just above.
    // ASSERT_DOUBLE_EQ(post_metadata_chunk->chunkStats.min.doubleval, 1.175494351E-38);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.doubleval,
              pre_metadata_chunk->chunkStats.max.doubleval);
    ASSERT_DOUBLE_EQ(post_metadata_chunk->chunkStats.min.doubleval,
                     pre_metadata_chunk->chunkStats.min.doubleval);
  });
}

TEST_F(MetadataUpdate, MetadataStringDict8Null) {
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
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, true);

    query("update presidents set presidents_copy=x;");
    auto post_metadata = get_metadata_vec("presidents", "presidents_copy"s);
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.intval, 0);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.intval, 3);
  });
}

TEST_F(MetadataUpdate, MetadataStringDict16Null) {
  make_table_cycler("safe_cities", "text encoding dict(16)")([&] {
    query("insert into safe_cities values ('El Paso');");
    query("insert into safe_cities values ('Pingyao');");
    query("insert into safe_cities values ('Veliky Novgorod');");
    query("insert into safe_cities values (NULL);");

    run_ddl_statement(
        "alter table safe_cities add column safe_cities_copy text encoding dict(16);");

    auto pre_metadata = get_metadata_vec("safe_cities", "safe_cities_copy"s);
    auto pre_metadata_chunk = pre_metadata[0].second;
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, true);

    query("update safe_cities set safe_cities_copy=x;");
    auto post_metadata = get_metadata_vec("safe_cities", "safe_cities_copy"s);
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.intval, 0);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.intval, 2);
  });
}

TEST_F(MetadataUpdate, MetadataStringDict32Null) {
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
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, true);

    query("update candidates set candidates_copy=x;");
    auto post_metadata = get_metadata_vec("candidates", "candidates_copy"s);
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.intval, 0);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.intval, 4);
  });
}

TEST_F(MetadataUpdate, MetadataSmallIntEncodedNull) {
  make_table_cycler("small_ints_null", "smallint encoding fixed(8)")([&] {
    query("insert into small_ints_null values (10);");
    query("insert into small_ints_null values (90);");

    auto pre_metadata = get_metadata_vec("small_ints_null");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    query("update small_ints_null set x = NULL where x < 50;");

    auto post_metadata = get_metadata_vec("small_ints_null");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->chunkStats.min.smallintval,
              post_metadata_chunk->chunkStats.min.smallintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.max.smallintval,
              post_metadata_chunk->chunkStats.max.smallintval);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    auto smallint_minval = post_metadata_chunk->chunkStats.min.smallintval;
    ASSERT_EQ(smallint_minval, 10);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);

    query("update small_ints_null set x=-127 where x is NULL;");
    query("update small_ints_null set x=127;");

    post_metadata = get_metadata_vec("small_ints_null");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    ASSERT_EQ(post_metadata_chunk->chunkStats.max.smallintval, 127);
    ASSERT_EQ(post_metadata_chunk->chunkStats.min.smallintval, -127);
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, true);
    ASSERT_GT(post_metadata_chunk->chunkStats.max.smallintval,
              pre_metadata_chunk->chunkStats.max.smallintval);
    ASSERT_LT(post_metadata_chunk->chunkStats.min.smallintval,
              pre_metadata_chunk->chunkStats.min.smallintval);
  });
};

TEST_F(MetadataUpdate, MetadataSmallIntEncodedNotNull) {
  make_table_cycler("small_ints_not_null", "smallint not null encoding fixed(8)")([&] {
    query("insert into small_ints_not_null values (10);");
    query("insert into small_ints_not_null values (90);");

    auto pre_metadata = get_metadata_vec("small_ints_not_null");
    ASSERT_EQ(pre_metadata.size(), 1U);
    auto pre_metadata_chunk = pre_metadata[0].second;

    ASSERT_EQ(pre_metadata_chunk->numElements, 2U);
    ASSERT_EQ(pre_metadata_chunk->chunkStats.has_nulls, false);

    EXPECT_THROW(query("update small_ints_not_null set x = NULL where x < 50;"),
                 std::runtime_error);
    auto post_metadata = get_metadata_vec("small_ints_not_null");
    ASSERT_EQ(post_metadata.size(), 1U);
    auto post_metadata_chunk = post_metadata[0].second;

    // This should never flip to true if it's a not-null table
    ASSERT_EQ(post_metadata_chunk->chunkStats.has_nulls, false);

    query("update small_ints_not_null set x=-127 where x < 50;");
    query("update small_ints_not_null set x=127 where x > 50;");

    post_metadata = get_metadata_vec("small_ints_not_null");
    ASSERT_EQ(post_metadata.size(), 1U);
    post_metadata_chunk = post_metadata[0].second;

    auto post_smallint_minval = post_metadata_chunk->chunkStats.min.smallintval;
    auto post_smallint_maxval = post_metadata_chunk->chunkStats.max.smallintval;
    ASSERT_EQ(post_smallint_minval, -127);
    ASSERT_EQ(post_smallint_maxval, 127);
  });
};

class OpportunisticMetadataUpdateTest : public testing::TestWithParam<SQLTypeInfo> {
 protected:
  void SetUp() override {
    run_ddl_statement("drop table if exists test_table;");
    run_ddl_statement("drop table if exists test_table_2;");
    g_enable_auto_metadata_update = true;
  }

  void TearDown() override {
    run_ddl_statement("drop table if exists test_table;");
    run_ddl_statement("drop table if exists test_table_2;");
    g_enable_auto_metadata_update = false;
  }

  void createTableForDataType() {
    std::string data_type;
    if (GetParam().is_decimal()) {
      data_type = "DECIMAL(2, 1)";
    } else {
      data_type = GetParam().get_type_name();
    }
    run_ddl_statement("create table test_table (i " + data_type + ");");
  }

  void insertRows() {
    auto type = GetParam().get_type();
    if (type == kINT) {
      insertRows(int_rows_);
    } else if (type == kFLOAT || type == kDOUBLE || type == kDECIMAL) {
      insertRows(float_rows_);
    } else if (type == kDATE) {
      insertRows(date_rows_);
    } else if (type == kTIME) {
      insertRows(time_rows_);
    } else if (type == kTIMESTAMP) {
      insertRows(timestamp_rows_);
    } else {
      UNREACHABLE();
    }
  }

  void insertRows(const std::array<std::string, 5>& rows) {
    for (const auto& row : rows) {
      query("insert into test_table values ('" + row + "');");
    }
  }

  void insertIntRange(const int start,
                      const int end,
                      const std::string& table_name = "test_table") {
    for (auto i = start; i <= end; i++) {
      query("insert into " + table_name + " values (" + std::to_string(i) + ");");
    }
  }

  std::string getRowValueAtIndex(int index) {
    std::string value;
    auto type = GetParam().get_type();
    if (type == kINT) {
      value = int_rows_[index];
    } else if (type == kFLOAT || type == kDOUBLE || type == kDECIMAL) {
      value = float_rows_[index];
    } else if (type == kDATE) {
      value = date_rows_[index];
    } else if (type == kTIME) {
      value = time_rows_[index];
    } else if (type == kTIMESTAMP) {
      value = timestamp_rows_[index];
    } else {
      UNREACHABLE();
    }
    return value;
  }

  std::string getLiteralValue(const std::string& value) {
    if (GetParam().is_number()) {
      return value;
    } else {
      return "'" + value + "'";
    }
  }

  void assertExpectedChunkMetadata(size_t num_elements,
                                   bool has_nulls,
                                   int min,
                                   int max) {
    auto metadata_vec = get_metadata_vec("test_table", "i");
    ASSERT_EQ(1U, metadata_vec.size());
    assertExpectedChunkMetadata(
        metadata_vec[0].second, num_elements, has_nulls, min, max);
  }

  void assertExpectedChunkMetadata(size_t num_elements,
                                   bool has_nulls,
                                   const std::string& min,
                                   const std::string& max) {
    auto metadata_vec = get_metadata_vec("test_table", "i");
    ASSERT_EQ(1U, metadata_vec.size());
    assertExpectedChunkMetadata(
        metadata_vec[0].second, num_elements, has_nulls, min, max);
  }

  void assertExpectedChunkMetadata(std::shared_ptr<const ChunkMetadata> chunk_metadata,
                                   size_t num_elements,
                                   bool has_nulls,
                                   int min,
                                   int max) {
    ASSERT_EQ(num_elements, chunk_metadata->numElements);
    ASSERT_EQ(has_nulls, chunk_metadata->chunkStats.has_nulls);
    ASSERT_EQ(min, chunk_metadata->chunkStats.min.intval);
    ASSERT_EQ(max, chunk_metadata->chunkStats.max.intval);
  }

  void assertExpectedChunkMetadata(std::shared_ptr<const ChunkMetadata> chunk_metadata,
                                   size_t num_elements,
                                   bool has_nulls,
                                   const std::string& min,
                                   const std::string& max) {
    ASSERT_EQ(num_elements, chunk_metadata->numElements);
    ASSERT_EQ(has_nulls, chunk_metadata->chunkStats.has_nulls);

    auto type = GetParam().get_type();
    if (type == kINT) {
      ASSERT_EQ(std::stoi(min), chunk_metadata->chunkStats.min.intval);
      ASSERT_EQ(std::stoi(max), chunk_metadata->chunkStats.max.intval);
    } else if (type == kFLOAT) {
      ASSERT_EQ(std::stof(min), chunk_metadata->chunkStats.min.floatval);
      ASSERT_EQ(std::stof(max), chunk_metadata->chunkStats.max.floatval);
    } else if (type == kDOUBLE) {
      ASSERT_EQ(std::stod(min), chunk_metadata->chunkStats.min.doubleval);
      ASSERT_EQ(std::stod(max), chunk_metadata->chunkStats.max.doubleval);
    } else if (type == kDECIMAL) {
      // Remove floating point from number before integer comparison.
      ASSERT_EQ(std::stoi(std::regex_replace(min, std::regex{"\\."}, "")),
                chunk_metadata->chunkStats.min.bigintval);
      ASSERT_EQ(std::stoi(std::regex_replace(max, std::regex{"\\."}, "")),
                chunk_metadata->chunkStats.max.bigintval);
    } else if (type == kDATE) {
      ASSERT_EQ(dateTimeParse<kDATE>(min, 0), chunk_metadata->chunkStats.min.bigintval);
      ASSERT_EQ(dateTimeParse<kDATE>(max, 0), chunk_metadata->chunkStats.max.bigintval);
    } else if (type == kTIME) {
      ASSERT_EQ(dateTimeParse<kTIME>(min, 0), chunk_metadata->chunkStats.min.bigintval);
      ASSERT_EQ(dateTimeParse<kTIME>(max, 0), chunk_metadata->chunkStats.max.bigintval);
    } else if (type == kTIMESTAMP) {
      ASSERT_EQ(dateTimeParse<kTIMESTAMP>(min, 0),
                chunk_metadata->chunkStats.min.bigintval);
      ASSERT_EQ(dateTimeParse<kTIMESTAMP>(max, 0),
                chunk_metadata->chunkStats.max.bigintval);
    } else {
      UNREACHABLE();
    }
  }

  void assertExpectedChunkMetadata(std::shared_ptr<const ChunkMetadata> chunk_metadata,
                                   size_t num_elements,
                                   bool has_nulls,
                                   float min,
                                   float max) {
    ASSERT_EQ(num_elements, chunk_metadata->numElements);
    ASSERT_EQ(has_nulls, chunk_metadata->chunkStats.has_nulls);
    ASSERT_EQ(min, chunk_metadata->chunkStats.min.floatval);
    ASSERT_EQ(max, chunk_metadata->chunkStats.max.floatval);
  }

  void recomputeMetadata() {
    auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
    auto catalog = QR::get()->getCatalog();
    auto td = catalog->getMetadataForTable("test_table");
    TableOptimizer optimizer(td, executor.get(), *catalog);
    optimizer.recomputeMetadata();
  }

  void setupTableWithMultipleFragments() {
    run_ddl_statement(
        "create table test_table (i float, t text) with (fragment_size = 3);");
    query("insert into test_table values (1.5, 'a');");
    query("insert into test_table values (2.5, 'b');");
    query("insert into test_table values (3.5, 'c');");
    query("insert into test_table values (4.5, 'd');");
    query("insert into test_table values (5.5, 'e');");
  }

  void setupShardedTable() {
    run_ddl_statement(
        "create table test_table (i integer, i2 integer, shard key(i)) with (shard_count "
        "= "
        "2, fragment_size = 2);");

    query("insert into test_table values (1, 1);");
    query("insert into test_table values (1, 2);");
    query("insert into test_table values (1, 3);");
    query("insert into test_table values (1, 4);");

    query("insert into test_table values (2, 1);");
    query("insert into test_table values (2, 2);");
    query("insert into test_table values (2, 3);");
    query("insert into test_table values (2, 4);");
  }

  // Variables storing test row content for data type parameterized tests.
  // Values are stored as strings in all cases for simplicity.
  inline static const std::array<std::string, 5> int_rows_{"1", "2", "3", "4", "5"};
  inline static const std::array<std::string, 5> float_rows_{"1.5",
                                                             "2.5",
                                                             "3.5",
                                                             "4.5",
                                                             "5.5"};
  inline static const std::array<std::string, 5> date_rows_{"2021-01-01",
                                                            "2021-02-01",
                                                            "2021-03-01",
                                                            "2021-04-01",
                                                            "2021-05-01"};
  inline static const std::array<std::string, 5> time_rows_{"00:00:00",
                                                            "00:01:00",
                                                            "00:02:00",
                                                            "00:03:00",
                                                            "00:04:00"};
  inline static const std::array<std::string, 5> timestamp_rows_{"2021-02-09 00:00:00",
                                                                 "2021-02-09 00:01:00",
                                                                 "2021-02-09 00:02:00",
                                                                 "2021-02-09 00:03:00",
                                                                 "2021-02-09 00:04:00"};
};

TEST_P(OpportunisticMetadataUpdateTest, MinValueReplaced) {
  createTableForDataType();
  insertRows();

  auto last_value = getRowValueAtIndex(4);
  auto second_value = getRowValueAtIndex(1);
  auto second_value_literal = getLiteralValue(second_value);
  query("update test_table set i = " + second_value_literal +
        " where i <= " + second_value_literal + ";");

  // Min should now be the second value while the max remains the same.
  assertExpectedChunkMetadata(5U, false, second_value, last_value);
}

TEST_P(OpportunisticMetadataUpdateTest, MaxValueReplaced) {
  createTableForDataType();
  insertRows();

  auto first_value = getRowValueAtIndex(0);
  auto fourth_value = getRowValueAtIndex(3);
  auto fourth_value_literal = getLiteralValue(fourth_value);
  query("update test_table set i = " + fourth_value_literal +
        " where i >= " + fourth_value_literal + ";");

  // Max should now be the fourth value while the min remains the same.
  assertExpectedChunkMetadata(5U, false, first_value, fourth_value);
}

TEST_P(OpportunisticMetadataUpdateTest, DuplicateMinMax) {
  createTableForDataType();
  insertRows();
  insertRows();

  auto first_value = getRowValueAtIndex(0);
  auto last_value = getRowValueAtIndex(4);
  auto first_value_literal = getLiteralValue(first_value);
  auto third_value_literal = getLiteralValue(getRowValueAtIndex(2));
  auto last_value_literal = getLiteralValue(last_value);
  query("update test_table set i = " + third_value_literal + " where (i = " +
        first_value_literal + " or i = " + last_value_literal + ") and rowid < 5;");

  // Max and min should remain the same.
  assertExpectedChunkMetadata(10U, false, first_value, last_value);
}

TEST_P(OpportunisticMetadataUpdateTest, ReplacedValuesNotMinOrMax) {
  createTableForDataType();
  insertRows();

  auto first_value = getRowValueAtIndex(0);
  auto last_value = getRowValueAtIndex(4);
  auto second_value_literal = getLiteralValue(getRowValueAtIndex(1));
  auto third_value_literal = getLiteralValue(getRowValueAtIndex(2));
  auto fourth_value_literal = getLiteralValue(getRowValueAtIndex(3));
  query("update test_table set i = " + third_value_literal +
        " where i = " + second_value_literal + " or i = " + fourth_value_literal + ";");

  // Max and min should remain the same.
  assertExpectedChunkMetadata(5U, false, first_value, last_value);
}

INSTANTIATE_TEST_SUITE_P(DataTypesTest,
                         OpportunisticMetadataUpdateTest,
                         testing::Values(SQLTypeInfo(kINT),
                                         SQLTypeInfo(kFLOAT),
                                         SQLTypeInfo(kDOUBLE),
                                         SQLTypeInfo(kDECIMAL),
                                         SQLTypeInfo(kDATE),
                                         SQLTypeInfo(kTIME),
                                         SQLTypeInfo(kTIMESTAMP)),
                         [](const auto& param_info) {
                           // Return type name string without parentheses.
                           return std::regex_replace(
                               param_info.param.get_type_name(), std::regex{"\\(.+"}, "");
                         });

TEST_F(OpportunisticMetadataUpdateTest, MultipleFragmentsAndColumns) {
  setupTableWithMultipleFragments();

  query("update test_table set i = i + 1, t = 'abc' where i = 1.5 or i = 4.5;");
  auto metadata_vec = get_metadata_vec("test_table", "i");
  ASSERT_EQ(2U, metadata_vec.size());

  assertExpectedChunkMetadata(metadata_vec[0].second, 3U, false, 2.5f, 3.5f);
  assertExpectedChunkMetadata(metadata_vec[1].second, 2U, false, 5.5f, 5.5f);
}

TEST_F(OpportunisticMetadataUpdateTest, DeletedRows) {
  run_ddl_statement("create table test_table (i integer);");
  insertIntRange(1, 5);

  query("delete from test_table where i = 5;");
  query("update test_table set i = i + 1 where i = 1;");
  assertExpectedChunkMetadata(5U, false, 2, 4);
}

TEST_F(OpportunisticMetadataUpdateTest, NullValues) {
  run_ddl_statement("create table test_table (i integer);");
  insertIntRange(1, 5);

  query("update test_table set i = null where i = 1 or i = 5;");
  assertExpectedChunkMetadata(5U, true, 2, 4);
}

TEST_F(OpportunisticMetadataUpdateTest, DeletedFragment) {
  setupTableWithMultipleFragments();
  query("delete from test_table where i < 4;");

  // Ensure metadata stats for the delete column is updated
  recomputeMetadata();

  query("update test_table set i = i + 1, t = 'abc' where i > 3;");

  auto metadata_vec = get_metadata_vec("test_table", "i");
  ASSERT_EQ(2U, metadata_vec.size());

  auto chunk_metadata = metadata_vec[0].second;
  assertExpectedChunkMetadata(metadata_vec[0].second, 3U, false, 1.5f, 3.5f);
  assertExpectedChunkMetadata(metadata_vec[1].second, 2U, false, 5.5f, 6.5f);

  auto catalog = QR::get()->getCatalog();
  auto& data_manager = catalog->getDataMgr();

  // Deleted chunk should not be loaded to CPU
  ASSERT_FALSE(
      data_manager.isBufferOnDevice(metadata_vec[0].first, MemoryLevel::CPU_LEVEL, 0));
  ASSERT_TRUE(
      data_manager.isBufferOnDevice(metadata_vec[1].first, MemoryLevel::CPU_LEVEL, 0));
}

TEST_F(OpportunisticMetadataUpdateTest, MultipleTables) {
  run_ddl_statement("create table test_table (i integer);");
  run_ddl_statement("create table test_table_2 (i integer);");
  insertIntRange(1, 5);
  insertIntRange(1, 5, "test_table_2");

  query(
      "update test_table set i = (select count(i) from test_table_2 where test_table.i < "
      "test_table_2.i);");
  assertExpectedChunkMetadata(5U, false, 0, 4);
}

TEST_F(OpportunisticMetadataUpdateTest, ShardedTable) {
  setupShardedTable();

  query("update test_table set i2 = i2 + 1 where i2 = 1 or i2 = 3;");
  auto metadata_vec = get_metadata_vec("test_table_shard_#1", "i2");
  ASSERT_EQ(2U, metadata_vec.size());

  assertExpectedChunkMetadata(metadata_vec[0].second, 2U, false, 2, 2);
  assertExpectedChunkMetadata(metadata_vec[1].second, 2U, false, 4, 4);

  metadata_vec = get_metadata_vec("test_table_shard_#2", "i2");
  ASSERT_EQ(2U, metadata_vec.size());

  assertExpectedChunkMetadata(metadata_vec[0].second, 2U, false, 2, 2);
  assertExpectedChunkMetadata(metadata_vec[1].second, 2U, false, 4, 4);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  logger::init(log_options);

  if (vm.count("keep-data")) {
    g_keep_test_data = true;
  }

  // Disable automatic metadata update in order to ensure
  // that metadata is not automatically updated for other
  // tests that do and assert metadata updates.
  g_enable_auto_metadata_update = false;
  g_vacuum_min_selectivity = 1.1;

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
