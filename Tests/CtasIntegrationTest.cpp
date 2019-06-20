/*
 * Copyright 2018, OmniSci, Inc.
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

#include <gtest/gtest.h>
#include <boost/program_options.hpp>

#include <thrift/Thrift.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TSocket.h>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;

#include "QueryEngine/TargetValue.h"
#include "Shared/ThriftClient.h"
#include "Shared/sqltypes.h"
#include "TestHelpers.h"
#include "gen-cpp/MapD.h"

#include <ctime>
#include <iostream>

// uncomment to run full test suite
// #define RUN_ALL_TEST

TSessionId g_session_id;
std::shared_ptr<MapDClient> g_client;

template <typename RETURN_TYPE, typename SOURCE_TYPE>
bool checked_get(size_t row,
                 const SOURCE_TYPE* boost_variant,
                 RETURN_TYPE& val,
                 RETURN_TYPE null_value) {
  const auto val_p = boost::get<RETURN_TYPE>(boost_variant);
  if (nullptr == val_p) {
    return false;
  }

  val = *val_p;
  return true;
}

template <>
bool checked_get(size_t row,
                 const ScalarTargetValue* boost_variant,
                 std::string& val,
                 std::string null_value) {
  const auto val_p = boost::get<NullableString>(boost_variant);
  if (nullptr == val_p) {
    return false;
  }

  const auto mapd_str_p = boost::get<std::string>(val_p);

  if (nullptr == mapd_str_p) {
    val = null_value;
  } else {
    val = *mapd_str_p;
  }

  return true;
}

template <>
bool checked_get(size_t row, const TDatum* datum, int64_t& val, int64_t null_value) {
  if (datum->is_null) {
    val = null_value;
  } else {
    val = datum->val.int_val;
  }

  return true;
}

template <>
bool checked_get(size_t row, const TDatum* datum, float& val, float null_value) {
  if (datum->is_null) {
    val = null_value;
  } else {
    val = (float)datum->val.real_val;
  }
  return true;
}

template <>
bool checked_get(size_t row, const TDatum* datum, double& val, double null_value) {
  if (datum->is_null) {
    val = null_value;
  } else {
    val = datum->val.real_val;
  }

  return true;
}

template <>
bool checked_get(size_t row,
                 const TDatum* datum,
                 std::string& val,
                 std::string null_value) {
  if (datum->is_null) {
    val = null_value;
  } else {
    val = datum->val.str_val;
  }

  return true;
}

class TestColumnDescriptor {
 public:
  virtual std::string get_column_definition() = 0;
  virtual std::string get_column_value(int row) = 0;
  virtual std::string get_update_column_value(int row) { return get_column_value(row); }

  virtual bool check_column_value(const int row, const TDatum* datum) = 0;
  virtual ~TestColumnDescriptor() = default;

  virtual bool skip_test(std::string name) { return false; }
};

template <typename T>
class NumberColumnDescriptor : public TestColumnDescriptor {
  std::string column_definition;
  SQLTypes rs_type;
  T null_value;

 public:
  NumberColumnDescriptor(std::string col_type, SQLTypes sql_type, T null)
      : column_definition(col_type), rs_type(sql_type), null_value(null){};

  bool skip_test(std::string name) override {
    if (kDECIMAL == rs_type) {
      return "Array.UpdateColumnByLiteral" == name;
    }
    if (kDOUBLE == rs_type || kFLOAT == rs_type) {
      return "Array.UpdateColumnByLiteral" == name;
    }
    return false;
  }

  std::string get_column_definition() override { return column_definition; };
  std::string get_column_value(int row) override {
    if (0 == row) {
      return "null";
    }

    return std::to_string(row);
  };

  bool check_column_value(int row, const TDatum* value) override {
    T mapd_val;

    if (!checked_get(row, value, mapd_val, null_value)) {
      return false;
    }

    T value_to_check = (T)row;
    if (row == 0) {
      value_to_check = null_value;
    }

    if (mapd_val == value_to_check) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << std::to_string(value_to_check)
               << " vs. " << std::to_string(mapd_val);
    return false;
  }
};

class BooleanColumnDescriptor : public TestColumnDescriptor {
  std::string column_definition;

 public:
  BooleanColumnDescriptor(std::string col_type, SQLTypes sql_type)
      : column_definition(col_type){};

  bool skip_test(std::string name) override {
    return "UpdateColumnByColumn" == name || "UpdateColumnByLiteral" == name ||
           "Array.UpdateColumnByLiteral" == name;
  }

  std::string get_column_definition() override { return column_definition; };
  std::string get_column_value(int row) override {
    if (0 == row) {
      return "null";
    }

    return (row % 2) ? "'true'" : "'false'";
  };

  bool check_column_value(int row, const TDatum* value) override {
    int64_t mapd_val;
    if (!checked_get(row, value, mapd_val, (int64_t)NULL_TINYINT)) {
      return false;
    }

    int64_t value_to_check = (row % 2);
    if (row == 0) {
      value_to_check = NULL_TINYINT;
    }

    if (mapd_val == value_to_check) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << std::to_string(value_to_check)
               << " vs. " << std::to_string(mapd_val);
    return false;
  }
};

class StringColumnDescriptor : public TestColumnDescriptor {
  std::string column_definition;
  std::string prefix;

 public:
  StringColumnDescriptor(std::string col_type, SQLTypes sql_type, std::string pfix)
      : column_definition(col_type), prefix(pfix){};

  bool skip_test(std::string name) override {
    return "Array.UpdateColumnByLiteral" == name;
  }

  std::string get_column_definition() override { return column_definition; };
  std::string get_column_value(int row) override {
    if (0 == row) {
      return "null";
    }

    return "'" + prefix + "_" + std::to_string(row) + "'";
  };
  bool check_column_value(int row, const TDatum* value) override {
    std::string mapd_val;

    if (!checked_get(row, value, mapd_val, std::string(""))) {
      return 0 == row;
    }

    if (row == 0) {
      if (mapd_val == "") {
        return true;
      }
    } else if (mapd_val == (prefix + "_" + std::to_string(row))) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " "
               << (prefix + "_" + std::to_string(row)) << " vs. " << mapd_val;
    return false;
  }
};

class DateTimeColumnDescriptor : public TestColumnDescriptor {
  std::string column_definition;
  SQLTypes rs_type;
  std::string format;
  long offset;
  long scale;

 public:
  DateTimeColumnDescriptor(std::string col_type,
                           SQLTypes sql_type,
                           std::string fmt,
                           long offset,
                           int scale)
      : column_definition(col_type)
      , rs_type(sql_type)
      , format(fmt)
      , offset(offset)
      , scale(scale){};

  bool skip_test(std::string name) override {
    return "Array.UpdateColumnByLiteral" == name;
  }

  std::string get_column_definition() override { return column_definition; };
  std::string get_column_value(int row) override {
    if (0 == row) {
      return "null";
    }

    return "'" + getValueAsString(row) + "'";
  };
  bool check_column_value(int row, const TDatum* value) override {
    int64_t mapd_val;

    if (!checked_get(row, value, mapd_val, NULL_BIGINT)) {
      return 0 == row;
    }

    int64_t value_to_check = (offset + (scale * row));
    if (rs_type == kDATE) {
      value_to_check /= (24 * 60 * 60);
      value_to_check *= (24 * 60 * 60);
    }
    if (row == 0) {
      value_to_check = NULL_BIGINT;
    }

    if (mapd_val == value_to_check) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " "
               << std::to_string((offset + (scale * row))) << " vs. "
               << std::to_string(mapd_val);
    return false;
  }

  std::string getValueAsString(int row) {
    std::tm tm_struct;
    time_t t = offset + (scale * row);
    gmtime_r(&t, &tm_struct);
    char buf[128];
    strftime(buf, 128, format.c_str(), &tm_struct);
    return std::string(buf);
  }
};

class ArrayColumnDescriptor : public TestColumnDescriptor {
 public:
  std::string column_definition;
  const std::shared_ptr<TestColumnDescriptor> element_descriptor;
  int fixed_array_length;

  ArrayColumnDescriptor(std::string def,
                        const std::shared_ptr<TestColumnDescriptor> columnDesc,
                        int fixed_len = 0)
      : column_definition(def +
                          (fixed_len ? "[" + std::to_string(fixed_len) + "]" : "[]"))
      , element_descriptor(columnDesc)
      , fixed_array_length(fixed_len) {}

  bool skip_test(std::string name) override {
    return element_descriptor->skip_test("Array." + name);
  }

  std::string get_column_definition() override { return column_definition; }

  std::string make_column_value(int rows, std::string prefix, std::string suffix) {
    std::string values = prefix;

    int i = 0;

    if (fixed_array_length) {
      i = rows;
      rows += fixed_array_length;
    }

    bool firstElementWritten = false;

    for (; i < rows; i++) {
      if (firstElementWritten) {
        values += ", ";
      }
      values += element_descriptor->get_column_value(i + 1);
      firstElementWritten = true;
    }
    values += suffix;

    return values;
  }

  std::string get_column_value(int row) override {
    return make_column_value(row, "{", "}");
  }

  std::string get_update_column_value(int row) override {
    return make_column_value(row, "ARRAY[", "]");
  }

  bool check_column_value(const int row, const TDatum* datum) override {
    if (row == 0 && datum->is_null) {
    } else if (datum->is_null) {
      return false;
    }

    int elementIndex = 1;

    if (fixed_array_length) {
      elementIndex += row;
    }

    for (auto& dv : datum->val.arr_val) {
      if (!element_descriptor->check_column_value(elementIndex, &dv)) {
        return false;
      }

      elementIndex++;
    }

    return true;
  }
};

class GeoPointColumnDescriptor : public TestColumnDescriptor {
  std::string prefix;

 public:
  GeoPointColumnDescriptor(SQLTypes sql_type = kPOINT){};

  bool skip_test(std::string name) override { return "CreateTableAsSelect" != name; }

  std::string get_column_definition() override { return "POINT"; };

  std::string getColumnWktStringValue(int row) {
    return "POINT (" + std::to_string(row) + " 0)";
  }
  std::string get_column_value(int row) override {
    return "'" + getColumnWktStringValue(row) + "'";
  };

  bool check_column_value(int row, const TDatum* value) override {
    std::string mapd_val;
    if (!checked_get(row, value, mapd_val, std::string(""))) {
      return false;
    }

    if (mapd_val == getColumnWktStringValue(row)) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << getColumnWktStringValue(row)
               << " vs. " << mapd_val;
    return false;
  }
};

class GeoLinestringColumnDescriptor : public TestColumnDescriptor {
  std::string prefix;

 public:
  GeoLinestringColumnDescriptor(SQLTypes sql_type = kLINESTRING){};

  bool skip_test(std::string name) override { return "CreateTableAsSelect" != name; }

  std::string get_column_definition() override { return "LINESTRING"; };

  std::string getColumnWktStringValue(int row) {
    std::string linestring = "LINESTRING (0 0";
    for (int i = 0; i <= row; i++) {
      linestring += "," + std::to_string(row) + " 0";
    }
    linestring += ")";
    return linestring;
  }
  std::string get_column_value(int row) override {
    return "'" + getColumnWktStringValue(row) + "'";
  };

  bool check_column_value(int row, const TDatum* value) override {
    std::string mapd_val;
    if (!checked_get(row, value, mapd_val, std::string(""))) {
      return false;
    }

    if (mapd_val == getColumnWktStringValue(row)) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << getColumnWktStringValue(row)
               << " vs. " << mapd_val;
    return false;
  }
};

class GeoMultiPolygonColumnDescriptor : public TestColumnDescriptor {
  std::string prefix;

 public:
  GeoMultiPolygonColumnDescriptor(SQLTypes sql_type = kMULTIPOLYGON){};

  bool skip_test(std::string name) override { return "CreateTableAsSelect" != name; }

  std::string get_column_definition() override { return "MULTIPOLYGON"; };

  std::string getColumnWktStringValue(int row) {
    std::string polygon =
        "MULTIPOLYGON (((0 " + std::to_string(row) + ",4 " + std::to_string(row) + ",4 " +
        std::to_string(row + 4) + ",0 " + std::to_string(row + 4) + ",0 " +
        std::to_string(row) + "),(1 " + std::to_string(row + 1) + ",1 " +
        std::to_string(row + 2) + ",2 " + std::to_string(row + 2) + ",2 " +
        std::to_string(row + 1) + ",1 " + std::to_string(row + 1) + ")))";
    return polygon;
  }

  std::string get_column_value(int row) override {
    return "'" + getColumnWktStringValue(row) + "'";
  };

  bool check_column_value(int row, const TDatum* value) override {
    std::string mapd_val;
    if (!checked_get(row, value, mapd_val, std::string(""))) {
      return false;
    }

    if (mapd_val == getColumnWktStringValue(row)) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << getColumnWktStringValue(row)
               << " vs. " << mapd_val;
    return false;
  }
};

class GeoPolygonColumnDescriptor : public TestColumnDescriptor {
  std::string prefix;

 public:
  GeoPolygonColumnDescriptor(SQLTypes sql_type = kPOLYGON){};

  bool skip_test(std::string name) override { return "CreateTableAsSelect" != name; }

  std::string get_column_definition() override { return "POLYGON"; };

  std::string getColumnWktStringValue(int row) {
    std::string polygon =
        "POLYGON ((0 " + std::to_string(row) + ",4 " + std::to_string(row) + ",4 " +
        std::to_string(row + 4) + ",0 " + std::to_string(row + 4) + ",0 " +
        std::to_string(row) + "),(1 " + std::to_string(row + 1) + ",1 " +
        std::to_string(row + 2) + ",2 " + std::to_string(row + 2) + ",2 " +
        std::to_string(row + 1) + ",1 " + std::to_string(row + 1) + "))";
    return polygon;
  }

  std::string get_column_value(int row) override {
    return "'" + getColumnWktStringValue(row) + "'";
  };

  bool check_column_value(int row, const TDatum* value) override {
    std::string mapd_val;
    if (!checked_get(row, value, mapd_val, std::string(""))) {
      return false;
    }

    if (mapd_val == getColumnWktStringValue(row)) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << getColumnWktStringValue(row)
               << " vs. " << mapd_val;
    return false;
  }
};

struct Ctas
    : testing::Test,
      testing::WithParamInterface<std::vector<std::shared_ptr<TestColumnDescriptor>>> {
  std::vector<std::shared_ptr<TestColumnDescriptor>> columnDescriptors;

  Ctas() { columnDescriptors = GetParam(); }
};

struct Itas
    : testing::Test,
      testing::WithParamInterface<std::vector<std::shared_ptr<TestColumnDescriptor>>> {
  std::vector<std::shared_ptr<TestColumnDescriptor>> columnDescriptors;

  Itas() { columnDescriptors = GetParam(); }
};

struct Update
    : testing::Test,
      testing::WithParamInterface<std::vector<std::shared_ptr<TestColumnDescriptor>>> {
  std::vector<std::shared_ptr<TestColumnDescriptor>> columnDescriptors;

  Update() { columnDescriptors = GetParam(); }
};

TQueryResult run_multiple_agg(std::string sql) {
  TQueryResult result;
  g_client->sql_execute(result, g_session_id, sql, false, "", -1, -1);
  return result;
}

void run_ddl_statement(std::string ddl) {
  run_multiple_agg(ddl);
}

TEST(Ctas, SyntaxCheck) {
  std::string ddl = "DROP TABLE IF EXISTS CTAS_SOURCE;";
  run_ddl_statement(ddl);
  ddl = "DROP TABLE IF EXISTS CTAS_TARGET;";

  run_ddl_statement(ddl);

  run_ddl_statement("CREATE TABLE CTAS_SOURCE (id int);");

  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  run_ddl_statement(ddl);
  EXPECT_THROW(run_ddl_statement(ddl), apache::thrift::TException);
  ddl = "DROP TABLE CTAS_TARGET;";
  run_ddl_statement(ddl);

  ddl = "CREATE TEMPORARY TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  run_ddl_statement(ddl);
  EXPECT_THROW(run_ddl_statement(ddl), apache::thrift::TException);
  ddl = "DROP TABLE CTAS_TARGET;";
  run_ddl_statement(ddl);

  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE WITH( FRAGMENT_SIZE=3 );";
  run_ddl_statement(ddl);
  EXPECT_THROW(run_ddl_statement(ddl), apache::thrift::TException);
  ddl = "DROP TABLE CTAS_TARGET;";
  run_ddl_statement(ddl);

  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE WITH( MAX_CHUNK_SIZE=3 );";
  run_ddl_statement(ddl);
  EXPECT_THROW(run_ddl_statement(ddl), apache::thrift::TException);
  ddl = "DROP TABLE CTAS_TARGET;";
  run_ddl_statement(ddl);
}

TEST_P(Ctas, CreateTableAsSelect) {
  run_ddl_statement("DROP TABLE IF EXISTS CTAS_SOURCE;");
  run_ddl_statement("DROP TABLE IF EXISTS CTAS_TARGET;");

  std::string create_sql = "CREATE TABLE CTAS_SOURCE (id int";
  for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
    auto tcd = columnDescriptors[col];
    if (tcd->skip_test("CreateTableAsSelect")) {
      LOG(ERROR) << "not supported... skipping";
      return;
    }

    create_sql += ", col_" + std::to_string(col) + " " + tcd->get_column_definition();
  }
  create_sql += ");";

  LOG(INFO) << create_sql;

  run_ddl_statement(create_sql);

  size_t num_rows = 25;

  // fill source table
  for (unsigned int row = 0; row < num_rows; row++) {
    std::string insert_sql = "INSERT INTO CTAS_SOURCE VALUES (" + std::to_string(row);
    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];
      insert_sql += ", " + tcd->get_column_value(row);
    }
    insert_sql += ");";

    //    LOG(INFO) << "insert_sql: " << insert_sql;

    run_multiple_agg(insert_sql);
  }

  // execute CTAS
  std::string create_ctas_sql = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  LOG(INFO) << create_ctas_sql;

  run_ddl_statement(create_ctas_sql);

  // check tables
  TTableDetails td_source;
  TTableDetails td_target;

  g_client->get_table_details(td_source, g_session_id, "CTAS_SOURCE");
  g_client->get_table_details(td_target, g_session_id, "CTAS_TARGET");
  ASSERT_EQ(td_source.row_desc.size(), td_target.row_desc.size());

  // compare source against CTAS
  std::string select_sql = "SELECT * FROM CTAS_SOURCE ORDER BY id;";
  std::string select_ctas_sql = "SELECT * FROM CTAS_TARGET ORDER BY id;";

  LOG(INFO) << select_sql;
  auto select_result = run_multiple_agg(select_sql);

  LOG(INFO) << select_ctas_sql;
  auto select_ctas_result = run_multiple_agg(select_ctas_sql);

  ASSERT_EQ(num_rows, select_result.row_set.rows.size());
  ASSERT_EQ(num_rows, select_ctas_result.row_set.rows.size());

  for (unsigned int row = 0; row < num_rows; row++) {
    const auto select_crt_row = select_result.row_set.rows[row];
    const auto select_ctas_crt_row = select_ctas_result.row_set.rows[row];

    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];

      {
        const auto mapd_variant = select_crt_row.cols[col + 1];
        ASSERT_EQ(true, tcd->check_column_value(row, &mapd_variant));
      }
      {
        const auto mapd_variant = select_ctas_crt_row.cols[col + 1];
        ASSERT_EQ(true, tcd->check_column_value(row, &mapd_variant));
      }
    }
  }
}

void itasTestBody(std::vector<std::shared_ptr<TestColumnDescriptor>>& columnDescriptors,
                  std::string targetPartitionScheme = ")") {
  run_ddl_statement("DROP TABLE IF EXISTS ITAS_SOURCE;");
  run_ddl_statement("DROP TABLE IF EXISTS ITAS_TARGET;");

  std::string create_source_sql = "CREATE TABLE ITAS_SOURCE (id int";
  std::string create_target_sql = "CREATE TABLE ITAS_TARGET (id int";
  for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
    auto tcd = columnDescriptors[col];
    if (tcd->skip_test("CreateTableAsSelect")) {
      LOG(ERROR) << "not supported... skipping";
      return;
    }

    create_source_sql +=
        ", col_" + std::to_string(col) + " " + tcd->get_column_definition();
    create_target_sql +=
        ", col_" + std::to_string(col) + " " + tcd->get_column_definition();
  }
  create_source_sql += ");";
  create_target_sql += targetPartitionScheme + ";";

  LOG(INFO) << create_source_sql;
  LOG(INFO) << create_target_sql;

  run_ddl_statement(create_source_sql);
  run_ddl_statement(create_target_sql);

  size_t num_rows = 25;

  // fill source table
  for (unsigned int row = 0; row < num_rows; row++) {
    std::string insert_sql = "INSERT INTO ITAS_SOURCE VALUES (" + std::to_string(row);
    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];
      insert_sql += ", " + tcd->get_column_value(row);
    }
    insert_sql += ");";

    //    LOG(INFO) << "insert_sql: " << insert_sql;

    run_multiple_agg(insert_sql);
  }

  // execute CTAS
  std::string insert_itas_sql = "INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;";
  LOG(INFO) << insert_itas_sql;

  run_ddl_statement(insert_itas_sql);

  // compare source against CTAS
  std::string select_sql = "SELECT * FROM ITAS_SOURCE ORDER BY id;";
  std::string select_itas_sql = "SELECT * FROM ITAS_TARGET ORDER BY id;";

  LOG(INFO) << select_sql;
  auto select_result = run_multiple_agg(select_sql);

  LOG(INFO) << select_itas_sql;
  auto select_itas_result = run_multiple_agg(select_itas_sql);

  ASSERT_EQ(num_rows, select_result.row_set.rows.size());
  ASSERT_EQ(num_rows, select_itas_result.row_set.rows.size());

  for (unsigned int row = 0; row < num_rows; row++) {
    const auto select_crt_row = select_result.row_set.rows[row];
    const auto select_itas_crt_row = select_itas_result.row_set.rows[row];

    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];

      {
        const auto mapd_variant = select_crt_row.cols[col + 1];
        ASSERT_EQ(true, tcd->check_column_value(row, &mapd_variant));
      }
      {
        const auto mapd_variant = select_itas_crt_row.cols[col + 1];
        ASSERT_EQ(true, tcd->check_column_value(row, &mapd_variant));
      }
    }
  }
}

TEST_P(Itas, InsertIntoTableFromSelect) {
  itasTestBody(columnDescriptors, ")");
}

TEST_P(Itas, InsertIntoTableFromSelectReplicated) {
  itasTestBody(columnDescriptors, ") WITH (partitions='REPLICATED')");
}

TEST_P(Itas, InsertIntoTableFromSelectSharded) {
  itasTestBody(columnDescriptors,
               ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')");
}

const std::shared_ptr<TestColumnDescriptor> STRING_NONE_BASE =
    std::make_shared<StringColumnDescriptor>("TEXT ENCODING NONE",
                                             kTEXT,
                                             "STRING_NONE_BASE");

#ifdef RUN_ALL_TEST

#define INSTANTIATE_DATA_INGESTION_TEST(CDT)                                           \
  INSTANTIATE_TEST_CASE_P(                                                             \
      CDT,                                                                             \
      Ctas,                                                                            \
      testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{CDT}));       \
  INSTANTIATE_TEST_CASE_P(                                                             \
      CDT,                                                                             \
      Itas,                                                                            \
      testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{CDT}));       \
  INSTANTIATE_TEST_CASE_P(                                                             \
      CDT,                                                                             \
      Update,                                                                          \
      testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{CDT}));       \
  INSTANTIATE_TEST_CASE_P(                                                             \
      VARLEN_TEXT_AND_##CDT,                                                           \
      Update,                                                                          \
      testing::Values(                                                                 \
          std::vector<std::shared_ptr<TestColumnDescriptor>>{STRING_NONE_BASE, CDT})); \
  INSTANTIATE_TEST_CASE_P(                                                             \
      CDT##_AND_VARLEN_TEXT,                                                           \
      Update,                                                                          \
      testing::Values(                                                                 \
          std::vector<std::shared_ptr<TestColumnDescriptor>>{CDT, STRING_NONE_BASE}))

#else

#define INSTANTIATE_DATA_INGESTION_TEST(CDT)

#endif

#define BOOLEAN_COLUMN_TEST(name, c_type, definition, sql_type, null)  \
  const std::shared_ptr<TestColumnDescriptor> name =                   \
      std::make_shared<BooleanColumnDescriptor>(definition, sql_type); \
  INSTANTIATE_DATA_INGESTION_TEST(name)

#define NUMBER_COLUMN_TEST(name, c_type, definition, sql_type, null)                \
  const std::shared_ptr<TestColumnDescriptor> name =                                \
      std::make_shared<NumberColumnDescriptor<c_type>>(definition, sql_type, null); \
  INSTANTIATE_DATA_INGESTION_TEST(name)

#define STRING_COLUMN_TEST(name, definition, sql_type)                       \
  const std::shared_ptr<TestColumnDescriptor> name =                         \
      std::make_shared<StringColumnDescriptor>(definition, sql_type, #name); \
  INSTANTIATE_DATA_INGESTION_TEST(name)

#define TIME_COLUMN_TEST(name, definition, sql_type, format, offset, scale) \
  const std::shared_ptr<TestColumnDescriptor> name =                        \
      std::make_shared<DateTimeColumnDescriptor>(                           \
          definition, sql_type, format, offset, scale);                     \
  INSTANTIATE_DATA_INGESTION_TEST(name)

#define ARRAY_COLUMN_TEST(name, definition)                            \
  const std::shared_ptr<TestColumnDescriptor> name##_ARRAY =           \
      std::make_shared<ArrayColumnDescriptor>(definition, name, 0);    \
  INSTANTIATE_DATA_INGESTION_TEST(name##_ARRAY);                       \
  const std::shared_ptr<TestColumnDescriptor> name##_FIXED_LEN_ARRAY = \
      std::make_shared<ArrayColumnDescriptor>(definition, name, 3);    \
  INSTANTIATE_DATA_INGESTION_TEST(name##_FIXED_LEN_ARRAY)

BOOLEAN_COLUMN_TEST(BOOLEAN, int64_t, "BOOLEAN", kBOOLEAN, NULL_TINYINT);
ARRAY_COLUMN_TEST(BOOLEAN, "BOOLEAN");

NUMBER_COLUMN_TEST(TINYINT, int64_t, "TINYINT", kTINYINT, NULL_TINYINT);
ARRAY_COLUMN_TEST(TINYINT, "TINYINT");

NUMBER_COLUMN_TEST(SMALLINT, int64_t, "SMALLINT", kSMALLINT, NULL_SMALLINT);
NUMBER_COLUMN_TEST(SMALLINT_8,
                   int64_t,
                   "SMALLINT ENCODING FIXED(8)",
                   kSMALLINT,
                   NULL_SMALLINT);
ARRAY_COLUMN_TEST(SMALLINT, "SMALLINT");

NUMBER_COLUMN_TEST(INTEGER, int64_t, "INTEGER", kINT, NULL_INT);
NUMBER_COLUMN_TEST(INTEGER_8, int64_t, "INTEGER ENCODING FIXED(8)", kINT, NULL_INT);
NUMBER_COLUMN_TEST(INTEGER_16, int64_t, "INTEGER ENCODING FIXED(16)", kINT, NULL_INT);
ARRAY_COLUMN_TEST(INTEGER, "INTEGER");

NUMBER_COLUMN_TEST(BIGINT, int64_t, "BIGINT", kBIGINT, NULL_BIGINT);
NUMBER_COLUMN_TEST(BIGINT_8, int64_t, "BIGINT ENCODING FIXED(8)", kBIGINT, NULL_BIGINT);
NUMBER_COLUMN_TEST(BIGINT_16, int64_t, "BIGINT ENCODING FIXED(16)", kBIGINT, NULL_BIGINT);
NUMBER_COLUMN_TEST(BIGINT_32, int64_t, "BIGINT ENCODING FIXED(32)", kBIGINT, NULL_BIGINT);
ARRAY_COLUMN_TEST(BIGINT, "BIGINT");

NUMBER_COLUMN_TEST(FLOAT, float, "FLOAT", kFLOAT, NULL_FLOAT);
ARRAY_COLUMN_TEST(FLOAT, "FLOAT");

NUMBER_COLUMN_TEST(DOUBLE, double, "DOUBLE", kDOUBLE, NULL_DOUBLE);
ARRAY_COLUMN_TEST(DOUBLE, "DOUBLE");

NUMBER_COLUMN_TEST(NUMERIC, double, "NUMERIC(18)", kNUMERIC, NULL_DOUBLE);
NUMBER_COLUMN_TEST(NUMERIC_32,
                   double,
                   "NUMERIC(9) ENCODING FIXED(32)",
                   kNUMERIC,
                   NULL_DOUBLE);
NUMBER_COLUMN_TEST(NUMERIC_16,
                   double,
                   "NUMERIC(4) ENCODING FIXED(16)",
                   kNUMERIC,
                   NULL_DOUBLE);
ARRAY_COLUMN_TEST(NUMERIC, "NUMERIC(18)");

NUMBER_COLUMN_TEST(DECIMAL, double, "DECIMAL(18,9)", kDECIMAL, NULL_DOUBLE);
NUMBER_COLUMN_TEST(DECIMAL_32,
                   double,
                   "DECIMAL(9,2) ENCODING FIXED(32)",
                   kDECIMAL,
                   NULL_DOUBLE);
NUMBER_COLUMN_TEST(DECIMAL_16,
                   double,
                   "DECIMAL(4,2) ENCODING FIXED(16)",
                   kDECIMAL,
                   NULL_DOUBLE);
ARRAY_COLUMN_TEST(DECIMAL, "DECIMAL(18,9)");

STRING_COLUMN_TEST(CHAR, "CHAR(100)", kCHAR);
STRING_COLUMN_TEST(CHAR_DICT, "CHAR(100) ENCODING DICT", kCHAR);
STRING_COLUMN_TEST(CHAR_DICT_8, "CHAR(100) ENCODING DICT(8)", kCHAR);
STRING_COLUMN_TEST(CHAR_DICT_16, "CHAR(100) ENCODING DICT(16)", kCHAR);
STRING_COLUMN_TEST(CHAR_NONE, "CHAR(100) ENCODING NONE", kCHAR);
ARRAY_COLUMN_TEST(CHAR, "CHAR(100)");

STRING_COLUMN_TEST(VARCHAR, "VARCHAR(100)", kCHAR);
STRING_COLUMN_TEST(VARCHAR_DICT, "VARCHAR(100) ENCODING DICT", kCHAR);
STRING_COLUMN_TEST(VARCHAR_DICT_8, "VARCHAR(100) ENCODING DICT(8)", kCHAR);
STRING_COLUMN_TEST(VARCHAR_DICT_16, "VARCHAR(100) ENCODING DICT(16)", kCHAR);
STRING_COLUMN_TEST(VARCHAR_NONE, "VARCHAR(100) ENCODING NONE", kCHAR);
ARRAY_COLUMN_TEST(VARCHAR, "VARCHAR(100)");

STRING_COLUMN_TEST(TEXT, "TEXT", kTEXT);
STRING_COLUMN_TEST(TEXT_DICT, "TEXT ENCODING DICT", kTEXT);
STRING_COLUMN_TEST(TEXT_DICT_8, "TEXT ENCODING DICT(8)", kTEXT);
STRING_COLUMN_TEST(TEXT_DICT_16, "TEXT ENCODING DICT(16)", kTEXT);
STRING_COLUMN_TEST(TEXT_NONE, "TEXT ENCODING NONE", kTEXT);
ARRAY_COLUMN_TEST(TEXT, "TEXT");

TIME_COLUMN_TEST(TIME, "TIME", kTIME, "%T", 0, 1);
TIME_COLUMN_TEST(TIME_32, "TIME ENCODING FIXED(32)", kTIME, "%T", 0, 1);
ARRAY_COLUMN_TEST(TIME, "TIME");

TIME_COLUMN_TEST(DATE, "DATE", kDATE, "%F", 0, 160 * 60 * 100);
TIME_COLUMN_TEST(DATE_16, "DATE ENCODING FIXED(16)", kDATE, "%F", 0, 160 * 60 * 100);
ARRAY_COLUMN_TEST(DATE, "DATE");

TIME_COLUMN_TEST(TIMESTAMP, "TIMESTAMP", kTIMESTAMP, "%F %T", 0, 160 * 60 * 100);
TIME_COLUMN_TEST(TIMESTAMP_32,
                 "TIMESTAMP ENCODING FIXED(32)",
                 kTIMESTAMP,
                 "%F %T",
                 0,
                 160 * 60 * 100);
ARRAY_COLUMN_TEST(TIMESTAMP, "TIMESTAMP");

const std::shared_ptr<TestColumnDescriptor> GEO_POINT =
    std::shared_ptr<TestColumnDescriptor>(new GeoPointColumnDescriptor(kPOINT));
INSTANTIATE_DATA_INGESTION_TEST(GEO_POINT);

const std::shared_ptr<TestColumnDescriptor> GEO_LINESTRING =
    std::shared_ptr<TestColumnDescriptor>(new GeoLinestringColumnDescriptor(kLINESTRING));
INSTANTIATE_DATA_INGESTION_TEST(GEO_LINESTRING);

const std::shared_ptr<TestColumnDescriptor> GEO_POLYGON =
    std::shared_ptr<TestColumnDescriptor>(new GeoPolygonColumnDescriptor(kPOLYGON));
INSTANTIATE_DATA_INGESTION_TEST(GEO_POLYGON);

const std::shared_ptr<TestColumnDescriptor> GEO_MULTI_POLYGON =
    std::shared_ptr<TestColumnDescriptor>(
        new GeoMultiPolygonColumnDescriptor(kMULTIPOLYGON));
INSTANTIATE_DATA_INGESTION_TEST(GEO_MULTI_POLYGON);

INSTANTIATE_TEST_CASE_P(
    MIXED_NO_GEO,
    Ctas,
    testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{BOOLEAN,
                                                                       TINYINT,
                                                                       SMALLINT,
                                                                       INTEGER,
                                                                       BIGINT,
                                                                       FLOAT,
                                                                       DOUBLE,
                                                                       NUMERIC,
                                                                       DECIMAL,
                                                                       CHAR,
                                                                       VARCHAR,
                                                                       TEXT,
                                                                       TIME,
                                                                       DATE,
                                                                       TIMESTAMP}));

INSTANTIATE_TEST_CASE_P(
    MIXED_NO_GEO,
    Itas,
    testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{BOOLEAN,
                                                                       TINYINT,
                                                                       SMALLINT,
                                                                       INTEGER,
                                                                       BIGINT,
                                                                       FLOAT,
                                                                       DOUBLE,
                                                                       NUMERIC,
                                                                       DECIMAL,
                                                                       CHAR,
                                                                       VARCHAR,
                                                                       TEXT,
                                                                       TIME,
                                                                       DATE,
                                                                       TIMESTAMP}));

INSTANTIATE_TEST_CASE_P(
    MIXED_WITH_GEO,
    Update,
    testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{TEXT,
                                                                       INTEGER,
                                                                       DOUBLE,
                                                                       GEO_POINT,
                                                                       GEO_LINESTRING,
                                                                       GEO_POLYGON,
                                                                       GEO_MULTI_POLYGON

    }));

int main(int argc, char* argv[]) {
  int err = 0;
  TestHelpers::init_logger_stderr_only(argc, argv);

  try {
    testing::InitGoogleTest(&argc, argv);

    namespace po = boost::program_options;

    po::options_description desc("Options");

    // these two are here to allow passing correctly google testing parameters
    desc.add_options()("gtest_list_tests", "list all test");
    desc.add_options()("gtest_filter", "filters tests, use --help for details");

    std::string host = "localhost";
    int port = 6274;
    std::string cert = "";

    std::string user = "admin";
    std::string pwd = "HyperInteractive";
    std::string db = "omnisci";

    desc.add_options()(
        "host",
        po::value<std::string>(&host)->default_value(host)->implicit_value(host),
        "hostname of target server");
    desc.add_options()("port",
                       po::value<int>(&port)->default_value(port)->implicit_value(port),
                       "tcp port of target server");
    desc.add_options()(
        "cert",
        po::value<std::string>(&cert)->default_value(cert)->implicit_value(cert),
        "tls/ssl certificate to use for contacting target server");
    desc.add_options()(
        "user",
        po::value<std::string>(&user)->default_value(user)->implicit_value(user),
        "user name to connect as");
    desc.add_options()(
        "pwd",
        po::value<std::string>(&pwd)->default_value(pwd)->implicit_value(pwd),
        "password to connect with");
    desc.add_options()("db",
                       po::value<std::string>(&db)->default_value(db)->implicit_value(db),
                       "db to connect to");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    auto transport = openBufferedClientTransport(host, port, cert);
    transport->open();
    auto protocol = std::make_shared<TBinaryProtocol>(transport);
    g_client = std::make_shared<MapDClient>(protocol);

    g_client->connect(g_session_id, user, pwd, db);

    err = RUN_ALL_TESTS();

  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = -1;
  }

  return err;
}
