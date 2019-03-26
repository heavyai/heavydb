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

#include "../QueryRunner/QueryRunner.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ctime>
#include <iostream>
#include "../Parser/parser.h"
#include "../QueryEngine/ArrowResultSet.h"
#include "../QueryEngine/Execute.h"
#include "../Shared/file_delete.h"

#include "../Shared/ConfigResolve.h"

// uncomment to run full test suite
// #define RUN_ALL_TEST

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;

class TestColumnDescriptor {
 public:
  virtual std::string get_column_definition() = 0;
  virtual std::string get_column_value(int row) = 0;
  virtual std::string get_update_column_value(int row) { return get_column_value(row); }

  virtual bool check_column_value(const int row,
                                  const SQLTypeInfo& type,
                                  const ScalarTargetValue* scalarValue) = 0;
  virtual bool check_column_value(const int row,
                                  const SQLTypeInfo& type,
                                  const TargetValue* value) {
    const auto scalar_mapd_variant = boost::get<ScalarTargetValue>(value);
    if (nullptr == scalar_mapd_variant) {
      return false;
    }

    return check_column_value(row, type, scalar_mapd_variant);
  }
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
  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* value) override {
    if (type.get_type() != rs_type) {
      return false;
    }

    const auto mapd_as_int_p = boost::get<T>(value);
    if (nullptr == mapd_as_int_p) {
      LOG(ERROR) << "row: null";
      return false;
    }

    const auto mapd_val = *mapd_as_int_p;

    T value_to_check = (T)row;
    if (row == 0) {
      value_to_check = null_value;
    } else if (kDECIMAL == rs_type) {
      // TODO: create own descriptor for decimals
      value_to_check *= pow10(type.get_scale());
    }

    if (mapd_val == value_to_check) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << std::to_string(value_to_check)
               << " vs. " << std::to_string(mapd_val);
    return false;
  }

  int64_t pow10(int scale) {
    int64_t pow = 1;
    for (int i = 0; i < scale; i++) {
      pow *= 10;
    }

    return pow;
  }
};

class BooleanColumnDescriptor : public TestColumnDescriptor {
  std::string column_definition;
  SQLTypes rs_type;

 public:
  BooleanColumnDescriptor(std::string col_type, SQLTypes sql_type)
      : column_definition(col_type), rs_type(sql_type){};

  bool skip_test(std::string name) override {
    return "UpdateColumnByLiteral" == name || "Array.UpdateColumnByLiteral" == name;
  }

  std::string get_column_definition() override { return column_definition; };
  std::string get_column_value(int row) override {
    if (0 == row) {
      return "null";
    }

    return (row % 2) ? "'true'" : "'false'";
  };
  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* value) override {
    if (type.get_type() != rs_type) {
      return false;
    }

    const auto mapd_as_int_p = boost::get<int64_t>(value);
    if (nullptr == mapd_as_int_p) {
      LOG(ERROR) << "row: null";
      return false;
    }

    const auto mapd_val = *mapd_as_int_p;

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
  SQLTypes rs_type;
  std::string prefix;

 public:
  StringColumnDescriptor(std::string col_type, SQLTypes sql_type, std::string pfix)
      : column_definition(col_type), rs_type(sql_type), prefix(pfix){};

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
  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* value) override {
    if (!(type.get_type() == rs_type || type.get_type() == kTEXT)) {
      return false;
    }

    const auto mapd_as_str_p = boost::get<NullableString>(value);
    if (nullptr == mapd_as_str_p) {
      return false;
    }

    const auto mapd_str_p = boost::get<std::string>(mapd_as_str_p);
    if (nullptr == mapd_str_p) {
      return 0 == row;
    }

    const auto mapd_val = *mapd_str_p;
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
  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* value) override {
    if (type.get_type() != rs_type) {
      return false;
    }

    const auto mapd_as_int_p = boost::get<int64_t>(value);
    if (nullptr == mapd_as_int_p) {
      return 0 == row;
    }

    const auto mapd_val = *mapd_as_int_p;

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
    return make_column_value(row + 1, "{", "}");
  }

  std::string get_update_column_value(int row) override {
    return make_column_value(row + 1, "ARRAY[", "]");
  }

  bool check_column_value(const int row,
                          const SQLTypeInfo& type,
                          const TargetValue* value) override {
    auto arrayValue = boost::get<ArrayTargetValue>(value);

    if (!arrayValue) {
      return false;
    }
    if (!arrayValue->is_initialized()) {
      return true;  // NULL array, nothing to check
    }

    const SQLTypeInfo subtype = type.get_elem_type();

    int elementIndex = 1;

    if (fixed_array_length) {
      elementIndex += row + 1;
    }

    const auto& vec = arrayValue->get();
    for (auto& scalarValue : vec) {
      if (!element_descriptor->check_column_value(elementIndex, subtype, &scalarValue)) {
        return false;
      }

      elementIndex++;
    }

    return true;
  }

  bool check_column_value(const int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* scalarValue) override {
    return false;
  }
};

class GeoPointColumnDescriptor : public TestColumnDescriptor {
  SQLTypes rs_type;
  std::string prefix;

 public:
  GeoPointColumnDescriptor(SQLTypes sql_type = kPOINT) : rs_type(sql_type){};

  bool skip_test(std::string name) override { return "CreateTableAsSelect" != name; }

  std::string get_column_definition() override { return "POINT"; };

  std::string getColumnWktStringValue(int row) {
    return "POINT (" + std::to_string(row) + " 0)";
  }
  std::string get_column_value(int row) override {
    return "'" + getColumnWktStringValue(row) + "'";
  };

  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* value) override {
    if (!(type.get_type() == rs_type)) {
      return false;
    }

    const auto mapd_as_str_p = boost::get<NullableString>(value);
    if (nullptr == mapd_as_str_p) {
      return false;
    }

    const auto mapd_str_p = boost::get<std::string>(mapd_as_str_p);
    if (nullptr == mapd_str_p) {
      return false;
    }

    const auto mapd_val = *mapd_str_p;
    if (mapd_val == getColumnWktStringValue(row)) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << getColumnWktStringValue(row)
               << " vs. " << mapd_val;
    return false;
  }
};

class GeoLinestringColumnDescriptor : public TestColumnDescriptor {
  SQLTypes rs_type;
  std::string prefix;

 public:
  GeoLinestringColumnDescriptor(SQLTypes sql_type = kLINESTRING) : rs_type(sql_type){};

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

  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* value) override {
    if (!(type.get_type() == rs_type)) {
      return false;
    }

    const auto mapd_as_str_p = boost::get<NullableString>(value);
    if (nullptr == mapd_as_str_p) {
      return false;
    }

    const auto mapd_str_p = boost::get<std::string>(mapd_as_str_p);
    if (nullptr == mapd_str_p) {
      return false;
    }

    const auto mapd_val = *mapd_str_p;
    if (mapd_val == getColumnWktStringValue(row)) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << getColumnWktStringValue(row)
               << " vs. " << mapd_val;
    return false;
  }
};

class GeoMultiPolygonColumnDescriptor : public TestColumnDescriptor {
  SQLTypes rs_type;
  std::string prefix;

 public:
  GeoMultiPolygonColumnDescriptor(SQLTypes sql_type = kMULTIPOLYGON)
      : rs_type(sql_type){};

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

  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* value) override {
    if (!(type.get_type() == rs_type)) {
      return false;
    }

    const auto mapd_as_str_p = boost::get<NullableString>(value);
    if (nullptr == mapd_as_str_p) {
      return false;
    }

    const auto mapd_str_p = boost::get<std::string>(mapd_as_str_p);
    if (nullptr == mapd_str_p) {
      return false;
    }

    const auto mapd_val = *mapd_str_p;
    if (mapd_val == getColumnWktStringValue(row)) {
      return true;
    }

    LOG(ERROR) << "row: " << std::to_string(row) << " " << getColumnWktStringValue(row)
               << " vs. " << mapd_val;
    return false;
  }
};

class GeoPolygonColumnDescriptor : public TestColumnDescriptor {
  SQLTypes rs_type;
  std::string prefix;

 public:
  GeoPolygonColumnDescriptor(SQLTypes sql_type = kPOLYGON) : rs_type(sql_type){};

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

  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const ScalarTargetValue* value) override {
    if (!(type.get_type() == rs_type)) {
      return false;
    }

    const auto mapd_as_str_p = boost::get<NullableString>(value);
    if (nullptr == mapd_as_str_p) {
      return false;
    }

    const auto mapd_str_p = boost::get<std::string>(mapd_as_str_p);
    if (nullptr == mapd_str_p) {
      return false;
    }

    const auto mapd_val = *mapd_str_p;
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

struct Update
    : testing::Test,
      testing::WithParamInterface<std::vector<std::shared_ptr<TestColumnDescriptor>>> {
  std::vector<std::shared_ptr<TestColumnDescriptor>> columnDescriptors;

  Update() { columnDescriptors = GetParam(); }
};

TEST(Ctas, SyntaxCheck) {
  std::string ddl = "DROP TABLE IF EXISTS CTAS_SOURCE;";
  QueryRunner::run_ddl_statement(ddl, g_session);
  ddl = "DROP TABLE IF EXISTS CTAS_TARGET;";
  QueryRunner::run_ddl_statement(ddl, g_session);

  QueryRunner::run_ddl_statement("CREATE TABLE CTAS_SOURCE (id int);", g_session);

  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  QueryRunner::run_ddl_statement(ddl, g_session);
  EXPECT_THROW(QueryRunner::run_ddl_statement(ddl, g_session), std::runtime_error);
  ddl = "CREATE TABLE IF NOT EXISTS CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  QueryRunner::run_ddl_statement(ddl, g_session);
  ddl = "DROP TABLE CTAS_TARGET;";
  QueryRunner::run_ddl_statement(ddl, g_session);

  ddl = "CREATE TEMPORARY TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  QueryRunner::run_ddl_statement(ddl, g_session);
  EXPECT_THROW(QueryRunner::run_ddl_statement(ddl, g_session), std::runtime_error);
  ddl = "CREATE TEMPORARY TABLE IF NOT EXISTS CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  QueryRunner::run_ddl_statement(ddl, g_session);
  ddl = "DROP TABLE CTAS_TARGET;";
  QueryRunner::run_ddl_statement(ddl, g_session);

  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE WITH( FRAGMENT_SIZE=3 );";
  QueryRunner::run_ddl_statement(ddl, g_session);
  EXPECT_THROW(QueryRunner::run_ddl_statement(ddl, g_session), std::runtime_error);
  ddl =
      "CREATE TABLE IF NOT EXISTS CTAS_TARGET AS SELECT * FROM CTAS_SOURCE WITH( "
      "FRAGMENT_SIZE=3 );";
  QueryRunner::run_ddl_statement(ddl, g_session);
  ddl = "DROP TABLE CTAS_TARGET;";
  QueryRunner::run_ddl_statement(ddl, g_session);

  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE WITH( MAX_CHUNK_SIZE=3 );";
  QueryRunner::run_ddl_statement(ddl, g_session);
  EXPECT_THROW(QueryRunner::run_ddl_statement(ddl, g_session), std::runtime_error);
  ddl =
      "CREATE TABLE IF NOT EXISTS CTAS_TARGET AS SELECT * FROM CTAS_SOURCE WITH( "
      "MAX_CHUNK_SIZE=3 );";
  QueryRunner::run_ddl_statement(ddl, g_session);
  ddl = "DROP TABLE CTAS_TARGET;";
  QueryRunner::run_ddl_statement(ddl, g_session);
}

TEST(Ctas, LiteralStringTest) {
  std::string ddl = "DROP TABLE IF EXISTS CTAS_SOURCE;";
  QueryRunner::run_ddl_statement(ddl, g_session);
  ddl = "DROP TABLE IF EXISTS CTAS_TARGET;";
  QueryRunner::run_ddl_statement(ddl, g_session);

  QueryRunner::run_ddl_statement("CREATE TABLE CTAS_SOURCE (id int, val int);",
                                 g_session);

  QueryRunner::run_multiple_agg("INSERT INTO CTAS_SOURCE VALUES(1,1); ",
                                g_session,
                                ExecutorDeviceType::CPU,
                                true,
                                true,
                                nullptr);
  QueryRunner::run_multiple_agg("INSERT INTO CTAS_SOURCE VALUES(2,2); ",
                                g_session,
                                ExecutorDeviceType::CPU,
                                true,
                                true,
                                nullptr);
  QueryRunner::run_multiple_agg("INSERT INTO CTAS_SOURCE VALUES(3,3); ",
                                g_session,
                                ExecutorDeviceType::CPU,
                                true,
                                true,
                                nullptr);

  ddl =
      "CREATE TABLE CTAS_TARGET AS select id, val, (case when val=1 then 'aa' else 'bb' "
      "end) as txt FROM CTAS_SOURCE;";
  QueryRunner::run_ddl_statement(ddl, g_session);

  auto check = [](int id, std::string txt) {
    auto select_result = QueryRunner::run_multiple_agg(
        "SELECT txt FROM CTAS_TARGET WHERE id=" + std::to_string(id) + ";",
        g_session,
        ExecutorDeviceType::CPU,
        true,
        true,
        nullptr);

    const auto select_crt_row = select_result->getNextRow(true, false);
    const auto mapd_variant = select_crt_row[0];
    const auto scalar_mapd_variant = boost::get<ScalarTargetValue>(&mapd_variant);
    const auto mapd_as_str_p = boost::get<NullableString>(scalar_mapd_variant);
    const auto mapd_str_p = boost::get<std::string>(mapd_as_str_p);
    const auto mapd_val = *mapd_str_p;
    ASSERT_EQ(txt, mapd_val);
  };

  check(1, "aa");
  check(2, "bb");
  check(3, "bb");
}

TEST_P(Ctas, CreateTableAsSelect) {
  QueryRunner::run_ddl_statement("DROP TABLE IF EXISTS CTAS_SOURCE;", g_session);
  QueryRunner::run_ddl_statement("DROP TABLE IF EXISTS CTAS_TARGET;", g_session);

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

  QueryRunner::run_ddl_statement(create_sql, g_session);

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

    QueryRunner::run_multiple_agg(
        insert_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);
  }

  // execute CTAS
  std::string create_ctas_sql = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  LOG(INFO) << create_ctas_sql;

  QueryRunner::run_ddl_statement(create_ctas_sql, g_session);

  // check tables
  Catalog_Namespace::Catalog& cat = g_session->getCatalog();
  const TableDescriptor* td_source = cat.getMetadataForTable("CTAS_SOURCE");
  const TableDescriptor* td_target = cat.getMetadataForTable("CTAS_TARGET");

  auto source_cols =
      cat.getAllColumnMetadataForTable(td_source->tableId, false, true, false);
  auto target_cols =
      cat.getAllColumnMetadataForTable(td_target->tableId, false, true, false);

  ASSERT_EQ(source_cols.size(), target_cols.size());

  while (source_cols.size()) {
    auto source_col = source_cols.back();
    auto target_col = target_cols.back();

    LOG(INFO) << "Checking: " << source_col->columnName << " vs. "
              << target_col->columnName << " ( " << source_col->columnType.get_type_name()
              << " vs. " << target_col->columnType.get_type_name() << " )";

    //    ASSERT_EQ(source_col->columnType.get_type(), target_col->columnType.get_type());
    //    ASSERT_EQ(source_col->columnType.get_elem_type(),
    //    target_col->columnType.get_elem_type());
    ASSERT_EQ(source_col->columnType.get_compression(),
              target_col->columnType.get_compression());
    ASSERT_EQ(source_col->columnType.get_size(), target_col->columnType.get_size());

    source_cols.pop_back();
    target_cols.pop_back();
  }

  // compare source against CTAS
  std::string select_sql = "SELECT * FROM CTAS_SOURCE ORDER BY id;";
  std::string select_ctas_sql = "SELECT * FROM CTAS_TARGET ORDER BY id;";

  LOG(INFO) << select_sql;
  auto select_result = QueryRunner::run_multiple_agg(
      select_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);

  LOG(INFO) << select_ctas_sql;
  auto select_ctas_result = QueryRunner::run_multiple_agg(
      select_ctas_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);

  ASSERT_EQ(num_rows, select_result->rowCount());
  ASSERT_EQ(num_rows, select_ctas_result->rowCount());

  for (unsigned int row = 0; row < num_rows; row++) {
    const auto select_crt_row = select_result->getNextRow(true, false);
    const auto select_ctas_crt_row = select_ctas_result->getNextRow(true, false);

    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];

      {
        const auto mapd_variant = select_crt_row[col + 1];
        auto mapd_ti = select_result->getColType(col + 1);
        ASSERT_EQ(true, tcd->check_column_value(row, mapd_ti, &mapd_variant));
      }
      {
        const auto mapd_variant = select_ctas_crt_row[col + 1];
        auto mapd_ti = select_ctas_result->getColType(col + 1);
        ASSERT_EQ(true, tcd->check_column_value(row, mapd_ti, &mapd_variant));
      }
    }
  }
}

TEST_P(Update, UpdateColumnByColumn) {
  // disable if varlen update is not enabled
  if (!is_feature_enabled<VarlenUpdates>()) {
    LOG(WARNING) << "skipping...";
    return;
  }

  QueryRunner::run_ddl_statement("DROP TABLE IF EXISTS update_test;", g_session);

  std::string create_sql = "CREATE TABLE update_test(id int";
  for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
    auto tcd = columnDescriptors[col];

    if (tcd->skip_test("UpdateColumnByColumn")) {
      LOG(ERROR) << "not supported... skipping";
      return;
    }

    create_sql += ", col_src_" + std::to_string(col) + " " + tcd->get_column_definition();
    create_sql += ", col_dst_" + std::to_string(col) + " " + tcd->get_column_definition();
  }
  create_sql += ") WITH (fragment_size=3);";

  LOG(INFO) << create_sql;

  QueryRunner::run_ddl_statement(create_sql, g_session);

  size_t num_rows = 10;

  // fill source table
  for (unsigned int row = 0; row < num_rows; row++) {
    std::string insert_sql = "INSERT INTO update_test VALUES (" + std::to_string(row);
    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];
      insert_sql += ", " + tcd->get_column_value(row);
      insert_sql += ", " + tcd->get_column_value(row + 1);
    }
    insert_sql += ");";

    //    LOG(INFO) << "insert_sql: " << insert_sql;

    QueryRunner::run_multiple_agg(
        insert_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);
  }

  // execute Updates
  std::string update_sql = "UPDATE update_test set ";
  for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
    update_sql +=
        " col_dst_" + std::to_string(col) + "=" + "col_src_" + std::to_string(col);
    if (col + 1 < columnDescriptors.size()) {
      update_sql += ",";
    }
  }
  update_sql += ";";

  LOG(INFO) << update_sql;

  QueryRunner::run_multiple_agg(
      update_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);

  // compare source against CTAS
  std::string select_sql = "SELECT id";
  for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
    select_sql += ", col_dst_" + std::to_string(col);
    select_sql += ", col_src_" + std::to_string(col);
  }
  select_sql += " FROM update_test ORDER BY id;";

  LOG(INFO) << select_sql;
  auto select_result = QueryRunner::run_multiple_agg(
      select_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);

  for (unsigned int row = 0; row < num_rows; row++) {
    const auto select_crt_row = select_result->getNextRow(true, false);

    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];

      {
        const auto mapd_variant = select_crt_row[(2 * col) + 1];
        auto mapd_ti = select_result->getColType((2 * col) + 1);
        ASSERT_EQ(true, tcd->check_column_value(row, mapd_ti, &mapd_variant));
      }
      {
        const auto mapd_variant = select_crt_row[(2 * col) + 2];
        auto mapd_ti = select_result->getColType((2 * col) + 2);
        ASSERT_EQ(true, tcd->check_column_value(row, mapd_ti, &mapd_variant));
      }
    }
  }
}

void updateColumnByLiteralTest(
    std::vector<std::shared_ptr<TestColumnDescriptor>>& columnDescriptors,
    size_t numColsToUpdate) {
  // disable if varlen update is not enabled
  if (!is_feature_enabled<VarlenUpdates>()) {
    LOG(WARNING) << "skipping...";
    return;
  }

  QueryRunner::run_ddl_statement("DROP TABLE IF EXISTS update_test;", g_session);

  std::string create_sql = "CREATE TABLE update_test(id int";
  for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
    auto tcd = columnDescriptors[col];

    if (col < numColsToUpdate) {
      if (tcd->skip_test("UpdateColumnByLiteral")) {
        LOG(ERROR) << "not supported... skipping";
        return;
      }
    }
    create_sql += ", col_dst_" + std::to_string(col) + " " + tcd->get_column_definition();
  }
  create_sql += ") WITH (fragment_size=3);";

  LOG(INFO) << create_sql;

  QueryRunner::run_ddl_statement(create_sql, g_session);

  size_t num_rows = 10;

  // fill source table
  for (unsigned int row = 0; row < num_rows; row++) {
    std::string insert_sql = "INSERT INTO update_test VALUES (" + std::to_string(row);
    for (unsigned int col = 0; col < numColsToUpdate; col++) {
      auto tcd = columnDescriptors[col];
      insert_sql += ", " + tcd->get_column_value(row + 1);
    }
    for (unsigned int col = numColsToUpdate; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];
      insert_sql += ", " + tcd->get_column_value(row);
    }
    insert_sql += ");";

    //    LOG(INFO) << "insert_sql: " << insert_sql;

    QueryRunner::run_multiple_agg(
        insert_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);
  }

  // execute Updates
  for (unsigned int row = 0; row < num_rows; row++) {
    std::string update_sql = "UPDATE update_test set ";
    for (unsigned int col = 0; col < numColsToUpdate; col++) {
      auto tcd = columnDescriptors[col];
      update_sql +=
          " col_dst_" + std::to_string(col) + "=" + tcd->get_update_column_value(row);
      if (col + 1 < numColsToUpdate) {
        update_sql += ",";
      }
    }
    update_sql += " WHERE id=" + std::to_string(row) + ";";
    LOG(INFO) << update_sql;
    QueryRunner::run_multiple_agg(
        update_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);
  }

  // compare source against CTAS
  std::string select_sql = "SELECT id";
  for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
    select_sql += ", col_dst_" + std::to_string(col);
  }
  select_sql += " FROM update_test ORDER BY id;";

  LOG(INFO) << select_sql;
  auto select_result = QueryRunner::run_multiple_agg(
      select_sql, g_session, ExecutorDeviceType::CPU, true, true, nullptr);

  for (unsigned int row = 0; row < num_rows; row++) {
    const auto select_crt_row = select_result->getNextRow(true, false);

    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];

      {
        const auto mapd_variant = select_crt_row[(1 * col) + 1];
        auto mapd_ti = select_result->getColType((1 * col) + 1);
        ASSERT_EQ(true, tcd->check_column_value(row, mapd_ti, &mapd_variant));
      }
    }
  }
}

TEST_P(Update, UpdateColumnByLiteral) {
  updateColumnByLiteralTest(columnDescriptors, columnDescriptors.size());
}

TEST_P(Update, UpdateFirstColumnByLiteral) {
  if (columnDescriptors.size() > 1) {
    updateColumnByLiteralTest(columnDescriptors, 1);
  }
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

NUMBER_COLUMN_TEST(NUMERIC, int64_t, "NUMERIC(18)", kNUMERIC, NULL_BIGINT);
NUMBER_COLUMN_TEST(NUMERIC_32,
                   int64_t,
                   "NUMERIC(9) ENCODING FIXED(32)",
                   kNUMERIC,
                   NULL_BIGINT);
NUMBER_COLUMN_TEST(NUMERIC_16,
                   int64_t,
                   "NUMERIC(4) ENCODING FIXED(16)",
                   kNUMERIC,
                   NULL_BIGINT);
ARRAY_COLUMN_TEST(NUMERIC, "NUMERIC(18)");

NUMBER_COLUMN_TEST(DECIMAL, int64_t, "DECIMAL(18,9)", kDECIMAL, NULL_BIGINT);
NUMBER_COLUMN_TEST(DECIMAL_32,
                   int64_t,
                   "DECIMAL(9,2) ENCODING FIXED(32)",
                   kDECIMAL,
                   NULL_BIGINT);
NUMBER_COLUMN_TEST(DECIMAL_16,
                   int64_t,
                   "DECIMAL(4,2) ENCODING FIXED(16)",
                   kDECIMAL,
                   NULL_BIGINT);
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

const std::vector<std::shared_ptr<TestColumnDescriptor>> ALL = {STRING_NONE_BASE,
                                                                BOOLEAN,
                                                                BOOLEAN_ARRAY,
                                                                BOOLEAN_FIXED_LEN_ARRAY,
                                                                TINYINT,
                                                                TINYINT_ARRAY,
                                                                TINYINT_FIXED_LEN_ARRAY,
                                                                SMALLINT_8,
                                                                SMALLINT,
                                                                SMALLINT_ARRAY,
                                                                SMALLINT_FIXED_LEN_ARRAY,
                                                                INTEGER_8,
                                                                INTEGER_16,
                                                                INTEGER,
                                                                INTEGER_ARRAY,
                                                                INTEGER_FIXED_LEN_ARRAY,
                                                                BIGINT_8,
                                                                BIGINT_16,
                                                                BIGINT_32,
                                                                BIGINT,
                                                                BIGINT_ARRAY,
                                                                BIGINT_FIXED_LEN_ARRAY,
                                                                FLOAT,
                                                                FLOAT_ARRAY,
                                                                FLOAT_FIXED_LEN_ARRAY,
                                                                DOUBLE,
                                                                DOUBLE_ARRAY,
                                                                DOUBLE_FIXED_LEN_ARRAY,
                                                                NUMERIC_16,
                                                                NUMERIC_32,
                                                                NUMERIC,
                                                                NUMERIC_ARRAY,
                                                                NUMERIC_FIXED_LEN_ARRAY,
                                                                DECIMAL_16,
                                                                DECIMAL_32,
                                                                DECIMAL,
                                                                DECIMAL_ARRAY,
                                                                DECIMAL_FIXED_LEN_ARRAY,
                                                                TEXT_NONE,
                                                                TEXT_DICT,
                                                                TEXT_DICT_8,
                                                                TEXT_DICT_16,
                                                                TEXT,
                                                                TEXT_ARRAY,
                                                                TEXT_FIXED_LEN_ARRAY,
                                                                TIME_32,
                                                                TIME,
                                                                TIME_ARRAY,
                                                                TIME_FIXED_LEN_ARRAY,
                                                                DATE_16,
                                                                DATE,
                                                                DATE_ARRAY,
                                                                DATE_FIXED_LEN_ARRAY,
                                                                TIMESTAMP_32,
                                                                TIMESTAMP,
                                                                TIMESTAMP_ARRAY,
                                                                TIMESTAMP_FIXED_LEN_ARRAY,
                                                                GEO_POINT,
                                                                GEO_LINESTRING,
                                                                GEO_POLYGON,
                                                                GEO_MULTI_POLYGON};

INSTANTIATE_TEST_CASE_P(MIXED_ALL, Ctas, testing::Values(ALL));

INSTANTIATE_TEST_CASE_P(MIXED_ALL, Update, testing::Values(ALL));

INSTANTIATE_TEST_CASE_P(
    MIXED_VARLEN_WITHOUT_GEO,
    Update,
    testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{
        STRING_NONE_BASE,
        BOOLEAN_ARRAY,
        BOOLEAN_FIXED_LEN_ARRAY,
        TINYINT_ARRAY,
        TINYINT_FIXED_LEN_ARRAY,
        SMALLINT_ARRAY,
        SMALLINT_FIXED_LEN_ARRAY,
        INTEGER_ARRAY,
        INTEGER_FIXED_LEN_ARRAY,
        BIGINT_ARRAY,
        BIGINT_FIXED_LEN_ARRAY,
        NUMERIC_ARRAY,
        NUMERIC_FIXED_LEN_ARRAY,
        TEXT_NONE,
        TEXT_ARRAY,
        TEXT_FIXED_LEN_ARRAY,
        TIME_ARRAY,
        TIME_FIXED_LEN_ARRAY,
        DATE_ARRAY,
        DATE_FIXED_LEN_ARRAY,
        TIMESTAMP_ARRAY,
        TIMESTAMP_FIXED_LEN_ARRAY

    }));

int main(int argc, char* argv[]) {
  int err = 0;

  try {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging(argv[0]);
    g_session.reset(QueryRunner::get_session(BASE_PATH));

    err = RUN_ALL_TESTS();

  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    err = -1;
  }

  return err;
}
