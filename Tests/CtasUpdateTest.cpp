/*
 * Copyright 2021, OmniSci, Inc.
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
#include <ctime>
#include <iostream>
#include "DBHandlerTestHelpers.h"
#include "TestHelpers.h"

// uncomment to run full test suite
// #define RUN_ALL_TEST

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

class TestColumnDescriptor {
 public:
  virtual std::string get_column_definition() = 0;
  virtual std::string get_column_value(int row) = 0;
  virtual std::string get_column_comparison(int row, std::string colname) = 0;
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

  std::string get_column_comparison(int row, std::string colname) override {
    if (0 == row) {
      return colname + " is null";
    }
    return colname + " = " + std::to_string(row);
  }

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
  std::string get_column_comparison(int row, std::string colname) override {
    if (0 == row) {
      return colname + " is null";
    }

    return colname + " = " + ((row % 2) ? "'true'" : "'false'");
  }

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
  std::string get_column_comparison(int row, std::string colname) override {
    if (0 == row) {
      return colname + " is null";
    }

    return colname + " = " + "'" + prefix + "_" + std::to_string(row) + "'";
  }

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
  std::string get_column_comparison(int row, std::string colname) override {
    if (0 == row) {
      return colname + " is null";
    }

    return colname + " = " + "'" + getValueAsString(row) + "'";
  }
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
    rows = fixupRowForDatatype(rows);

    if (0 == rows) {
      return "null";
    }

    std::string values = prefix;

    rows -= 1;
    int i = 0;
    auto elementOffset = fixupElementIndexOffset();

    if (fixed_array_length) {
      i = rows;
      rows += fixed_array_length;
    }

    bool firstElementWritten = false;

    for (; i < rows; i++) {
      if (firstElementWritten) {
        values += ", ";
      }
      values += element_descriptor->get_column_value(i + elementOffset);
      firstElementWritten = true;
    }
    values += suffix;

    return values;
  }

  std::string get_column_value(int row) override {
    return make_column_value(row, "{", "}");
  }
  std::string make_column_comparison(int rows, std::string colname) {
    rows = fixupRowForDatatype(rows);

    if (0 == rows) {
      return colname + " is null";
    }

    std::string values;

    rows -= 1;
    int i = 0;
    auto elementOffset = fixupElementIndexOffset();

    if (fixed_array_length) {
      i = rows;
      rows += fixed_array_length;
    }

    bool firstElementWritten = false;
    int count = 1;
    for (; i < rows; i++) {
      if (firstElementWritten) {
        values += " AND ";
      }
      values += colname + "[" + std::to_string(count) +
                "] = " + element_descriptor->get_column_value(i + elementOffset);
      count++;
      firstElementWritten = true;
    }

    return values;
  }

  std::string get_column_comparison(int row, std::string colname) override {
    return make_column_comparison(row, colname);
  }
  std::string get_update_column_value(int row) override {
    return make_column_value(row, "ARRAY[", "]");
  }

  int fixupRowForDatatype(int row) {
    if (fixed_array_length) {
      auto def = element_descriptor->get_column_definition();
      if (def == "TEXT" || def == "CHAR(100)" || def == "VARCHAR(100)") {
        return row + 1;
      }
    }
    return row;
  }

  int fixupElementIndexOffset() {
    if ("BOOLEAN" == element_descriptor->get_column_definition()) {
      return 1;  // null
    }
    return 0;
  }

  bool check_column_value(int row,
                          const SQLTypeInfo& type,
                          const TargetValue* value) override {
    const auto actual_row = row;
    row = fixupRowForDatatype(row);

    auto arrayValue = boost::get<ArrayTargetValue>(value);

    if (!arrayValue) {
      return false;
    }

    if (0 == row) {
      return !arrayValue->is_initialized();
    }

    if (!arrayValue->is_initialized()) {
      return false;
    }

    const SQLTypeInfo subtype = type.get_elem_type();

    int elementIndex = 0;

    if (fixed_array_length) {
      elementIndex += row - 1;
    }
    auto elementOffset = fixupElementIndexOffset();

    const auto& vec = arrayValue->get();
    for (auto& scalarValue : vec) {
      if (!element_descriptor->check_column_value(
              elementIndex + elementOffset, subtype, &scalarValue)) {
        LOG(ERROR) << get_column_definition() << " row (" << actual_row
                   << ") -> expected " << get_column_value(actual_row);
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
  std::string get_column_comparison(int row, std::string colname) override {
    return colname + " = ST_GeomFromText('" + getColumnWktStringValue(row) + "')";
  }
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
  std::string get_column_comparison(int row, std::string colname) override {
    return colname + " = ST_GeomFromText('" + getColumnWktStringValue(row) + "')";
    ;
  }
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
  std::string get_column_comparison(int row, std::string colname) override {
    return colname + " = ST_GeomFromText('" + getColumnWktStringValue(row) + "')";
  }

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
  std::string get_column_comparison(int row, std::string colname) override {
    return colname + " = ST_GeomFromText('" + getColumnWktStringValue(row) + "')";
  }

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

class Itas : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    // Default connection string outside of thrift
  }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  void itas_syntax_check(const std::string& sql_in,
                         const std::string& create_src_ddl,
                         const std::string& create_dst_ddl,
                         bool throws) {
    std::string drop_src_ddl = "DROP TABLE IF EXISTS ITAS_SOURCE;";
    std::string drop_target_ddl = "DROP TABLE IF EXISTS ITAS_TARGET;";
    sql(drop_src_ddl);
    sql(drop_target_ddl);
    sql(create_src_ddl);
    sql(create_dst_ddl);
    if (throws) {
      EXPECT_ANY_THROW(sql(sql_in));
    } else {
      EXPECT_NO_THROW(sql(sql_in));
    }
  }

  void create_itas_tables(
      std::vector<std::shared_ptr<TestColumnDescriptor>>& columnDescriptors,
      std::string sourcePartitionScheme,
      std::string targetPartitionScheme,
      std::string targetTempTable,
      size_t n_rows) {
    sql("DROP TABLE IF EXISTS ITAS_SOURCE;");
    sql("DROP TABLE IF EXISTS ITAS_TARGET;");

    std::string create_source_sql = "CREATE TABLE ITAS_SOURCE (id int";
    std::string create_target_sql =
        "CREATE " + targetTempTable + " TABLE ITAS_TARGET (id int";
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
    create_source_sql += sourcePartitionScheme + ";";
    create_target_sql += targetPartitionScheme + ";";

    LOG(INFO) << create_source_sql;
    LOG(INFO) << create_target_sql;

    sql(create_source_sql);
    sql(create_target_sql);

    // fill source table
    for (unsigned int row = 0; row < n_rows; row++) {
      std::string insert_sql = "INSERT INTO ITAS_SOURCE VALUES (" + std::to_string(row);
      for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
        auto tcd = columnDescriptors[col];
        insert_sql += ", " + tcd->get_column_value(row);
      }
      insert_sql += ");";
      LOG(INFO) << insert_sql;
      sql(insert_sql);
    }
  }

  void itasTestBody(std::vector<std::shared_ptr<TestColumnDescriptor>>& columnDescriptors,
                    std::string sourcePartitionScheme = ")",
                    std::string targetPartitionScheme = ")",
                    std::string targetTempTable = "") {
    size_t num_rows = 25;
    create_itas_tables(columnDescriptors,
                       sourcePartitionScheme,
                       targetPartitionScheme,
                       targetTempTable,
                       num_rows);

    // execute ITAS
    std::string insert_itas_sql = "INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;";
    LOG(INFO) << insert_itas_sql;
    sql(insert_itas_sql);

    // compare source against CTAS
    std::string select_sql = "SELECT * FROM ITAS_SOURCE ORDER BY id;";
    std::string select_itas_sql = "SELECT * FROM ITAS_TARGET ORDER BY id;";

    LOG(INFO) << select_sql;
    TQueryResult select_result;
    sql(select_result, select_sql);

    LOG(INFO) << select_itas_sql;
    TQueryResult select_itas_result;
    sql(select_itas_result, select_itas_sql);

    // check we have a columnar result
    ASSERT_EQ(true, select_itas_result.row_set.is_columnar);
    ASSERT_EQ(true, select_result.row_set.is_columnar);

    auto columns = select_result.row_set.columns;
    auto columns_itas = select_itas_result.row_set.columns;

    // check expected result count
    ASSERT_EQ(num_rows, columns[0].nulls.size());
    ASSERT_EQ(num_rows, columns_itas[0].nulls.size());

    // check same number of columns
    ASSERT_EQ(columns.size(), columns_itas.size());

    std::string col_details = " id INT ";
    for (size_t c = 0; c < columns.size(); c++) {
      if (c > 1) {
        col_details = "col_" + std::to_string(c - 1) + " " +
                      columnDescriptors[c - 1]->get_column_definition();
      }
      for (size_t r = 0; r < num_rows; r++) {
        ASSERT_EQ(columns[c].nulls[r], columns_itas[c].nulls[r])
            << col_details << " Column " << std::to_string(c) << " row "
            << std::to_string(r);
        if (columns[c].data.int_col.size() > 0) {
          ASSERT_EQ(columns[c].data.int_col[r], columns_itas[c].data.int_col[r])
              << col_details << " Column " << std::to_string(c) << " row "
              << std::to_string(r);
        }
        if (columns[c].data.real_col.size() > 0) {
          ASSERT_EQ(columns[c].data.real_col[r], columns_itas[c].data.real_col[r])
              << col_details << " Column " << std::to_string(c) << " row "
              << std::to_string(r);
        }
        if (columns[c].data.str_col.size() > 0) {
          ASSERT_EQ(columns[c].data.str_col[r], columns_itas[c].data.str_col[r])
              << col_details << " Column " << std::to_string(c) << " row "
              << std::to_string(r);
        }
        if (columns[c].data.arr_col.size() > 0) {
          ASSERT_EQ(columns[c].data.arr_col[r], columns_itas[c].data.arr_col[r])
              << col_details << " Column " << std::to_string(c) << " row "
              << std::to_string(r);
        }
      }
    }
  }

  void create_partial_itas_tables(const std::string& target_partitioning) {
    sql("DROP TABLE IF EXISTS ITAS_SOURCE;");
    sql("DROP TABLE IF EXISTS ITAS_TARGET;");
    std::string create_source_sql =
        "CREATE TABLE itas_source (id int, "
        "not_nullable_col int NOT NULL,"
        "nullable_col int,"
        "text_array text[])";
    std::string create_target_sql =
        "CREATE TABLE itas_target (id int, "
        "not_nullable_col int NOT NULL,"
        "nullable_col int,"
        "text_array text[]" +
        target_partitioning;
    sql(create_source_sql);
    sql(create_target_sql);
    sql("INSERT INTO itas_source VALUES(1, 10, 100, ARRAY['A', 'B']);");
    sql("INSERT INTO itas_source VALUES(2, 20, 200, ARRAY['Aa', 'Bb']);");
    sql("INSERT INTO itas_source VALUES(3, 30, 300, ARRAY['Aaa', 'Bbb']);");
    sql("INSERT INTO itas_source VALUES(4, 40, 400, ARRAY['Aaaa', 'Bbbb']);");
    sql("INSERT INTO itas_source VALUES(5, 50, 500, ARRAY['Aaaaa', 'Bbbbb']);");
  }

  void populate_partial_itas_target(std::vector<bool> column_is_used) {
    sql("TRUNCATE TABLE itas_target;");
    std::string insert_itas_sql = "INSERT INTO itas_target(id";
    std::string select_from_sql = "SELECT id";
    for (size_t i = 0; i < column_is_used.size(); ++i) {
      if (column_is_used[i]) {
        std::string append_str = ", col_" + std::to_string(i);
        insert_itas_sql += append_str;
        select_from_sql += append_str;
      } else {
        // else we omit the column
      }
    }
    select_from_sql += " FROM itas_source";
    insert_itas_sql += ") " + select_from_sql + ";";
    LOG(INFO) << insert_itas_sql;
    sql(insert_itas_sql);
  }

  void validate_partial_itas_results(
      std::vector<bool> column_is_used,
      std::vector<std::shared_ptr<TestColumnDescriptor>>& cds,
      size_t num_rows) {
    // compare source against original
    std::string select_sql = "SELECT * FROM itas_source ORDER BY id;";
    std::string select_itas_sql = "SELECT * FROM itas_target ORDER BY id;";

    LOG(INFO) << select_sql;
    TQueryResult select_result;
    sql(select_result, select_sql);

    LOG(INFO) << select_itas_sql;
    TQueryResult select_itas_result;
    sql(select_itas_result, select_itas_sql);

    auto columns = select_result.row_set.columns;
    auto columns_itas = select_itas_result.row_set.columns;

    // check expected result count
    ASSERT_EQ(num_rows, columns[0].nulls.size());
    ASSERT_EQ(num_rows, columns_itas[0].nulls.size());

    // check same number of columns
    ASSERT_EQ(columns.size(), columns_itas.size());

    ASSERT_EQ(columns.size(), (column_is_used.size() + 1));

    for (size_t i = 0; i < column_is_used.size(); i++) {
      for (size_t r = 0; r < num_rows; r++) {
        // if column was not populated null should be true
        size_t c = i + 1;  // column is offset by id column
        if (!column_is_used[i]) {
          ASSERT_EQ(columns_itas[c].nulls[r], true);
        } else {
          ASSERT_EQ(columns[c].nulls[r], columns_itas[c].nulls[r]);
          if (columns[c].data.int_col.size() > 0) {
            ASSERT_EQ(columns[c].data.int_col[r], columns_itas[c].data.int_col[r]);
          }
          if (columns[c].data.real_col.size() > 0) {
            ASSERT_EQ(columns[c].data.real_col[r], columns_itas[c].data.real_col[r]);
          }
          if (columns[c].data.str_col.size() > 0) {
            ASSERT_EQ(columns[c].data.str_col[r], columns_itas[c].data.str_col[r]);
          }
          if (columns[c].data.arr_col.size() > 0) {
            ASSERT_EQ(columns[c].data.arr_col[r], columns_itas[c].data.arr_col[r]);
          }
        }
      }
    }
  }

  void test_partial_columns(std::vector<bool> column_is_used,
                            std::vector<std::shared_ptr<TestColumnDescriptor>>& cds,
                            size_t num_rows) {
    populate_partial_itas_target(column_is_used);
    validate_partial_itas_results(column_is_used, cds, num_rows);
  }

  void partial_itas_test_body(
      std::vector<std::shared_ptr<TestColumnDescriptor>>& columnDescriptors,
      std::string sourcePartitionScheme = ")",
      std::string targetPartitionScheme = ")",
      std::string targetTempTable = "") {
    size_t num_rows = 25;
    create_itas_tables(columnDescriptors,
                       sourcePartitionScheme,
                       targetPartitionScheme,
                       targetTempTable,
                       num_rows);

    std::vector<bool> use_even_columns, use_odd_columns;
    use_even_columns.resize(columnDescriptors.size(), false);
    use_odd_columns.resize(columnDescriptors.size(), false);
    for (size_t c = 0; c < columnDescriptors.size(); ++c) {
      if (c % 2 == 0) {
        use_even_columns[c] = true;
      } else {
        use_odd_columns[c] = true;
      }
    }
    test_partial_columns(use_even_columns, columnDescriptors, num_rows);
    test_partial_columns(use_odd_columns, columnDescriptors, num_rows);
  }
};

class Itas_P : public Itas,
               public testing::WithParamInterface<
                   std::vector<std::shared_ptr<TestColumnDescriptor>>> {
 public:
  std::vector<std::shared_ptr<TestColumnDescriptor>> columnDescriptors;
  Itas_P() { columnDescriptors = GetParam(); }
};

class Update : public DBHandlerTestFixture,
               public testing::WithParamInterface<
                   std::vector<std::shared_ptr<TestColumnDescriptor>>> {
 public:
  std::vector<std::shared_ptr<TestColumnDescriptor>> columnDescriptors;

  Update() { columnDescriptors = GetParam(); }
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    // Default connection string outside of thrift
  }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  void updateColumnByLiteralTest(
      std::vector<std::shared_ptr<TestColumnDescriptor>>& columnDescriptors,
      size_t numColsToUpdate) {
    sql("DROP TABLE IF EXISTS update_test;");
    sql("DROP TABLE IF EXISTS update_canonical;");
    std::string create_sql = "CREATE TABLE update_test(id int";
    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];

      if (col < numColsToUpdate) {
        if (tcd->skip_test("UpdateColumnByLiteral")) {
          LOG(ERROR) << "not supported... skipping";
          return;
        }
      }
      create_sql +=
          ", col_dst_" + std::to_string(col) + " " + tcd->get_column_definition();
    }
    create_sql += ") WITH (fragment_size=3);";

    LOG(INFO) << create_sql;
    sql(create_sql);

    {
      std::string create_sql = "CREATE TABLE update_canonical(id int";
      for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
        auto tcd = columnDescriptors[col];

        if (col < numColsToUpdate) {
          if (tcd->skip_test("UpdateColumnByLiteral")) {
            LOG(ERROR) << "not supported... skipping";
            return;
          }
        }
        create_sql +=
            ", col_dst_" + std::to_string(col) + " " + tcd->get_column_definition();
      }
      create_sql += ") WITH (fragment_size=3);";

      LOG(INFO) << create_sql;
      sql(create_sql);
    }
    size_t num_rows = 10;

    // fill canonical table
    for (unsigned int row = 0; row < num_rows; row++) {
      std::string insert_sql =
          "INSERT INTO update_canonical VALUES (" + std::to_string(row);

      for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
        auto tcd = columnDescriptors[col];
        insert_sql += ", " + tcd->get_column_value(row);
      }
      insert_sql += ");";

      LOG(INFO) << insert_sql;
      sql(insert_sql);
    }

    // fill update table
    for (size_t row = 0; row < num_rows; row++) {
      std::string insert_sql = "INSERT INTO update_test VALUES (" + std::to_string(row);
      // place non 'natural' value in for this row as it will be updated
      for (unsigned int col = 0; col < numColsToUpdate; col++) {
        auto tcd = columnDescriptors[col];
        insert_sql += ", " + tcd->get_column_value(row + 1);
      }
      for (unsigned int col = numColsToUpdate; col < columnDescriptors.size(); col++) {
        auto tcd = columnDescriptors[col];
        insert_sql += ", " + tcd->get_column_value(row);
      }
      insert_sql += ");";

      LOG(INFO) << insert_sql;
      sql(insert_sql);
    }

    // execute Updates
    for (size_t row = 0; row < num_rows; row++) {
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
      sql(update_sql);
    }

    std::string select_sql = "SELECT * FROM update_canonical ORDER BY id;";
    std::string select_update_sql = "SELECT * FROM update_test ORDER BY id;";

    LOG(INFO) << select_sql;
    TQueryResult select_result;
    sql(select_result, select_sql);

    LOG(INFO) << select_update_sql;
    TQueryResult select_update_result;
    sql(select_update_result, select_update_sql);

    auto columns = select_result.row_set.columns;
    auto columns_update = select_update_result.row_set.columns;

    // check expected result count
    ASSERT_EQ(num_rows, columns[0].nulls.size());
    ASSERT_EQ(num_rows, columns_update[0].nulls.size());

    // check same number of columns
    ASSERT_EQ(columns.size(), columns_update.size());

    for (size_t c = 0; c < columns.size(); c++) {
      for (size_t r = 0; r < num_rows; r++) {
        // if column was not populated null should be true
        ASSERT_EQ(columns[c].nulls[r], columns_update[c].nulls[r]);
        if (columns[c].data.int_col.size() > 0) {
          ASSERT_EQ(columns[c].data.int_col[r], columns_update[c].data.int_col[r]);
        }
        if (columns[c].data.real_col.size() > 0) {
          ASSERT_EQ(columns[c].data.real_col[r], columns_update[c].data.real_col[r]);
        }
        if (columns[c].data.str_col.size() > 0) {
          ASSERT_EQ(columns[c].data.str_col[r], columns_update[c].data.str_col[r]);
        }
        if (columns[c].data.arr_col.size() > 0) {
          ASSERT_EQ(columns[c].data.arr_col[r], columns_update[c].data.arr_col[r]);
        }
      }
    }

    // select from table to get expected result
    // TODO cant quite select everything yet
    // will leave here in the hope oneday we can select  with a where on
    // geo
    if (false) {
      for (size_t row = 0; row < num_rows; row++) {
        std::string select_sql = "SELECT id  FROM update_test where ";
        for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
          auto tcd = columnDescriptors[col];
          if (col > 0) {
            select_sql += " AND ";  // + ("col_dst_" + std::to_string(col)) + "++ ";
          }
          select_sql +=
              tcd->get_column_comparison(row, ("col_dst_" + std::to_string(col)));
        }
        int64_t number;
        LOG(INFO) << select_sql;
        TQueryResult select_result;
        sql(select_result, select_sql);
        assertResultSetEqual({{number}}, select_result);
      }
    }
  }
};

class Ctas : public DBHandlerTestFixture,
             public testing::WithParamInterface<
                 std::vector<std::shared_ptr<TestColumnDescriptor>>> {
 public:
  std::vector<std::shared_ptr<TestColumnDescriptor>> columnDescriptors;

  Ctas() { columnDescriptors = GetParam(); }
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    // Default connection string outside of thrift
  }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  void runCtasTest(std::vector<std::shared_ptr<TestColumnDescriptor>>& columnDescriptors,
                   std::string create_ctas_sql,
                   size_t num_rows,
                   size_t num_rows_to_check,
                   std::string sourcePartitionScheme = ")") {
    sql("DROP TABLE IF EXISTS CTAS_SOURCE;");
    sql("DROP TABLE IF EXISTS CTAS_TARGET;");

    std::string create_sql = "CREATE TABLE CTAS_SOURCE (id int";
    for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
      auto tcd = columnDescriptors[col];
      if (tcd->skip_test("CreateTableAsSelect")) {
        LOG(ERROR) << "not supported... skipping";
        return;
      }

      create_sql += ", col_" + std::to_string(col) + " " + tcd->get_column_definition();
    }
    create_sql += sourcePartitionScheme + ";";

    LOG(INFO) << create_sql;

    sql(create_sql);

    // fill source table
    for (size_t row = 0; row < num_rows; row++) {
      std::string insert_sql = "INSERT INTO CTAS_SOURCE VALUES (" + std::to_string(row);
      for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
        auto tcd = columnDescriptors[col];
        insert_sql += ", " + tcd->get_column_value(row);
      }
      insert_sql += ");";

      sql(insert_sql);
    }

    // execute CTAS
    LOG(INFO) << create_ctas_sql;

    sql(create_ctas_sql);

    // check tables
    auto& cat = getCatalog();
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
                << target_col->columnName << " ( "
                << source_col->columnType.get_type_name() << " vs. "
                << target_col->columnType.get_type_name() << " )";

      //    ASSERT_EQ(source_col->columnType.get_type(),
      //    target_col->columnType.get_type());
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
    TQueryResult select_result;
    sql(select_result, select_sql);

    LOG(INFO) << select_ctas_sql;
    TQueryResult select_ctas_result;
    sql(select_ctas_result, select_ctas_sql);

    auto columns = select_result.row_set.columns;
    auto columns_ctas = select_ctas_result.row_set.columns;

    // check expected result count
    ASSERT_EQ(num_rows, columns[0].nulls.size());
    ASSERT_EQ(num_rows_to_check, columns_ctas[0].nulls.size());

    // check same number of columns
    ASSERT_EQ(columns.size(), columns_ctas.size());

    for (size_t c = 0; c < columns.size(); c++) {
      for (size_t r = 0; r < num_rows_to_check; r++) {
        ASSERT_EQ(columns[c].nulls[r], columns_ctas[c].nulls[r]);
        if (columns[c].data.int_col.size() > 0) {
          ASSERT_EQ(columns[c].data.int_col[r], columns_ctas[c].data.int_col[r]);
        }
        if (columns[c].data.real_col.size() > 0) {
          ASSERT_EQ(columns[c].data.real_col[r], columns_ctas[c].data.real_col[r]);
        }
        if (columns[c].data.str_col.size() > 0) {
          ASSERT_EQ(columns[c].data.str_col[r], columns_ctas[c].data.str_col[r]);
        }
        if (columns[c].data.arr_col.size() > 0) {
          ASSERT_EQ(columns[c].data.arr_col[r], columns_ctas[c].data.arr_col[r]);
        }
      }
    }
  }
};

TEST_P(Ctas, SyntaxCheck) {
  sql("DROP TABLE IF EXISTS CTAS_SOURCE;");

  sql("DROP TABLE IF EXISTS CTAS_SOURCE_WITH;");
  sql("DROP TABLE IF EXISTS CTAS_SOURCE_TEXT;");
  sql("DROP TABLE IF EXISTS CTAS_TARGET;");

  sql("CREATE TABLE CTAS_SOURCE (id int);");
  sql("CREATE TABLE CTAS_SOURCE_WITH (id int);");

  std::string ddl = "CREATE TABLE CTAS_TARGET AS SELECT \n * \r FROM CTAS_SOURCE;";
  sql(ddl);
  queryAndAssertPartialException(
      ddl, "Table CTAS_TARGET already exists and no data was loaded");
  ddl = "DROP TABLE CTAS_TARGET;";
  sql(ddl);

  ddl = "CREATE TEMPORARY TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  sql(ddl);
  queryAndAssertPartialException(
      ddl, "Table CTAS_TARGET already exists and no data was loaded");
  ddl = "DROP TABLE CTAS_TARGET;";
  sql(ddl);

  ddl =
      "CREATE TABLE CTAS_TARGET AS SELECT * \n FROM \r CTAS_SOURCE WITH( FRAGMENT_SIZE=3 "
      ");";
  sql(ddl);
  queryAndAssertPartialException(
      ddl, "Table CTAS_TARGET already exists and no data was loaded");
  ddl = "DROP TABLE CTAS_TARGET;";
  sql(ddl);

  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE WITH( MAX_CHUNK_SIZE=3 );";
  sql(ddl);
  queryAndAssertPartialException(
      ddl, "Table CTAS_TARGET already exists and no data was loaded");
  ddl = "DROP TABLE CTAS_TARGET;";
  sql(ddl);

  ddl =
      "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE_WITH WITH( MAX_CHUNK_SIZE=3 "
      ");";
  sql(ddl);
  queryAndAssertPartialException(
      ddl, "Table CTAS_TARGET already exists and no data was loaded");
  ddl = "DROP TABLE CTAS_TARGET;";
  sql(ddl);

  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE_WITH;";
  sql(ddl);
  queryAndAssertPartialException(
      ddl, "Table CTAS_TARGET already exists and no data was loaded");
  ddl = "DROP TABLE CTAS_TARGET;";
  sql(ddl);

  sql("CREATE TABLE CTAS_SOURCE_TEXT (id text);");
  ddl =
      "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE_TEXT WITH( "
      "USE_SHARED_DICTIONARIES='FALSE' );";
  sql(ddl);

  {
    auto& cat = getCatalog();
    auto td_source = cat.getMetadataForTable("CTAS_SOURCE_TEXT");
    auto cd_source = cat.getMetadataForColumn(td_source->tableId, "id");

    auto td_target = cat.getMetadataForTable("CTAS_TARGET");
    auto cd_target = cat.getMetadataForColumn(td_target->tableId, "id");

    ASSERT_TRUE(cd_source->columnType.get_comp_param() !=
                cd_target->columnType.get_comp_param());
  }

  ddl = "DROP TABLE CTAS_TARGET;";
  sql(ddl);
  ddl = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE_TEXT;";
  sql(ddl);

  {
    auto& cat = getCatalog();
    auto td_source = cat.getMetadataForTable("CTAS_SOURCE_TEXT");
    auto cd_source = cat.getMetadataForColumn(td_source->tableId, "id");

    auto td_target = cat.getMetadataForTable("CTAS_TARGET");
    auto cd_target = cat.getMetadataForColumn(td_target->tableId, "id");

    ASSERT_EQ(cd_source->columnType.get_comp_param(),
              cd_target->columnType.get_comp_param());
  }

  ddl = "DROP TABLE CTAS_TARGET;";
  sql(ddl);
  ddl =
      "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE_TEXT WITH( "
      "USE_SHARED_DICTIONARIES='TRUE' );";
  sql(ddl);

  {
    auto& cat = getCatalog();
    auto td_source = cat.getMetadataForTable("CTAS_SOURCE_TEXT");
    auto cd_source = cat.getMetadataForColumn(td_source->tableId, "id");

    auto td_target = cat.getMetadataForTable("CTAS_TARGET");
    auto cd_target = cat.getMetadataForColumn(td_target->tableId, "id");

    ASSERT_EQ(cd_source->columnType.get_comp_param(),
              cd_target->columnType.get_comp_param());
  }
}

TEST_P(Ctas, LiteralStringTest) {
  std::string ddl = "DROP TABLE IF EXISTS CTAS_SOURCE;";
  sql(ddl);
  ddl = "DROP TABLE IF EXISTS CTAS_TARGET;";
  sql(ddl);

  sql("CREATE TABLE CTAS_SOURCE (id int, val int);");

  sql("INSERT INTO CTAS_SOURCE VALUES(1,1); ");
  sql("INSERT INTO CTAS_SOURCE VALUES(2,2); ");
  sql("INSERT INTO CTAS_SOURCE VALUES(3,3); ");

  ddl =
      "CREATE TABLE CTAS_TARGET AS select id, val, (case when val=1 then 'aa' else 'bb' "
      "end) as txt FROM CTAS_SOURCE;";
  sql(ddl);

  auto check = [this](int id, std::string txt) {
    TQueryResult result;
    std::string query =
        "SELECT txt FROM CTAS_TARGET WHERE id=" + std::to_string(id) + ";";
    sql(result, query);
    assertResultSetEqual({{txt}}, result);
  };

  check(1, "aa");
  check(2, "bb");
  check(3, "bb");
}

TEST_P(Ctas, ValidationCheck) {
  sql("DROP TABLE IF EXISTS ctas_source;");
  sql("DROP TABLE IF EXISTS ctas_target;");
  sql("CREATE TABLE ctas_source (id int, dd DECIMAL(17,2));");
  sql("INSERT INTO ctas_source VALUES(1, 10000);");
  ASSERT_ANY_THROW(
      sql("CREATE TABLE ctas_target AS SELECT id, CEIL(dd*10000) FROM ctas_source;"));
}

TEST_P(Ctas, GeoTest) {
  std::string ddl = "DROP TABLE IF EXISTS CTAS_SOURCE;";
  sql(ddl);
  ddl = "DROP TABLE IF EXISTS CTAS_TARGET;";
  sql(ddl);

  sql("CREATE TABLE CTAS_SOURCE ("
      "pu GEOMETRY(POINT, 4326) ENCODING NONE, "
      "pc GEOMETRY(POINT, 4326) ENCODING COMPRESSED(32), "
      "lc GEOMETRY(LINESTRING, 4326), "
      "poly GEOMETRY(POLYGON), "
      "mpoly GEOMETRY(MULTIPOLYGON, 4326)"
      ");");

  sql("INSERT INTO CTAS_SOURCE VALUES("
      "'POINT (-118.480499954187 34.2662998541567)', "
      "'POINT (-118.480499954187 34.2662998541567)', "
      "'LINESTRING (-118.480499954187 34.2662998541567, "
      "             -117.480499954187 35.2662998541567)', "
      "'POLYGON ((-118.480499954187 34.2662998541567, "
      "           -117.480499954187 35.2662998541567, "
      "           -110.480499954187 45.2662998541567))', "
      "'MULTIPOLYGON (((-118.480499954187 34.2662998541567, "
      "                 -117.480499954187 35.2662998541567, "
      "                 -110.480499954187 45.2662998541567)))' "
      "); ");

  ddl = "CREATE TABLE CTAS_TARGET AS select * FROM CTAS_SOURCE;";
  sql(ddl);

  TQueryResult rows;
  sql(rows, "SELECT * FROM CTAS_TARGET;");
  assertResultSetEqual(
      {{"POINT (-118.480499954187 34.2662998541567)",
        "POINT (-118.480499954187 34.2662998541567)",
        "LINESTRING (-118.480499954187 34.2662998541567,-117.480499929507 "
        "35.2662998369272)",
        "POLYGON ((-118.480499954187 34.2662998541567,-117.480499954187 "
        "35.2662998541567,-110.480499954187 45.2662998541567,-118.480499954187 "
        "34.2662998541567))",
        "MULTIPOLYGON (((-118.480499954187 34.2662998541567,-117.480499929507 "
        "35.2662998369272,-110.480499924384 45.2662998322706,-118.480499954187 "
        "34.2662998541567)))"}},
      rows);
}

TEST_P(Ctas, CreateTableAsSelect_IfNotExists) {
  sql("DROP TABLE IF EXISTS CTAS_SOURCE;");
  sql("DROP TABLE IF EXISTS CTAS_TARGET;");
  sql("CREATE TABLE CTAS_SOURCE(a INT);");
  sql("CREATE TABLE CTAS_TARGET(a INT);");
  ASSERT_ANY_THROW(sql("CREATE TABLE CTAS_TARGET AS (SELECT * FROM CTAS_SOURCE);"));
  ASSERT_NO_THROW(
      sql("CREATE TABLE IF NOT EXISTS CTAS_TARGET AS (SELECT * FROM CTAS_SOURCE);"));
}

TEST_P(Ctas, CreateTableAsSelect) {
  // execute CTAS
  std::string create_ctas_sql = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  int num_rows = 25;
  int num_rows_to_check = num_rows;
  runCtasTest(columnDescriptors, create_ctas_sql, num_rows, num_rows_to_check, ")");
}

TEST_P(Ctas, CreateTableFromSelectFragments) {
  // execute CTAS
  std::string create_ctas_sql = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  int num_rows = 25;
  int num_rows_to_check = num_rows;
  runCtasTest(columnDescriptors,
              create_ctas_sql,
              num_rows,
              num_rows_to_check,
              ") WITH (FRAGMENT_SIZE=3)");
}

TEST_P(Ctas, CreateTableFromSelectReplicated) {
  // execute CTAS
  std::string create_ctas_sql = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  int num_rows = 25;
  int num_rows_to_check = num_rows;
  runCtasTest(columnDescriptors,
              create_ctas_sql,
              num_rows,
              num_rows_to_check,
              ") WITH (FRAGMENT_SIZE=3, partitions='REPLICATED')");
}

TEST_P(Ctas, CreateTableFromSelectSharded) {
  // execute CTAS
  std::string create_ctas_sql = "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE;";
  int num_rows = 25;
  int num_rows_to_check = num_rows;
  runCtasTest(
      columnDescriptors,
      create_ctas_sql,
      num_rows,
      num_rows_to_check,
      ", SHARD KEY (id)) WITH (FRAGMENT_SIZE=3, shard_count = 4, partitions='SHARDED')");
}

TEST_P(Ctas, CreateTableAsSelectWithLimit) {
  // execute CTAS
  std::string create_ctas_sql =
      "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE ORDER BY id LIMIT 20;";
  int num_rows = 25;
  int num_rows_to_check = 20;
  runCtasTest(columnDescriptors, create_ctas_sql, num_rows, num_rows_to_check);
}

TEST_P(Ctas, CreateTableAsSelectWithZeroLimit) {
  // execute CTAS
  std::string create_ctas_sql =
      "CREATE TABLE CTAS_TARGET AS SELECT * FROM CTAS_SOURCE ORDER BY id LIMIT 0;";
  int num_rows = 5;
  int num_rows_to_check = 0;
  runCtasTest(columnDescriptors, create_ctas_sql, num_rows, num_rows_to_check);
}

TEST_P(Ctas, Parmtest) {
  std::string ddl = "DROP TABLE IF EXISTS CTAS_SOURCE;";
  sql(ddl);
  if (columnDescriptors.size() > 1) {
    // do nothing
  }
}

TEST_F(Itas, SyntaxCheck) {
  std::string select_star_sql = "INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;";
  itas_syntax_check(select_star_sql,
                    "CREATE TABLE ITAS_SOURCE (id int, val int);",
                    "CREATE TABLE ITAS_TARGET (id int);",
                    true);
  itas_syntax_check(select_star_sql,
                    "CREATE TABLE ITAS_SOURCE (id int);",
                    "CREATE TABLE ITAS_TARGET (id int, val int);",
                    true);
  itas_syntax_check(select_star_sql,
                    "CREATE TABLE ITAS_SOURCE (id int);",
                    "CREATE TABLE ITAS_TARGET (id int encoding FIXED(8));",
                    false);
  itas_syntax_check(select_star_sql,
                    "CREATE TABLE ITAS_SOURCE (id int encoding FIXED(8));",
                    "CREATE TABLE ITAS_TARGET (id int);",
                    false);
  itas_syntax_check(select_star_sql,
                    "CREATE TABLE ITAS_SOURCE (id int, val timestamp(0));",
                    "CREATE TABLE ITAS_TARGET (id int, val timestamp(3));",
                    true);
  itas_syntax_check(select_star_sql,
                    "CREATE TABLE ITAS_SOURCE (id int, val text encoding none);",
                    "CREATE TABLE ITAS_TARGET (id int, val text);",
                    true);
  itas_syntax_check(select_star_sql,
                    "CREATE TABLE ITAS_SOURCE (id int, val decimal(10,2));",
                    "CREATE TABLE ITAS_TARGET (id int, val decimal(10,3));",
                    true);
  itas_syntax_check("INSERT INTO ITAS_TARGET SELECT id FROM ITAS_SOURCE;",
                    "CREATE TABLE ITAS_SOURCE (id int, val int);",
                    "CREATE TABLE ITAS_TARGET (id int);",
                    false);
  itas_syntax_check("INSERT INTO ITAS_TARGET(id) SELECT id FROM ITAS_SOURCE;",
                    "CREATE TABLE ITAS_SOURCE (id int);",
                    "CREATE TABLE ITAS_TARGET (id int, val int);",
                    false);
  itas_syntax_check("INSERT INTO ITAS_TARGET(id2) SELECT id FROM ITAS_SOURCE;",
                    "CREATE TABLE ITAS_SOURCE (id int);",
                    "CREATE TABLE ITAS_TARGET (id2 int, val int);",
                    false);
  itas_syntax_check(select_star_sql,
                    "CREATE TABLE ITAS_SOURCE (id int);",
                    "CREATE TABLE ITAS_TARGET (id2 int);",
                    false);
}

TEST_F(Itas, DifferentColumnNames) {
  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");

  sql("CREATE TABLE ITAS_SOURCE (id int, val int);");

  sql("INSERT INTO ITAS_SOURCE VALUES(1,10); ");
  sql("INSERT INTO ITAS_SOURCE VALUES(2,20); ");
  sql("INSERT INTO ITAS_SOURCE VALUES(3,30); ");

  auto check = [this](int id, int64_t val) {
    TQueryResult result;
    std::string query =
        "SELECT target_val FROM ITAS_TARGET WHERE target_id=" + std::to_string(id) + ";";
    sql(result, query);
    assertResultSetEqual({{val}}, result);
  };

  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("CREATE TABLE ITAS_TARGET (target_id int, target_val int);");
  sql("INSERT INTO ITAS_TARGET SELECT id, val FROM ITAS_SOURCE;");

  check(1, 10);
  check(2, 20);
  check(3, 30);

  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("CREATE TABLE ITAS_TARGET (target_id int, target_val int);");
  sql("INSERT INTO ITAS_TARGET (target_id, target_val) SELECT id, val FROM ITAS_SOURCE;");

  check(1, 10);
  check(2, 20);
  check(3, 30);

  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("CREATE TABLE ITAS_TARGET (target_id int, target_val int);");
  sql("INSERT INTO ITAS_TARGET (target_val, target_id) SELECT val, id FROM ITAS_SOURCE;");

  check(1, 10);
  check(2, 20);
  check(3, 30);

  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("CREATE TABLE ITAS_TARGET (target_id int, target_val int);");
  sql("INSERT INTO ITAS_TARGET (target_id, target_val) SELECT val, id FROM ITAS_SOURCE;");

  check(10, 1);
  check(20, 2);
  check(30, 3);

  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("CREATE TABLE ITAS_TARGET (target_id int, target_val int);");
  sql("INSERT INTO ITAS_TARGET (target_val, target_id) SELECT id, val FROM ITAS_SOURCE;");

  check(10, 1);
  check(20, 2);
  check(30, 3);
}

TEST_F(Itas, AllowDifferentFixedEncodings) {
  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");

  sql("CREATE TABLE ITAS_SOURCE (id int, val int);");
  sql("CREATE TABLE ITAS_TARGET (id int, val bigint);");

  EXPECT_NO_THROW(sql("INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;"));

  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");

  sql("CREATE TABLE ITAS_SOURCE (id int, val bigint);");
  sql("CREATE TABLE ITAS_TARGET (id int, val bigint encoding fixed(8));");

  EXPECT_NO_THROW(sql("INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;"));

  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");

  sql("CREATE TABLE ITAS_SOURCE (id int, val timestamp);");
  sql("CREATE TABLE ITAS_TARGET (id int, val timestamp encoding fixed(32));");

  EXPECT_NO_THROW(sql("INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;"));

  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");

  sql("CREATE TABLE ITAS_SOURCE (id int, val time);");
  sql("CREATE TABLE ITAS_TARGET (id int, val time encoding fixed(32));");

  EXPECT_NO_THROW(sql("INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;"));

  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");

  sql("CREATE TABLE ITAS_SOURCE (id int, val date);");
  sql("CREATE TABLE ITAS_TARGET (id int, val date encoding fixed(16));");

  EXPECT_NO_THROW(sql("INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;"));

  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");

  sql("CREATE TABLE ITAS_SOURCE (id int, val decimal(17, 2));");
  sql("CREATE TABLE ITAS_TARGET (id int, val decimal( 5, 2));");

  EXPECT_NO_THROW(sql("INSERT INTO ITAS_TARGET SELECT * FROM ITAS_SOURCE;"));
}

TEST_F(Itas, SelectStar) {
  sql("DROP TABLE IF EXISTS ITAS_SOURCE_1;");
  sql("DROP TABLE IF EXISTS ITAS_SOURCE_2;");
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");

  sql("CREATE TABLE ITAS_SOURCE_1 (id int);");

  if (isDistributedMode()) {
    sql("CREATE TABLE ITAS_SOURCE_2 (id int, val int) with (partitions = 'REPLICATED');");
  } else {
    sql("CREATE TABLE ITAS_SOURCE_2 (id int, val int);");
  }

  sql("CREATE TABLE ITAS_TARGET (id int, val int);");

  sql("INSERT INTO ITAS_SOURCE_1 VALUES(1); ");
  sql("INSERT INTO ITAS_SOURCE_2 VALUES(1, 2); ");

  EXPECT_NO_THROW(
      sql("INSERT INTO ITAS_TARGET SELECT ITAS_SOURCE_1.*, ITAS_SOURCE_2.val FROM "
          "ITAS_SOURCE_1 JOIN ITAS_SOURCE_2 on ITAS_SOURCE_1.id = ITAS_SOURCE_2.id;"));

  sql("DROP TABLE ITAS_SOURCE_1;");
  sql("DROP TABLE ITAS_SOURCE_2;");
  sql("DROP TABLE ITAS_TARGET;");
}

TEST_F(Itas, UnsupportedBooleanCast) {
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");

  sql("CREATE TABLE ITAS_TARGET (id boolean);");
  sql("CREATE TABLE ITAS_SOURCE (id int, str text, val timestamp(3), g point);");

  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT id FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT str FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT val FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT g FROM ITAS_SOURCE);"));
  EXPECT_NO_THROW(
      sql("INSERT INTO ITAS_TARGET (SELECT CAST(id AS boolean) FROM ITAS_SOURCE);"));

  sql("DROP TABLE ITAS_TARGET;");
  sql("DROP TABLE ITAS_SOURCE;");
}

TEST_F(Itas, UnsupportedGeo) {
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");

  sql("CREATE TABLE ITAS_TARGET (p point);");
  sql("CREATE TABLE ITAS_SOURCE (id int, str text, val timestamp(3), g linestring);");

  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT id FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT str FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT val FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT g FROM ITAS_SOURCE);"));

  sql("DROP TABLE ITAS_TARGET;");
  sql("DROP TABLE ITAS_SOURCE;");
}

TEST_F(Itas, UnsupportedDateTime) {
  // time
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");

  sql("CREATE TABLE ITAS_TARGET (t time);");
  sql("CREATE TABLE ITAS_SOURCE (id int, str text, val timestamp(3), d date);");

  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT id FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT str FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT val FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT d FROM ITAS_SOURCE);"));

  sql("DROP TABLE ITAS_TARGET;");
  sql("DROP TABLE ITAS_SOURCE;");

  // date
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");

  sql("CREATE TABLE ITAS_TARGET (t date);");
  sql("CREATE TABLE ITAS_SOURCE (id int, str text, val timestamp(3), d time);");

  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT id FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT str FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT val FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT d FROM ITAS_SOURCE);"));

  sql("DROP TABLE ITAS_TARGET;");
  sql("DROP TABLE ITAS_SOURCE;");

  // timestamp
  sql("DROP TABLE IF EXISTS ITAS_TARGET;");
  sql("DROP TABLE IF EXISTS ITAS_SOURCE;");

  sql("CREATE TABLE ITAS_TARGET (t timestamp);");
  sql("CREATE TABLE ITAS_SOURCE (id int, str text, val timestamp(3), d date);");

  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT id FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT str FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT val FROM ITAS_SOURCE);"));
  EXPECT_NO_THROW(
      sql("INSERT INTO ITAS_TARGET (SELECT CAST(val AS TIMESTAMP) FROM ITAS_SOURCE);"));
  EXPECT_ANY_THROW(sql("INSERT INTO ITAS_TARGET (SELECT d FROM ITAS_SOURCE);"));

  sql("DROP TABLE ITAS_TARGET;");
  sql("DROP TABLE ITAS_SOURCE;");
}

TEST_P(Itas_P, InsertIntoTableFromSelect) {
  itasTestBody(columnDescriptors, ")", ")");
}

TEST_P(Itas_P, InsertIntoTableFromSelectFragments) {
  itasTestBody(columnDescriptors, ") WITH (FRAGMENT_SIZE=3)", ")");
}

TEST_P(Itas_P, InsertIntoFragmentsTableFromSelect) {
  itasTestBody(columnDescriptors, ")", ") WITH (FRAGMENT_SIZE=3)");
}

TEST_P(Itas_P, InsertIntoFragmentsTableFromSelectFragments) {
  itasTestBody(columnDescriptors, ") WITH (FRAGMENT_SIZE=3)", ") WITH (FRAGMENT_SIZE=3)");
}

TEST_P(Itas_P, InsertIntoTableFromSelectReplicated) {
  itasTestBody(
      columnDescriptors, ") WITH (FRAGMENT_SIZE=3, partitions='REPLICATED')", ")");
}

TEST_P(Itas_P, InsertIntoTableFromSelectSharded) {
  itasTestBody(
      columnDescriptors,
      ", SHARD KEY (id)) WITH (FRAGMENT_SIZE=3, shard_count = 4, partitions='SHARDED')",
      ")");
}

TEST_P(Itas_P, InsertIntoReplicatedTableFromSelect) {
  itasTestBody(columnDescriptors, ")", ") WITH (partitions='REPLICATED')");
}

TEST_P(Itas_P, InsertIntoShardedTableFromSelect) {
  itasTestBody(columnDescriptors,
               ")",
               ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')");
}

TEST_P(Itas_P, InsertIntoReplicatedTableFromSelectReplicated) {
  itasTestBody(columnDescriptors,
               ") WITH (partitions='REPLICATED')",
               ") WITH (partitions='REPLICATED')");
}

TEST_P(Itas_P, InsertIntoReplicatedTableFromSelectSharded) {
  itasTestBody(columnDescriptors,
               ") WITH (partitions='REPLICATED')",
               ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')");
}

TEST_P(Itas_P, InsertIntoShardedTableFromSelectSharded) {
  itasTestBody(columnDescriptors,
               ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')",
               ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')");
}

TEST_P(Itas_P, InsertIntoShardedTableFromSelectReplicated) {
  itasTestBody(columnDescriptors,
               ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')",
               ") WITH (partitions='REPLICATED')");
}

TEST_P(Itas_P, OmitNotNullableColumn) {
  std::vector<std::string> partitioning_schemes = {
      ")", ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')"};
  for (auto target_partitioning : partitioning_schemes) {
    create_partial_itas_tables(target_partitioning);
    std::string itas_omitting_nullable_column =
        "INSERT INTO ITAS_TARGET(id, not_nullable_col, text_array) "
        "SELECT id, not_nullable_col, text_array FROM ITAS_SOURCE";
    std::string itas_omitting_not_nullable_column =
        "INSERT INTO ITAS_TARGET(id, nullable_col, text_array) "
        "SELECT id, nullable_col, text_array FROM ITAS_SOURCE";
    EXPECT_NO_THROW(sql(itas_omitting_nullable_column));

    queryAndAssertPartialException(itas_omitting_not_nullable_column,
                                   "NULL for column not_nullable_col");
  }
}

TEST_F(Itas, OmitShardingColumn) {
  create_partial_itas_tables(
      ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')");
  std::string itas_omitting_nullable_column =
      "INSERT INTO ITAS_TARGET(id, not_nullable_col, text_array) "
      "SELECT id, not_nullable_col, text_array FROM ITAS_SOURCE";
  std::string itas_omitting_sharding_column =
      "INSERT INTO ITAS_TARGET(not_nullable_col, nullable_col, text_array) "
      "SELECT not_nullable_col, nullable_col, text_array FROM ITAS_SOURCE";
  EXPECT_NO_THROW(sql(itas_omitting_nullable_column));
  EXPECT_NO_THROW(sql(itas_omitting_sharding_column));
}

TEST_F(Itas, OmitDictionaryEncodedArrayColumn) {
  std::vector<std::string> partitioning_schemes = {
      ")", ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')"};
  for (auto target_partitioning : partitioning_schemes) {
    create_partial_itas_tables(target_partitioning);
    std::string itas_omitting_nullable_column =
        "INSERT INTO ITAS_TARGET(id, not_nullable_col, text_array) "
        "SELECT id, not_nullable_col, text_array FROM ITAS_SOURCE";
    std::string itas_omitting_array_column =
        "INSERT INTO ITAS_TARGET(id, not_nullable_col, nullable_col) "
        "SELECT id, not_nullable_col, nullable_col FROM ITAS_SOURCE";
    EXPECT_NO_THROW(sql(itas_omitting_nullable_column));

    queryAndAssertPartialException(itas_omitting_array_column,
                                   "omitting TEXT arrays is not supported yet");
  }
}

class Export : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    // Default connection string outside of thrift
  }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  void exportTestBody(std::string sourcePartitionScheme = ")") {
    sql("DROP TABLE IF EXISTS EXPORT_SOURCE;");

    std::string create_sql =
        "CREATE TABLE EXPORT_SOURCE ( id int, val int " + sourcePartitionScheme + ";";
    LOG(INFO) << create_sql;

    sql(create_sql);

    size_t num_rows = 25;
    std::vector<std::string> expected_rows;

    // fill source table
    for (unsigned int row = 0; row < num_rows; row++) {
      std::string insert_sql = "INSERT INTO EXPORT_SOURCE VALUES (" +
                               std::to_string(row) + "," + std::to_string(row) + ");";
      expected_rows.push_back(std::to_string(row) + "," + std::to_string(row));

      sql(insert_sql);
    }

    boost::filesystem::path temp =
        boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();

    std::string export_file_name = temp.make_preferred().string() + std::string(".csv");

    // execute CTAS
    std::string export_sql = "COPY (SELECT * FROM EXPORT_SOURCE) TO '" +
                             export_file_name +
                             "' with (header='false', quoted='false');";
    LOG(INFO) << export_sql;

    sql(export_sql);

    std::ifstream export_file(export_file_name);

    std::vector<std::string> exported_rows;

    std::copy(std::istream_iterator<std::string>(export_file),
              std::istream_iterator<std::string>(),
              std::back_inserter(exported_rows));

    export_file.close();
    remove(export_file_name.c_str());

    std::sort(exported_rows.begin(), exported_rows.end());
    std::sort(expected_rows.begin(), expected_rows.end());

    ASSERT_EQ(expected_rows.size(), num_rows);
    ASSERT_EQ(exported_rows.size(), num_rows);

    for (unsigned int row = 0; row < num_rows; row++) {
      ASSERT_EQ(exported_rows[row], expected_rows[row]);
    }
  }
};

TEST_F(Export, ExportFromSelect) {
  exportTestBody(")");
}

TEST_F(Export, ExportFromSelectFragments) {
  exportTestBody(") WITH (FRAGMENT_SIZE=3)");
}

TEST_F(Export, ExportFromSelectReplicated) {
  exportTestBody(") WITH (FRAGMENT_SIZE=3, partitions='REPLICATED')");
}

TEST_F(Export, ExportFromSelectSharded) {
  exportTestBody(
      ", SHARD KEY (id)) WITH (FRAGMENT_SIZE=3, shard_count = 4, "
      "partitions='SHARDED')");
}

TEST_P(Update, InvalidTextArrayAssignment) {
  sql("DROP TABLE IF EXISTS arr;");
  sql("CREATE TABLE arr (id int, ia text[3]);");
  sql("INSERT INTO arr VALUES(1 , ARRAY[null,null,null]); ");
  ASSERT_ANY_THROW(sql("INSERT INTO arr VALUES(0 , null); "));
  ASSERT_ANY_THROW(sql("UPDATE arr set ia = NULL;"));
  ASSERT_ANY_THROW(sql("UPDATE arr set ia = ARRAY[];"));
  ASSERT_ANY_THROW(sql("UPDATE arr set ia = ARRAY[null];"));
  ASSERT_ANY_THROW(sql("UPDATE arr set ia = ARRAY['one'];"));
  ASSERT_ANY_THROW(sql("UPDATE arr set ia = ARRAY['one', 'two', 'three', 'four'];"));
}

TEST_P(Update, UpdateColumnByColumn) {
  if (isDistributedMode()) {
    GTEST_SKIP();
  }
  sql("DROP TABLE IF EXISTS update_test;");

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

  sql(create_sql);

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

    sql(insert_sql);
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

  sql(update_sql);

  // compare source against CTAS
  std::string select_sql = "SELECT id";
  for (unsigned int col = 0; col < columnDescriptors.size(); col++) {
    select_sql += ", col_dst_" + std::to_string(col);
    select_sql += ", col_src_" + std::to_string(col);
  }
  select_sql += " FROM update_test ORDER BY id;";

  LOG(INFO) << select_sql;
  TQueryResult select_result;
  sql(select_result, select_sql);

  auto columns = select_result.row_set.columns;

  // check expected result count
  ASSERT_EQ(num_rows, columns[0].nulls.size());

  // we compare each column with the column 'next' to it to check the update replace the
  // value
  for (unsigned int c = 0; c < columnDescriptors.size(); c++) {
    std::string col_details =
        "col_" + std::to_string(c) + " " + columnDescriptors[c]->get_column_definition();
    for (size_t r = 0; r < num_rows; r++) {
      ASSERT_EQ(columns[2 * c + 1].nulls[r], columns[2 * c + 2].nulls[r])
          << col_details << " Column " << std::to_string(c) << " row "
          << std::to_string(r);
      if (columns[2 * c + 1].data.int_col.size() > 0) {
        ASSERT_EQ(columns[2 * c + 1].data.int_col[r], columns[2 * c + 2].data.int_col[r])
            << col_details << " Column " << std::to_string(c) << " row "
            << std::to_string(r);
      }
      if (columns[2 * c + 1].data.real_col.size() > 0) {
        ASSERT_EQ(columns[2 * c + 1].data.real_col[r],
                  columns[2 * c + 2].data.real_col[r])
            << col_details << " Column " << std::to_string(c) << " row "
            << std::to_string(r);
      }
      if (columns[2 * c + 1].data.str_col.size() > 0) {
        ASSERT_EQ(columns[2 * c + 1].data.str_col[r], columns[2 * c + 2].data.str_col[r])
            << col_details << " Column " << std::to_string(c) << " row "
            << std::to_string(r);
      }
      if (columns[2 * c + 1].data.arr_col.size() > 0) {
        ASSERT_EQ(columns[2 * c + 1].data.arr_col[r], columns[2 * c + 2].data.arr_col[r])
            << col_details << " Column " << std::to_string(c) << " row "
            << std::to_string(r);
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
  INSTANTIATE_TEST_SUITE_P(                                                            \
      CDT,                                                                             \
      Ctas,                                                                            \
      testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{CDT}));       \
  INSTANTIATE_TEST_SUITE_P(                                                            \
      CDT,                                                                             \
      Itas,                                                                            \
      testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{CDT}));       \
  INSTANTIATE_TEST_SUITE_P(                                                            \
      CDT,                                                                             \
      Update,                                                                          \
      testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{CDT}));       \
  INSTANTIATE_TEST_SUITE_P(                                                            \
      VARLEN_TEXT_AND_##CDT,                                                           \
      Update,                                                                          \
      testing::Values(                                                                 \
          std::vector<std::shared_ptr<TestColumnDescriptor>>{STRING_NONE_BASE, CDT})); \
  INSTANTIATE_TEST_SUITE_P(                                                            \
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
NUMBER_COLUMN_TEST(NOT_NULL_INTEGER, int64_t, "INTEGER NOT NULL", kINT, NULL_INT);
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

const std::vector<std::shared_ptr<TestColumnDescriptor>> ALL = {
    STRING_NONE_BASE,
    BOOLEAN,
    BOOLEAN_ARRAY,
    // removed until fixed in distributed
    //        BOOLEAN_FIXED_LEN_ARRAY,
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

INSTANTIATE_TEST_SUITE_P(MIXED_ALL, Ctas, testing::Values(ALL));
INSTANTIATE_TEST_SUITE_P(MIXED_ALL, Itas_P, testing::Values(ALL));
INSTANTIATE_TEST_SUITE_P(MIXED_ALL, Update, testing::Values(ALL));

INSTANTIATE_TEST_SUITE_P(
    MIXED_VARLEN_WITHOUT_GEO,
    Update,
    testing::Values(std::vector<std::shared_ptr<TestColumnDescriptor>>{
        STRING_NONE_BASE,
        BOOLEAN_ARRAY,
        // removed until fixed in distributed
        //        BOOLEAN_FIXED_LEN_ARRAY,
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

// TODO 4 May TEMP tables not being found
// calcite not being updated as it should
// probably an artifact of how dbhandler is being run
// I suspect it is reading from sqlite directly and
// temp tables details are not in the sqllite tables
// needs further investigation
TEST_F(Itas, DISABLED_InsertIntoTempTableFromSelect) {
  std::vector<std::shared_ptr<TestColumnDescriptor>> columnDescriptors = {
      BOOLEAN,
      TINYINT,
      SMALLINT,
      INTEGER,
      BIGINT,
      NUMERIC,
      TEXT,
      TIME,
      DATE,
      TIMESTAMP,
  };
  itasTestBody(columnDescriptors, ")", ")", "TEMPORARY");
}

namespace {
std::vector<std::shared_ptr<TestColumnDescriptor>> partialDescriptors = {
    STRING_NONE_BASE,
    BOOLEAN,
    TINYINT,
    BIGINT,
    BIGINT_ARRAY,
    FLOAT,
    DECIMAL,
    TEXT_NONE,
    TEXT_DICT,
    // TODO(max): TypedImportBuffers don't work with NULL text arrays yet
    // TEXT_ARRAY,
    TIME,
    DATE,
    TIMESTAMP,
    GEO_POINT,
    GEO_LINESTRING,
    GEO_POLYGON,
    GEO_MULTI_POLYGON};
}  // namespace

TEST_P(Itas_P, PartialInsertIntoTableFromSelect) {
  partial_itas_test_body(partialDescriptors, ")", ")");
}

TEST_P(Itas_P, PartialInsertIntoTableFromSelectFragments) {
  partial_itas_test_body(partialDescriptors, ") WITH (FRAGMENT_SIZE=3)", ")");
}

TEST_P(Itas_P, PartialInsertIntoFragmentsTableFromSelect) {
  partial_itas_test_body(partialDescriptors, ")", ") WITH (FRAGMENT_SIZE=3)");
}

TEST_P(Itas_P, PartialInsertIntoFragmentsTableFromSelectFragments) {
  partial_itas_test_body(
      partialDescriptors, ") WITH (FRAGMENT_SIZE=3)", ") WITH (FRAGMENT_SIZE=3)");
}

TEST_P(Itas_P, PartialInsertIntoTableFromSelectReplicated) {
  partial_itas_test_body(
      partialDescriptors, ") WITH (FRAGMENT_SIZE=3, partitions='REPLICATED')", ")");
}

TEST_P(Itas_P, PartialInsertIntoTableFromSelectSharded) {
  partial_itas_test_body(
      partialDescriptors,
      ", SHARD KEY (id)) WITH (FRAGMENT_SIZE=3, shard_count = 4, partitions='SHARDED')",
      ")");
}

TEST_P(Itas_P, PartialInsertIntoReplicatedTableFromSelect) {
  partial_itas_test_body(partialDescriptors, ")", ") WITH (partitions='REPLICATED')");
}

TEST_P(Itas_P, PartialInsertIntoShardedTableFromSelect) {
  partial_itas_test_body(
      partialDescriptors,
      ")",
      ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')");
}

TEST_P(Itas_P, PartialInsertIntoReplicatedTableFromSelectReplicated) {
  partial_itas_test_body(partialDescriptors,
                         ") WITH (partitions='REPLICATED')",
                         ") WITH (partitions='REPLICATED')");
}

TEST_P(Itas_P, PartialInsertIntoReplicatedTableFromSelectSharded) {
  itasTestBody(partialDescriptors,
               ") WITH (partitions='REPLICATED')",
               ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')");
}

TEST_P(Itas_P, PartialInsertIntoShardedTableFromSelectSharded) {
  partial_itas_test_body(
      partialDescriptors,
      ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')",
      ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')");
}

TEST_P(Itas_P, PartialInsertIntoShardedTableFromSelectReplicated) {
  partial_itas_test_body(partialDescriptors,
                         ", SHARD KEY (id)) WITH (shard_count = 4, partitions='SHARDED')",
                         ") WITH (partitions='REPLICATED')");
}

class Select : public DBHandlerTestFixture {
 public:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    // Default connection string outside of thrift
  }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }
};

TEST_F(Select, CtasItasValidation) {
  auto drop_table = []() {
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_1;");
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_2;");
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_3;");
  };

  auto drop_ctas_itas_table = []() {
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_RES_1;");
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_RES_2;");
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_RES_3;");
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_RES_4;");
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_RES_5;");
    sql("DROP TABLE IF EXISTS CTAS_ITAS_VALIDATION_RES_6;");
  };

  auto create_itas_table = []() {
    sql("CREATE TABLE CTAS_ITAS_VALIDATION_RES_4 (i1 INT);");
    sql("CREATE TABLE CTAS_ITAS_VALIDATION_RES_5 (i1 INT);");
    sql("CREATE TABLE CTAS_ITAS_VALIDATION_RES_6 (i1 INT) WITH (FRAGMENT_SIZE = "
        "100000);");
  };

  auto create_table = []() {
    sql("CREATE TABLE CTAS_ITAS_VALIDATION_1 (i1 INT);");
    sql("CREATE TABLE CTAS_ITAS_VALIDATION_2 (i1 INT);");
    sql("CREATE TABLE CTAS_ITAS_VALIDATION_3 (i1 INT) WITH (FRAGMENT_SIZE = 100000);");
  };

  // write a temporary datafile used in the test
  // because "INSERT INTO ..." stmt for this takes too much time
  // and add pre-generated dataset increases meaningless LOC of this test code
  const auto data1_path =
      boost::filesystem::path("../../Tests/Import/datafiles/ctas_itas_validation_1.csv");
  if (boost::filesystem::exists(data1_path)) {
    boost::filesystem::remove(data1_path);
  }
  std::ofstream out1(data1_path.string());
  for (int i = 0; i < 75000; i++) {
    if (out1.is_open()) {
      out1 << i << "\n";
    }
  }
  out1.close();

  const auto data2_path =
      boost::filesystem::path("../../Tests/Import/datafiles/ctas_itas_validation_2.csv");
  if (boost::filesystem::exists(data2_path)) {
    boost::filesystem::remove(data2_path);
  }
  std::ofstream out2(data2_path.string());
  for (int i = 0; i < 750000; i++) {
    if (out2.is_open()) {
      out2 << i << "\n";
    }
  }
  out2.close();

  drop_table();

  create_table();

  auto copy_data1_str = "COPY CTAS_ITAS_VALIDATION_1 FROM \'" + data1_path.string() +
                        "\' WITH (HEADER=\'f\');";
  auto copy_data2_str = "COPY CTAS_ITAS_VALIDATION_2 FROM \'" + data2_path.string() +
                        "\' WITH (HEADER=\'f\');";
  auto copy_data3_str = "COPY CTAS_ITAS_VALIDATION_3 FROM \'" + data2_path.string() +
                        "\' WITH (HEADER=\'f\');";

  sql(copy_data1_str);
  sql(copy_data2_str);
  sql(copy_data3_str);

  boost::filesystem::remove(data1_path);
  boost::filesystem::remove(data2_path);

  drop_ctas_itas_table();
  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_1;",
                      {{i(75000)}});

  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_2",
                      {{i(750000)}});

  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_3",
                      {{i(750000)}});
  ASSERT_NO_THROW(
      sql("CREATE TABLE CTAS_ITAS_VALIDATION_RES_1 AS SELECT * FROM "
          "CTAS_ITAS_VALIDATION_1;"));
  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_RES_1",
                      {{i(75000)}});

  ASSERT_NO_THROW(
      sql("CREATE TABLE CTAS_ITAS_VALIDATION_RES_2 AS SELECT * FROM "
          "CTAS_ITAS_VALIDATION_2;"));
  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_RES_2",
                      {{i(750000)}});
  ASSERT_NO_THROW(
      sql("CREATE TABLE CTAS_ITAS_VALIDATION_RES_3 AS SELECT * FROM "
          "CTAS_ITAS_VALIDATION_3;"));
  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_RES_3",
                      {{i(750000)}});
  create_itas_table();
  ASSERT_NO_THROW(
      sql("INSERT INTO CTAS_ITAS_VALIDATION_RES_4 SELECT * FROM "
          "CTAS_ITAS_VALIDATION_1;"));
  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_RES_4",
                      {{i(75000)}});
  ASSERT_NO_THROW(
      sql("INSERT INTO CTAS_ITAS_VALIDATION_RES_5 SELECT * FROM "
          "CTAS_ITAS_VALIDATION_2;"));
  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_RES_5",
                      {{i(750000)}});
  ASSERT_NO_THROW(
      sql("INSERT INTO CTAS_ITAS_VALIDATION_RES_6 SELECT * FROM "
          "CTAS_ITAS_VALIDATION_2;"));
  sqlAndCompareResult("SELECT COUNT(DISTINCT i1) FROM CTAS_ITAS_VALIDATION_RES_6",
                      {{i(750000)}});
  drop_table();
  drop_ctas_itas_table();
}

TEST_F(Select, CtasItasNullGeoPoint) {
  // TODO 3 May CTAS and ITAS with null points
  // doesnt seem to work in distributed at all
  if (isDistributedMode()) {
    GTEST_SKIP();
  }

  auto run_test = [this](const std::string col_type) {
    auto drop_table = []() {
      sql("DROP TABLE IF EXISTS T_With_Null_GeoPoint;");
      sql("DROP TABLE IF EXISTS CTAS_GeoNull;");
      sql("DROP TABLE IF EXISTS ITAS_GeoNull;");
    };

    auto create_table = [&col_type]() {
      sql("CREATE TABLE T_With_Null_GeoPoint (pt " + col_type + ");");
      sql("CREATE TABLE ITAS_GeoNull (pt " + col_type + ");");
    };

    drop_table();
    create_table();

    sql("INSERT INTO T_With_Null_GeoPoint VALUES (\'POINT(1 1)\');");
    sql("INSERT INTO T_With_Null_GeoPoint VALUES (NULL);");
    sqlAndCompareResult(
        "SELECT COUNT(1) FROM T_With_Null_GeoPoint WHERE ST_X(pt) is not null;",
        {{i(1)}});
    sqlAndCompareResult(
        "SELECT COUNT(1) FROM T_With_Null_GeoPoint WHERE ST_X(pt) is null;", {{i(1)}});
    sql("INSERT INTO ITAS_GeoNull SELECT * FROM T_With_Null_GeoPoint;");
    sqlAndCompareResult("SELECT COUNT(1) FROM ITAS_GeoNull WHERE ST_X(pt) is not null;",
                        {{i(1)}});
    sqlAndCompareResult("SELECT COUNT(1) FROM ITAS_GeoNull WHERE ST_X(pt) is null;",
                        {{i(1)}});
    sql("CREATE TABLE CTAS_GeoNull AS SELECT * FROM T_With_Null_GeoPoint;");
    sqlAndCompareResult("SELECT COUNT(1) FROM CTAS_GeoNull WHERE ST_X(pt) is not null;",
                        {{i(1)}});
    sqlAndCompareResult("SELECT COUNT(1) FROM CTAS_GeoNull WHERE ST_X(pt) is null;",
                        {{i(1)}});
    sql("INSERT INTO T_With_Null_GeoPoint SELECT * FROM T_With_Null_GeoPoint;");
    sqlAndCompareResult("SELECT COUNT(1) FROM T_With_Null_GeoPoint", {{i(4)}});
    sqlAndCompareResult(
        "SELECT COUNT(1) FROM T_With_Null_GeoPoint WHERE ST_X(pt) is not null;",
        {{i(2)}});
    sqlAndCompareResult(
        "SELECT COUNT(1) FROM T_With_Null_GeoPoint WHERE ST_X(pt) is null;", {{i(2)}});

    drop_table();
  };

  std::vector<std::string> geo_point_types{
      "POINT",
      "GEOMETRY(POINT)",
      "GEOMETRY(POINT, 4326)",
      "GEOMETRY(POINT, 4326)",
      "GEOMETRY(POINT, 4326) ENCODING COMPRESSED(32)",
      "GEOMETRY(POINT, 4326) ENCODING NONE",
      "GEOMETRY(POINT, 900913)",
      "GEOMETRY(POINT, 900913) ENCODING NONE"};
  for (auto& col_type : geo_point_types) {
    run_test(col_type);
  }
}

TEST_F(Select, CtasFixedLenBooleanArrayCrash) {
  auto drop_table = [this]() { sql("DROP TABLE IF EXISTS CtasFBoolArrayCrash;"); };
  auto prepare_table = []() {
    sql("CREATE TABLE CtasFBoolArrayCrash (src boolean[3], dst boolean[3]);");
    sql("INSERT INTO CtasFBoolArrayCrash VALUES (null, {\'true\', \'false\', "
        "\'true\'});");
    sql("INSERT INTO CtasFBoolArrayCrash VALUES ({\'true\', \'false\', \'true\'}, "
        "{\'false\', \'true\', \'false\'});");
    sql("UPDATE CtasFBoolArrayCrash set dst = src;");
  };
  drop_table();

  prepare_table();
  sql("SELECT src FROM CtasFBoolArrayCrash;");
  drop_table();

  prepare_table();
  sql("SELECT dst FROM CtasFBoolArrayCrash;");
  drop_table();

  prepare_table();
  sql("UPDATE CtasFBoolArrayCrash set dst = src;");
  drop_table();
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  DBHandlerTestFixture::initTestArgs(argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
