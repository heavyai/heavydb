/*
 * Copyright 2017 MapD Technologies, Inc.
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

/**
 * @file SqliteConnector.cpp
 * @author Todd Mostak <todd@mapd.com>
 *
 */

#include "SqliteConnector.h"

#include <iostream>

#include "Logger/Logger.h"

using std::cout;
using std::endl;
using std::runtime_error;
using std::string;

SqliteConnector::SqliteConnector(const string& dbName, const string& dir)
    : dbName_(dbName) {
  string connectString(dir);
  if (connectString.size() > 0 && connectString[connectString.size() - 1] != '/') {
    connectString.push_back('/');
  }
  connectString += dbName;
  int returnCode = sqlite3_open(connectString.c_str(), &db_);
  if (returnCode != SQLITE_OK) {
    throwError();
  }
}

SqliteConnector::SqliteConnector(sqlite3* db) : db_(db) {}

SqliteConnector::~SqliteConnector() {
  if (!dbName_.empty()) {
    sqlite3_close(db_);
  }
}

void SqliteConnector::throwError() {
  string errorMsg(sqlite3_errmsg(db_));
  throw runtime_error("Sqlite3 Error: " + errorMsg);
}

std::string get_column_datum(int column_type, sqlite3_stmt* stmt, size_t column_index) {
  const char* datum_ptr;
  if (column_type == SQLITE_BLOB) {
    datum_ptr = static_cast<const char*>(sqlite3_column_blob(stmt, column_index));
  } else {
    datum_ptr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, column_index));
  }
  size_t datum_size = sqlite3_column_bytes(stmt, column_index);
  return {datum_ptr, datum_size};
}

void SqliteConnector::query_with_text_params(const std::string& queryString,
                                             const std::vector<std::string>& text_params,
                                             const std::vector<BindType>& bind_types) {
  if (!bind_types.empty()) {
    CHECK_EQ(text_params.size(), bind_types.size());
  }

  atFirstResult_ = true;
  numRows_ = 0;
  numCols_ = 0;
  columnNames.clear();
  columnTypes.clear();
  results_.clear();
  sqlite3_stmt* stmt;
  int returnCode = sqlite3_prepare_v2(db_, queryString.c_str(), -1, &stmt, nullptr);
  if (returnCode != SQLITE_OK) {
    throwError();
  }

  int num_params = 1;
  for (auto text_param : text_params) {
    if (!bind_types.empty() && bind_types[num_params - 1] == BindType::BLOB) {
      returnCode = sqlite3_bind_blob(
          stmt, num_params++, text_param.c_str(), text_param.size(), SQLITE_TRANSIENT);
    } else if (!bind_types.empty() && bind_types[num_params - 1] == BindType::NULL_TYPE) {
      returnCode = sqlite3_bind_null(stmt, num_params++);
    } else {
      returnCode = sqlite3_bind_text(
          stmt, num_params++, text_param.c_str(), text_param.size(), SQLITE_TRANSIENT);
    }
    if (returnCode != SQLITE_OK) {
      throwError();
    }
  }

  do {
    returnCode = sqlite3_step(stmt);
    if (returnCode != SQLITE_ROW && returnCode != SQLITE_DONE) {
      throwError();
    }
    if (returnCode == SQLITE_DONE) {
      break;
    }
    if (atFirstResult_) {
      numCols_ = sqlite3_column_count(stmt);
      for (size_t c = 0; c < numCols_; ++c) {
        columnNames.emplace_back(sqlite3_column_name(stmt, c));
        columnTypes.push_back(sqlite3_column_type(stmt, c));
      }
      results_.resize(numCols_);
      atFirstResult_ = false;
    }
    numRows_++;
    for (size_t c = 0; c < numCols_; ++c) {
      auto column_type = sqlite3_column_type(stmt, c);
      bool is_null = (column_type == SQLITE_NULL);
      auto col_text = get_column_datum(column_type, stmt, c);
      if (is_null) {
        CHECK(col_text.empty());
      }
      results_[c].emplace_back(NullableResult{col_text, is_null});
    }
  } while (1 == 1);  // Loop control in break statement above

  sqlite3_finalize(stmt);
}

void SqliteConnector::query_with_text_params(
    const std::string& queryString,
    const std::vector<std::string>& text_params) {
  query_with_text_params(queryString, text_params, {});
}

void SqliteConnector::query_with_text_param(const std::string& queryString,
                                            const std::string& text_param) {
  query_with_text_params(queryString, std::vector<std::string>{text_param});
}

void SqliteConnector::query(const std::string& queryString) {
  query_with_text_params(queryString, std::vector<std::string>{});
}

void SqliteConnector::batch_insert(const std::string& table_name,
                                   std::vector<std::vector<std::string>>& insert_vals) {
  const size_t num_rows = insert_vals.size();
  if (!num_rows) {
    return;
  }
  const size_t num_cols(insert_vals[0].size());
  if (!num_cols) {
    return;
  }
  std::string paramertized_query = "INSERT INTO " + table_name + " VALUES(";
  for (size_t col_idx = 0; col_idx < num_cols - 1; ++col_idx) {
    paramertized_query += "?, ";
  }
  paramertized_query += "?)";

  query("BEGIN TRANSACTION");

  sqlite3_stmt* stmt;
  int returnCode =
      sqlite3_prepare_v2(db_, paramertized_query.c_str(), -1, &stmt, nullptr);
  if (returnCode != SQLITE_OK) {
    throwError();
  }

  for (size_t r = 0; r < num_rows; ++r) {
    const auto& row_insert_vals = insert_vals[r];
    int num_params = 1;
    for (const auto& insert_field : row_insert_vals) {
      returnCode = sqlite3_bind_text(stmt,
                                     num_params++,
                                     insert_field.c_str(),
                                     insert_field.size(),
                                     SQLITE_TRANSIENT);
      if (returnCode != SQLITE_OK) {
        throwError();
      }
    }
    returnCode = sqlite3_step(stmt);
    if (returnCode != SQLITE_DONE) {
      throwError();
    }
    sqlite3_reset(stmt);
  }
  sqlite3_finalize(stmt);
  query("END TRANSACTION");
}
