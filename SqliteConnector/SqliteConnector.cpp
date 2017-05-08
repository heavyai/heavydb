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

using std::cout;
using std::endl;
using std::string;
using std::runtime_error;

SqliteConnector::SqliteConnector(const string& dbName, const string& dir) : dbName_(dbName) {
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

SqliteConnector::~SqliteConnector() {
  sqlite3_close(db_);
}

void SqliteConnector::throwError() {
  string errorMsg(sqlite3_errmsg(db_));
  throw runtime_error("Sqlite3 Error: " + errorMsg);
}

void SqliteConnector::query_with_text_params(const std::string& queryString,
                                             const std::vector<std::string>& text_params) {
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

  int numParams_ = 1;
  for (auto text_param : text_params) {
    returnCode = sqlite3_bind_text(stmt, numParams_++, text_param.c_str(), text_param.size(), SQLITE_TRANSIENT);
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
        columnNames.push_back(sqlite3_column_name(stmt, c));
        columnTypes.push_back(sqlite3_column_type(stmt, c));
      }
      results_.resize(numCols_);
      atFirstResult_ = false;
    }
    numRows_++;
    for (size_t c = 0; c < numCols_; ++c) {
      auto col_text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, c));
      bool is_null = sqlite3_column_type(stmt, c) == SQLITE_NULL;
      assert(is_null == !col_text);
      results_[c].push_back(NullableResult{is_null ? "" : col_text, is_null});
    }
  } while (1 == 1);  // Loop control in break statement above

  sqlite3_finalize(stmt);
}

void SqliteConnector::query_with_text_param(const std::string& queryString, const std::string& text_param) {
  query_with_text_params(queryString, std::vector<std::string>{text_param});
}

void SqliteConnector::query(const std::string& queryString) {
  query_with_text_params(queryString, std::vector<std::string>{});
}
