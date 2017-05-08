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
 * @file		SqliteConnector.h
 * @author		Todd Mostak <todd@map-d.com>
 */

#ifndef SQLITE_CONNECTOR
#define SQLITE_CONNECTOR

#include <string>
#include <vector>
#include <assert.h>
#include <boost/lexical_cast.hpp>

#include "sqlite3.h"

class SqliteConnector {
 public:
  SqliteConnector(const std::string& dbName, const std::string& dir = ".");
  ~SqliteConnector();
  void query(const std::string& queryString);
  void query_with_text_params(const std::string& queryString, const std::vector<std::string>& text_param);
  void query_with_text_param(const std::string& queryString, const std::string& text_param);

  size_t getNumRows() const { return numRows_; }
  size_t getNumCols() const { return numCols_; }

  template <typename T>
  T getData(const int row, const int col) {
    assert(row < static_cast<int>(numRows_));
    assert(col < static_cast<int>(numCols_));
    return boost::lexical_cast<T>(results_[col][row].result);
  }

  bool isNull(const int row, const int col) const {
    assert(row < static_cast<int>(numRows_));
    assert(col < static_cast<int>(numCols_));
    return results_[col][row].is_null;
  }

  std::vector<std::string> columnNames;  // make this public for easy access
  std::vector<int> columnTypes;

 private:
  struct NullableResult {
    const std::string result;
    const bool is_null;
  };

  void throwError();

  sqlite3* db_;
  std::string dbName_;
  bool atFirstResult_;
  std::vector<std::vector<NullableResult>> results_;
  size_t numCols_;
  size_t numRows_;
};

#endif  // SQLITE_CONNECTOR
