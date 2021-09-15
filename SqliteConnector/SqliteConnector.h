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

#include <boost/lexical_cast.hpp>
#include <cassert>
#include <string>
#include <vector>

#include <sqlite3.h>

class SqliteConnector {
 public:
  SqliteConnector(const std::string& dbName, const std::string& dir = ".");
  SqliteConnector(sqlite3* db);
  SqliteConnector() {}
  virtual ~SqliteConnector();
  virtual void query(const std::string& queryString);

  virtual void query_with_text_params(std::string const& query_only) {
    query(query_only);
  }
  template <typename STRING_CONTAINER>
  void query_with_text_params(STRING_CONTAINER const& query_and_text_params) {
    query_with_text_params(
        *query_and_text_params.begin(),
        std::vector<std::string>{std::next(query_and_text_params.begin()),
                                 query_and_text_params.end()});
  }
  virtual void query_with_text_params(const std::string& queryString,
                                      const std::vector<std::string>& text_param);

  enum class BindType { TEXT = 1, BLOB, NULL_TYPE };
  virtual void query_with_text_params(const std::string& queryString,
                                      const std::vector<std::string>& text_params,
                                      const std::vector<BindType>& bind_types);

  virtual void query_with_text_param(const std::string& queryString,
                                     const std::string& text_param);

  virtual void batch_insert(const std::string& table_name,
                            std::vector<std::vector<std::string>>& insert_vals);

  virtual size_t getNumRows() const { return numRows_; }
  virtual size_t getNumCols() const { return numCols_; }

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

  auto getSqlitePtr() const { return db_; }

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
