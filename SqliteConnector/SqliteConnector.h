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

  inline size_t getNumRows() { return numRows_; }
  inline size_t getNumCols() { return numCols_; }

  template <typename T>
  T getData(const int row, const int col) {
    assert(row < static_cast<int>(numRows_));
    assert(col < static_cast<int>(numCols_));
    return boost::lexical_cast<T>(results_[col][row]);
  }
  std::vector<std::string> columnNames;  // make this public for easy access
  std::vector<int> columnTypes;

 private:
  void throwError();

  sqlite3* db_;
  std::string dbName_;
  bool atFirstResult_;
  std::vector<std::vector<std::string>> results_;
  size_t numCols_;
  size_t numRows_;
};

#endif  // SQLITE_CONNECTOR
