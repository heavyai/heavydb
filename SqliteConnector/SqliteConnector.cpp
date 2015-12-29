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

int resultCallback(void* connObj, int argc, char** argv, char** colNames) {
  cout << "callback" << endl;
  SqliteConnector* sqliteConnector = reinterpret_cast<SqliteConnector*>(connObj);
  if (sqliteConnector->atFirstResult_) {
    sqliteConnector->numCols_ = argc;
    for (int c = 0; c < argc; ++c) {
      sqliteConnector->columnNames.push_back(colNames[c]);
    }
    sqliteConnector->results_.resize(argc);
    sqliteConnector->atFirstResult_ = false;
  }
  sqliteConnector->numRows_++;
  for (int c = 0; c < argc; ++c) {
    sqliteConnector->results_[c].push_back(argv[c]);
  }
  return 0;
}

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
      if (col_text) {
        results_[c].push_back(
            col_text);  // b/c sqlite returns unsigned char* which can't be used in constructor of string
      } else {
        results_[c].push_back("");  // interpret nulls as empty string
      }
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

void SqliteConnector::queryWithCallback(const std::string& queryString) {
  atFirstResult_ = true;
  numRows_ = 0;
  numCols_ = 0;
  columnNames.clear();
  columnTypes.clear();
  results_.clear();
  char* errorMsg;
  int returnCode = sqlite3_exec(db_, queryString.c_str(), resultCallback, this, &errorMsg);
  if (returnCode != SQLITE_OK) {
    throwError();
  }
}
/*

int main () {
    SqliteConnector sqlConnector ("test");
    sqlConnector.query("select * from tables");
    int numRows = sqlConnector.getNumRows();
    int numCols = sqlConnector.getNumCols();
    cout << "Num rows: " << numRows << endl;
    cout << "Num cols: " << numCols << endl;
    for (int r = 0; r < numRows; ++r) {
        cout << sqlConnector.getData<string>(r,0);
        cout << " ";
        cout << sqlConnector.getData<int>(r,1);
        cout << endl;
    }
}
*/
