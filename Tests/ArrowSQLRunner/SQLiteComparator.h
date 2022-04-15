/*
 * Copyright 2021 OmniSci, Inc.
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

#pragma once

#include "QueryEngine/ArrowResultSet.h"
#include "Shared/sqltypes.h"
#include "SqliteConnector/SqliteConnector.h"

namespace TestHelpers {

constexpr double EPS = 1.25e-5;

class SQLiteComparator {
 public:
  SQLiteComparator(bool use_row_iterator = true)
      : connector_("sqliteTestDB", ""), use_row_iterator_(use_row_iterator) {}

  void query(const std::string& query_string) { connector_.query(query_string); }

  void compare(ResultSetPtr omnisci_results,
               const std::string& query_string,
               const ExecutorDeviceType device_type);

  void compare_arrow_output(std::unique_ptr<ArrowResultSet>& arrow_omnisci_results,
                            const std::string& sqlite_query_string,
                            const ExecutorDeviceType device_type);

  // added to deal with time shift for now testing
  void compare_timstamp_approx(ResultSetPtr omnisci_results,
                               const std::string& query_string,
                               const ExecutorDeviceType device_type);

 private:
  SqliteConnector connector_;
  bool use_row_iterator_;
};

}  // namespace TestHelpers