/*
 * Copyright 2020 OmniSci, Inc.
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

#include <arrow/table.h>
#include "DBETypes.h"

namespace EmbeddedDatabase {

class Cursor {
 public:
  virtual ~Cursor() {}
  size_t getColCount();
  size_t getRowCount();
  Row getNextRow();
  ColumnType getColType(uint32_t col_num);
  std::shared_ptr<arrow::RecordBatch> getArrowRecordBatch();

 protected:
  Cursor() {}
  Cursor(const Cursor&) = delete;
  Cursor& operator=(const Cursor&) = delete;
};

class DBEngine {
 public:
  virtual ~DBEngine() {}
  void executeDDL(const std::string& query);
  std::shared_ptr<Cursor> executeDML(const std::string& query);
  std::shared_ptr<Cursor> executeRA(const std::string& query);
  void importArrowTable(const std::string& name,
                        std::shared_ptr<arrow::Table>& table,
                        uint64_t fragment_size = 0);
  static std::shared_ptr<DBEngine> create(const std::string& cmd_line);
  std::vector<std::string> getTables();
  std::vector<ColumnDetails> getTableDetails(const std::string& table_name);
  void createUser(const std::string& user_name, const std::string& password);
  void dropUser(const std::string& user_name);
  void createDatabase(const std::string& db_name);
  void dropDatabase(const std::string& db_name);
  bool setDatabase(std::string& db_name);
  bool login(std::string& db_name, std::string& user_name, const std::string& password);

 protected:
  DBEngine() {}
  DBEngine(const DBEngine&) = delete;
  DBEngine& operator=(const DBEngine&) = delete;
};
}  // namespace EmbeddedDatabase
