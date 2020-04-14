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

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "QueryEngine/TargetValue.h"

namespace EmbeddedDatabase {

class Row {
 public:
  Row();
  Row(std::vector<TargetValue>& row);
  int64_t getInt(size_t col_num);
  double getDouble(size_t col_num);
  std::string getStr(size_t col_num);

 private:
  std::vector<TargetValue> row_;
};

class Cursor {
 public:
  size_t getColCount();
  size_t getRowCount();
  Row getNextRow();
  int getColType(uint32_t col_num);
};

class DBEngine {
 public:
  void reset();
  void executeDDL(std::string query);
  Cursor* executeDML(std::string query);
  static DBEngine* create(std::string path);

 protected:
  DBEngine() {}
};
}  // namespace EmbeddedDatabase
