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

#ifndef __DB_ENGINE_H
#define __DB_ENGINE_H

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
  int64_t GetInt(size_t col);
  double GetDouble(size_t col);
  std::string GetStr(size_t col);

 private:
  std::vector<TargetValue> m_row;
};

class Cursor {
 public:
  size_t GetColCount();
  size_t GetRowCount();
  Row GetNextRow();
  int GetColType(uint32_t nPos);
};

class DBEngine {
 public:
  void Reset();
  void ExecuteDDL(std::string sQuery);
  Cursor* ExecuteDML(std::string sQuery);
  static DBEngine* Create(std::string sPath);

 protected:
  DBEngine() {}
};
}  // namespace EmbeddedDatabase

#endif  // __DB_ENGINE_H
