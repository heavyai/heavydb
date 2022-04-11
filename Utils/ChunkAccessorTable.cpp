/*
 * Copyright 2018 MapD Technologies, Inc.
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

/*
 * @file ChunkAccessorTable.cpp
 * @author Simon Eves <simon.eves@mapd.com>
 */

#include "ChunkAccessorTable.h"

ChunkAccessorTable getChunkAccessorTable(const Catalog_Namespace::Catalog& cat,
                                         const TableDescriptor* td,
                                         const std::vector<std::string>& columnNames) {
  UNREACHABLE();
}

ChunkIterVector& getChunkItersAndRowOffset(ChunkAccessorTable& table,
                                           size_t rowid,
                                           size_t& rowOffset) {
  rowOffset = 0;
  for (auto& entry : table) {
    if (rowid < std::get<0>(entry)) {
      return std::get<2>(entry);
    }
    rowOffset = std::get<0>(entry);
  }
  CHECK(false);
  static ChunkIterVector emptyChunkIterVector;
  return emptyChunkIterVector;
}
