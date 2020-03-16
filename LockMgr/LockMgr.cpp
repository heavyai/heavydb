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

#include "LockMgr/LockMgrImpl.h"

#include "LockMgr/LegacyLockMgr.h"

#include "Fragmenter/InsertOrderFragmenter.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryRunner/QueryRunner.h"
#include "gen-cpp/CalciteServer.h"

namespace lockmgr {

namespace helpers {

ChunkKey chunk_key_for_table(const Catalog_Namespace::Catalog& cat,
                             const std::string& tableName) {
  if (const auto tdp = cat.getMetadataForTable(tableName, false)) {
    ChunkKey chunk_key{cat.getCurrentDB().dbId, tdp->tableId};
    return chunk_key;
  } else {
    throw std::runtime_error("Table " + tableName + " does not exist.");
  }
}

}  // namespace helpers

}  // namespace lockmgr
