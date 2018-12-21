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
  ChunkAccessorTable table;

  // get fragments
  const auto tableInfo = td->fragmenter->getFragmentsForQuery();
  if (tableInfo.fragments.size() == 0) {
    throw std::runtime_error("No fragments in table '" + td->tableName + "'");
  }

  // for each fragment...
  for (const auto& fragment : tableInfo.fragments) {
    // add a table entry for it
    table.emplace_back();
    std::get<0>(table.back()) = 0;

    // for each column...
    bool isFirstColumn = true;
    for (const auto& columnName : columnNames) {
      // get column descriptor
      const auto cd = cat.getMetadataForColumn(td->tableId, columnName);
      if (!cd) {
        throw std::runtime_error("Failed to find physical column '" + columnName + "'");
      }

      // find the chunk
      ChunkKey chunkKey{
          cat.getCurrentDB().dbId, td->tableId, cd->columnId, fragment.fragmentId};
      auto chunkMetaIt = fragment.getChunkMetadataMap().find(cd->columnId);
      if (chunkMetaIt == fragment.getChunkMetadataMap().end()) {
        throw std::runtime_error("Failed to find the chunk for column: " +
                                 cd->columnName + " in table: " + td->tableName +
                                 ". The column was likely deleted via a table truncate.");
      }

      // get the chunk
      std::shared_ptr<Chunk_NS::Chunk> chunk =
          Chunk_NS::Chunk::getChunk(cd,
                                    &cat.getDataMgr(),
                                    chunkKey,
                                    Data_Namespace::CPU_LEVEL,
                                    0,
                                    chunkMetaIt->second.numBytes,
                                    chunkMetaIt->second.numElements);
      CHECK(chunk);

      // the size
      size_t chunkSize = chunkMetaIt->second.numElements;

      // and an iterator
      ChunkIter chunkIt = chunk->begin_iterator(chunkMetaIt->second);

      // populate table entry
      if (isFirstColumn) {
        // store the size
        std::get<0>(table.back()) = chunkSize;
        isFirstColumn = false;
      } else {
        // all columns chunks must be the same size
        CHECK(std::get<0>(table.back()) == chunkSize);
      }
      std::get<1>(table.back()).push_back(chunk);
      std::get<2>(table.back()).push_back(chunkIt);
    }
  }

  // prefix-sum the per-fragment counts
  // these are now "first row of next fragment"
  size_t sum = 0;
  for (auto& entry : table) {
    sum += std::get<0>(entry);
    std::get<0>(entry) = sum;
  }

  // done
  return table;
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
