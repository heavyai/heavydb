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

#ifndef FRAGMENTER_H
#define FRAGMENTER_H

#include <deque>
#include <list>
#include <map>
#include <mutex>
#include "../Catalog/ColumnDescriptor.h"
#include "../DataMgr/ChunkMetadata.h"
#include "../Shared/mapd_shared_mutex.h"
#include "../Shared/types.h"

namespace Data_Namespace {
class AbstractBuffer;
}

class ResultSet;

namespace Fragmenter_Namespace {
class InsertOrderFragmenter;

/**
 * @enum FragmenterType
 * stores the type of a child class of
 * AbstractTableFragmenter
 */

enum FragmenterType {
  INSERT_ORDER = 0  // these values persist in catalog.  make explicit
};

/**
 * @struct InsertData
 * @brief The data to be inserted using the fragment manager.
 *
 * The data being inserted is assumed to be in columnar format, and so the offset
 * to the beginning of each column can be calculated by multiplying the column size
 * by the number of rows.
 *
 * @todo support for variable-length data types
 */

struct InsertData {
  int databaseId;  /// identifies the database into which the data is being inserted
  int tableId;     /// identifies the table into which the data is being inserted
  std::vector<int> columnIds;  /// a vector of column ids for the row(s) being inserted
  size_t numRows;              /// the number of rows being inserted
  std::vector<DataBlockPtr> data;  /// points to the start of the data block per column
                                   /// for the row(s) being inserted
  std::vector<bool> is_default;    /// data for such columns consist of a single
                                   /// replicated element (NULL or DEFAULT)
};

/**
 * @class FragmentInfo
 * @brief Used by Fragmenter classes to store info about each
 * fragment - the fragment id and number of tuples(rows)
 * currently stored by that fragment
 */

class FragmentInfo {
 public:
  FragmentInfo()
      : fragmentId(-1)
      , shadowNumTuples(0)
      , physicalTableId(-1)
      , resultSet(nullptr)
      , numTuples(0)
      , synthesizedNumTuplesIsValid(false)
      , synthesizedMetadataIsValid(false) {}

  void setChunkMetadataMap(const ChunkMetadataMap& chunk_metadata_map) {
    this->chunkMetadataMap = chunk_metadata_map;
  }

  void setChunkMetadata(const int col, std::shared_ptr<ChunkMetadata> chunkMetadata) {
    chunkMetadataMap[col] = chunkMetadata;
  }

  const ChunkMetadataMap& getChunkMetadataMap() const;

  const ChunkMetadataMap& getChunkMetadataMapPhysical() const { return chunkMetadataMap; }

  ChunkMetadataMap getChunkMetadataMapPhysicalCopy() const;

  size_t getNumTuples() const;

  size_t getPhysicalNumTuples() const { return numTuples; }

  bool isEmptyPhysicalFragment() const { return physicalTableId >= 0 && !numTuples; }

  void setPhysicalNumTuples(const size_t physNumTuples) { numTuples = physNumTuples; }

  void invalidateChunkMetadataMap() const { synthesizedMetadataIsValid = false; };
  void invalidateNumTuples() const { synthesizedNumTuplesIsValid = false; }

  // for unit tests
  static void setUnconditionalVacuum(const double unconditionalVacuum) {
    unconditionalVacuum_ = unconditionalVacuum;
  }

  int fragmentId;
  size_t shadowNumTuples;
  std::vector<int> deviceIds;
  int physicalTableId;
  ChunkMetadataMap shadowChunkMetadataMap;
  mutable ResultSet* resultSet;
  mutable std::shared_ptr<std::mutex> resultSetMutex;

 private:
  mutable size_t numTuples;
  mutable ChunkMetadataMap chunkMetadataMap;
  mutable bool synthesizedNumTuplesIsValid;
  mutable bool synthesizedMetadataIsValid;

  friend class InsertOrderFragmenter;
  static bool unconditionalVacuum_;
};

/**
 * @struct QueryInfo
 * @brief returned by Fragmenter classes in
 * getFragmentsForQuery - tells Executor which
 * fragments to scan from which fragmenter
 * (fragmenter id and fragment id needed for building
 * ChunkKey)
 */

class TableInfo {
 public:
  TableInfo() : numTuples(0) {}

  size_t getNumTuples() const;

  size_t getNumTuplesUpperBound() const;

  size_t getPhysicalNumTuples() const { return numTuples; }

  void setPhysicalNumTuples(const size_t physNumTuples) { numTuples = physNumTuples; }

  size_t getFragmentNumTuplesUpperBound() const;

  std::vector<int> chunkKeyPrefix;
  std::vector<FragmentInfo> fragments;

 private:
  mutable size_t numTuples;
};

}  // namespace Fragmenter_Namespace

#endif  // FRAGMENTER_H
