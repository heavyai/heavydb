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
#ifndef UPDELROLL_H
#define UPDELROLL_H

#include <map>
#include <mutex>
#include <set>
#include <utility>

#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ChunkMetadata.h"
#include "DataMgr/MemoryLevel.h"

namespace Fragmenter_Namespace {
class InsertOrderFragmenter;
class FragmentInfo;
}  // namespace Fragmenter_Namespace

namespace Catalog_Namespace {
class Catalog;
}

struct TableDescriptor;

using MetaDataKey =
    std::pair<const TableDescriptor*, Fragmenter_Namespace::FragmentInfo*>;

// this roll records stuff that need to be roll back/forw after upd/del fails or finishes
struct UpdelRoll {
  ~UpdelRoll() {
    if (dirty_chunks.size()) {
      cancelUpdate();
    }
  }

  // on aggregater it's possible that updateColumn is never called but
  // commitUpdate is still called, so this nullptr is a protection
  const Catalog_Namespace::Catalog* catalog = nullptr;
  int logicalTableId;
  Data_Namespace::MemoryLevel memoryLevel{Data_Namespace::MemoryLevel::CPU_LEVEL};

  bool is_varlen_update = false;
  const TableDescriptor* table_descriptor{nullptr};

  void cancelUpdate();

  // Commits/checkpoints data and metadata updates. A boolean that indicates whether or
  // not data update actually occurred is returned.
  bool commitUpdate();

  // Writes chunks at the CPU memory level to storage without checkpointing at the storage
  // level.
  void stageUpdate();

  void addDirtyChunk(std::shared_ptr<Chunk_NS::Chunk> chunk, int fragment_id);

  std::shared_ptr<ChunkMetadata> getChunkMetadata(
      const MetaDataKey& key,
      int32_t column_id,
      Fragmenter_Namespace::FragmentInfo& fragment_info);

  ChunkMetadataMap getChunkMetadataMap(const MetaDataKey& key) const;

  size_t getNumTuple(const MetaDataKey& key) const;

  void setNumTuple(const MetaDataKey& key, size_t num_tuple);

 private:
  void updateFragmenterAndCleanupChunks();

  void initializeUnsetMetadata(const TableDescriptor* td,
                               Fragmenter_Namespace::FragmentInfo& fragment_info);

  // Used to guard internal data structures that track chunk/chunk metadata updates
  mutable mapd_shared_mutex chunk_update_tracker_mutex;

  // chunks changed during this query
  std::map<ChunkKey, std::shared_ptr<Chunk_NS::Chunk>> dirty_chunks;

  // new FragmentInfo.numTuples
  std::map<MetaDataKey, size_t> num_tuples;

  // new FragmentInfo.ChunkMetadata;
  std::map<MetaDataKey, ChunkMetadataMap> chunk_metadata_map_per_fragment;
};

#endif
