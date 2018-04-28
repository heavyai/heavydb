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
#include <set>
#include <mutex>
#include <utility>

#include "../Chunk/Chunk.h"
#include "../DataMgr/ChunkMetadata.h"
#include "../DataMgr/MemoryLevel.h"

namespace Fragmenter_Namespace {
class InsertOrderFragmenter;
}

namespace Catalog_Namespace {
class Catalog;
}

struct TableDescriptor;

using MetaDataKey = std::pair<const TableDescriptor*, int>;

// this roll records stuff that need to be roll back/forw after upd/del fails or finishes
struct UpdelRoll {
  ~UpdelRoll() {
    if (dirtyChunks.size())
      cancelUpdate();
  }
  std::mutex mutex;

  // chunks changed during this query
  std::map<Chunk_NS::Chunk*, std::shared_ptr<Chunk_NS::Chunk>> dirtyChunks;
  std::set<ChunkKey> dirtyChunkeys;

  // new FragmentInfo.numTuples
  std::map<MetaDataKey, size_t> numTuples;

  // new FragmentInfo.ChunkMetadata;
  std::map<MetaDataKey, std::map<int, ChunkMetadata>> chunkMetadata;

  // helper members
  const Catalog_Namespace::Catalog* catalog;
  std::set<const TableDescriptor*> tableDescriptors;
  Data_Namespace::MemoryLevel memoryLevel{Data_Namespace::MemoryLevel::CPU_LEVEL};

  void cancelUpdate();
  void commitUpdate();
};

#endif
