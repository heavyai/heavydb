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

#include "../Chunk/Chunk.h"
#include "../DataMgr/ChunkMetadata.h"

namespace Fragmenter_Namespace {
class InsertOrderFragmenter;
}

namespace Catalog_Namespace {
class Catalog;
}

struct TableDescriptor;

// this roll records stuff that need to be roll back/forw after upd/del fails or finishes
struct UpdelRoll {
  ~UpdelRoll() {
    if (dirty_chunks.size())
      cancelUpdate();
  }
  std::mutex mutex;

  // chunks changed during this query
  std::map<Chunk_NS::Chunk*, std::shared_ptr<Chunk_NS::Chunk>> dirty_chunks;

  // new FragmentInfo.numTuples
  std::map<int, size_t> numTuples;

  // new FragmentInfo.ChunkMetadata;
  std::map<int, std::map<int, ChunkMetadata>> chunkMetadata;

  // helper members
  const Catalog_Namespace::Catalog* catalog;
  const TableDescriptor* tableDescriptor;
  Fragmenter_Namespace::InsertOrderFragmenter* insertOrderFragmenter;

  void cancelUpdate();
  void commitUpdate();
};

#endif
