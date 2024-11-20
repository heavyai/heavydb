/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include <map>

#include "Catalog/CatalogFwd.h"
#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/Chunk/Chunk.h"

namespace foreign_storage {

struct IterativeFileScanParameters {
  IterativeFileScanParameters(
      std::map<std::pair<int, int>, Chunk_NS::Chunk>& column_id_to_chunk_map,
      int32_t fragment_id,
      const std::vector<AbstractBuffer*>& delete_buffers)
      : column_id_and_batch_id_to_chunk_map(column_id_to_chunk_map)
      , fragment_id(fragment_id)
      , delete_buffers(delete_buffers) {}

  std::map<std::pair<int, int>, Chunk_NS::Chunk>& column_id_and_batch_id_to_chunk_map;

  int32_t fragment_id;
  const std::vector<AbstractBuffer*>& delete_buffers;
};

}  // namespace foreign_storage
