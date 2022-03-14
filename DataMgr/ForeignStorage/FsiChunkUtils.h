/*
 * Copyright 2021 OmniSci, Inc.
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

#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ChunkMetadata.h"

namespace foreign_storage {
void init_chunk_for_column(
    const ChunkKey& chunk_key,
    const std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& chunk_metadata_map,
    const std::map<ChunkKey, AbstractBuffer*>& buffers,
    Chunk_NS::Chunk& chunk);

// Construct default metadata for given column descriptor with num_elements
std::shared_ptr<ChunkMetadata> get_placeholder_metadata(const ColumnDescriptor* column,
                                                        size_t num_elements);
}  // namespace foreign_storage
