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

/**
 * @file ChunkAccessorTable.h
 * @author Simon Eves <simon.eves@mapd.com>
 *
 */
#ifndef _CHUNK_ACCESSOR_TABLE_H_
#define _CHUNK_ACCESSOR_TABLE_H_

#include <Catalog/Catalog.h>
#include <Chunk/Chunk.h>

#include <vector>

// convenience functions for multi-fragment support in multi-threaded worker functions (poly rendering, importer)

using ChunkIterVector = std::vector<ChunkIter>;
using ChunkAccessorTable = std::vector<std::tuple<size_t, std::vector<std::shared_ptr<Chunk_NS::Chunk>>, ChunkIterVector>>;

ChunkAccessorTable getChunkAccessorTable(const Catalog_Namespace::Catalog& cat,
                                         const TableDescriptor* td,
                                         const std::vector<std::string>& columnNames);

ChunkIterVector& getChunkItersAndRowOffset(ChunkAccessorTable& table,
                                           size_t rowid,
                                           size_t& rowOffset);

#endif // _CHUNK_ACCESSOR_TABLE_H_
