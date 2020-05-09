/*
 * Copyright 2020 OmniSci, Inc.
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

#include "Catalog/ColumnDescriptor.h"
#include "Catalog/ForeignServer.h"
#include "DataMgr/ChunkMetadata.h"
#include "ForeignStorageBuffer.h"
#include "Shared/types.h"

namespace foreign_storage {
class ForeignDataWrapper {
 public:
  ForeignDataWrapper() = default;
  virtual ~ForeignDataWrapper() = default;

  /**
   * Gets the buffer for chunk identified by given chunk key.
   *
   * @param chunk_key - key for chunk whose buffer will be returned
   * @return buffer for chunk identified by given chunk key
   */
  virtual ForeignStorageBuffer* getChunkBuffer(const ChunkKey& chunk_key) = 0;

  /**
   * Populates given chunk_metadata_vector with metadata for all chunks with the given
   * prefix.
   *
   * @param chunk_key_prefix - key prefix for chunks whose metadata will be provided
   * @param chunk_metadata_vector - vector that will be populated with chunk metadata
   */
  virtual void populateMetadataForChunkKeyPrefix(
      const ChunkKey& chunk_key_prefix,
      ChunkMetadataVector& chunk_metadata_vector) = 0;
};
}  // namespace foreign_storage
