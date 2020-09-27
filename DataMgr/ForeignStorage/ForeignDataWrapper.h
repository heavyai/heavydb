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
   * Populates given chunk metadata vector with metadata for all chunks in related
   * foreign table.
   *
   * @param chunk_metadata_vector - vector that will be populated with chunk metadata
   */
  virtual void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) = 0;

  /**
   * Populates given chunk buffers identified by chunk keys. All provided chunk
   * buffers are expected to be for the same fragment.
   *
   * @param required_buffers - chunk buffers that must always be populated
   * @param optional_buffers - chunk buffers that can be optionally populated,
   * if the data wrapper has to scan through chunk data anyways (typically for
   * row wise data formats)
   */

  virtual void populateChunkBuffers(
      std::map<ChunkKey, AbstractBuffer*>& required_buffers,
      std::map<ChunkKey, AbstractBuffer*>& optional_buffers) = 0;

  /**
   * Serialize internal state of wrapper into file at given path if implemented
   * @param file_path - location to save file to
   */
  virtual void serializeDataWrapperInternals(const std::string& file_path) const = 0;

  /**
   * Restore internal state of datawrapper
   * @param file_path - location of file created by serializeMetadata
   * @param chunk_metadata_vector - vector of chunk metadata recovered from disk
   */
  virtual void restoreDataWrapperInternals(const std::string& file_path,
                                           const ChunkMetadataVector& chunk_metadata) = 0;

  // For testing, is this data wrapper restored from disk
  virtual bool isRestored() const = 0;
};
}  // namespace foreign_storage
