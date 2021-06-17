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

#include "DataMgr/ChunkMetadata.h"
#include "ForeignStorageBuffer.h"
#include "Shared/types.h"

struct ColumnDescriptor;
namespace foreign_storage {
struct ForeignServer;
struct ForeignTable;
struct UserMapping;
using ChunkToBufferMap = std::map<ChunkKey, AbstractBuffer*>;

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

  virtual void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                                    const ChunkToBufferMap& optional_buffers) = 0;

  /**
   * Serialize internal state of wrapper into file at given path if implemented
   */
  virtual std::string getSerializedDataWrapper() const = 0;

  /**
   * Restore internal state of datawrapper
   * @param file_path - location of file created by serializeMetadata
   * @param chunk_metadata_vector - vector of chunk metadata recovered from disk
   */
  virtual void restoreDataWrapperInternals(const std::string& file_path,
                                           const ChunkMetadataVector& chunk_metadata) = 0;

  // For testing, is this data wrapper restored from disk
  virtual bool isRestored() const = 0;

  /**
   * Checks that the options for the given foreign server object are valid.
   * @param foreign_server - foreign server object containing options to be validated
   */
  virtual void validateServerOptions(const ForeignServer* foreign_server) const = 0;

  /**
   * Checks that the options for the given foreign table object are valid.
   * @param foreign_table - foreign table object containing options to be validated
   */
  virtual void validateTableOptions(const ForeignTable* foreign_table) const = 0;

  /**
   * Gets the set of supported table options for the data wrapper.
   */
  virtual const std::set<std::string_view>& getSupportedTableOptions() const = 0;

  /**
   * Checks that the options for the given user mapping object are valid.
   * @param user_mapping - user mapping object containing options to be validated
   */
  virtual void validateUserMappingOptions(const UserMapping* user_mapping,
                                          const ForeignServer* foreign_server) const = 0;

  /**
   * Gets the set of supported user mapping options for the data wrapper.
   */
  virtual const std::set<std::string_view>& getSupportedUserMappingOptions() const = 0;

  /**
    Verifies the schema is supported by this foreign table
    * @param columns - column descriptors for this table
   */
  virtual void validateSchema(const std::list<ColumnDescriptor>& columns) const {};

  /**
   * ParallelismLevel describes the desired level of parallelism of the data
   * wrapper. This level controls which `optional_buffers` are passed to
   * `populateChunkBuffers` with the following behaviour:
   *
   * NONE - no additional optional buffers are passed in
   *
   * INTRA_FRAGMENT - additional optional buffers which are in the same fragment as the
   * required buffers
   *
   * INTER_FRAGMENT - additional optional buffers which may be in
   * different fragments than those of the required buffers
   *
   * Note, the optional buffers are passed in with the intention of
   * allowing the data wrapper to employ parallelism in retrieving them. Each subsequent
   * level allows for a greater degree of parallelism but does not have to be supported.
   */
  enum ParallelismLevel { NONE, INTRA_FRAGMENT, INTER_FRAGMENT };

  /**
   * Gets the desired level of parallelism for the data wrapper when a cache is
   * in use. This affects the optional buffers that the data wrapper is made
   * aware of during data requests.
   */
  virtual ParallelismLevel getCachedParallelismLevel() const { return NONE; }

  /**
   * Gets the desired level of parallelism for the data wrapper when no cache
   * is in use. This affects the optional buffers that the data wrapper is made
   * aware of during data requests.
   */
  virtual ParallelismLevel getNonCachedParallelismLevel() const { return NONE; }
};
}  // namespace foreign_storage
