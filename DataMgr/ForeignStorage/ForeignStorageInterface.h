/*
 * Copyright 2019 OmniSci, Inc.
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

#include "../AbstractBufferMgr.h"
#include "Catalog/Catalog.h"

#include <unordered_map>

struct ForeignStorageColumnBuffer {
  const ChunkKey chunk_key;
  const SQLTypeInfo sql_type;
  const std::vector<int8_t> buff;
};

class PersistentForeignStorageInterface {
 public:
  virtual void append(const std::vector<ForeignStorageColumnBuffer>& column_buffers) = 0;
  virtual void read(const ChunkKey& chunk_key,
                    const SQLTypeInfo& sql_type,
                    int8_t* dest,
                    const size_t num_bytes) = 0;
  virtual std::string getType() const = 0;
};

class ForeignStorageInterface {
 public:
  static Data_Namespace::AbstractBufferMgr* lookupBufferManager(const int db_id,
                                                                const int table_id);

  static void registerPersistentStorageInterface(
      PersistentForeignStorageInterface* persistent_foreign_storage);

  static void registerTable(const int db_id, const int table_id, const std::string& type);

 private:
  static std::unordered_map<std::string,
                            std::unique_ptr<PersistentForeignStorageInterface>>
      persistent_storage_interfaces_;
  static std::map<std::pair<int, int>, PersistentForeignStorageInterface*>
      table_persistent_storage_interface_map_;
  static std::mutex persistent_storage_interfaces_mutex_;
};
