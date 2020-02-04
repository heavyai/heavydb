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

#include "ForeignStorageInterface.h"

class DummyPersistentForeignStorage : public PersistentForeignStorageInterface {
 public:
  void append(const std::vector<ForeignStorageColumnBuffer>& column_buffers) override;

  void read(const ChunkKey& chunk_key,
            const SQLTypeInfo& sql_type,
            int8_t* dest,
            const size_t numBytes) override;

  std::string getType() const override;

 private:
  void append(const ChunkKey& chunk_key,
              const SQLTypeInfo& sql_type,
              const int8_t* src,
              const size_t numBytes);

  std::map<ChunkKey, std::vector<int8_t>> files_;
  std::mutex files_mutex_;
};
