/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
