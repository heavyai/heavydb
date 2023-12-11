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

/**
 * @file    CpuMgrArenaAllocator.h
 * @brief   Allocate CPU memory using CpuBuffers via DataMgr.
 */

#pragma once

#include "DataMgr/Allocators/ArenaAllocator.h"

namespace Data_Namespace {
class AbstractBuffer;
class DataMgr;
}  // namespace Data_Namespace

class CpuMgrArenaAllocator : public Arena {
 public:
  CpuMgrArenaAllocator();

  ~CpuMgrArenaAllocator() override;

  void* allocate(size_t num_bytes) override;

  void* allocateAndZero(const size_t num_bytes) override;

  size_t bytesUsed() const override;

  size_t totalBytes() const override;

  MemoryType getMemoryType() const override;

 private:
  Data_Namespace::DataMgr& data_mgr_;
  std::vector<Data_Namespace::AbstractBuffer*> allocated_buffers_;
  size_t size_;
};
