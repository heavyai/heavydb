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
 * @file    AggModeHashTableGpu.h
 * @brief   CPU code to manage the GPU hash tables for calculating mode.
 *
 */

#pragma once

#include "DataMgr/Allocators/FastAllocator.h"
#include "QueryEngine/AggMode.h"

#include <cstdint>
#include <optional>
#include <vector>

class CudaAllocator;

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

namespace heavyai {
namespace agg_mode {
namespace detail {

// Holds warpcore instances that can be lightly copied to the GPU.
// hash_tables_ - holds pointers to the warpcore hash tables, one per mode target.
// Pointers are void* since class AggModeHashTableGpu is only defined in cuda files.
class AggModeHashTablesGpu {
 public:
  using Allocator = FastAllocator<int8_t>;
#ifdef HAVE_CUDA
  ~AggModeHashTablesGpu();
  void init(CudaAllocator*, CUstream, size_t nhash_tables);
  std::vector<int8_t> serialize() const;
  AggMode::Map moveToHost(size_t const index);
#endif
  size_t size() const {
    return hash_tables_.size();
  }

 private:
  std::optional<Allocator> allocator_;
  std::vector<void*> hash_tables_;  // memory owners
};

}  // namespace detail
}  // namespace agg_mode
}  // namespace heavyai

using heavyai::agg_mode::detail::AggModeHashTablesGpu;
