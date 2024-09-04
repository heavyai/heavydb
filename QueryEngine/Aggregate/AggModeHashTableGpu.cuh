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

#pragma once

#include "AggModeHashTableGpu.h"

#include <warpcore/counting_hash_table.cuh>

namespace heavyai {
namespace agg_mode {
namespace detail {

using Value = uint64_t;  // must be uint32_t or uint64_t
using Count = uint64_t;

struct CudaHostDeleter {
  void operator()(void* ptr) const { cudaFreeHost(ptr); }
};

template <typename T>
std::unique_ptr<T, CudaHostDeleter> cuda_malloc_host_unique_ptr(size_t const size) {
  T* ptr;
  cudaError_t const err = cudaMallocHost(&ptr, sizeof(T) * size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed call to cuda_malloc_host_unique_ptr w/ T=" +
                             std::string(typeid(T).name()) +
                             " and size=" + std::to_string(size));
  }
  return std::unique_ptr<T, CudaHostDeleter>(ptr);
}

// Sentinel value used by warpcore::CountingHashTable to designate an empty slot.
constexpr Value kEmptyValue = Value(1) << 63;  // BIGINT NULL.

// The kTombstoneValue is only used in calls to erase() which we don't use but need
// to set a value for. Set it to kEmptyValue so that we don't lose a sentinel value.
constexpr Value kTombstoneValue = kEmptyValue;

// Necessary for our kernel launches due to early branching of active threads.  Thus any
// thread in a cooperative group with size greater than 1 will not be guaranteed active
// and can result in a hanging kernel.
constexpr size_t kCooperativeGroupSize = 1u;

using ProbingScheme = warpcore::defaults::probing_scheme_t<Value, kCooperativeGroupSize>;

// This is the default TableStore for CountingHashTable, but using our Allocator.
using TableStorage =
    warpcore::storage::key_value::AoSStore<Value, Count, AggModeHashTablesGpu::Allocator>;

using CountingHashTable = warpcore::CountingHashTable<Value,  // key type
                                                      Count,  // value type
                                                      AggModeHashTablesGpu::Allocator,
                                                      kEmptyValue,
                                                      kTombstoneValue,
                                                      ProbingScheme,
                                                      TableStorage>;

// Initial hash table capacity. Capacities are limited to warpcore::detail::primes.
constexpr CountingHashTable::index_type kMinCapacity = warpcore::detail::primes.front();

// Override default probing_length = ~index_t(0) - 1024 = 18446744073709550591
constexpr CountingHashTable::index_type kProbingLength = kMinCapacity;

// Count after which new occurrences are ignored.
constexpr auto kMaxCount = std::numeric_limits<Count>::max();
// Initialize CountingHashTable by calling init(custream_) rather than by the constructor
// so that the custream can be set to a non-default value.
constexpr bool kNoInit = true;

// Wrapper class around CountingHashTable stores cudaStream_t known at initialization.
class AggModeHashTableGpu : public CountingHashTable {
 public:
  HOSTQUALIFIER INLINEQUALIFIER
  AggModeHashTableGpu(cudaStream_t const custream,
                      AggModeHashTablesGpu::Allocator* const allocator,
                      Value const seed) noexcept
      : CountingHashTable(kMinCapacity, allocator, seed, kMaxCount, kNoInit)
      , custream_(custream) {
    CountingHashTable::init(custream_);
  }

  HOSTQUALIFIER INLINEQUALIFIER index_type capacity() const noexcept {
    return CountingHashTable::capacity();
  }

  HOSTQUALIFIER INLINEQUALIFIER status_type peek_status() const noexcept {
    return CountingHashTable::peek_status(custream_);
  }

  DEVICEQUALIFIER INLINEQUALIFIER status_type insert(int64_t const value) noexcept {
    auto const coop_group = cooperative_groups::tiled_partition<cg_size()>(
        cooperative_groups::this_thread_block());
    status_type const status =
        CountingHashTable::insert(static_cast<Value>(value), coop_group, kProbingLength);
    return status;
  }

  HOSTDEVICEQUALIFIER INLINEQUALIFIER bool is_copy() const noexcept {
    return CountingHashTable::is_copy();
  }

  HOSTQUALIFIER INLINEQUALIFIER Count retrieve_all(Value* values_out,
                                                   Count* counts_out) const noexcept {
    auto num_out = cuda_malloc_host_unique_ptr<index_type>(1);
    CountingHashTable::retrieve_all(values_out, counts_out, *num_out, custream_);
    cudaStreamSynchronize(custream_);
    return static_cast<Count>(*num_out);
  }

  HOSTQUALIFIER INLINEQUALIFIER index_type size() const noexcept {
    return CountingHashTable::size(custream_);
  }

 private:
  cudaStream_t custream_;
};

// 8-byte alignment is necessary when AggModeHashTableGpu objects are memcopied.
static_assert(sizeof(AggModeHashTableGpu) % sizeof(int64_t) == 0);

}  // namespace detail
}  // namespace agg_mode
}  // namespace heavyai

using heavyai::agg_mode::detail::AggModeHashTableGpu;
