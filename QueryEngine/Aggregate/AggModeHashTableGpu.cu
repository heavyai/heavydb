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

#include "AggModeHashTableGpu.cuh"
#include "DataMgr/Allocators/CudaAllocator.h"

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

#include <cstdio>
#include <cstring>
#include <exception>

namespace heavyai {
namespace agg_mode {
namespace detail {

void AggModeHashTablesGpu::init(CudaAllocator* cuda_allocator,
                                CUstream cuda_stream,
                                size_t const nhash_tables) {
  constexpr size_t init_capacity = kMinCapacity;
  constexpr size_t temp_memory_bytes = warpcore::defaults::temp_memory_bytes();
  constexpr size_t entry_size = sizeof(Value) + sizeof(Count);
  constexpr size_t size_per_table = temp_memory_bytes + entry_size * init_capacity;
  static_assert(size_per_table % 8u == 0u);
  size_t const capacity = nhash_tables * size_per_table;
  int8_t* const buffer = cuda_allocator->alloc(capacity);
  allocator_.emplace(buffer, capacity);
  hash_tables_.reserve(nhash_tables);
  for (size_t i = 0; i < nhash_tables; ++i) {
    hash_tables_.push_back(new AggModeHashTableGpu(cuda_stream, &*allocator_, i));
  }
}

AggModeHashTablesGpu::~AggModeHashTablesGpu() {
  // Deleting in reverse of allocation may help reduce memory fragmentation.
  for (auto itr = hash_tables_.rbegin(); itr != hash_tables_.rend(); ++itr) {
    delete static_cast<AggModeHashTableGpu*>(*itr);
  }
}

std::vector<int8_t> AggModeHashTablesGpu::serialize() const {
  constexpr size_t hash_table_size = sizeof(AggModeHashTableGpu);
  std::vector<int8_t> retval(hash_table_size * hash_tables_.size());
  for (size_t i = 0; i < hash_tables_.size(); ++i) {
    auto* const hash_table = static_cast<AggModeHashTableGpu*>(hash_tables_[i]);
    memcpy(retval.data() + i * hash_table_size, hash_table, hash_table_size);
  }
  return retval;
}

namespace {

// robin_hood::unordered_map can only be initialized by iterators that point to std::pair
// so this functor transforms boost::tuples into them, as well as casting key type
// uint64_t (needed for warpcore) to int64_t (needed for heavydb).
template <typename OutValue, typename OutCount, typename InValue, typename InCount>
struct StdPair {
  std::pair<OutValue, OutCount> operator()(
      boost::tuple<InValue, InCount> const& tuple) const {
    return {static_cast<OutValue>(boost::get<0>(tuple)),
            static_cast<OutCount>(boost::get<1>(tuple))};
  }
};

// Return (begin, end) pair of zip iterators by transforming
// (keys iterator, values iterator) pair into an iterator of (key, value) pairs.
template <typename MapType, typename T, typename U>
auto make_zip_iterator_pair(size_t const size, T const* ptr0, U const* ptr1) {
  return std::make_pair(
      boost::make_transform_iterator(
          boost::make_zip_iterator(boost::make_tuple(ptr0, ptr1)),
          StdPair<typename MapType::key_type, typename MapType::mapped_type, T, U>{}),
      boost::make_transform_iterator(
          boost::make_zip_iterator(boost::make_tuple(ptr0 + size, ptr1 + size)),
          StdPair<typename MapType::key_type, typename MapType::mapped_type, T, U>{}));
}
}  // namespace

// Return a CPU robin_hood::unordered_map based on GPU warpcore hash table.
AggMode::Map AggModeHashTablesGpu::moveToHost(size_t const index) {
  assert(index < hash_tables_.size());
  std::unique_ptr<AggModeHashTableGpu> hash_table(
      static_cast<AggModeHashTableGpu*>(hash_tables_[index]));
  hash_tables_[index] = nullptr;
#ifndef NDEBUG
  assert(hash_table);
#endif
  CountingHashTable::status_type const status = hash_table->peek_status();
  if (status.has_any_errors()) {  // index_overflow() or out_of_memory()
    // This exception will be caught and logged and a QueryMustRunOnCpu() thrown instead
    // in QueryMemoryInitializer::copyFromDeviceForMode().
    throw std::runtime_error("AggMode::status=" + std::to_string(status.base()));
  }
  size_t const hash_table_size = hash_table->size();
#ifndef NDEBUG
  // To distinguish size from a typical 48-bit memory address.
  // If hash_table is invalid, it will often have an excessively large size().
  constexpr size_t hash_table_size_upper_bound = size_t(1) << 44;
  assert(hash_table_size < hash_table_size_upper_bound);
#endif
  if (hash_table_size == 0) {
    return {};  // empty hash table
  } else {
    auto values = cuda_malloc_host_unique_ptr<Value>(hash_table_size);
    auto counts = cuda_malloc_host_unique_ptr<Count>(hash_table_size);
    auto const output_size = hash_table->retrieve_all(values.get(), counts.get());
#ifndef NDEBUG
    assert(hash_table_size == output_size);
#endif
    auto const zip_pair =
        make_zip_iterator_pair<AggMode::Map>(output_size, values.get(), counts.get());
    return {zip_pair.first, zip_pair.second};  // {begin, end}
  }
}

}  // namespace detail
}  // namespace agg_mode
}  // namespace heavyai
