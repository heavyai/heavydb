/*
 * Copyright 2021 OmniSci, Inc.
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

#include <boost/noncopyable.hpp>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/Allocators/ArenaAllocator.h"
#include "DataMgr/DataMgr.h"
#include "DataProvider/DataProvider.h"
#include "Logger/Logger.h"
#include "QueryEngine/StringDictionaryGenerations.h"
#include "Shared/quantile.h"
#include "StringDictionary/StringDictionaryProxy.h"
#include "ThirdParty/robin_hood.h"

class ResultSet;

/**
 * Handles allocations and outputs for all stages in a query, either explicitly or via a
 * managed allocator object
 */
class RowSetMemoryOwner final : public SimpleAllocator, boost::noncopyable {
 public:
  RowSetMemoryOwner(DataProvider* data_provider,
                    const size_t arena_block_size,
                    const size_t num_kernel_threads = 0)
      : data_provider_(data_provider), arena_block_size_(arena_block_size) {
    for (size_t i = 0; i < num_kernel_threads + 1; i++) {
      allocators_.emplace_back(std::make_unique<Arena>(arena_block_size));
    }
    CHECK(!allocators_.empty());
  }

  int8_t* allocate(const size_t num_bytes, const size_t thread_idx = 0) override {
    CHECK_LT(thread_idx, allocators_.size());
    auto allocator = allocators_[thread_idx].get();
    std::lock_guard<std::mutex> lock(state_mutex_);
    return reinterpret_cast<int8_t*>(allocator->allocate(num_bytes));
  }

  int8_t* allocateCountDistinctBuffer(const size_t num_bytes,
                                      const size_t thread_idx = 0) {
    int8_t* buffer = allocate(num_bytes, thread_idx);
    std::memset(buffer, 0, num_bytes);
    addCountDistinctBuffer(buffer, num_bytes, /*physical_buffer=*/true);
    return buffer;
  }

  void addCountDistinctBuffer(int8_t* count_distinct_buffer,
                              const size_t bytes,
                              const bool physical_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    count_distinct_bitmaps_.emplace_back(
        CountDistinctBitmapBuffer{count_distinct_buffer, bytes, physical_buffer});
  }

  void addCountDistinctSet(robin_hood::unordered_set<int64_t>* count_distinct_set) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    count_distinct_sets_.push_back(count_distinct_set);
  }

  void addGroupByBuffer(int64_t* group_by_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    group_by_buffers_.push_back(group_by_buffer);
  }

  void addVarlenBuffer(void* varlen_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    varlen_buffers_.push_back(varlen_buffer);
  }

  /**
   * Adds a GPU buffer containing a variable length input column. Variable length inputs
   * on GPU are referenced in output projected targets and should not be freed until the
   * query results have been resolved.
   */
  void addVarlenInputBuffer(Data_Namespace::AbstractBuffer* buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    CHECK_EQ(buffer->getType(), Data_Namespace::MemoryLevel::GPU_LEVEL);
    varlen_input_buffers_.push_back(buffer);
  }

  std::string* addString(const std::string& str) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    strings_.emplace_back(str);
    return &strings_.back();
  }

  std::vector<int64_t>* addArray(const std::vector<int64_t>& arr) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    arrays_.emplace_back(arr);
    return &arrays_.back();
  }

  StringDictionaryProxy* addStringDict(std::shared_ptr<StringDictionary> str_dict,
                                       const int dict_id,
                                       const int64_t generation) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = str_dict_proxy_owned_.find(dict_id);
    if (it != str_dict_proxy_owned_.end()) {
      CHECK_EQ(it->second->getDictionary(), str_dict.get());
      it->second->updateGeneration(generation);
      return it->second.get();
    }
    it = str_dict_proxy_owned_
             .emplace(
                 dict_id,
                 std::make_shared<StringDictionaryProxy>(str_dict, dict_id, generation))
             .first;
    return it->second.get();
  }

  StringDictionaryProxyTranslationMap* addStringProxyTranslationMap(
      const StringDictionaryProxy* source_proxy,
      const StringDictionaryProxy* dest_proxy) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const auto map_key = std::make_pair(source_proxy->getDictionary()->getDictId(),
                                        dest_proxy->getDictionary()->getDictId());
    auto it = str_proxy_translation_maps_owned_.find(map_key);
    if (it == str_proxy_translation_maps_owned_.end()) {
      it =
          str_proxy_translation_maps_owned_
              .emplace(map_key, source_proxy->buildTranslationMapToOtherProxy(dest_proxy))
              .first;
    }
    return it->second.get();
  }

  StringDictionaryProxy* getStringDictProxy(const int dict_id) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = str_dict_proxy_owned_.find(dict_id);
    CHECK(it != str_dict_proxy_owned_.end());
    return it->second.get();
  }

  StringDictionaryProxy* getOrAddStringDictProxy(const int db_id,
                                                 const int dict_id_in,
                                                 const bool with_generation);

  void addLiteralStringDictProxy(
      std::shared_ptr<StringDictionaryProxy> lit_str_dict_proxy) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    lit_str_dict_proxy_ = lit_str_dict_proxy;
  }

  StringDictionaryProxy* getLiteralStringDictProxy() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return lit_str_dict_proxy_.get();
  }

  StringDictionaryProxyTranslationMap* getOrAddStringProxyTranslationMap(
      const int db_id,
      const int source_dict_id_in,
      const int dest_dict_id_in,
      const bool with_generation);

  void addColBuffer(const void* col_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    col_buffers_.push_back(const_cast<void*>(col_buffer));
  }

  ~RowSetMemoryOwner() {
    for (auto count_distinct_set : count_distinct_sets_) {
      delete count_distinct_set;
    }
    for (auto group_by_buffer : group_by_buffers_) {
      free(group_by_buffer);
    }
    for (auto varlen_buffer : varlen_buffers_) {
      free(varlen_buffer);
    }
    for (auto varlen_input_buffer : varlen_input_buffers_) {
      CHECK(varlen_input_buffer);
      varlen_input_buffer->unPin();
    }
    for (auto col_buffer : col_buffers_) {
      free(col_buffer);
    }
  }

  std::shared_ptr<RowSetMemoryOwner> cloneStrDictDataOnly() {
    auto rtn = std::make_shared<RowSetMemoryOwner>(
        data_provider_, arena_block_size_, /*num_kernels=*/1);
    rtn->str_dict_proxy_owned_ = str_dict_proxy_owned_;
    rtn->lit_str_dict_proxy_ = lit_str_dict_proxy_;
    return rtn;
  }

  void setDictionaryGenerations(StringDictionaryGenerations generations) {
    string_dictionary_generations_ = generations;
  }

  StringDictionaryGenerations& getStringDictionaryGenerations() {
    return string_dictionary_generations_;
  }

  quantile::TDigest* nullTDigest(double const q);

 private:
  struct CountDistinctBitmapBuffer {
    int8_t* ptr;
    const size_t size;
    const bool physical_buffer;
  };

  std::vector<CountDistinctBitmapBuffer> count_distinct_bitmaps_;
  std::vector<robin_hood::unordered_set<int64_t>*> count_distinct_sets_;
  std::vector<int64_t*> group_by_buffers_;
  std::vector<void*> varlen_buffers_;
  std::list<std::string> strings_;
  std::list<std::vector<int64_t>> arrays_;
  std::unordered_map<int, std::shared_ptr<StringDictionaryProxy>> str_dict_proxy_owned_;
  std::map<std::pair<int, int>, std::shared_ptr<StringDictionaryProxyTranslationMap>>
      str_proxy_translation_maps_owned_;
  std::shared_ptr<StringDictionaryProxy> lit_str_dict_proxy_;
  StringDictionaryGenerations string_dictionary_generations_;
  std::vector<void*> col_buffers_;
  std::vector<Data_Namespace::AbstractBuffer*> varlen_input_buffers_;
  std::vector<std::unique_ptr<quantile::TDigest>> t_digests_;

  DataProvider* data_provider_;  // for metadata lookups
  size_t arena_block_size_;      // for cloning
  std::vector<std::unique_ptr<Arena>> allocators_;

  mutable std::mutex state_mutex_;

  friend class ResultSet;
  friend class QueryExecutionContext;
};
