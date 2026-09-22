/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/DatumFetchers.h"
#include "StringDictionary/StringDictionaryProxy.h"
#include "StringOps/StringOps.h"

#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_arena.h>
#include <algorithm>
#include <bitset>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/sort/spreadsort/string_sort.hpp>
#include <functional>
#include <future>
#include <iostream>
#include <numeric>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

// TODO(adb): fixup
#include <sys/fcntl.h>

#include "Logger/Logger.h"
#include "Shared/heavyai_fs.h"
#include "Shared/measure.h"
#include "Shared/sqltypes.h"
#include "Shared/thread_count.h"
#include "Utils/Regexp.h"
#include "Utils/StringLike.h"

bool g_cache_string_hash{true};
size_t g_max_concurrent_llm_transform_call{16};
int64_t g_llm_transform_max_num_unique_value{1000};

namespace {

const int SYSTEM_PAGE_SIZE = heavyai::get_page_size();

int checked_open(const char* path, const bool recover) {
  auto fd = heavyai::open(path, O_RDWR | O_CREAT | (recover ? O_APPEND : O_TRUNC), 0644);
  if (fd > 0) {
    return fd;
  }
  auto err = std::string("Dictionary path ") + std::string(path) +
             std::string(" does not exist.");
  LOG(ERROR) << err;
  throw DictPayloadUnavailable(err);
}

const uint64_t round_up_p2(const uint64_t num) {
  uint64_t in = num;
  in--;
  in |= in >> 1;
  in |= in >> 2;
  in |= in >> 4;
  in |= in >> 8;
  in |= in >> 16;
  in++;
  // TODO MAT deal with case where filesize has been increased but reality is
  // we are constrained to 2^31.
  // In that situation this calculation will wrap to zero
  if (in == 0 || (in > (UINT32_MAX))) {
    in = UINT32_MAX;
  }
  return in;
}

string_dict_hash_t hash_string(const std::string_view& str) {
  string_dict_hash_t str_hash = 1;
  // rely on fact that unsigned overflow is defined and wraps
  for (size_t i = 0; i < str.size(); ++i) {
    str_hash = str_hash * 997 + str[i];
  }
  return str_hash;
}

class PercentLiteralLikeMatcher {
 public:
  PercentLiteralLikeMatcher(const std::string_view pattern, const char escape)
      : pattern_(pattern)
      , has_percent_(pattern.find('%') != std::string_view::npos)
      , leading_percent_(!pattern.empty() && pattern.front() == '%')
      , trailing_percent_(!pattern.empty() && pattern.back() == '%') {
    if (pattern.find(escape) != std::string_view::npos ||
        pattern.find('_') != std::string_view::npos ||
        pattern.find('[') != std::string_view::npos) {
      return;
    }

    size_t begin = 0;
    while (begin < pattern.size()) {
      const auto end = pattern.find('%', begin);
      const auto literal_end = end == std::string_view::npos ? pattern.size() : end;
      if (literal_end > begin) {
        literals_.push_back(pattern.substr(begin, literal_end - begin));
      }
      if (end == std::string_view::npos) {
        break;
      }
      begin = end + 1;
    }
    supported_ = true;
  }

  bool supported() const { return supported_; }

  bool matches(const std::string_view str) const {
    if (!has_percent_) {
      return str == pattern_;
    }
    if (literals_.empty()) {
      return true;
    }

    size_t search_begin = 0;
    size_t search_end = str.size();
    size_t first_literal = 0;
    size_t last_literal = literals_.size();

    if (!leading_percent_) {
      const auto prefix = literals_.front();
      if (str.size() < prefix.size() || str.substr(0, prefix.size()) != prefix) {
        return false;
      }
      search_begin = prefix.size();
      ++first_literal;
    }

    if (!trailing_percent_) {
      const auto suffix = literals_.back();
      if (str.size() < suffix.size() ||
          str.substr(str.size() - suffix.size()) != suffix) {
        return false;
      }
      search_end = str.size() - suffix.size();
      --last_literal;
    }

    if (search_begin > search_end) {
      return false;
    }
    for (size_t literal_index = first_literal; literal_index < last_literal;
         ++literal_index) {
      const auto literal = literals_[literal_index];
      const auto position = str.find(literal, search_begin);
      if (position == std::string_view::npos || position > search_end ||
          literal.size() > search_end - position) {
        return false;
      }
      search_begin = position + literal.size();
    }
    return search_begin <= search_end;
  }

 private:
  std::string_view pattern_;
  std::vector<std::string_view> literals_;
  bool has_percent_;
  bool leading_percent_;
  bool trailing_percent_;
  bool supported_{false};
};

struct ThreadInfo {
  int64_t num_threads{0};
  int64_t num_elems_per_thread;

  ThreadInfo(const int64_t max_thread_count,
             const int64_t num_elems,
             const int64_t target_elems_per_thread) {
    num_threads =
        std::min(std::max(max_thread_count, int64_t(1)),
                 ((num_elems + target_elems_per_thread - 1) / target_elems_per_thread));
    num_elems_per_thread =
        std::max((num_elems + num_threads - 1) / num_threads, int64_t(1));
  }
};

void try_parallelize_llm_transform(ThreadInfo& thread_info,
                                   const StringOps_Namespace::StringOps& string_ops,
                                   int num_source_strings) {
  const auto& ops = string_ops.getStringOps();
  auto const has_llm_transform_expr =
      std::any_of(ops.begin(), ops.end(), [](auto const& op) {
        return op->getOpInfo().getOpKind() == SqlStringOpKind::LLM_TRANSFORM;
      });
  if (has_llm_transform_expr) {
    if (num_source_strings > g_llm_transform_max_num_unique_value) {
      std::ostringstream oss;
      oss << "The number of entries of a string dictionary of the input argument of the "
             "LLM_TRANSFORM (="
          << num_source_strings
          << ") is larger than a threshold (=" << g_llm_transform_max_num_unique_value
          << ")";
      throw std::runtime_error(oss.str());
    }
    thread_info.num_threads = g_max_concurrent_llm_transform_call;
    thread_info.num_elems_per_thread = 1;
  }
}

}  // namespace

bool SortedStringPermutation::operator()(const int32_t lhs, const int32_t rhs) const {
  if (lhs == rhs) {
    return false;
  }

  if (lhs == inline_int_null_value<int32_t>()) {
    return !sort_descending;
  }
  if (rhs == inline_int_null_value<int32_t>()) {
    return sort_descending;
  }

  const std::pair<int32_t, int32_t> lhs_permutation = get_permutation(lhs);
  const std::pair<int32_t, int32_t> rhs_permutation = get_permutation(rhs);

  bool lhs_lt_rhs = lhs_permutation.first != rhs_permutation.first
                        ? lhs_permutation.first < rhs_permutation.first
                        : lhs_permutation.second < rhs_permutation.second;

  return sort_descending != lhs_lt_rhs;
}

bool g_enable_stringdict_parallel{false};
bool g_enable_stringdict_parallel_sort{false};
bool g_enable_lazy_string_dictionary_hash_recovery{false};
constexpr int32_t StringDictionary::INVALID_STR_ID;
constexpr size_t StringDictionary::MAX_STRLEN;
constexpr size_t StringDictionary::MAX_STRCOUNT;

StringDictionary::StringDictionary(const shared::StringDictKey& dict_key,
                                   const std::string& folder,
                                   const bool isTemp,
                                   const bool recover,
                                   const bool materializeHashes,
                                   size_t initial_capacity)
    : dict_key_(dict_key)
    , folder_(folder)
    , str_count_(0)
    , string_id_string_dict_hash_table_(initial_capacity, INVALID_STR_ID)
    , hash_cache_(initial_capacity)
    , isTemp_(isTemp)
    , materialize_hashes_(materializeHashes)
    , payload_fd_(-1)
    , offset_fd_(-1)
    , offset_map_(nullptr)
    , payload_map_(nullptr)
    , offset_file_size_(0)
    , payload_file_size_(0)
    , payload_file_off_(0)
    , like_cache_size_(0)
    , regex_cache_size_(0)
    , equal_cache_size_(0)
    , compare_cache_size_(0)
    , strings_cache_(nullptr)
    , strings_cache_size_(0) {
  if (!isTemp && folder.empty()) {
    return;
  }

  // initial capacity must be a power of two for efficient bucket computation
  CHECK_EQ(size_t(0), (initial_capacity & (initial_capacity - 1)));
  if (!isTemp_) {
    boost::filesystem::path storage_path(folder);
    offsets_path_ = (storage_path / boost::filesystem::path("DictOffsets")).string();
    const auto payload_path =
        (storage_path / boost::filesystem::path("DictPayload")).string();
    payload_fd_ = checked_open(payload_path.c_str(), recover);
    offset_fd_ = checked_open(offsets_path_.c_str(), recover);
    payload_file_size_ = heavyai::file_size(payload_fd_);
    offset_file_size_ = heavyai::file_size(offset_fd_);
  }
  bool storage_is_empty = false;
  if (payload_file_size_ == 0) {
    storage_is_empty = true;
    addPayloadCapacity();
  }
  if (offset_file_size_ == 0) {
    addOffsetCapacity();
  }
  if (!isTemp_) {  // we never mmap or recover temp dictionaries
    payload_map_ =
        reinterpret_cast<char*>(heavyai::checked_mmap(payload_fd_, payload_file_size_));
    offset_map_ = reinterpret_cast<StringIdxEntry*>(
        heavyai::checked_mmap(offset_fd_, offset_file_size_));
    total_mmap_size += payload_file_size_ + offset_file_size_;
    if (recover) {
      const size_t bytes = heavyai::file_size(offset_fd_);
      if (bytes % sizeof(StringIdxEntry) != 0) {
        LOG(WARNING) << "Offsets " << offsets_path_ << " file is truncated";
      }
      str_count_ =
          storage_is_empty ? 0 : getNumStringsFromStorage(bytes / sizeof(StringIdxEntry));
      if (str_count_ > 0) {
        const auto& final_entry = offset_map_[str_count_ - 1];
        payload_file_off_ = final_entry.off + final_entry.size;
      }
      if (g_enable_lazy_string_dictionary_hash_recovery && str_count_ > 0) {
        hash_table_recovered_.store(false, std::memory_order_relaxed);
        return;
      }
      recoverHashTableFromStorageUnlocked();
      VLOG(1) << "Opened string dictionary " << folder << " # Strings: " << str_count_
              << " Hash table size: " << string_id_string_dict_hash_table_.size()
              << " Fill rate: "
              << static_cast<double>(str_count_) * 100.0 /
                     string_id_string_dict_hash_table_.size()
              << "% Collisions: " << collisions_;
    }
  }
}

// Call serial_callback for each (string_view, string_id). Must be called serially.
void StringDictionary::eachStringSerially(int64_t const generation,
                                          StringCallback& serial_callback) const {
  size_t const n = std::min(static_cast<size_t>(generation), str_count_);
  CHECK_LE(n, static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1);
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  for (unsigned id = 0; id < n; ++id) {
    serial_callback(getStringFromStorageFast(static_cast<int>(id)), id);
  }
}

void StringDictionary::processDictionaryFutures(
    std::vector<std::future<std::vector<std::pair<string_dict_hash_t, unsigned int>>>>&
        dictionary_futures) {
  for (auto& dictionary_future : dictionary_futures) {
    dictionary_future.wait();
    const auto hashVec = dictionary_future.get();
    for (const auto& hash : hashVec) {
      const uint32_t bucket =
          computeUniqueBucketWithHash(hash.first, string_id_string_dict_hash_table_);
      string_id_string_dict_hash_table_[bucket] = static_cast<int32_t>(hash.second);
      if (materialize_hashes_) {
        hash_cache_[hash.second] = hash.first;
      }
    }
  }
  dictionary_futures.clear();
}

void StringDictionary::recoverHashTableFromStorageUnlocked() {
  collisions_ = 0;
  const uint64_t max_entries = std::max(
      round_up_p2(str_count_ * 2 + 1),
      round_up_p2(std::max(string_id_string_dict_hash_table_.size(), size_t(1))));
  std::vector<int32_t> new_str_ids(max_entries, INVALID_STR_ID);
  string_id_string_dict_hash_table_.swap(new_str_ids);
  if (materialize_hashes_) {
    std::vector<string_dict_hash_t> new_hash_cache(max_entries / 2);
    hash_cache_.swap(new_hash_cache);
  }
  if (str_count_ == 0) {
    hash_table_recovered_.store(true, std::memory_order_release);
    return;
  }

  const auto thread_count = std::max(1u, std::thread::hardware_concurrency());
  const uint32_t items_per_thread = std::max<uint32_t>(
      2000, std::min<uint32_t>(200000, (str_count_ / thread_count) + 1));
  std::vector<std::future<std::vector<std::pair<string_dict_hash_t, unsigned int>>>>
      dictionary_futures;
  uint32_t thread_inits{0};
  for (uint32_t string_id = 0; string_id < str_count_; string_id += items_per_thread) {
    dictionary_futures.emplace_back(
        std::async(std::launch::async, [string_id, items_per_thread, this] {
          std::vector<std::pair<string_dict_hash_t, unsigned int>> hash_vec;
          for (uint32_t curr_id = string_id;
               curr_id < string_id + items_per_thread && curr_id < str_count_;
               ++curr_id) {
            const auto recovered = getStringFromStorage(curr_id);
            if (recovered.canary) {
              break;
            }
            const std::string_view string_view(recovered.c_str_ptr, recovered.size);
            hash_vec.emplace_back(hash_string(string_view), curr_id);
          }
          return hash_vec;
        }));
    if (++thread_inits % thread_count == 0) {
      processDictionaryFutures(dictionary_futures);
    }
  }
  if (!dictionary_futures.empty()) {
    processDictionaryFutures(dictionary_futures);
  }
  hash_table_recovered_.store(true, std::memory_order_release);
}

void StringDictionary::ensureHashTableRecovered() const {
  if (hash_table_recovered_.load(std::memory_order_acquire)) {
    return;
  }
  auto mutable_this = const_cast<StringDictionary*>(this);
  std::lock_guard<std::shared_mutex> write_lock(mutable_this->rw_mutex_);
  if (!mutable_this->hash_table_recovered_.load(std::memory_order_relaxed)) {
    mutable_this->recoverHashTableFromStorageUnlocked();
  }
}

bool StringDictionary::isHashTableRecovered() const noexcept {
  return hash_table_recovered_.load(std::memory_order_acquire);
}

size_t StringDictionary::lookupStringsByScanWithoutHash(
    const std::vector<std::string_view>& lookup_strings,
    int32_t* string_ids,
    const int64_t generation) const {
  CHECK(string_ids);
  if (lookup_strings.empty()) {
    return 0;
  }

  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  const int64_t dictionary_generation = generation >= 0 ? generation : str_count_;
  CHECK_GE(dictionary_generation, 0L);
  CHECK_LE(dictionary_generation, static_cast<int64_t>(str_count_));

  std::unordered_map<std::string_view, size_t> candidate_index;
  candidate_index.reserve(lookup_strings.size());
  std::vector<std::string_view> candidates;
  candidates.reserve(lookup_strings.size());
  for (size_t lookup_idx = 0; lookup_idx < lookup_strings.size(); ++lookup_idx) {
    const auto lookup_string = lookup_strings[lookup_idx];
    if (lookup_string.empty()) {
      string_ids[lookup_idx] = inline_int_null_value<int32_t>();
      continue;
    }
    const auto [candidate_it, inserted] =
        candidate_index.emplace(lookup_string, candidates.size());
    if (inserted) {
      candidates.push_back(lookup_string);
    }
  }

  std::vector<int32_t> candidate_ids(candidates.size(), INVALID_STR_ID);
  std::vector<bool> candidate_is_cached(candidates.size(), false);
  constexpr size_t max_cacheable_lookup_batch{1024};
  const bool use_scan_lookup_cache = candidates.size() <= max_cacheable_lookup_batch;
  if (use_scan_lookup_cache) {
    std::lock_guard<std::mutex> cache_lock(scan_lookup_cache_mutex_);
    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
      const auto cache_it = scan_lookup_cache_.find(candidates[candidate_idx]);
      if (cache_it != scan_lookup_cache_.end() &&
          cache_it->second.generation == dictionary_generation) {
        candidate_ids[candidate_idx] = cache_it->second.string_id;
        candidate_is_cached[candidate_idx] = true;
      }
    }
  }

  std::unordered_map<std::string_view, size_t> scan_candidate_index;
  scan_candidate_index.reserve(candidates.size());
  std::unordered_set<size_t> candidate_lengths;
  candidate_lengths.reserve(candidates.size());
  for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
    if (!candidate_is_cached[candidate_idx]) {
      scan_candidate_index.emplace(candidates[candidate_idx], candidate_idx);
      candidate_lengths.insert(candidates[candidate_idx].size());
    }
  }

  if (!scan_candidate_index.empty() && dictionary_generation > 0) {
    constexpr int64_t target_strings_per_thread{200000};
    ThreadInfo thread_info(std::thread::hardware_concurrency(),
                           dictionary_generation,
                           target_strings_per_thread);
    tbb::concurrent_vector<std::pair<size_t, int32_t>> persisted_matches;
    tbb::task_arena limited_arena(thread_info.num_threads);
    limited_arena.execute([&] {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(
              0, dictionary_generation, thread_info.num_elems_per_thread),
          [&](const tbb::blocked_range<int32_t>& range) {
            for (int32_t string_id = range.begin(); string_id != range.end();
                 ++string_id) {
              if (!candidate_lengths.count(offset_map_[string_id].size)) {
                continue;
              }
              const auto candidate_it =
                  scan_candidate_index.find(getStringFromStorageFast(string_id));
              if (candidate_it != scan_candidate_index.end()) {
                persisted_matches.emplace_back(candidate_it->second, string_id);
              }
            }
          },
          tbb::simple_partitioner());
    });
    for (const auto& [candidate_idx, persisted_id] : persisted_matches) {
      auto& candidate_id = candidate_ids[candidate_idx];
      candidate_id = candidate_id == INVALID_STR_ID
                         ? persisted_id
                         : std::min(candidate_id, persisted_id);
    }
  }

  if (use_scan_lookup_cache) {
    constexpr size_t max_scan_lookup_cache_entries{4096};
    std::lock_guard<std::mutex> cache_lock(scan_lookup_cache_mutex_);
    if (scan_lookup_cache_.size() + candidates.size() > max_scan_lookup_cache_entries) {
      scan_lookup_cache_.clear();
      scan_lookup_cache_size_ = 0;
    }
    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
      const auto [cache_it, inserted] = scan_lookup_cache_.insert_or_assign(
          std::string(candidates[candidate_idx]),
          ScanLookupCacheEntry{dictionary_generation, candidate_ids[candidate_idx]});
      if (inserted) {
        scan_lookup_cache_size_ += cache_it->first.size() + sizeof(ScanLookupCacheEntry);
      }
    }
  }

  size_t num_strings_not_found{0};
  for (size_t lookup_idx = 0; lookup_idx < lookup_strings.size(); ++lookup_idx) {
    const auto lookup_string = lookup_strings[lookup_idx];
    if (lookup_string.empty()) {
      continue;
    }
    const auto candidate_it = candidate_index.find(lookup_string);
    CHECK(candidate_it != candidate_index.end());
    string_ids[lookup_idx] = candidate_ids[candidate_it->second];
    num_strings_not_found += string_ids[lookup_idx] == INVALID_STR_ID;
  }
  return num_strings_not_found;
}

const shared::StringDictKey& StringDictionary::getDictKey() const noexcept {
  return dict_key_;
}

/**
 * Method to retrieve number of strings in storage via a binary search for the first
 * canary
 * @param storage_slots number of storage entries we should search to find the minimum
 * canary
 * @return number of strings in storage
 */
size_t StringDictionary::getNumStringsFromStorage(
    const size_t storage_slots) const noexcept {
  if (storage_slots == 0) {
    return 0;
  }
  // Must use signed integers since final binary search step can wrap to max size_t value
  // if dictionary is empty
  int64_t min_bound = 0;
  int64_t max_bound = storage_slots - 1;
  int64_t guess{0};
  while (min_bound <= max_bound) {
    guess = (max_bound + min_bound) / 2;
    CHECK_GE(guess, 0);
    if (getStringFromStorage(guess).canary) {
      max_bound = guess - 1;
    } else {
      min_bound = guess + 1;
    }
  }
  CHECK_GE(guess + (min_bound > guess ? 1 : 0), 0);
  return guess + (min_bound > guess ? 1 : 0);
}

StringDictionary::~StringDictionary() noexcept {
  if (payload_map_) {
    if (!isTemp_) {
      CHECK(offset_map_);
      heavyai::checked_munmap(payload_map_, payload_file_size_);
      heavyai::checked_munmap(offset_map_, offset_file_size_);
      total_mmap_size -= payload_file_size_ + offset_file_size_;
      CHECK_GE(payload_fd_, 0);
      heavyai::close(payload_fd_);
      CHECK_GE(offset_fd_, 0);
      heavyai::close(offset_fd_);
    } else {
      CHECK(offset_map_);
      free(payload_map_);
      free(offset_map_);
      total_temp_size -= payload_file_size_ + offset_file_size_;
    }
  }
}

int32_t StringDictionary::getOrAdd(const std::string& str) noexcept {
  ensureHashTableRecovered();
  return getOrAddImpl(str);
}

std::vector<std::string> StringDictionary::getStringsForRange(
    int32_t start_id,
    int32_t end_id,
    const StringOps_Namespace::StringOps& string_ops,
    const std::function<bool(int32_t)>& mask_functor) const {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_LE(start_id, end_id);
  std::vector<std::string> result(end_id - start_id);
  if (start_id == end_id) {
    return result;
  }

  const bool has_string_ops = string_ops.size() > 0;

  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  tbb::parallel_for(tbb::blocked_range<int32_t>(start_id, end_id),
                    [&](const tbb::blocked_range<int32_t>& range) {
                      for (int32_t id = range.begin(); id != range.end(); ++id) {
                        if (mask_functor(id)) {
                          if (id >= 0 && static_cast<size_t>(id) < str_count_) {
                            std::string str = getStringUnlocked(id);
                            if (has_string_ops) {
                              str = string_ops(str);
                            }
                            result[id - start_id] = std::move(str);
                          }
                        }
                      }
                    });

  return result;
}

void StringDictionary::fillStringOpUnionTranslationMap(
    int32_t* translated_ids,
    const int64_t source_generation,
    const StringOps_Namespace::StringOps& string_ops,
    const std::function<bool(int32_t)>& mask_functor,
    const StringAddCallback& add_transient_callback,
    const StringIdLookupCallback& lookup_transient_callback) const {
  CHECK(translated_ids);
  CHECK_GE(source_generation, 0L);
  if (source_generation == 0L) {
    return;
  }

  tbb::concurrent_unordered_set<std::string> unique_strings;
  const bool has_string_ops = string_ops.size() > 0;
  constexpr int64_t target_strings_per_thread{1000};
  ThreadInfo thread_info(
      std::thread::hardware_concurrency(), source_generation, target_strings_per_thread);
  try_parallelize_llm_transform(thread_info, string_ops, source_generation);
  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);

  auto process_source_string = [&](const int32_t source_string_id,
                                   std::string& string_ops_storage) {
    const auto source_string = getStringFromStorageFast(source_string_id);
    return has_string_ops ? string_ops(source_string, string_ops_storage) : source_string;
  };

  tbb::task_arena limited_arena(thread_info.num_threads);
  limited_arena.execute([&] {
    std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
    tbb::parallel_for(
        tbb::blocked_range<int32_t>(
            0, source_generation, thread_info.num_elems_per_thread /* tbb grain_size */),
        [&](const tbb::blocked_range<int32_t>& range) {
          std::string string_ops_storage;
          for (int32_t source_string_id = range.begin(); source_string_id != range.end();
               ++source_string_id) {
            if (!mask_functor(source_string_id)) {
              continue;
            }
            const auto processed_string =
                process_source_string(source_string_id, string_ops_storage);
            if (!processed_string.empty()) {
              unique_strings.insert(std::string(processed_string));
            }
          }
        },
        tbb::simple_partitioner());
  });

  for (const auto& str : unique_strings) {
    add_transient_callback(str);
  }

  limited_arena.execute([&] {
    std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
    tbb::parallel_for(
        tbb::blocked_range<int32_t>(
            0, source_generation, thread_info.num_elems_per_thread /* tbb grain_size */),
        [&](const tbb::blocked_range<int32_t>& range) {
          std::string string_ops_storage;
          for (int32_t source_string_id = range.begin(); source_string_id != range.end();
               ++source_string_id) {
            if (!mask_functor(source_string_id)) {
              continue;
            }
            const auto processed_string =
                process_source_string(source_string_id, string_ops_storage);
            translated_ids[source_string_id] =
                processed_string.empty() ? inline_int_null_value<int32_t>()
                                         : lookup_transient_callback(processed_string);
          }
        },
        tbb::simple_partitioner());
  });
}

bool StringDictionary::tryBuildSelfStringOpUnionTranslationMapWithoutHash(
    int32_t* translated_ids,
    const int64_t generation,
    const StringOps_Namespace::StringOps& string_ops,
    const StringAddCallback& add_transient_callback,
    const StringIdLookupCallback& lookup_transient_callback,
    size_t& num_untranslated_strings) const {
  CHECK(translated_ids);
  CHECK_GE(generation, 0L);
  CHECK_GT(string_ops.size(), 0UL);
  CHECK_LE(generation, static_cast<int64_t>(str_count_));
  if (generation == 0 || isHashTableRecovered()) {
    num_untranslated_strings = 0;
    return generation == 0;
  }

  const auto& string_op_pipeline = string_ops.getStringOps();
  if (std::any_of(string_op_pipeline.begin(),
                  string_op_pipeline.end(),
                  [](const auto& string_op) {
                    return string_op->getOpInfo().getOpKind() ==
                           SqlStringOpKind::LLM_TRANSFORM;
                  })) {
    return false;
  }

  constexpr size_t max_unique_transformed_strings{1'000'000};
  constexpr int64_t target_strings_per_thread{1000};
  ThreadInfo thread_info(
      std::thread::hardware_concurrency(), generation, target_strings_per_thread);
  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);

  using CandidateIndexMap = tbb::concurrent_hash_map<std::string, int32_t>;
  using LocalCandidateIndexMap = std::unordered_map<std::string_view, int32_t>;
  using SourceLengthSet = std::bitset<MAX_STRLEN + 1>;
  CandidateIndexMap candidate_indices;
  tbb::concurrent_vector<std::string> candidates;
  tbb::enumerable_thread_specific<LocalCandidateIndexMap> local_candidate_indices;
  tbb::enumerable_thread_specific<SourceLengthSet> source_lengths;
  std::atomic<bool> candidate_limit_exceeded{false};
  tbb::task_arena limited_arena(thread_info.num_threads);
  {
    std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
    limited_arena.execute([&] {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(0, generation, thread_info.num_elems_per_thread),
          [&](const tbb::blocked_range<int32_t>& range) {
            constexpr size_t max_local_candidates{4096};
            auto& local_candidates = local_candidate_indices.local();
            auto& local_source_lengths = source_lengths.local();
            std::string string_ops_storage;
            for (int32_t string_id = range.begin(); string_id != range.end();
                 ++string_id) {
              if (candidate_limit_exceeded.load(std::memory_order_relaxed)) {
                break;
              }
              const auto source_string = getStringFromStorageFast(string_id);
              local_source_lengths.set(source_string.size());
              const auto transformed_string =
                  string_ops.evalViewOrCopy(source_string, string_ops_storage);
              if (transformed_string.empty()) {
                translated_ids[string_id] = inline_int_null_value<int32_t>();
                continue;
              }

              auto local_candidate_it = local_candidates.find(transformed_string);
              int32_t candidate_idx{INVALID_STR_ID};
              if (local_candidate_it != local_candidates.end()) {
                candidate_idx = local_candidate_it->second;
              } else {
                const std::string transformed_key(transformed_string);
                CandidateIndexMap::const_accessor read_accessor;
                if (candidate_indices.find(read_accessor, transformed_key)) {
                  candidate_idx = read_accessor->second;
                  read_accessor.release();
                } else {
                  CandidateIndexMap::accessor write_accessor;
                  if (candidate_indices.insert(write_accessor, transformed_key)) {
                    const auto candidate_it = candidates.push_back(write_accessor->first);
                    const auto new_candidate_idx = candidate_it - candidates.begin();
                    if (new_candidate_idx >= static_cast<decltype(new_candidate_idx)>(
                                                 max_unique_transformed_strings)) {
                      write_accessor->second = INVALID_STR_ID;
                      candidate_limit_exceeded.store(true, std::memory_order_relaxed);
                      break;
                    }
                    write_accessor->second = static_cast<int32_t>(new_candidate_idx);
                  }
                  candidate_idx = write_accessor->second;
                  write_accessor.release();
                }
                if (candidate_idx == INVALID_STR_ID) {
                  candidate_limit_exceeded.store(true, std::memory_order_relaxed);
                  break;
                }
                if (local_candidates.size() < max_local_candidates) {
                  local_candidates.emplace(candidates[candidate_idx], candidate_idx);
                }
              }
              translated_ids[string_id] = candidate_idx;
            }
          },
          tbb::simple_partitioner());
    });
  }
  if (candidate_limit_exceeded.load(std::memory_order_relaxed)) {
    return false;
  }
  local_candidate_indices.clear();
  candidate_indices.clear();

  std::vector<int32_t> candidate_ids(candidates.size(), INVALID_STR_ID);
  std::vector<std::string_view> candidate_views;
  candidate_views.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    candidate_views.emplace_back(candidate);
  }
  SourceLengthSet observed_source_lengths;
  for (const auto& local_source_lengths : source_lengths) {
    observed_source_lengths |= local_source_lengths;
  }
  const bool candidate_may_be_persisted = std::any_of(
      candidate_views.begin(), candidate_views.end(), [&](const auto candidate) {
        return candidate.size() <= MAX_STRLEN &&
               observed_source_lengths.test(candidate.size());
      });
  if (candidate_may_be_persisted) {
    lookupStringsByScanWithoutHash(candidate_views, candidate_ids.data(), generation);
  }

  std::vector<uint8_t> candidate_was_initially_available(candidates.size(), false);
  size_t num_new_transients{0};
  for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
    if (candidate_ids[candidate_idx] != INVALID_STR_ID) {
      candidate_was_initially_available[candidate_idx] = true;
      continue;
    }
    const auto transient_id = lookup_transient_callback(candidates[candidate_idx]);
    if (transient_id != INVALID_STR_ID) {
      candidate_ids[candidate_idx] = transient_id;
      candidate_was_initially_available[candidate_idx] = true;
    } else {
      ++num_new_transients;
    }
  }
  constexpr size_t max_allowed_transients =
      static_cast<size_t>(std::numeric_limits<int32_t>::max() - 2);
  if (num_new_transients > max_allowed_transients) {
    throw std::runtime_error(
        "Self string-operation translation exceeds the transient string ID domain");
  }

  for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
    if (candidate_ids[candidate_idx] == INVALID_STR_ID) {
      add_transient_callback(candidates[candidate_idx]);
      candidate_ids[candidate_idx] = lookup_transient_callback(candidates[candidate_idx]);
      CHECK_LT(candidate_ids[candidate_idx], INVALID_STR_ID);
    }
  }

  std::atomic<size_t> untranslated_count{0};
  limited_arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<int32_t>(0, generation, thread_info.num_elems_per_thread),
        [&](const tbb::blocked_range<int32_t>& range) {
          size_t local_untranslated_count{0};
          for (int32_t string_id = range.begin(); string_id != range.end(); ++string_id) {
            const auto candidate_idx = translated_ids[string_id];
            if (candidate_idx == inline_int_null_value<int32_t>()) {
              continue;
            }
            CHECK_GE(candidate_idx, 0);
            CHECK_LT(static_cast<size_t>(candidate_idx), candidate_ids.size());
            translated_ids[string_id] = candidate_ids[candidate_idx];
            local_untranslated_count += !candidate_was_initially_available[candidate_idx];
          }
          untranslated_count.fetch_add(local_untranslated_count,
                                       std::memory_order_relaxed);
        },
        tbb::simple_partitioner());
  });
  num_untranslated_strings = untranslated_count.load(std::memory_order_relaxed);
  return true;
}

namespace {

template <class T>
void throw_encoding_error(std::string_view str, const shared::StringDictKey& dict_key) {
  std::ostringstream oss;
  oss << "The text encoded column using dictionary " << dict_key
      << " has exceeded it's limit of " << sizeof(T) * 8 << " bits ("
      << static_cast<size_t>(max_valid_int_value<T>() + 1) << " unique values) "
      << "while attempting to add the new string '" << str << "'. ";

  if (sizeof(T) < 4) {
    // Todo: Implement automatic type widening for dictionary-encoded text
    // columns/all fixed length columm types (at least if not defined
    //  with fixed encoding size), or short of that, ALTER TABLE
    // COLUMN TYPE to at least allow the user to do this manually
    // without re-creating the table

    oss << "To load more data, please re-create the table with "
        << "this column as type TEXT ENCODING DICT(" << sizeof(T) * 2 * 8 << ") ";
    if (sizeof(T) == 1) {
      oss << "or TEXT ENCODING DICT(32) ";
    }
    oss << "and reload your data.";
  } else {
    // Todo: Implement TEXT ENCODING DICT(64) type which should essentially
    // preclude overflows.
    oss << "Currently dictionary-encoded text columns support a maximum of "
        << StringDictionary::MAX_STRCOUNT
        << " strings. Consider recreating the table with "
        << "this column as type TEXT ENCODING NONE and reloading your data.";
  }
  LOG(ERROR) << oss.str();
  throw std::runtime_error(oss.str());
}

void throw_string_too_long_error(std::string_view str,
                                 const shared::StringDictKey& dict_key) {
  std::ostringstream oss;
  oss << "The string '" << str << " could not be inserted into the dictionary "
      << dict_key << " because it exceeded the maximum allowable "
      << "length of " << StringDictionary::MAX_STRLEN << " characters (string was "
      << str.size() << " characters).";
  LOG(ERROR) << oss.str();
  throw std::runtime_error(oss.str());
}

}  // namespace

template <class String>
void StringDictionary::getOrAddBulkArray(
    const std::vector<std::vector<String>>& string_array_vec,
    std::vector<std::vector<int32_t>>& ids_array_vec) {
  ids_array_vec.resize(string_array_vec.size());
  for (size_t i = 0; i < string_array_vec.size(); i++) {
    auto& strings = string_array_vec[i];
    auto& ids = ids_array_vec[i];
    ids.resize(strings.size());
    getOrAddBulk(strings, &ids[0]);
  }
}

template void StringDictionary::getOrAddBulkArray(
    const std::vector<std::vector<std::string>>& string_array_vec,
    std::vector<std::vector<int32_t>>& ids_array_vec);

template void StringDictionary::getOrAddBulkArray(
    const std::vector<std::vector<std::string_view>>& string_array_vec,
    std::vector<std::vector<int32_t>>& ids_array_vec);

/**
 * Method to hash a vector of strings in parallel.
 * @param string_vec input vector of strings to be hashed
 * @param hashes space for the output - should be pre-sized to match string_vec size
 */
template <class String>
void StringDictionary::hashStrings(
    const std::vector<String>& string_vec,
    std::vector<string_dict_hash_t>& hashes) const noexcept {
  CHECK_EQ(string_vec.size(), hashes.size());

  tbb::parallel_for(tbb::blocked_range<size_t>(0, string_vec.size()),
                    [&string_vec, &hashes](const tbb::blocked_range<size_t>& r) {
                      for (size_t curr_id = r.begin(); curr_id != r.end(); ++curr_id) {
                        if (string_vec[curr_id].empty()) {
                          continue;
                        }
                        hashes[curr_id] = hash_string(string_vec[curr_id]);
                      }
                    });
}

template <class T, class String>
size_t StringDictionary::getBulk(const std::vector<String>& string_vec,
                                 T* encoded_vec) const {
  return getBulk(string_vec, encoded_vec, -1L /* generation */);
}

template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint8_t* encoded_vec) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint16_t* encoded_vec) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          int32_t* encoded_vec) const;

template <class T, class String>
size_t StringDictionary::getBulk(const std::vector<String>& string_vec,
                                 T* encoded_vec,
                                 const int64_t generation) const {
  constexpr int64_t target_strings_per_thread{1000};
  const int64_t num_lookup_strings = string_vec.size();
  if (num_lookup_strings == 0) {
    return 0;
  }
  constexpr size_t max_scan_lookup_strings{1024};
  if (g_enable_lazy_string_dictionary_hash_recovery && !isHashTableRecovered() &&
      static_cast<size_t>(num_lookup_strings) <= max_scan_lookup_strings) {
    std::vector<std::string_view> lookup_strings;
    lookup_strings.reserve(num_lookup_strings);
    for (const auto& input_string : string_vec) {
      if (input_string.size() > StringDictionary::MAX_STRLEN) {
        throw_string_too_long_error(input_string, dict_key_);
      }
      lookup_strings.emplace_back(input_string);
    }
    std::vector<int32_t> scanned_ids(num_lookup_strings);
    const auto num_strings_not_found =
        lookupStringsByScanWithoutHash(lookup_strings, scanned_ids.data(), generation);
    for (int64_t string_idx = 0; string_idx < num_lookup_strings; ++string_idx) {
      encoded_vec[string_idx] =
          scanned_ids[string_idx] == inline_int_null_value<int32_t>()
              ? inline_int_null_value<T>()
              : static_cast<T>(scanned_ids[string_idx]);
    }
    return num_strings_not_found;
  }
  ensureHashTableRecovered();

  const ThreadInfo thread_info(
      std::thread::hardware_concurrency(), num_lookup_strings, target_strings_per_thread);
  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);

  std::vector<size_t> num_strings_not_found_per_thread(thread_info.num_threads, 0UL);

  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  const int64_t num_dict_strings = generation >= 0 ? generation : storageEntryCount();
  const bool dictionary_is_empty = (num_dict_strings == 0);
  if (dictionary_is_empty) {
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_lookup_strings),
                      [&](const tbb::blocked_range<int64_t>& r) {
                        const int64_t start_idx = r.begin();
                        const int64_t end_idx = r.end();
                        for (int64_t string_idx = start_idx; string_idx < end_idx;
                             ++string_idx) {
                          encoded_vec[string_idx] = StringDictionary::INVALID_STR_ID;
                        }
                      });
    return num_lookup_strings;
  }
  // If we're here the generation-capped dictionary has strings in it
  // that we need to look up against

  tbb::task_arena limited_arena(thread_info.num_threads);
  limited_arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(
            0, num_lookup_strings, thread_info.num_elems_per_thread /* tbb grain_size */),
        [&](const tbb::blocked_range<int64_t>& r) {
          const int64_t start_idx = r.begin();
          const int64_t end_idx = r.end();
          size_t num_strings_not_found = 0;
          for (int64_t string_idx = start_idx; string_idx != end_idx; ++string_idx) {
            const auto& input_string = string_vec[string_idx];
            if (input_string.empty()) {
              encoded_vec[string_idx] = inline_int_null_value<T>();
              continue;
            }
            if (input_string.size() > StringDictionary::MAX_STRLEN) {
              throw_string_too_long_error(input_string, dict_key_);
            }
            const string_dict_hash_t input_string_hash = hash_string(input_string);
            uint32_t hash_bucket = computeBucket(
                input_string_hash, input_string, string_id_string_dict_hash_table_);
            // Will either be legit id or INVALID_STR_ID
            const auto string_id = string_id_string_dict_hash_table_[hash_bucket];
            if (string_id == StringDictionary::INVALID_STR_ID ||
                string_id >= num_dict_strings) {
              encoded_vec[string_idx] = StringDictionary::INVALID_STR_ID;
              num_strings_not_found++;
              continue;
            }
            encoded_vec[string_idx] = string_id;
          }
          const size_t tbb_thread_idx = tbb::this_task_arena::current_thread_index();
          num_strings_not_found_per_thread[tbb_thread_idx] = num_strings_not_found;
        },
        tbb::simple_partitioner());
  });

  size_t num_strings_not_found = 0;
  for (int64_t thread_idx = 0; thread_idx < thread_info.num_threads; ++thread_idx) {
    num_strings_not_found += num_strings_not_found_per_thread[thread_idx];
  }
  return num_strings_not_found;
}

template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint8_t* encoded_vec,
                                          const int64_t generation) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          uint16_t* encoded_vec,
                                          const int64_t generation) const;
template size_t StringDictionary::getBulk(const std::vector<std::string>& string_vec,
                                          int32_t* encoded_vec,
                                          const int64_t generation) const;

template <class T, class String>
void StringDictionary::getOrAddBulk(const std::vector<String>& input_strings,
                                    T* output_string_ids) {
  if (g_enable_stringdict_parallel) {
    getOrAddBulkParallel(input_strings, output_string_ids);
    return;
  }
  ensureHashTableRecovered();
  // Single-thread path.
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);

  const size_t initial_str_count = str_count_;
  size_t idx = 0;
  for (const auto& input_string : input_strings) {
    if (input_string.empty()) {
      output_string_ids[idx++] = inline_int_null_value<T>();
      continue;
    }
    CHECK(input_string.size() <= MAX_STRLEN);

    const string_dict_hash_t input_string_hash = hash_string(input_string);
    uint32_t hash_bucket =
        computeBucket(input_string_hash, input_string, string_id_string_dict_hash_table_);
    if (string_id_string_dict_hash_table_[hash_bucket] != INVALID_STR_ID) {
      output_string_ids[idx++] = string_id_string_dict_hash_table_[hash_bucket];
      continue;
    }
    // need to add record to dictionary
    // check there is room
    if (str_count_ > static_cast<size_t>(max_valid_int_value<T>())) {
      throw_encoding_error<T>(input_string, dict_key_);
    }
    CHECK_LT(str_count_, MAX_STRCOUNT)
        << "Maximum number (" << str_count_
        << ") of Dictionary encoded Strings reached for this column, offset path "
           "for column is  "
        << offsets_path_;
    if (fillRateIsHigh(str_count_)) {
      // resize when more than 50% is full
      increaseHashTableCapacity();
      hash_bucket = computeBucket(
          input_string_hash, input_string, string_id_string_dict_hash_table_);
    }
    // TODO(Misiu): It's appending the strings one at a time?  This seems slow...
    appendToStorage(input_string);

    if (materialize_hashes_) {
      hash_cache_[str_count_] = input_string_hash;
    }
    const int32_t string_id = static_cast<int32_t>(str_count_);
    string_id_string_dict_hash_table_[hash_bucket] = string_id;
    output_string_ids[idx++] = string_id;
    ++str_count_;
  }
  const size_t num_strings_added = str_count_ - initial_str_count;
  if (num_strings_added > 0) {
    invalidateInvertedIndex();
  }
}

template <class T, class String>
void StringDictionary::getOrAddBulkParallel(const std::vector<String>& input_strings,
                                            T* output_string_ids) {
  ensureHashTableRecovered();
  // Compute hashes of the input strings up front, and in parallel,
  // as the string hashing does not need to be behind the subsequent write_lock
  std::vector<string_dict_hash_t> input_strings_hashes(input_strings.size());
  hashStrings(input_strings, input_strings_hashes);

  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  size_t shadow_str_count =
      str_count_;  // Need to shadow str_count_ now with bulk add methods
  const size_t storage_high_water_mark = shadow_str_count;
  std::vector<size_t> string_memory_ids;
  size_t sum_new_string_lengths = 0;
  string_memory_ids.reserve(input_strings.size());
  size_t input_string_idx{0};
  for (const auto& input_string : input_strings) {
    // Currently we make empty strings null
    if (input_string.empty()) {
      output_string_ids[input_string_idx++] = inline_int_null_value<T>();
      continue;
    }
    // TODO: Recover gracefully if an input string is too long
    CHECK(input_string.size() <= MAX_STRLEN);

    if (fillRateIsHigh(shadow_str_count)) {
      // resize when more than 50% is full
      increaseHashTableCapacityFromStorageAndMemory(shadow_str_count,
                                                    storage_high_water_mark,
                                                    input_strings,
                                                    string_memory_ids,
                                                    input_strings_hashes);
    }
    // Compute the hash for this input_string
    const string_dict_hash_t input_string_hash = input_strings_hashes[input_string_idx];

    const uint32_t hash_bucket =
        computeBucketFromStorageAndMemory(input_string_hash,
                                          input_string,
                                          string_id_string_dict_hash_table_,
                                          storage_high_water_mark,
                                          input_strings,
                                          string_memory_ids);

    // If the hash bucket is not empty, that is our string id
    // (computeBucketFromStorageAndMemory) already checked to ensure the input string and
    // bucket string are equal)
    if (string_id_string_dict_hash_table_[hash_bucket] != INVALID_STR_ID) {
      output_string_ids[input_string_idx++] =
          string_id_string_dict_hash_table_[hash_bucket];
      continue;
    }
    // Did not find string, so need to add record to dictionary
    // First check there is room
    if (shadow_str_count > static_cast<size_t>(max_valid_int_value<T>())) {
      throw_encoding_error<T>(input_string, dict_key_);
    }
    CHECK_LT(shadow_str_count, MAX_STRCOUNT)
        << "Maximum number (" << shadow_str_count
        << ") of Dictionary encoded Strings reached for this column, offset path "
           "for column is  "
        << offsets_path_;

    string_memory_ids.push_back(input_string_idx);
    sum_new_string_lengths += input_string.size();
    string_id_string_dict_hash_table_[hash_bucket] =
        static_cast<int32_t>(shadow_str_count);
    if (materialize_hashes_) {
      hash_cache_[shadow_str_count] = input_string_hash;
    }
    output_string_ids[input_string_idx++] = shadow_str_count++;
  }
  appendToStorageBulk(input_strings, string_memory_ids, sum_new_string_lengths);
  const size_t num_strings_added = shadow_str_count - str_count_;
  str_count_ = shadow_str_count;
  if (num_strings_added > 0) {
    invalidateInvertedIndex();
  }
}
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec,
                                             int32_t* encoded_vec);

template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(
    const std::vector<std::string_view>& string_vec,
    int32_t* encoded_vec);

template <class String>
int32_t StringDictionary::getIdOfString(const String& str) const {
  ensureHashTableRecovered();
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  return getUnlocked(str);
}

template int32_t StringDictionary::getIdOfString(const std::string&) const;
template int32_t StringDictionary::getIdOfString(const std::string_view&) const;

int32_t StringDictionary::getUnlocked(const std::string_view sv) const noexcept {
  const string_dict_hash_t hash = hash_string(sv);
  auto str_id = string_id_string_dict_hash_table_[computeBucket(
      hash, sv, string_id_string_dict_hash_table_)];
  return str_id;
}

std::string StringDictionary::getString(int32_t string_id) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  return getStringUnlocked(string_id);
}

std::string_view StringDictionary::getStringView(int32_t string_id) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  return getStringViewUnlocked(string_id);
}

std::string StringDictionary::getStringUnlocked(int32_t string_id) const noexcept {
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringChecked(string_id);
}

std::string_view StringDictionary::getStringViewUnlocked(
    int32_t string_id) const noexcept {
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringViewChecked(string_id);
}

std::pair<char*, size_t> StringDictionary::getStringBytes(
    int32_t string_id) const noexcept {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  CHECK_LE(0, string_id);
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringBytesChecked(string_id);
}

size_t StringDictionary::storageEntryCountUnlocked() const {
  return str_count_;
}

size_t StringDictionary::storageEntryCount() const {
  return storageEntryCountUnlocked();
}

bool StringDictionary::isSortedPermutationCacheComplete() const {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  return sorted_permutation_cache_.size() == str_count_;
}

template <typename T>
std::vector<T> StringDictionary::getLikeImpl(const std::string& pattern,
                                             const bool icase,
                                             const bool is_simple,
                                             const char escape,
                                             const size_t generation) const {
  CHECK_LE(generation, static_cast<size_t>(str_count_));
  constexpr size_t grain_size{1000};
  auto is_like_impl = icase       ? is_simple ? string_ilike_simple : string_ilike
                      : is_simple ? string_like_simple
                                  : string_like;
  auto const num_threads = static_cast<size_t>(cpu_threads());
  std::vector<std::vector<T>> worker_results(num_threads);
  tbb::task_arena limited_arena(num_threads);
  const auto populate_worker_results = [&](const auto& matches_at) {
    limited_arena.execute([&] {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, generation, grain_size),
          [&matches_at, &worker_results](const tbb::blocked_range<size_t>& range) {
            auto& result_vector =
                worker_results[tbb::this_task_arena::current_thread_index()];
            for (size_t i = range.begin(); i < range.end(); ++i) {
              if (matches_at(i)) {
                result_vector.push_back(i);
              }
            }
          });
    });
  };
  if (g_enable_lazy_string_dictionary_hash_recovery) {
    const PercentLiteralLikeMatcher percent_literal_matcher(pattern, escape);
    if (!icase && !is_simple && percent_literal_matcher.supported()) {
      populate_worker_results([&](const size_t string_id) {
        return percent_literal_matcher.matches(
            getStringFromStorageFast(static_cast<int32_t>(string_id)));
      });
    } else {
      populate_worker_results([&](const size_t string_id) {
        const auto str = getStringFromStorageFast(static_cast<int32_t>(string_id));
        return is_like_impl(
            str.data(), str.size(), pattern.c_str(), pattern.size(), escape);
      });
    }
  } else {
    populate_worker_results([&](const size_t string_id) {
      const auto str = getStringUnlocked(static_cast<int32_t>(string_id));
      return is_like_impl(
          str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape);
    });
  }
  // partial_sum to get 1) a start offset for each thread and 2) the total # elems
  std::vector<size_t> start_offsets(num_threads + 1, 0);
  auto vec_size = [](std::vector<T> const& vec) { return vec.size(); };
  auto begin = boost::make_transform_iterator(worker_results.begin(), vec_size);
  auto end = boost::make_transform_iterator(worker_results.end(), vec_size);
  std::partial_sum(begin, end, start_offsets.begin() + 1);  // first element is 0

  std::vector<T> result(start_offsets[num_threads]);
  limited_arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_threads, 1),
        [&worker_results, &result, &start_offsets](
            const tbb::blocked_range<size_t>& range) {
          auto& result_vector = worker_results[range.begin()];
          auto const start_offset = start_offsets[range.begin()];
          std::copy(
              result_vector.begin(), result_vector.end(), result.begin() + start_offset);
        },
        tbb::static_partitioner());
  });
  return result;
}
template <>
std::shared_ptr<const std::vector<int32_t>> StringDictionary::getLikeShared<int32_t>(
    const std::string& pattern,
    const bool icase,
    const bool is_simple,
    const char escape,
    const size_t generation) const {
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  const auto cache_key = std::make_tuple(pattern, icase, is_simple, escape, generation);
  const auto it = like_i32_cache_.find(cache_key);
  if (it != like_i32_cache_.end()) {
    return it->second;
  }

  auto result = std::make_shared<const std::vector<int32_t>>(
      getLikeImpl<int32_t>(pattern, icase, is_simple, escape, generation));
  const auto it_ok = like_i32_cache_.emplace(cache_key, result);
  like_cache_size_ +=
      pattern.size() + 3 + sizeof(generation) + result->size() * sizeof(int32_t);

  CHECK(it_ok.second);

  return result;
}

template <>
std::vector<int32_t> StringDictionary::getLike<int32_t>(const std::string& pattern,
                                                        const bool icase,
                                                        const bool is_simple,
                                                        const char escape,
                                                        const size_t generation) const {
  return *getLikeShared<int32_t>(pattern, icase, is_simple, escape, generation);
}

template <>
std::shared_ptr<const std::vector<int64_t>> StringDictionary::getLikeShared<int64_t>(
    const std::string& pattern,
    const bool icase,
    const bool is_simple,
    const char escape,
    const size_t generation) const {
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  const auto cache_key = std::make_tuple(pattern, icase, is_simple, escape, generation);
  const auto it = like_i64_cache_.find(cache_key);
  if (it != like_i64_cache_.end()) {
    return it->second;
  }

  auto result = std::make_shared<const std::vector<int64_t>>(
      getLikeImpl<int64_t>(pattern, icase, is_simple, escape, generation));
  const auto it_ok = like_i64_cache_.emplace(cache_key, result);
  like_cache_size_ +=
      pattern.size() + 3 + sizeof(generation) + result->size() * sizeof(int64_t);

  CHECK(it_ok.second);

  return result;
}

template <>
std::vector<int64_t> StringDictionary::getLike<int64_t>(const std::string& pattern,
                                                        const bool icase,
                                                        const bool is_simple,
                                                        const char escape,
                                                        const size_t generation) const {
  return *getLikeShared<int64_t>(pattern, icase, is_simple, escape, generation);
}

std::vector<int32_t> StringDictionary::getEquals(std::string pattern,
                                                 std::string comp_operator,
                                                 size_t generation) {
  std::vector<int32_t> result;
  auto eq_id_itr = equal_cache_.find(pattern);
  int32_t eq_id = MAX_STRLEN + 1;
  int32_t cur_size = str_count_;
  if (eq_id_itr != equal_cache_.end()) {
    eq_id = eq_id_itr->second;
    if (comp_operator == "=") {
      result.push_back(eq_id);
    } else {
      for (int32_t idx = 0; idx <= cur_size; idx++) {
        if (idx == eq_id) {
          continue;
        }
        result.push_back(idx);
      }
    }
  } else {
    std::vector<std::thread> workers;
    int worker_count = cpu_threads();
    CHECK_GT(worker_count, 0);
    std::vector<std::vector<int32_t>> worker_results(worker_count);
    CHECK_LE(generation, str_count_);
    for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
      workers.emplace_back(
          [&worker_results, &pattern, generation, worker_idx, worker_count, this]() {
            for (size_t string_id = worker_idx; string_id < generation;
                 string_id += worker_count) {
              const auto str = getStringUnlocked(string_id);
              if (str == pattern) {
                worker_results[worker_idx].push_back(string_id);
              }
            }
          });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    for (const auto& worker_result : worker_results) {
      result.insert(result.end(), worker_result.begin(), worker_result.end());
    }
    if (result.size() > 0) {
      const auto it_ok = equal_cache_.insert(std::make_pair(pattern, result[0]));
      equal_cache_size_ += (pattern.size() + (result.size() * sizeof(int32_t)));
      CHECK(it_ok.second);
      eq_id = result[0];
    }
    if (comp_operator == "<>") {
      for (int32_t idx = 0; idx <= cur_size; idx++) {
        if (idx == eq_id) {
          continue;
        }
        result.push_back(idx);
      }
    }
  }
  return result;
}

std::vector<int32_t> StringDictionary::getPersistedSortedPermutation() {
  permuteSortedCache();
  return sorted_permutation_cache_;
}

void StringDictionary::permuteSortedCache() {
  auto timer = DEBUG_TIMER(__func__);
  // not thread safe, only called with write lock around parent method

  const auto cur_cache_size = sorted_permutation_cache_.size();
  if (cur_cache_size == str_count_) {
    return;
  }
  buildSortedCache();
  sorted_permutation_cache_.resize(sorted_cache_.size());
  CHECK_LE(sorted_cache_.size(),
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  tbb::parallel_for(tbb::blocked_range<size_t>(0, sorted_cache_.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        sorted_permutation_cache_[sorted_cache_[i]] =
                            static_cast<int32_t>(i);
                      }
                    });
}

std::vector<std::pair<int32_t, int32_t>> StringDictionary::getTransientSortPermutation(
    const std::vector<std::pair<std::string, int32_t>>& transient_strings_to_ids) const {
  std::vector<std::pair<int32_t, int32_t>> transient_string_permutations(
      transient_strings_to_ids.size());

  int32_t last_persisted_rank =
      std::numeric_limits<int32_t>::max();  // impossible value for current signed 32 bit
                                            // dictionary
  int32_t transient_rank = 0;  // To order transient strings with same persisted rank

  for (const auto& transient_string_to_id : transient_strings_to_ids) {
    const auto sorted_cache_itr = std::lower_bound(
        sorted_cache_.begin(),
        sorted_cache_.end(),
        transient_string_to_id.first,
        [this](decltype(sorted_cache_)::value_type const& a, const std::string& b) {
          auto a_str = this->getStringFromStorage(a);
          return string_lt(a_str.c_str_ptr, a_str.size, b.c_str(), b.size());
        });
    int32_t persisted_rank = sorted_cache_itr != sorted_cache_.end()
                                 ? std::distance(sorted_cache_.begin(), sorted_cache_itr)
                                 : sorted_cache_.size();
    if (persisted_rank != last_persisted_rank) {
      last_persisted_rank = persisted_rank;
      transient_rank = 0;
    }
    transient_string_permutations[translateTransientIdToIndex(
        transient_string_to_id.second)] =
        std::make_pair(std::distance(sorted_cache_.begin(), sorted_cache_itr),
                       transient_rank++);
  }
  return transient_string_permutations;
}

SortedStringPermutation StringDictionary::getSortedPermutation(
    const std::vector<std::pair<std::string, int32_t>>& transient_string_to_id_map,
    const bool should_sort_descending) {
  auto timer = DEBUG_TIMER(__func__);
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);

  permuteSortedCache();
  SortedStringPermutation sorted_string_permutation(should_sort_descending);
  sorted_string_permutation.persisted_permutation = sorted_permutation_cache_;
  if (!transient_string_to_id_map.empty()) {
    sorted_string_permutation.transient_permutation =
        getTransientSortPermutation(transient_string_to_id_map);
  }
  return sorted_string_permutation;
}

std::vector<int32_t> StringDictionary::getCompare(const std::string& pattern,
                                                  const std::string& comp_operator,
                                                  const size_t generation) {
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  std::vector<int32_t> ret;
  if (str_count_ == 0) {
    return ret;
  }
  if (sorted_cache_.size() < str_count_) {
    if (comp_operator == "=" || comp_operator == "<>") {
      return getEquals(pattern, comp_operator, generation);
    }

    buildSortedCache();
  }
  auto cache_index = compare_cache_.get(pattern);

  if (!cache_index) {
    cache_index = std::make_shared<StringDictionary::compare_cache_value_t>();
    const auto cache_itr = std::lower_bound(
        sorted_cache_.begin(),
        sorted_cache_.end(),
        pattern,
        [this](decltype(sorted_cache_)::value_type const& a, decltype(pattern)& b) {
          auto a_str = this->getStringFromStorage(a);
          return string_lt(a_str.c_str_ptr, a_str.size, b.c_str(), b.size());
        });

    if (cache_itr == sorted_cache_.end()) {
      cache_index->index = sorted_cache_.size() - 1;
      cache_index->diff = 1;
    } else {
      const auto cache_str = getStringFromStorage(*cache_itr);
      if (!string_eq(
              cache_str.c_str_ptr, cache_str.size, pattern.c_str(), pattern.size())) {
        cache_index->index = cache_itr - sorted_cache_.begin() - 1;
        cache_index->diff = 1;
      } else {
        cache_index->index = cache_itr - sorted_cache_.begin();
        cache_index->diff = 0;
      }
    }

    compare_cache_.put(pattern, cache_index);
    compare_cache_size_ += (pattern.size() + sizeof(cache_index));
  }

  // since we have a cache in form of vector of ints which is sorted according to
  // corresponding strings in the dictionary all we need is the index of the element
  // which equal to the pattern that we are trying to match or the index of “biggest”
  // element smaller than the pattern, to perform all the comparison operators over
  // string. The search function guarantees we have such index so now it is just the
  // matter to include all the elements in the result vector.

  // For < operator if the index that we have points to the element which is equal to
  // the pattern that we are searching for we simply get all the elements less than the
  // index. If the element pointed by the index is not equal to the pattern we are
  // comparing with we also need to include that index in result vector, except when the
  // index points to 0 and the pattern is lesser than the smallest value in the string
  // dictionary.

  if (comp_operator == "<") {
    size_t idx = cache_index->index;
    if (cache_index->diff) {
      idx = cache_index->index + 1;
      if (cache_index->index == 0 && cache_index->diff > 0) {
        idx = cache_index->index;
      }
    }
    for (size_t i = 0; i < idx; i++) {
      ret.push_back(sorted_cache_[i]);
    }

    // For <= operator if the index that we have points to the element which is equal to
    // the pattern that we are searching for we want to include the element pointed by
    // the index in the result set. If the element pointed by the index is not equal to
    // the pattern we are comparing with we just want to include all the ids with index
    // less than the index that is cached, except when pattern that we are searching for
    // is smaller than the smallest string in the dictionary.

  } else if (comp_operator == "<=") {
    size_t idx = cache_index->index + 1;
    if (cache_index == 0 && cache_index->diff > 0) {
      idx = cache_index->index;
    }
    for (size_t i = 0; i < idx; i++) {
      ret.push_back(sorted_cache_[i]);
    }

    // For > operator we want to get all the elements with index greater than the index
    // that we have except, when the pattern we are searching for is lesser than the
    // smallest string in the dictionary we also want to include the id of the index
    // that we have.

  } else if (comp_operator == ">") {
    size_t idx = cache_index->index + 1;
    if (cache_index->index == 0 && cache_index->diff > 0) {
      idx = cache_index->index;
    }
    for (size_t i = idx; i < sorted_cache_.size(); i++) {
      ret.push_back(sorted_cache_[i]);
    }

    // For >= operator when the indexed element that we have points to element which is
    // equal to the pattern we are searching for we want to include that in the result
    // vector. If the index that we have does not point to the string which is equal to
    // the pattern we are searching we don’t want to include that id into the result
    // vector except when the index is 0.

  } else if (comp_operator == ">=") {
    size_t idx = cache_index->index;
    if (cache_index->diff) {
      idx = cache_index->index + 1;
      if (cache_index->index == 0 && cache_index->diff > 0) {
        idx = cache_index->index;
      }
    }
    for (size_t i = idx; i < sorted_cache_.size(); i++) {
      ret.push_back(sorted_cache_[i]);
    }
  } else if (comp_operator == "=") {
    if (!cache_index->diff) {
      ret.push_back(sorted_cache_[cache_index->index]);
    }

    // For <> operator it is simple matter of not including id of string which is equal
    // to pattern we are searching for.
  } else if (comp_operator == "<>") {
    if (!cache_index->diff) {
      size_t idx = cache_index->index;
      for (size_t i = 0; i < idx; i++) {
        ret.push_back(sorted_cache_[i]);
      }
      ++idx;
      for (size_t i = idx; i < sorted_cache_.size(); i++) {
        ret.push_back(sorted_cache_[i]);
      }
    } else {
      for (size_t i = 0; i < sorted_cache_.size(); i++) {
        ret.insert(ret.begin(), sorted_cache_.begin(), sorted_cache_.end());
      }
    }

  } else {
    std::runtime_error("Unsupported string comparison operator");
  }
  return ret;
}

namespace {

bool is_regexp_like(const std::string& str,
                    const std::string& pattern,
                    const char escape) {
  return regexp_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape);
}

}  // namespace

std::vector<int32_t> StringDictionary::getRegexpLike(const std::string& pattern,
                                                     const char escape,
                                                     const size_t generation) const {
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  const auto cache_key = std::make_pair(pattern, escape);
  const auto it = regex_cache_.find(cache_key);
  if (it != regex_cache_.end()) {
    return it->second;
  }
  std::vector<int32_t> result;
  std::vector<std::thread> workers;
  int worker_count = cpu_threads();
  CHECK_GT(worker_count, 0);
  std::vector<std::vector<int32_t>> worker_results(worker_count);
  CHECK_LE(generation, str_count_);
  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&worker_results,
                          &pattern,
                          generation,
                          escape,
                          worker_idx,
                          worker_count,
                          this]() {
      for (size_t string_id = worker_idx; string_id < generation;
           string_id += worker_count) {
        const auto str = getStringUnlocked(string_id);
        if (is_regexp_like(str, pattern, escape)) {
          worker_results[worker_idx].push_back(string_id);
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& worker_result : worker_results) {
    result.insert(result.end(), worker_result.begin(), worker_result.end());
  }
  const auto it_ok = regex_cache_.insert(std::make_pair(cache_key, result));
  regex_cache_size_ += (pattern.size() + 1 + (result.size() * sizeof(int32_t)));
  CHECK(it_ok.second);

  return result;
}

std::vector<std::string> StringDictionary::copyStrings() const {
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);

  if (strings_cache_) {
    return *strings_cache_;
  }

  strings_cache_ = std::make_shared<std::vector<std::string>>();
  strings_cache_->reserve(str_count_);
  const bool multithreaded = str_count_ > 10000;
  const auto worker_count =
      multithreaded ? static_cast<size_t>(cpu_threads()) : size_t(1);
  CHECK_GT(worker_count, 0UL);
  std::vector<std::vector<std::string>> worker_results(worker_count);
  std::vector<size_t> string_size(worker_count, 0);
  auto copy = [this, &string_size](std::vector<std::string>& str_list,
                                   const size_t worker_idx,
                                   const size_t start_id,
                                   const size_t end_id) {
    CHECK_LE(start_id, end_id);
    str_list.reserve(end_id - start_id);
    for (size_t string_id = start_id; string_id < end_id; ++string_id) {
      auto str = getStringUnlocked(string_id);
      string_size[worker_idx] += str.size();
      str_list.push_back(str);
    }
  };
  if (multithreaded) {
    std::vector<std::future<void>> workers;
    const auto stride = (str_count_ + (worker_count - 1)) / worker_count;
    for (size_t worker_idx = 0, start = 0, end = std::min(start + stride, str_count_);
         worker_idx < worker_count && start < str_count_;
         ++worker_idx, start += stride, end = std::min(start + stride, str_count_)) {
      workers.push_back(std::async(std::launch::async,
                                   copy,
                                   std::ref(worker_results[worker_idx]),
                                   worker_idx,
                                   start,
                                   end));
    }
    for (auto& worker : workers) {
      worker.get();
    }
  } else {
    CHECK_EQ(worker_results.size(), size_t(1));
    copy(worker_results[0], 0, 0, str_count_);
  }

  for (const auto& worker_result : worker_results) {
    strings_cache_->insert(
        strings_cache_->end(), worker_result.begin(), worker_result.end());
  }
  strings_cache_size_ +=
      std::accumulate(string_size.begin(), string_size.end(), size_t(0));
  return *strings_cache_;
}

bool StringDictionary::fillRateIsHigh(const size_t num_strings) const noexcept {
  return string_id_string_dict_hash_table_.size() <= num_strings * 2;
}

void StringDictionary::increaseHashTableCapacity() noexcept {
  std::vector<int32_t> new_str_ids(string_id_string_dict_hash_table_.size() * 2,
                                   INVALID_STR_ID);

  if (materialize_hashes_) {
    for (size_t i = 0; i != str_count_; ++i) {
      const string_dict_hash_t hash = hash_cache_[i];
      const uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = i;
    }
    hash_cache_.resize(hash_cache_.size() * 2);
  } else {
    for (size_t i = 0; i != str_count_; ++i) {
      const auto str = getStringChecked(i);
      const string_dict_hash_t hash = hash_string(str);
      const uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = i;
    }
  }
  string_id_string_dict_hash_table_.swap(new_str_ids);
}

template <class String>
void StringDictionary::increaseHashTableCapacityFromStorageAndMemory(
    const size_t str_count,  // str_count_ is only persisted strings, so need transient
                             // shadow count
    const size_t storage_high_water_mark,
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids,
    const std::vector<string_dict_hash_t>& input_strings_hashes) noexcept {
  std::vector<int32_t> new_str_ids(string_id_string_dict_hash_table_.size() * 2,
                                   INVALID_STR_ID);
  if (materialize_hashes_) {
    for (size_t i = 0; i != str_count; ++i) {
      const string_dict_hash_t hash = hash_cache_[i];
      const uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = i;
    }
    hash_cache_.resize(hash_cache_.size() * 2);
  } else {
    for (size_t storage_idx = 0; storage_idx != storage_high_water_mark; ++storage_idx) {
      const auto storage_string = getStringChecked(storage_idx);
      const string_dict_hash_t hash = hash_string(storage_string);
      const uint32_t bucket = computeUniqueBucketWithHash(hash, new_str_ids);
      new_str_ids[bucket] = storage_idx;
    }
    for (size_t memory_idx = 0; memory_idx != string_memory_ids.size(); ++memory_idx) {
      const size_t string_memory_id = string_memory_ids[memory_idx];
      const uint32_t bucket = computeUniqueBucketWithHash(
          input_strings_hashes[string_memory_id], new_str_ids);
      new_str_ids[bucket] = storage_high_water_mark + memory_idx;
    }
  }
  string_id_string_dict_hash_table_.swap(new_str_ids);
}

int32_t StringDictionary::getOrAddImpl(const std::string_view& str) noexcept {
  // @TODO(wei) treat empty string as NULL for now
  if (str.size() == 0) {
    return inline_int_null_value<int32_t>();
  }
  CHECK(str.size() <= MAX_STRLEN);
  const string_dict_hash_t hash = hash_string(str);
  {
    std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
    const uint32_t bucket = computeBucket(hash, str, string_id_string_dict_hash_table_);
    if (string_id_string_dict_hash_table_[bucket] != INVALID_STR_ID) {
      return string_id_string_dict_hash_table_[bucket];
    }
  }
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  if (fillRateIsHigh(str_count_)) {
    // resize when more than 50% is full
    increaseHashTableCapacity();
  }
  // need to recalculate the bucket in case it changed before
  // we got the lock
  const uint32_t bucket = computeBucket(hash, str, string_id_string_dict_hash_table_);
  if (string_id_string_dict_hash_table_[bucket] == INVALID_STR_ID) {
    CHECK_LT(str_count_, MAX_STRCOUNT)
        << "Maximum number (" << str_count_
        << ") of Dictionary encoded Strings reached for this column, offset path "
           "for column is  "
        << offsets_path_;
    appendToStorage(str);
    string_id_string_dict_hash_table_[bucket] = static_cast<int32_t>(str_count_);
    if (materialize_hashes_) {
      hash_cache_[str_count_] = hash;
    }
    ++str_count_;
    invalidateInvertedIndex();
  }
  return string_id_string_dict_hash_table_[bucket];
}

std::string StringDictionary::getStringChecked(const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!str_canary.canary);
  return std::string(str_canary.c_str_ptr, str_canary.size);
}

std::string_view StringDictionary::getStringViewChecked(
    const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!str_canary.canary);
  return std::string_view{str_canary.c_str_ptr, str_canary.size};
}

std::pair<char*, size_t> StringDictionary::getStringBytesChecked(
    const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!str_canary.canary);
  return std::make_pair(str_canary.c_str_ptr, str_canary.size);
}

template <class String>
uint32_t StringDictionary::computeBucket(
    const string_dict_hash_t hash,
    const String& input_string,
    const std::vector<int32_t>& string_id_string_dict_hash_table) const noexcept {
  const size_t string_dict_hash_table_size = string_id_string_dict_hash_table.size();
  uint32_t bucket = hash & (string_dict_hash_table_size - 1);
  while (true) {
    const int32_t candidate_string_id = string_id_string_dict_hash_table[bucket];
    if (candidate_string_id ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    if ((materialize_hashes_ && hash == hash_cache_[candidate_string_id]) ||
        !materialize_hashes_) {
      const auto candidate_string = getStringFromStorageFast(candidate_string_id);
      if (input_string.size() == candidate_string.size() &&
          !memcmp(input_string.data(), candidate_string.data(), input_string.size())) {
        // found the string
        break;
      }
    }
    // wrap around
    if (++bucket == string_dict_hash_table_size) {
      bucket = 0;
    }
  }
  return bucket;
}

template <class String>
uint32_t StringDictionary::computeBucketFromStorageAndMemory(
    const string_dict_hash_t input_string_hash,
    const String& input_string,
    const std::vector<int32_t>& string_id_string_dict_hash_table,
    const size_t storage_high_water_mark,
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids) const noexcept {
  uint32_t bucket = input_string_hash & (string_id_string_dict_hash_table.size() - 1);
  while (true) {
    const int32_t candidate_string_id = string_id_string_dict_hash_table[bucket];
    if (candidate_string_id ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    if (!materialize_hashes_ || (input_string_hash == hash_cache_[candidate_string_id])) {
      if (candidate_string_id >= 0 &&
          static_cast<size_t>(candidate_string_id) >= storage_high_water_mark) {
        // The candidate string is not in storage yet but in our string_memory_ids temp
        // buffer
        size_t memory_offset =
            static_cast<size_t>(candidate_string_id - storage_high_water_mark);
        const String candidate_string = input_strings[string_memory_ids[memory_offset]];
        if (input_string.size() == candidate_string.size() &&
            !memcmp(input_string.data(), candidate_string.data(), input_string.size())) {
          // found the string in the temp memory buffer
          break;
        }
      } else {
        // The candidate string is in storage, need to fetch it for comparison
        const auto candidate_storage_string =
            getStringFromStorageFast(candidate_string_id);
        if (input_string.size() == candidate_storage_string.size() &&
            !memcmp(input_string.data(),
                    candidate_storage_string.data(),
                    input_string.size())) {
          //! memcmp(input_string.data(), candidate_storage_string.c_str_ptr,
          //! input_string.size())) {
          // found the string in storage
          break;
        }
      }
    }
    if (++bucket == string_id_string_dict_hash_table.size()) {
      bucket = 0;
    }
  }
  return bucket;
}

uint32_t StringDictionary::computeUniqueBucketWithHash(
    const string_dict_hash_t hash,
    const std::vector<int32_t>& string_id_string_dict_hash_table) noexcept {
  const size_t string_dict_hash_table_size = string_id_string_dict_hash_table.size();
  uint32_t bucket = hash & (string_dict_hash_table_size - 1);
  while (true) {
    if (string_id_string_dict_hash_table[bucket] ==
        INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    collisions_++;
    // wrap around
    if (++bucket == string_dict_hash_table_size) {
      bucket = 0;
    }
  }
  return bucket;
}

void StringDictionary::checkAndConditionallyIncreasePayloadCapacity(
    const size_t write_length) {
  if (payload_file_off_ + write_length > payload_file_size_) {
    const size_t min_capacity_needed =
        write_length - (payload_file_size_ - payload_file_off_);
    if (!isTemp_) {
      CHECK_GE(payload_fd_, 0);
#ifdef __linux__
      const auto old_payload_file_size = payload_file_size_;
      addPayloadCapacity(min_capacity_needed);
      CHECK(payload_file_off_ + write_length <= payload_file_size_);
      payload_map_ = reinterpret_cast<char*>(heavyai::checked_mremap(
          payload_map_, old_payload_file_size, payload_file_size_));
      total_mmap_size += payload_file_size_ - old_payload_file_size;
#else
      heavyai::checked_munmap(payload_map_, payload_file_size_);
      total_mmap_size -= payload_file_size_;
      addPayloadCapacity(min_capacity_needed);
      CHECK(payload_file_off_ + write_length <= payload_file_size_);
      payload_map_ =
          reinterpret_cast<char*>(heavyai::checked_mmap(payload_fd_, payload_file_size_));
      total_mmap_size += payload_file_size_;
#endif
    } else {
      addPayloadCapacity(min_capacity_needed);
      CHECK(payload_file_off_ + write_length <= payload_file_size_);
    }
  }
}

void StringDictionary::checkAndConditionallyIncreaseOffsetCapacity(
    const size_t write_length) {
  const size_t offset_file_off = str_count_ * sizeof(StringIdxEntry);
  if (offset_file_off + write_length >= offset_file_size_) {
    const size_t min_capacity_needed =
        write_length - (offset_file_size_ - offset_file_off);
    if (!isTemp_) {
      CHECK_GE(offset_fd_, 0);
#ifdef __linux__
      const auto old_offset_file_size = offset_file_size_;
      addOffsetCapacity(min_capacity_needed);
      CHECK(offset_file_off + write_length <= offset_file_size_);
      offset_map_ = reinterpret_cast<StringIdxEntry*>(
          heavyai::checked_mremap(offset_map_, old_offset_file_size, offset_file_size_));
      total_mmap_size += offset_file_size_ - old_offset_file_size;
#else
      heavyai::checked_munmap(offset_map_, offset_file_size_);
      total_mmap_size -= offset_file_size_;
      addOffsetCapacity(min_capacity_needed);
      CHECK(offset_file_off + write_length <= offset_file_size_);
      offset_map_ = reinterpret_cast<StringIdxEntry*>(
          heavyai::checked_mmap(offset_fd_, offset_file_size_));
      total_mmap_size += offset_file_size_;
#endif
    } else {
      addOffsetCapacity(min_capacity_needed);
      CHECK(offset_file_off + write_length <= offset_file_size_);
    }
  }
}

template <class String>
void StringDictionary::appendToStorage(const String str) noexcept {
  // write the payload
  checkAndConditionallyIncreasePayloadCapacity(str.size());
  memcpy(payload_map_ + payload_file_off_, str.data(), str.size());

  // write the offset and length
  StringIdxEntry str_meta{static_cast<uint64_t>(payload_file_off_), str.size()};
  payload_file_off_ += str.size();  // Need to increment after we've defined str_meta

  checkAndConditionallyIncreaseOffsetCapacity(sizeof(str_meta));
  memcpy(offset_map_ + str_count_, &str_meta, sizeof(str_meta));
}

template <class String>
void StringDictionary::appendToStorageBulk(
    const std::vector<String>& input_strings,
    const std::vector<size_t>& string_memory_ids,
    const size_t sum_new_strings_lengths) noexcept {
  const size_t num_strings = string_memory_ids.size();

  checkAndConditionallyIncreasePayloadCapacity(sum_new_strings_lengths);
  checkAndConditionallyIncreaseOffsetCapacity(sizeof(StringIdxEntry) * num_strings);

  for (size_t i = 0; i < num_strings; ++i) {
    const size_t string_idx = string_memory_ids[i];
    const String str = input_strings[string_idx];
    const size_t str_size(str.size());
    memcpy(payload_map_ + payload_file_off_, str.data(), str_size);
    StringIdxEntry str_meta{static_cast<uint64_t>(payload_file_off_), str_size};
    payload_file_off_ += str_size;  // Need to increment after we've defined str_meta
    memcpy(offset_map_ + str_count_ + i, &str_meta, sizeof(str_meta));
  }
}

std::string_view StringDictionary::getStringFromStorageFast(
    const int string_id) const noexcept {
  const StringIdxEntry* str_meta = offset_map_ + string_id;
  return {payload_map_ + str_meta->off, str_meta->size};
}

StringDictionary::PayloadString StringDictionary::getStringFromStorage(
    const int string_id) const noexcept {
  if (!isTemp_) {
    CHECK_GE(payload_fd_, 0);
    CHECK_GE(offset_fd_, 0);
  }
  CHECK_GE(string_id, 0);
  const StringIdxEntry* str_meta = offset_map_ + string_id;
  if (str_meta->size == 0xffff) {
    // hit the canary
    return {nullptr, 0, true};
  }
  return {payload_map_ + str_meta->off, str_meta->size, false};
}

void StringDictionary::addPayloadCapacity(const size_t min_capacity_requested) noexcept {
  if (!isTemp_) {
    payload_file_size_ += addStorageCapacity(payload_fd_, min_capacity_requested);
  } else {
    payload_map_ = static_cast<char*>(
        addMemoryCapacity(payload_map_, payload_file_size_, min_capacity_requested));
  }
}

void StringDictionary::addOffsetCapacity(const size_t min_capacity_requested) noexcept {
  if (!isTemp_) {
    offset_file_size_ += addStorageCapacity(offset_fd_, min_capacity_requested);
  } else {
    offset_map_ = static_cast<StringIdxEntry*>(
        addMemoryCapacity(offset_map_, offset_file_size_, min_capacity_requested));
  }
}

namespace {

static constexpr size_t FOUR_MB = 4 * 1024 * 1024;

constexpr size_t align_up(size_t size, size_t alignment) {
  return ((size + alignment - 1) / alignment) * alignment;
}

}  // namespace

size_t StringDictionary::addStorageCapacity(
    int fd,
    const size_t min_capacity_requested) noexcept {
  CHECK_NE(lseek(fd, 0, SEEK_END), -1);

  // Round capacity-increase request size up to system page size multiple.
  const size_t request_size =
      std::max(align_up(FOUR_MB, SYSTEM_PAGE_SIZE),
               align_up(min_capacity_requested, SYSTEM_PAGE_SIZE));

  int write_return = -1;
  if (request_size > canary_buffer.size) {
    // If the request is larger than the default canary buffer, then allocate a new
    // temporary buffer for this request; otherwise use default size.
    const auto canary = string_dictionary::CanaryBuffer(request_size);
    write_return = write(fd, canary.buffer, request_size);
  } else {
    write_return = write(fd, canary_buffer.buffer, request_size);
  }
  CHECK_GT(write_return, 0) << "Write failed with error: " << strerror(errno);
  CHECK_EQ(static_cast<size_t>(write_return), request_size);
  return request_size;
}

void* StringDictionary::addMemoryCapacity(void* addr,
                                          size_t& mem_size,
                                          const size_t min_capacity_requested) noexcept {
  // Round capacity-increase request size up to system page size multiple.
  const size_t request_size =
      std::max(align_up(FOUR_MB, SYSTEM_PAGE_SIZE),
               align_up(min_capacity_requested, SYSTEM_PAGE_SIZE));
  void* new_addr = realloc(addr, mem_size + request_size);
  CHECK(new_addr);
  void* write_addr = reinterpret_cast<void*>(static_cast<char*>(new_addr) + mem_size);

  // If the request is larger than the default canary buffer, then allocate a new
  // temporary buffer for this request; otherwise use default size.
  if (request_size > canary_buffer.size) {
    const auto canary = string_dictionary::CanaryBuffer(request_size);
    CHECK(memcpy(write_addr, canary.buffer, request_size));
  } else {
    CHECK(memcpy(write_addr, canary_buffer.buffer, request_size));
  }
  mem_size += request_size;
  return new_addr;
}

void StringDictionary::invalidateInvertedIndex() noexcept {
  if (!like_i32_cache_.empty()) {
    decltype(like_i32_cache_)().swap(like_i32_cache_);
  }
  if (!like_i64_cache_.empty()) {
    decltype(like_i64_cache_)().swap(like_i64_cache_);
  }
  if (!regex_cache_.empty()) {
    decltype(regex_cache_)().swap(regex_cache_);
  }
  if (!equal_cache_.empty()) {
    decltype(equal_cache_)().swap(equal_cache_);
  }
  {
    std::lock_guard<std::mutex> cache_lock(scan_lookup_cache_mutex_);
    scan_lookup_cache_.clear();
    scan_lookup_cache_size_ = 0;
  }
  compare_cache_.invalidateInvertedIndex();

  like_cache_size_ = 0;
  regex_cache_size_ = 0;
  equal_cache_size_ = 0;
  compare_cache_size_ = 0;
}

// TODO 5 Mar 2021 Nothing will undo the writes to dictionary currently on a failed
// load.  The next write to the dictionary that does checkpoint will make the
// uncheckpointed data be written to disk. Only option is a table truncate, and thats
// assuming not replicated dictionary
bool StringDictionary::checkpoint() noexcept {
  CHECK(!isTemp_);
  bool ret = true;
  ret = ret &&
        (heavyai::msync((void*)offset_map_, offset_file_size_, /*async=*/false) == 0);
  ret = ret &&
        (heavyai::msync((void*)payload_map_, payload_file_size_, /*async=*/false) == 0);
  ret = ret && (heavyai::fsync(offset_fd_) == 0);
  ret = ret && (heavyai::fsync(payload_fd_) == 0);
  return ret;
}

std::vector<int32_t> StringDictionary::fillIndexVector(const size_t start_idx,
                                                       const size_t end_idx) const {
  std::vector<int32_t> index_vector(end_idx - start_idx);
  std::iota(index_vector.begin(), index_vector.end(), start_idx);
  return index_vector;
}

void StringDictionary::buildSortedCache() {
  auto timer = DEBUG_TIMER(__func__);
  // This method is not thread-safe.
  const auto cur_cache_size = sorted_cache_.size();
  if (cur_cache_size == str_count_) {
    return;
  }
  if (sorted_cache_.empty()) {
    sorted_cache_ = fillIndexVector(cur_cache_size, str_count_);
    sortCache(sorted_cache_);
  } else {
    std::vector<int32_t> temp_sorted_cache = fillIndexVector(cur_cache_size, str_count_);
    sortCache(temp_sorted_cache);
    mergeSortedCache(temp_sorted_cache);
  }
}

void StringDictionary::sortCache(std::vector<int32_t>& cache) {
  auto timer = DEBUG_TIMER(__func__);
  // This method is not thread-safe.

  // this boost sort is creating some problems when we use UTF-8 encoded strings.
  // TODO (vraj): investigate What is wrong with boost sort and try to mitigate it.

  const auto string_id_less = [this](int32_t a, int32_t b) {
    auto a_str = this->getStringFromStorage(a);
    auto b_str = this->getStringFromStorage(b);
    return string_lt(a_str.c_str_ptr, a_str.size, b_str.c_str_ptr, b_str.size);
  };
  if (g_enable_stringdict_parallel_sort) {
    tbb::parallel_sort(cache.begin(), cache.end(), string_id_less);
  } else {
    std::sort(cache.begin(), cache.end(), string_id_less);
  }
}

void StringDictionary::mergeSortedCache(std::vector<int32_t>& temp_sorted_cache) {
  // this method is not thread safe
  auto timer = DEBUG_TIMER(__func__);
  std::vector<int32_t> updated_cache(temp_sorted_cache.size() + sorted_cache_.size());
  size_t t_idx = 0, s_idx = 0, idx = 0;
  for (; t_idx < temp_sorted_cache.size() && s_idx < sorted_cache_.size(); idx++) {
    auto t_string = getStringFromStorage(temp_sorted_cache[t_idx]);
    auto s_string = getStringFromStorage(sorted_cache_[s_idx]);
    const auto insert_from_temp_cache =
        string_lt(t_string.c_str_ptr, t_string.size, s_string.c_str_ptr, s_string.size);
    if (insert_from_temp_cache) {
      updated_cache[idx] = temp_sorted_cache[t_idx++];
    } else {
      updated_cache[idx] = sorted_cache_[s_idx++];
    }
  }
  while (t_idx < temp_sorted_cache.size()) {
    updated_cache[idx++] = temp_sorted_cache[t_idx++];
  }
  while (s_idx < sorted_cache_.size()) {
    updated_cache[idx++] = sorted_cache_[s_idx++];
  }
  sorted_cache_.swap(updated_cache);
}

void StringDictionary::populate_string_ids(
    std::vector<int32_t>& dest_ids,
    StringDictionary* dest_dict,
    const std::vector<int32_t>& source_ids,
    const StringDictionary* source_dict,
    const std::vector<std::string const*>& transient_string_vec) {
  std::vector<std::string> strings;

  for (const int32_t source_id : source_ids) {
    if (source_id == std::numeric_limits<int32_t>::min()) {
      strings.emplace_back("");
    } else if (source_id < 0) {
      unsigned const string_index = StringDictionaryProxy::transientIdToIndex(source_id);
      CHECK_LT(string_index, transient_string_vec.size()) << "source_id=" << source_id;
      strings.emplace_back(*transient_string_vec[string_index]);
    } else {
      strings.push_back(source_dict->getString(source_id));
    }
  }

  dest_ids.resize(strings.size());
  dest_dict->getOrAddBulk(strings, &dest_ids[0]);
}

void StringDictionary::populate_string_array_ids(
    std::vector<std::vector<int32_t>>& dest_array_ids,
    StringDictionary* dest_dict,
    const std::vector<std::vector<int32_t>>& source_array_ids,
    const StringDictionary* source_dict) {
  dest_array_ids.resize(source_array_ids.size());

  std::atomic<size_t> row_idx{0};
  auto processor = [&row_idx, &dest_array_ids, dest_dict, &source_array_ids, source_dict](
                       int thread_id) {
    for (;;) {
      auto row = row_idx.fetch_add(1);

      if (row >= dest_array_ids.size()) {
        return;
      }
      const auto& source_ids = source_array_ids[row];
      auto& dest_ids = dest_array_ids[row];
      populate_string_ids(dest_ids, dest_dict, source_ids, source_dict);
    }
  };

  const int num_worker_threads = std::thread::hardware_concurrency();

  if (source_array_ids.size() / num_worker_threads > 10) {
    std::vector<std::future<void>> worker_threads;
    for (int i = 0; i < num_worker_threads; ++i) {
      worker_threads.push_back(std::async(std::launch::async, processor, i));
    }

    for (auto& child : worker_threads) {
      child.wait();
    }
    for (auto& child : worker_threads) {
      child.get();
    }
  } else {
    processor(0);
  }
}

std::vector<std::string_view> StringDictionary::getStringViews(
    const size_t generation) const {
  auto timer = DEBUG_TIMER(__func__);
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  const int64_t num_strings = generation >= 0 ? generation : storageEntryCount();
  CHECK_LE(num_strings, static_cast<int64_t>(StringDictionary::MAX_STRCOUNT));
  // The CHECK_LE below is currently redundant with the check
  // above against MAX_STRCOUNT, however given we iterate using
  // int32_t types for efficiency (to match type expected by
  // getStringFromStorageFast, check that the # of strings is also
  // in the int32_t range in case MAX_STRCOUNT is changed

  // Todo(todd): consider aliasing the max logical type width
  // (currently int32_t) throughout StringDictionary
  CHECK_LE(num_strings, std::numeric_limits<int32_t>::max());

  std::vector<std::string_view> string_views(num_strings);
  // We can bail early if the generation-specified dictionary is empty
  if (num_strings == 0) {
    return string_views;
  }
  constexpr int64_t tbb_parallel_threshold{1000};
  if (num_strings < tbb_parallel_threshold) {
    // Use int32_t to match type expected by getStringFromStorageFast
    for (int32_t string_idx = 0; string_idx < num_strings; ++string_idx) {
      string_views[string_idx] = getStringFromStorageFast(string_idx);
    }
  } else {
    constexpr int64_t target_strings_per_thread{1000};
    const ThreadInfo thread_info(
        std::thread::hardware_concurrency(), num_strings, target_strings_per_thread);
    CHECK_GE(thread_info.num_threads, 1L);
    CHECK_GE(thread_info.num_elems_per_thread, 1L);

    tbb::task_arena limited_arena(thread_info.num_threads);
    limited_arena.execute([&] {
      tbb::parallel_for(
          tbb::blocked_range<int64_t>(
              0, num_strings, thread_info.num_elems_per_thread /* tbb grain_size */),
          [&](const tbb::blocked_range<int64_t>& r) {
            // r should be in range of int32_t per CHECK above
            const int32_t start_idx = r.begin();
            const int32_t end_idx = r.end();
            for (int32_t string_idx = start_idx; string_idx != end_idx; ++string_idx) {
              string_views[string_idx] = getStringFromStorageFast(string_idx);
            }
          },
          tbb::simple_partitioner());
    });
  }
  return string_views;
}

std::vector<std::string_view> StringDictionary::getStringViews() const {
  return getStringViews(storageEntryCount());
}

std::vector<int32_t> StringDictionary::buildDictionaryTranslationMap(
    const std::shared_ptr<StringDictionary> dest_dict,
    StringLookupCallback const& dest_transient_lookup_callback) const {
  auto timer = DEBUG_TIMER(__func__);
  const size_t num_source_strings = storageEntryCount();
  const size_t num_dest_strings = dest_dict->storageEntryCount();
  std::vector<int32_t> translated_ids(num_source_strings);

  buildDictionaryTranslationMap(dest_dict.get(),
                                translated_ids.data(),
                                num_source_strings,
                                num_dest_strings,
                                true,  // Just assume true for dest_has_transients as this
                                       // function is only used for testing currently
                                dest_transient_lookup_callback,
                                {});
  return translated_ids;
}

void order_translation_locks(const shared::StringDictKey& source_dict_key,
                             const shared::StringDictKey& dest_dict_key,
                             std::shared_lock<std::shared_mutex>& source_read_lock,
                             std::shared_lock<std::shared_mutex>& dest_read_lock) {
  const bool dicts_are_same = (source_dict_key == dest_dict_key);
  const bool source_dict_is_locked_first = (source_dict_key < dest_dict_key);
  if (dicts_are_same) {
    // dictionaries are same, only take one write lock
    dest_read_lock.lock();
  } else if (source_dict_is_locked_first) {
    source_read_lock.lock();
    dest_read_lock.lock();
  } else {
    dest_read_lock.lock();
    source_read_lock.lock();
  }
}

size_t StringDictionary::buildDictionaryTranslationMap(
    const StringDictionary* dest_dict,
    int32_t* translated_ids,
    const int64_t source_generation,
    const int64_t dest_generation,
    const bool dest_has_transients,
    StringLookupCallback const& dest_transient_lookup_callback,
    const StringOps_Namespace::StringOps& string_ops) const {
  auto timer = DEBUG_TIMER(__func__);
  const bool has_string_ops = string_ops.size() > 0;
  if (materialize_hashes_ && !has_string_ops) {
    ensureHashTableRecovered();
  }
  dest_dict->ensureHashTableRecovered();
  CHECK_GE(source_generation, 0L);
  CHECK_GE(dest_generation, 0L);
  const int64_t num_source_strings = source_generation;
  const int64_t num_dest_strings = dest_generation;

  // We can bail early if there are no source strings to translate
  if (num_source_strings == 0L) {
    return 0;
  }

  // Sort this/source dict and dest dict on folder_ so we can enforce
  // lock ordering and avoid deadlocks
  std::shared_lock<std::shared_mutex> source_read_lock(rw_mutex_, std::defer_lock);
  std::shared_lock<std::shared_mutex> dest_read_lock(dest_dict->rw_mutex_,
                                                     std::defer_lock);
  order_translation_locks(
      getDictKey(), dest_dict->getDictKey(), source_read_lock, dest_read_lock);

  // For both source and destination dictionaries we cap the max
  // entries to be translated/translated to at the supplied
  // generation arguments, if valid (i.e. >= 0), otherwise just the
  // size of each dictionary

  CHECK_LE(num_source_strings, static_cast<int64_t>(str_count_));
  CHECK_LE(num_dest_strings, static_cast<int64_t>(dest_dict->str_count_));
  const bool dest_dictionary_is_empty = (num_dest_strings == 0);

  constexpr int64_t target_strings_per_thread{1000};
  ThreadInfo thread_info(
      std::thread::hardware_concurrency(), num_source_strings, target_strings_per_thread);
  try_parallelize_llm_transform(thread_info, string_ops, num_source_strings);
  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);

  // We use a tbb::task_arena to cap the number of threads, has been
  // in other contexts been shown to exhibit better performance when low
  // numbers of threads are needed than just letting tbb figure the number of threads,
  // but should benchmark in this specific context

  tbb::task_arena limited_arena(thread_info.num_threads);
  std::vector<size_t> num_strings_not_translated_per_thread(thread_info.num_threads, 0UL);
  constexpr bool short_circuit_empty_dictionary_translations{false};
  limited_arena.execute([&] {
    if (short_circuit_empty_dictionary_translations && dest_dictionary_is_empty) {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(
              0,
              num_source_strings,
              thread_info.num_elems_per_thread /* tbb grain_size */),
          [&](const tbb::blocked_range<int32_t>& r) {
            const int32_t start_idx = r.begin();
            const int32_t end_idx = r.end();
            for (int32_t string_idx = start_idx; string_idx != end_idx; ++string_idx) {
              translated_ids[string_idx] = INVALID_STR_ID;
            }
          },
          tbb::simple_partitioner());
      num_strings_not_translated_per_thread[0] += num_source_strings;
    } else {
      // The below logic, by executing low-level private variable accesses on both
      // dictionaries, is less clean than a previous variant that simply called
      // `getStringViews` from the source dictionary and then called `getBulk` on the
      // destination dictionary, but this version gets significantly better performance
      // (~2X), likely due to eliminating the overhead of writing out the string views and
      // then reading them back in (along with the associated cache misses)
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(
              0,
              num_source_strings,
              thread_info.num_elems_per_thread /* tbb grain_size */),
          [&](const tbb::blocked_range<int32_t>& r) {
            const int32_t start_idx = r.begin();
            const int32_t end_idx = r.end();
            size_t num_strings_not_translated = 0;
            std::string string_ops_storage;  // Needs to be thread local to back
                                             // string_view returned by string_ops()
            for (int32_t source_string_id = start_idx; source_string_id != end_idx;
                 ++source_string_id) {
              const std::string_view source_str =
                  has_string_ops ? string_ops(getStringFromStorageFast(source_string_id),
                                              string_ops_storage)
                                 : getStringFromStorageFast(source_string_id);

              if (source_str.empty()) {
                translated_ids[source_string_id] = inline_int_null_value<int32_t>();
                continue;
              }
              // Get the hash from this/the source dictionary's cache, as the function
              // will be the same for the dest_dict, sparing us having to recompute it

              // Todo(todd): Remove option to turn string hash cache off or at least
              // make a constexpr to avoid these branches when we expect it to be always
              // on going forward
              const string_dict_hash_t hash = (materialize_hashes_ && !has_string_ops)
                                                  ? hash_cache_[source_string_id]
                                                  : hash_string(source_str);
              const uint32_t hash_bucket = dest_dict->computeBucket(
                  hash, source_str, dest_dict->string_id_string_dict_hash_table_);
              const auto translated_string_id =
                  dest_dict->string_id_string_dict_hash_table_[hash_bucket];
              translated_ids[source_string_id] = translated_string_id;

              if (translated_string_id == StringDictionary::INVALID_STR_ID ||
                  translated_string_id >= num_dest_strings) {
                if (dest_has_transients) {
                  num_strings_not_translated +=
                      dest_transient_lookup_callback(source_str, source_string_id);
                } else {
                  num_strings_not_translated++;
                }
                continue;
              }
            }
            const size_t tbb_thread_idx = tbb::this_task_arena::current_thread_index();
            num_strings_not_translated_per_thread[tbb_thread_idx] +=
                num_strings_not_translated;
          },
          tbb::simple_partitioner());
    }
  });
  size_t total_num_strings_not_translated = 0;
  for (int64_t thread_idx = 0; thread_idx < thread_info.num_threads; ++thread_idx) {
    total_num_strings_not_translated += num_strings_not_translated_per_thread[thread_idx];
  }
  return total_num_strings_not_translated;
}

void StringDictionary::buildDictionaryNumericTranslationMap(
    Datum* translated_ids,
    const int64_t source_generation,
    const StringOps_Namespace::StringOps& string_ops) const {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_GE(source_generation, 0L);
  CHECK_GT(string_ops.size(), 0UL);
  CHECK(!string_ops.getLastStringOpsReturnType().is_string());
  const int64_t num_source_strings = source_generation;

  // We can bail early if there are no source strings to translate
  if (num_source_strings == 0L) {
    return;
  }

  std::shared_lock<std::shared_mutex> source_read_lock(rw_mutex_);

  // For source dictionary we cap the number of entries
  // to be translated/translated to at the supplied
  // generation arguments, if valid (i.e. >= 0), otherwise
  // just the size of each dictionary

  CHECK_LE(num_source_strings, static_cast<int64_t>(str_count_));

  constexpr int64_t target_strings_per_thread{1000};
  ThreadInfo thread_info(
      std::thread::hardware_concurrency(), num_source_strings, target_strings_per_thread);
  try_parallelize_llm_transform(thread_info, string_ops, num_source_strings);

  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);

  // We use a tbb::task_arena to cap the number of threads, has been
  // in other contexts been shown to exhibit better performance when low
  // numbers of threads are needed than just letting tbb figure the number of threads,
  // but should benchmark in this specific context

  CHECK_GT(string_ops.size(), 0UL);

  tbb::task_arena limited_arena(thread_info.num_threads);
  // The below logic, by executing low-level private variable accesses on both
  // dictionaries, is less clean than a previous variant that simply called
  // `getStringViews` from the source dictionary and then called `getBulk` on the
  // destination dictionary, but this version gets significantly better performance
  // (~2X), likely due to eliminating the overhead of writing out the string views and
  // then reading them back in (along with the associated cache misses)
  limited_arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<int32_t>(
            0, num_source_strings, thread_info.num_elems_per_thread /* tbb grain_size */),
        [&](const tbb::blocked_range<int32_t>& r) {
          const int32_t start_idx = r.begin();
          const int32_t end_idx = r.end();
          for (int32_t source_string_id = start_idx; source_string_id != end_idx;
               ++source_string_id) {
            const std::string source_str =
                std::string(getStringFromStorageFast(source_string_id));
            translated_ids[source_string_id] = string_ops.numericEval(source_str);
          }
        });
  });
}

size_t StringDictionary::computeCacheSize() const {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  std::lock_guard<std::mutex> scan_cache_lock(scan_lookup_cache_mutex_);
  return string_id_string_dict_hash_table_.size() * sizeof(int32_t) +
         hash_cache_.size() * sizeof(string_dict_hash_t) +
         sorted_cache_.size() * sizeof(int32_t) + like_cache_size_ + regex_cache_size_ +
         equal_cache_size_ + compare_cache_size_ + strings_cache_size_ +
         scan_lookup_cache_size_;
}

StringDictionary::StringDictMemoryUsage StringDictionary::getStringDictMemoryUsage() {
  return {total_mmap_size, total_temp_size, canary_buffer.size};
}

std::ostream& operator<<(std::ostream& os,
                         const StringDictionary::StringDictMemoryUsage& mu) {
  return os << "\"StringDictMemoryUsage\": {"
            << "\"mmap_size MB\": " << mu.mmap_size / (1024 * 1024)
            << ", \"temp_size MB\": " << mu.temp_dict_size / (1024 * 1024)
            << ", \"canary_size MB\": " << mu.canary_buffer_size / (1024 * 1024) << "}";
}
