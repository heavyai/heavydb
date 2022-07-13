/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "StringDictionary/StringDictionaryProxy.h"
#include "StringOps/StringOps.h"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <algorithm>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/sort/spreadsort/string_sort.hpp>
#include <functional>
#include <future>
#include <iostream>
#include <string_view>
#include <thread>
#include <type_traits>

// TODO(adb): fixup
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <sys/fcntl.h>
#endif

#include "Logger/Logger.h"
#include "OSDependent/heavyai_fs.h"
#include "Shared/sqltypes.h"
#include "Shared/thread_count.h"
#include "StringDictionaryClient.h"
#include "Utils/Regexp.h"
#include "Utils/StringLike.h"

#include "LeafHostInfo.h"

bool g_cache_string_hash{true};

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

}  // namespace

bool g_enable_stringdict_parallel{false};
constexpr int32_t StringDictionary::INVALID_STR_ID;
constexpr size_t StringDictionary::MAX_STRLEN;
constexpr size_t StringDictionary::MAX_STRCOUNT;

StringDictionary::StringDictionary(const DictRef& dict_ref,
                                   const std::string& folder,
                                   const bool isTemp,
                                   const bool recover,
                                   const bool materializeHashes,
                                   size_t initial_capacity)
    : dict_ref_(dict_ref)
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
    , strings_cache_(nullptr) {
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
    if (recover) {
      const size_t bytes = heavyai::file_size(offset_fd_);
      if (bytes % sizeof(StringIdxEntry) != 0) {
        LOG(WARNING) << "Offsets " << offsets_path_ << " file is truncated";
      }
      const uint64_t str_count =
          storage_is_empty ? 0 : getNumStringsFromStorage(bytes / sizeof(StringIdxEntry));
      collisions_ = 0;
      // at this point we know the size of the StringDict we need to load
      // so lets reallocate the vector to the correct size
      const uint64_t max_entries =
          std::max(round_up_p2(str_count * 2 + 1),
                   round_up_p2(std::max(initial_capacity, static_cast<size_t>(1))));
      std::vector<int32_t> new_str_ids(max_entries, INVALID_STR_ID);
      string_id_string_dict_hash_table_.swap(new_str_ids);
      if (materialize_hashes_) {
        std::vector<string_dict_hash_t> new_hash_cache(max_entries / 2);
        hash_cache_.swap(new_hash_cache);
      }
      // Bail early if we know we don't have strings to add (i.e. a new or empty
      // dictionary)
      if (str_count == 0) {
        return;
      }

      unsigned string_id = 0;
      std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);

      uint32_t thread_inits = 0;
      const auto thread_count = std::thread::hardware_concurrency();
      const uint32_t items_per_thread = std::max<uint32_t>(
          2000, std::min<uint32_t>(200000, (str_count / thread_count) + 1));
      std::vector<std::future<std::vector<std::pair<string_dict_hash_t, unsigned int>>>>
          dictionary_futures;
      for (string_id = 0; string_id < str_count; string_id += items_per_thread) {
        dictionary_futures.emplace_back(std::async(
            std::launch::async, [string_id, str_count, items_per_thread, this] {
              std::vector<std::pair<string_dict_hash_t, unsigned int>> hashVec;
              for (uint32_t curr_id = string_id;
                   curr_id < string_id + items_per_thread && curr_id < str_count;
                   curr_id++) {
                const auto recovered = getStringFromStorage(curr_id);
                if (recovered.canary) {
                  // hit the canary, recovery finished
                  break;
                } else {
                  std::string_view temp(recovered.c_str_ptr, recovered.size);
                  hashVec.emplace_back(std::make_pair(hash_string(temp), temp.size()));
                }
              }
              return hashVec;
            }));
        thread_inits++;
        if (thread_inits % thread_count == 0) {
          processDictionaryFutures(dictionary_futures);
        }
      }
      // gather last few threads
      if (dictionary_futures.size() != 0) {
        processDictionaryFutures(dictionary_futures);
      }
      VLOG(1) << "Opened string dictionary " << folder << " # Strings: " << str_count_
              << " Hash table size: " << string_id_string_dict_hash_table_.size()
              << " Fill rate: "
              << static_cast<double>(str_count_) * 100.0 /
                     string_id_string_dict_hash_table_.size()
              << "% Collisions: " << collisions_;
    }
  }
}

namespace {
class MapMaker : public StringDictionary::StringCallback {
  std::unordered_map<std::string, int32_t> map_;

 public:
  void operator()(std::string const& str, int32_t const string_id) override {
    auto const emplaced = map_.emplace(str, string_id);
    CHECK(emplaced.second) << "str(" << str << ") string_id(" << string_id << ')';
  }
  void operator()(std::string_view const, int32_t const string_id) override {
    UNREACHABLE() << "MapMaker must be called with a std::string.";
  }
  std::unordered_map<std::string, int32_t> moveMap() { return std::move(map_); }
};
}  // namespace

std::function<int32_t(std::string const&)> StringDictionary::makeLambdaStringToId()
    const {
  CHECK(isClient());
  constexpr size_t big_gen = static_cast<size_t>(std::numeric_limits<size_t>::max());
  MapMaker map_maker;
  eachStringSerially(big_gen, map_maker);
  return [map{map_maker.moveMap()}](std::string const& str) {
    auto const itr = map.find(str);
    return itr == map.cend() ? INVALID_STR_ID : itr->second;
  };
}

// Call serial_callback for each (string/_view, string_id). Must be called serially.
void StringDictionary::eachStringSerially(int64_t const generation,
                                          StringCallback& serial_callback) const {
  if (isClient()) {
    // copyStrings() is not supported when isClient().
    std::string str;  // Import buffer. Placing outside of loop should reduce allocations.
    size_t const n = std::min(static_cast<size_t>(generation), storageEntryCount());
    CHECK_LE(n, static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1);
    for (unsigned id = 0; id < n; ++id) {
      {
        std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
        client_->get_string(str, id);
      }
      serial_callback(str, id);
    }
  } else {
    size_t const n = std::min(static_cast<size_t>(generation), str_count_);
    CHECK_LE(n, static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1);
    std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
    for (unsigned id = 0; id < n; ++id) {
      serial_callback(getStringFromStorageFast(static_cast<int>(id)), id);
    }
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
      payload_file_off_ += hash.second;
      string_id_string_dict_hash_table_[bucket] = static_cast<int32_t>(str_count_);
      if (materialize_hashes_) {
        hash_cache_[str_count_] = hash.first;
      }
      ++str_count_;
    }
  }
  dictionary_futures.clear();
}

int32_t StringDictionary::getDbId() const noexcept {
  return dict_ref_.dbId;
}

int32_t StringDictionary::getDictId() const noexcept {
  return dict_ref_.dictId;
}

/**
 * Method to retrieve number of strings in storage via a binary search for the first
 * canary
 * @param storage_slots number of storage entries we should search to find the minimum
 * canary
 * @return number of strings in storage
 */
size_t StringDictionary::getNumStringsFromStorage(const size_t storage_slots) const
    noexcept {
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

StringDictionary::StringDictionary(const LeafHostInfo& host, const DictRef dict_ref)
    : dict_ref_(dict_ref)
    , folder_("DB_" + std::to_string(dict_ref.dbId) + "_DICT_" +
              std::to_string(dict_ref.dictId))
    , strings_cache_(nullptr)
    , client_(new StringDictionaryClient(host, dict_ref, true))
    , client_no_timeout_(new StringDictionaryClient(host, dict_ref, false)) {}

StringDictionary::~StringDictionary() noexcept {
  free(CANARY_BUFFER);
  if (isClient()) {
    return;
  }
  if (payload_map_) {
    if (!isTemp_) {
      CHECK(offset_map_);
      heavyai::checked_munmap(payload_map_, payload_file_size_);
      heavyai::checked_munmap(offset_map_, offset_file_size_);
      CHECK_GE(payload_fd_, 0);
      heavyai::close(payload_fd_);
      CHECK_GE(offset_fd_, 0);
      heavyai::close(offset_fd_);
    } else {
      CHECK(offset_map_);
      free(payload_map_);
      free(offset_map_);
    }
  }
}

void StringDictionary::update_leaf(const LeafHostInfo& new_host) {
}

int32_t StringDictionary::getOrAdd(const std::string& str) noexcept {
  if (isClient()) {
    std::vector<int32_t> string_ids;
    client_->get_or_add_bulk(string_ids, std::vector<std::string>{str});
    CHECK_EQ(size_t(1), string_ids.size());
    return string_ids.front();
  }
  return getOrAddImpl(str);
}

namespace {

template <class T>
void throw_encoding_error(std::string_view str, const DictRef& dict_ref) {
  std::ostringstream oss;
  oss << "The text encoded column using dictionary " << dict_ref.toString()
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

void throw_string_too_long_error(std::string_view str, const DictRef& dict_ref) {
  std::ostringstream oss;
  oss << "The string '" << str << " could not be inserted into the dictionary "
      << dict_ref.toString() << " because it exceeded the maximum allowable "
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
  if (client_no_timeout_) {
    client_no_timeout_->get_or_add_bulk_array(ids_array_vec, string_array_vec);
    return;
  }

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
void StringDictionary::hashStrings(const std::vector<String>& string_vec,
                                   std::vector<string_dict_hash_t>& hashes) const
    noexcept {
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
    CHECK_LE(tbb::this_task_arena::max_concurrency(), thread_info.num_threads);
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
              throw_string_too_long_error(input_string, dict_ref_);
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
      throw_encoding_error<T>(input_string, dict_ref_);
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
      throw_encoding_error<T>(input_string, dict_ref_);
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
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  if (isClient()) {
    if constexpr (std::is_same_v<std::string, std::decay_t<String>>) {
      return client_->get(str);
    } else {
      return client_->get(std::string(str));
    }
  }
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
  if (isClient()) {
    std::string ret;
    client_->get_string(ret, string_id);
    return ret;
  }
  return getStringUnlocked(string_id);
}

std::string StringDictionary::getStringUnlocked(int32_t string_id) const noexcept {
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringChecked(string_id);
}

std::pair<char*, size_t> StringDictionary::getStringBytes(int32_t string_id) const
    noexcept {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  CHECK(!isClient());
  CHECK_LE(0, string_id);
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringBytesChecked(string_id);
}

size_t StringDictionary::storageEntryCount() const {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  if (isClient()) {
    return client_->storage_entry_count();
  }
  return str_count_;
}

namespace {

bool is_like(const std::string& str,
             const std::string& pattern,
             const bool icase,
             const bool is_simple,
             const char escape) {
  return icase
             ? (is_simple ? string_ilike_simple(
                                str.c_str(), str.size(), pattern.c_str(), pattern.size())
                          : string_ilike(str.c_str(),
                                         str.size(),
                                         pattern.c_str(),
                                         pattern.size(),
                                         escape))
             : (is_simple ? string_like_simple(
                                str.c_str(), str.size(), pattern.c_str(), pattern.size())
                          : string_like(str.c_str(),
                                        str.size(),
                                        pattern.c_str(),
                                        pattern.size(),
                                        escape));
}

}  // namespace

std::vector<int32_t> StringDictionary::getLike(const std::string& pattern,
                                               const bool icase,
                                               const bool is_simple,
                                               const char escape,
                                               const size_t generation) const {
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  if (isClient()) {
    return client_->get_like(pattern, icase, is_simple, escape, generation);
  }
  const auto cache_key = std::make_tuple(pattern, icase, is_simple, escape);
  const auto it = like_cache_.find(cache_key);
  if (it != like_cache_.end()) {
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
                          icase,
                          is_simple,
                          escape,
                          worker_idx,
                          worker_count,
                          this]() {
      for (size_t string_id = worker_idx; string_id < generation;
           string_id += worker_count) {
        const auto str = getStringUnlocked(string_id);
        if (is_like(str, pattern, icase, is_simple, escape)) {
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
  // place result into cache for reuse if similar query
  const auto it_ok = like_cache_.insert(std::make_pair(cache_key, result));

  CHECK(it_ok.second);

  return result;
}

std::vector<int32_t> StringDictionary::getEquals(std::string pattern,
                                                 std::string comp_operator,
                                                 size_t generation) {
  std::vector<int32_t> result;
  auto eq_id_itr = equal_cache_.find(pattern);
  int32_t eq_id = MAX_STRLEN + 1;
  int32_t cur_size = str_count_;
  if (eq_id_itr != equal_cache_.end()) {
    auto eq_id = eq_id_itr->second;
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

std::vector<int32_t> StringDictionary::getCompare(const std::string& pattern,
                                                  const std::string& comp_operator,
                                                  const size_t generation) {
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  if (isClient()) {
    return client_->get_compare(pattern, comp_operator, generation);
  }
  std::vector<int32_t> ret;
  if (str_count_ == 0) {
    return ret;
  }
  if (sorted_cache.size() < str_count_) {
    if (comp_operator == "=" || comp_operator == "<>") {
      return getEquals(pattern, comp_operator, generation);
    }

    buildSortedCache();
  }
  auto cache_index = compare_cache_.get(pattern);

  if (!cache_index) {
    cache_index = std::make_shared<StringDictionary::compare_cache_value_t>();
    const auto cache_itr = std::lower_bound(
        sorted_cache.begin(),
        sorted_cache.end(),
        pattern,
        [this](decltype(sorted_cache)::value_type const& a, decltype(pattern)& b) {
          auto a_str = this->getStringFromStorage(a);
          return string_lt(a_str.c_str_ptr, a_str.size, b.c_str(), b.size());
        });

    if (cache_itr == sorted_cache.end()) {
      cache_index->index = sorted_cache.size() - 1;
      cache_index->diff = 1;
    } else {
      const auto cache_str = getStringFromStorage(*cache_itr);
      if (!string_eq(
              cache_str.c_str_ptr, cache_str.size, pattern.c_str(), pattern.size())) {
        cache_index->index = cache_itr - sorted_cache.begin() - 1;
        cache_index->diff = 1;
      } else {
        cache_index->index = cache_itr - sorted_cache.begin();
        cache_index->diff = 0;
      }
    }

    compare_cache_.put(pattern, cache_index);
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
      ret.push_back(sorted_cache[i]);
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
      ret.push_back(sorted_cache[i]);
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
    for (size_t i = idx; i < sorted_cache.size(); i++) {
      ret.push_back(sorted_cache[i]);
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
    for (size_t i = idx; i < sorted_cache.size(); i++) {
      ret.push_back(sorted_cache[i]);
    }
  } else if (comp_operator == "=") {
    if (!cache_index->diff) {
      ret.push_back(sorted_cache[cache_index->index]);
    }

    // For <> operator it is simple matter of not including id of string which is equal
    // to pattern we are searching for.
  } else if (comp_operator == "<>") {
    if (!cache_index->diff) {
      size_t idx = cache_index->index;
      for (size_t i = 0; i < idx; i++) {
        ret.push_back(sorted_cache[i]);
      }
      ++idx;
      for (size_t i = idx; i < sorted_cache.size(); i++) {
        ret.push_back(sorted_cache[i]);
      }
    } else {
      for (size_t i = 0; i < sorted_cache.size(); i++) {
        ret.insert(ret.begin(), sorted_cache.begin(), sorted_cache.end());
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
  if (isClient()) {
    return client_->get_regexp_like(pattern, escape, generation);
  }
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
  CHECK(it_ok.second);

  return result;
}

std::vector<std::string> StringDictionary::copyStrings() const {
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  if (isClient()) {
    // TODO(miyu): support remote string dictionary
    throw std::runtime_error(
        "copying dictionaries from remote server is not supported yet.");
  }

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
  auto copy = [this](std::vector<std::string>& str_list,
                     const size_t start_id,
                     const size_t end_id) {
    CHECK_LE(start_id, end_id);
    str_list.reserve(end_id - start_id);
    for (size_t string_id = start_id; string_id < end_id; ++string_id) {
      str_list.push_back(getStringUnlocked(string_id));
    }
  };
  if (multithreaded) {
    std::vector<std::future<void>> workers;
    const auto stride = (str_count_ + (worker_count - 1)) / worker_count;
    for (size_t worker_idx = 0, start = 0, end = std::min(start + stride, str_count_);
         worker_idx < worker_count && start < str_count_;
         ++worker_idx, start += stride, end = std::min(start + stride, str_count_)) {
      workers.push_back(std::async(
          std::launch::async, copy, std::ref(worker_results[worker_idx]), start, end));
    }
    for (auto& worker : workers) {
      worker.get();
    }
  } else {
    CHECK_EQ(worker_results.size(), size_t(1));
    copy(worker_results[0], 0, str_count_);
  }

  for (const auto& worker_result : worker_results) {
    strings_cache_->insert(
        strings_cache_->end(), worker_result.begin(), worker_result.end());
  }
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
      if (candidate_string_id > 0 &&
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
      heavyai::checked_munmap(payload_map_, payload_file_size_);
      addPayloadCapacity(min_capacity_needed);
      CHECK(payload_file_off_ + write_length <= payload_file_size_);
      payload_map_ =
          reinterpret_cast<char*>(heavyai::checked_mmap(payload_fd_, payload_file_size_));
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
      heavyai::checked_munmap(offset_map_, offset_file_size_);
      addOffsetCapacity(min_capacity_needed);
      CHECK(offset_file_off + write_length <= offset_file_size_);
      offset_map_ = reinterpret_cast<StringIdxEntry*>(
          heavyai::checked_mmap(offset_fd_, offset_file_size_));
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

std::string_view StringDictionary::getStringFromStorageFast(const int string_id) const
    noexcept {
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

size_t StringDictionary::addStorageCapacity(
    int fd,
    const size_t min_capacity_requested) noexcept {
  const size_t canary_buff_size_to_add =
      std::max(static_cast<size_t>(1024 * SYSTEM_PAGE_SIZE),
               (min_capacity_requested / SYSTEM_PAGE_SIZE + 1) * SYSTEM_PAGE_SIZE);

  if (canary_buffer_size < canary_buff_size_to_add) {
    CANARY_BUFFER = static_cast<char*>(realloc(CANARY_BUFFER, canary_buff_size_to_add));
    canary_buffer_size = canary_buff_size_to_add;
    CHECK(CANARY_BUFFER);
    memset(CANARY_BUFFER, 0xff, canary_buff_size_to_add);
  }

  CHECK_NE(lseek(fd, 0, SEEK_END), -1);
  const auto write_return = write(fd, CANARY_BUFFER, canary_buff_size_to_add);
  CHECK(write_return > 0 &&
        (static_cast<size_t>(write_return) == canary_buff_size_to_add));
  return canary_buff_size_to_add;
}

void* StringDictionary::addMemoryCapacity(void* addr,
                                          size_t& mem_size,
                                          const size_t min_capacity_requested) noexcept {
  const size_t canary_buff_size_to_add =
      std::max(static_cast<size_t>(1024 * SYSTEM_PAGE_SIZE),
               (min_capacity_requested / SYSTEM_PAGE_SIZE + 1) * SYSTEM_PAGE_SIZE);
  if (canary_buffer_size < canary_buff_size_to_add) {
    CANARY_BUFFER =
        reinterpret_cast<char*>(realloc(CANARY_BUFFER, canary_buff_size_to_add));
    canary_buffer_size = canary_buff_size_to_add;
    CHECK(CANARY_BUFFER);
    memset(CANARY_BUFFER, 0xff, canary_buff_size_to_add);
  }
  void* new_addr = realloc(addr, mem_size + canary_buff_size_to_add);
  CHECK(new_addr);
  void* write_addr = reinterpret_cast<void*>(static_cast<char*>(new_addr) + mem_size);
  CHECK(memcpy(write_addr, CANARY_BUFFER, canary_buff_size_to_add));
  mem_size += canary_buff_size_to_add;
  return new_addr;
}

void StringDictionary::invalidateInvertedIndex() noexcept {
  if (!like_cache_.empty()) {
    decltype(like_cache_)().swap(like_cache_);
  }
  if (!regex_cache_.empty()) {
    decltype(regex_cache_)().swap(regex_cache_);
  }
  if (!equal_cache_.empty()) {
    decltype(equal_cache_)().swap(equal_cache_);
  }
  compare_cache_.invalidateInvertedIndex();
}

// TODO 5 Mar 2021 Nothing will undo the writes to dictionary currently on a failed
// load.  The next write to the dictionary that does checkpoint will make the
// uncheckpointed data be written to disk. Only option is a table truncate, and thats
// assuming not replicated dictionary
bool StringDictionary::checkpoint() noexcept {
  if (isClient()) {
    try {
      return client_->checkpoint();
    } catch (...) {
      return false;
    }
  }
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

bool StringDictionary::isClient() const noexcept {
  return static_cast<bool>(client_);
}

void StringDictionary::buildSortedCache() {
  // This method is not thread-safe.
  const auto cur_cache_size = sorted_cache.size();
  std::vector<int32_t> temp_sorted_cache;
  for (size_t i = cur_cache_size; i < str_count_; i++) {
    temp_sorted_cache.push_back(i);
  }
  sortCache(temp_sorted_cache);
  mergeSortedCache(temp_sorted_cache);
}

void StringDictionary::sortCache(std::vector<int32_t>& cache) {
  // This method is not thread-safe.

  // this boost sort is creating some problems when we use UTF-8 encoded strings.
  // TODO (vraj): investigate What is wrong with boost sort and try to mitigate it.

  std::sort(cache.begin(), cache.end(), [this](int32_t a, int32_t b) {
    auto a_str = this->getStringFromStorage(a);
    auto b_str = this->getStringFromStorage(b);
    return string_lt(a_str.c_str_ptr, a_str.size, b_str.c_str_ptr, b_str.size);
  });
}

void StringDictionary::mergeSortedCache(std::vector<int32_t>& temp_sorted_cache) {
  // this method is not thread safe
  std::vector<int32_t> updated_cache(temp_sorted_cache.size() + sorted_cache.size());
  size_t t_idx = 0, s_idx = 0, idx = 0;
  for (; t_idx < temp_sorted_cache.size() && s_idx < sorted_cache.size(); idx++) {
    auto t_string = getStringFromStorage(temp_sorted_cache[t_idx]);
    auto s_string = getStringFromStorage(sorted_cache[s_idx]);
    const auto insert_from_temp_cache =
        string_lt(t_string.c_str_ptr, t_string.size, s_string.c_str_ptr, s_string.size);
    if (insert_from_temp_cache) {
      updated_cache[idx] = temp_sorted_cache[t_idx++];
    } else {
      updated_cache[idx] = sorted_cache[s_idx++];
    }
  }
  while (t_idx < temp_sorted_cache.size()) {
    updated_cache[idx++] = temp_sorted_cache[t_idx++];
  }
  while (s_idx < sorted_cache.size()) {
    updated_cache[idx++] = sorted_cache[s_idx++];
  }
  sorted_cache.swap(updated_cache);
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
    CHECK_LE(tbb::this_task_arena::max_concurrency(), thread_info.num_threads);
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

void order_translation_locks(const int32_t source_db_id,
                             const int32_t source_dict_id,
                             const int32_t dest_db_id,
                             const int32_t dest_dict_id,
                             std::shared_lock<std::shared_mutex>& source_read_lock,
                             std::shared_lock<std::shared_mutex>& dest_read_lock) {
  const bool dicts_are_same =
      source_db_id == dest_db_id && source_dict_id == dest_dict_id;
  const bool source_dict_is_locked_first =
      source_db_id < dest_db_id ||
      (source_db_id == dest_db_id && source_dict_id < dest_dict_id);
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
    const std::vector<StringOps_Namespace::StringOpInfo>& string_op_infos) const {
  auto timer = DEBUG_TIMER(__func__);
  CHECK_GE(source_generation, 0L);
  CHECK_GE(dest_generation, 0L);
  const int64_t num_source_strings = source_generation;
  const int64_t num_dest_strings = dest_generation;

  // We can bail early if there are no source strings to translate
  if (num_source_strings == 0L) {
    return 0;
  }

  // If here we should should have local dictionaries.
  // Note case of transient source dictionaries that aren't
  // seen as remote (they have no client_no_timeout_) is covered
  // by early bail above on num_source_strings == 0
  if (dest_dict->client_no_timeout_) {
    throw std::runtime_error(
        "Cannot translate between a local source and remote destination dictionary.");
  }

  // Sort this/source dict and dest dict on folder_ so we can enforce
  // lock ordering and avoid deadlocks
  std::shared_lock<std::shared_mutex> source_read_lock(rw_mutex_, std::defer_lock);
  std::shared_lock<std::shared_mutex> dest_read_lock(dest_dict->rw_mutex_,
                                                     std::defer_lock);
  order_translation_locks(getDbId(),
                          getDictId(),
                          dest_dict->getDbId(),
                          dest_dict->getDictId(),
                          source_read_lock,
                          dest_read_lock);

  // For both source and destination dictionaries we cap the max
  // entries to be translated/translated to at the supplied
  // generation arguments, if valid (i.e. >= 0), otherwise just the
  // size of each dictionary

  CHECK_LE(num_source_strings, static_cast<int64_t>(str_count_));
  CHECK_LE(num_dest_strings, static_cast<int64_t>(dest_dict->str_count_));
  const bool dest_dictionary_is_empty = (num_dest_strings == 0);

  constexpr int64_t target_strings_per_thread{1000};
  const ThreadInfo thread_info(
      std::thread::hardware_concurrency(), num_source_strings, target_strings_per_thread);
  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);

  // We use a tbb::task_arena to cap the number of threads, has been
  // in other contexts been shown to exhibit better performance when low
  // numbers of threads are needed than just letting tbb figure the number of threads,
  // but should benchmark in this specific context

  const StringOps_Namespace::StringOps string_ops(string_op_infos);
  const bool has_string_ops = string_ops.size();

  tbb::task_arena limited_arena(thread_info.num_threads);
  std::vector<size_t> num_strings_not_translated_per_thread(thread_info.num_threads, 0UL);
  constexpr bool short_circuit_empty_dictionary_translations{false};
  limited_arena.execute([&] {
    CHECK_LE(tbb::this_task_arena::max_concurrency(), thread_info.num_threads);
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

void translate_string_ids(std::vector<int32_t>& dest_ids,
                          const LeafHostInfo& dict_server_host,
                          const DictRef dest_dict_ref,
                          const std::vector<int32_t>& source_ids,
                          const DictRef source_dict_ref,
                          const int32_t dest_generation) {
  DictRef temp_dict_ref(-1, -1);
  StringDictionaryClient string_client(dict_server_host, temp_dict_ref, false);
  string_client.translate_string_ids(
      dest_ids, dest_dict_ref, source_ids, source_dict_ref, dest_generation);
}
