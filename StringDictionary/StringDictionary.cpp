/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "StringDictionary.h"
#include "StringDictionaryClient.h"
#include "../Shared/sqltypes.h"
#include "../Utils/StringLike.h"
#include "../Utils/Regexp.h"
#include "Shared/thread_count.h"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <glog/logging.h>
#include <sys/fcntl.h>

#include <thread>
#include <future>

namespace {
const int PAGE_SIZE = getpagesize();

size_t file_size(const int fd) {
  struct stat buf;
  int err = fstat(fd, &buf);
  CHECK_EQ(0, err);
  return buf.st_size;
}

int checked_open(const char* path, const bool recover) {
  auto fd = open(path, O_RDWR | O_CREAT | (recover ? O_APPEND : O_TRUNC), 0644);
  CHECK_GE(fd, 0);
  return fd;
}

void* checked_mmap(const int fd, const size_t sz) {
  auto ptr = mmap(nullptr, sz, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
  CHECK(ptr != reinterpret_cast<void*>(-1));
#ifdef __linux__
#ifdef MADV_HUGEPAGE
  madvise(ptr, sz, MADV_RANDOM | MADV_WILLNEED | MADV_HUGEPAGE);
#else
  madvise(ptr, sz, MADV_RANDOM | MADV_WILLNEED);
#endif
#endif
  return ptr;
}

void checked_munmap(void* addr, size_t length) {
  CHECK_EQ(0, munmap(addr, length));
}

const uint32_t round_up_p2(const size_t num) {
  uint32_t in = num;
  in--;
  in |= in >> 1;
  in |= in >> 2;
  in |= in >> 4;
  in |= in >> 8;
  in |= in >> 16;
  in++;
  // TODO MAT deal with case where filesize has been increased but reality is
  // we are constrained to 2^30.
  // In that situation this calculation will wrap to zero
  if (in == 0) {
    in = 1 << 31;
  }
  return in;
}

size_t rk_hash(const std::string& str) {
  size_t str_hash = 1;
  for (size_t i = 0; i < str.size(); ++i) {
    str_hash = str_hash * 997 + str[i];
  }
  return str_hash;
}
}  // namespace

const int32_t StringDictionary::INVALID_STR_ID{-1};

StringDictionary::StringDictionary(const std::string& folder,
                                   const bool isTemp,
                                   const bool recover,
                                   size_t initial_capacity) noexcept : str_count_(0),
                                                                       str_ids_(initial_capacity, INVALID_STR_ID),
                                                                       isTemp_(isTemp),
                                                                       payload_fd_(-1),
                                                                       offset_fd_(-1),
                                                                       offset_map_(nullptr),
                                                                       payload_map_(nullptr),
                                                                       offset_file_size_(0),
                                                                       payload_file_size_(0),
                                                                       payload_file_off_(0),
                                                                       strings_cache_(nullptr) {
  if (!isTemp && folder.empty()) {
    return;
  }
  // initial capacity must be a power of two for efficient bucket computation
  CHECK_EQ(size_t(0), (initial_capacity & (initial_capacity - 1)));
  if (!isTemp_) {
    boost::filesystem::path storage_path(folder);
    offsets_path_ = (storage_path / boost::filesystem::path("DictOffsets")).string();
    const auto payload_path = (storage_path / boost::filesystem::path("DictPayload")).string();
    payload_fd_ = checked_open(payload_path.c_str(), recover);
    offset_fd_ = checked_open(offsets_path_.c_str(), recover);
    payload_file_size_ = file_size(payload_fd_);
    offset_file_size_ = file_size(offset_fd_);
  }

  if (payload_file_size_ == 0) {
    addPayloadCapacity();
  }
  if (offset_file_size_ == 0) {
    addOffsetCapacity();
  }
  if (!isTemp_) {  // we never mmap or recover temp dictionaries
    payload_map_ = reinterpret_cast<char*>(checked_mmap(payload_fd_, payload_file_size_));
    offset_map_ = reinterpret_cast<StringIdxEntry*>(checked_mmap(offset_fd_, offset_file_size_));
    if (recover) {
      const size_t bytes = file_size(offset_fd_);
      const unsigned str_count = bytes / sizeof(StringIdxEntry);
      // at this point we know the size of the StringDict we need to load
      // so lets reallocate the vector to the correct size
      const uint32_t max_entries = round_up_p2(str_count * 2 + 1);
      std::vector<int32_t> new_str_ids(max_entries, INVALID_STR_ID);
      str_ids_.swap(new_str_ids);
      unsigned string_id = 0;
      mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
      for (string_id = 0; string_id < str_count; ++string_id) {
        const auto recovered = getStringFromStorage(string_id);
        if (std::get<2>(recovered)) {
          // hit the canary, recovery finished
          break;
        }
        getOrAddImpl(std::string(std::get<0>(recovered), std::get<1>(recovered)), true);
      }
      if (bytes % sizeof(StringIdxEntry) != 0) {
        LOG(WARNING) << "Offsets " << offsets_path_ << " file is truncated";
      }
    }
  }
}

StringDictionary::StringDictionary(const LeafHostInfo& host, const int dict_id)
    : strings_cache_(nullptr),
      client_(new StringDictionaryClient(host, dict_id, true)),
      client_no_timeout_(new StringDictionaryClient(host, dict_id, false)) {}

StringDictionary::~StringDictionary() noexcept {
  if (client_) {
    return;
  }
  if (payload_map_) {
    if (!isTemp_) {
      CHECK(offset_map_);
      checked_munmap(payload_map_, payload_file_size_);
      checked_munmap(offset_map_, offset_file_size_);
      CHECK_GE(payload_fd_, 0);
      close(payload_fd_);
      CHECK_GE(offset_fd_, 0);
      close(offset_fd_);
    } else {
      CHECK(offset_map_);
      free(payload_map_);
      free(offset_map_);
    }
  }
}

int32_t StringDictionary::getOrAdd(const std::string& str) noexcept {
  if (client_) {
    std::vector<int32_t> string_ids;
    client_->get_or_add_bulk(string_ids, {str});
    CHECK_EQ(size_t(1), string_ids.size());
    return string_ids.front();
  }
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  return getOrAddImpl(str, false);
}

namespace {

template <class T>
void log_encoding_error(const std::string& str) {
  LOG(ERROR) << "Could not encode string: " << str << ", the encoded value doesn't fit in " << sizeof(T) * 8
             << " bits. Will store NULL instead.";
}

}  // namespace

template <class T>
void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec, T* encoded_vec) {
  if (client_no_timeout_) {
    getOrAddBulkRemote(string_vec, encoded_vec);
    return;
  }
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  size_t out_idx{0};
  for (const auto& str : string_vec) {
    const auto string_id = getOrAddImpl(str, false);
    const bool invalid = string_id > max_valid_int_value<T>();
    if (invalid || string_id == inline_int_null_value<int32_t>()) {
      if (invalid) {
        log_encoding_error<T>(str);
      }
      encoded_vec[out_idx++] = inline_int_null_value<T>();
      continue;
    }
    encoded_vec[out_idx++] = string_id;
  }
}

template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec, uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec, uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec, int32_t* encoded_vec);

template <class T>
void StringDictionary::getOrAddBulkRemote(const std::vector<std::string>& string_vec, T* encoded_vec) {
  CHECK(client_no_timeout_);
  std::vector<int32_t> string_ids;
  client_no_timeout_->get_or_add_bulk(string_ids, string_vec);
  size_t out_idx{0};
  for (size_t i = 0; i < string_ids.size(); ++i) {
    const auto string_id = string_ids[i];
    const bool invalid = string_id > max_valid_int_value<T>();
    if (invalid || string_id == inline_int_null_value<int32_t>()) {
      if (invalid) {
        log_encoding_error<T>(string_vec[i]);
      }
      encoded_vec[out_idx++] = inline_int_null_value<T>();
      continue;
    }
    encoded_vec[out_idx++] = string_id;
  }
}

template void StringDictionary::getOrAddBulkRemote(const std::vector<std::string>& string_vec, uint8_t* encoded_vec);
template void StringDictionary::getOrAddBulkRemote(const std::vector<std::string>& string_vec, uint16_t* encoded_vec);
template void StringDictionary::getOrAddBulkRemote(const std::vector<std::string>& string_vec, int32_t* encoded_vec);

int32_t StringDictionary::getIdOfString(const std::string& str) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  if (client_) {
    return client_->get(str);
  }
  return getUnlocked(str);
}

int32_t StringDictionary::getUnlocked(const std::string& str) const noexcept {
  auto str_id = str_ids_[computeBucket(str, str_ids_, false)];
  return str_id;
}

std::string StringDictionary::getString(int32_t string_id) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  if (client_) {
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

std::pair<char*, size_t> StringDictionary::getStringBytes(int32_t string_id) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  CHECK(!client_);
  CHECK_LE(0, string_id);
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringBytesChecked(string_id);
}

size_t StringDictionary::storageEntryCount() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  if (client_) {
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
  return icase ? (is_simple ? string_ilike_simple(str.c_str(), str.size(), pattern.c_str(), pattern.size())
                            : string_ilike(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape))
               : (is_simple ? string_like_simple(str.c_str(), str.size(), pattern.c_str(), pattern.size())
                            : string_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape));
}

}  // namespace

std::vector<int32_t> StringDictionary::getLike(const std::string& pattern,
                                               const bool icase,
                                               const bool is_simple,
                                               const char escape,
                                               const size_t generation) const {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  if (client_) {
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
    workers.emplace_back(
        [&worker_results, &pattern, generation, icase, is_simple, escape, worker_idx, worker_count, this]() {
          for (size_t string_id = worker_idx; string_id < generation; string_id += worker_count) {
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

namespace {

bool is_regexp_like(const std::string& str, const std::string& pattern, const char escape) {
  return regexp_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape);
}

}  // namespace

std::vector<int32_t> StringDictionary::getRegexpLike(const std::string& pattern,
                                                     const char escape,
                                                     const size_t generation) const {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  if (client_) {
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
    workers.emplace_back([&worker_results, &pattern, generation, escape, worker_idx, worker_count, this]() {
      for (size_t string_id = worker_idx; string_id < generation; string_id += worker_count) {
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

std::shared_ptr<const std::vector<std::string>> StringDictionary::copyStrings() const {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  if (client_) {
    // TODO(miyu): support remote string dictionary
    throw std::runtime_error("copying dictionaries from remote server is not supported yet.");
  }

  if (strings_cache_) {
    return strings_cache_;
  }

  strings_cache_ = std::make_shared<std::vector<std::string>>();
  strings_cache_->reserve(str_count_);
  const bool multithreaded = str_count_ > 10000;
  const auto worker_count = multithreaded ? static_cast<size_t>(cpu_threads()) : size_t(1);
  CHECK_GT(worker_count, 0);
  std::vector<std::vector<std::string>> worker_results(worker_count);
  auto copy = [this](std::vector<std::string>& str_list, const size_t start_id, const size_t end_id) {
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
      workers.push_back(std::async(std::launch::async, copy, std::ref(worker_results[worker_idx]), start, end));
    }
    for (auto& worker : workers) {
      worker.get();
    }
  } else {
    CHECK_EQ(worker_results.size(), size_t(1));
    copy(worker_results[0], 0, str_count_);
  }

  for (const auto& worker_result : worker_results) {
    strings_cache_->insert(strings_cache_->end(), worker_result.begin(), worker_result.end());
  }
  return strings_cache_;
}

bool StringDictionary::fillRateIsHigh() const noexcept {
  return str_ids_.size() <= str_count_ * 2;
}

void StringDictionary::increaseCapacity() noexcept {
  const size_t MAX_STRCOUNT = 1 << 30;
  if (str_count_ >= MAX_STRCOUNT) {
    LOG(FATAL) << "Maximum number (" << str_count_
               << ") of Dictionary encoded Strings reached for this column, offset path for column is  "
               << offsets_path_;
  }
  std::vector<int32_t> new_str_ids(str_ids_.size() * 2, INVALID_STR_ID);
  for (size_t i = 0; i < str_count_; ++i) {
    const auto str = getStringChecked(i);
    int32_t bucket = computeBucket(str, new_str_ids, true);
    new_str_ids[bucket] = i;
  }
  str_ids_.swap(new_str_ids);
}

int32_t StringDictionary::getOrAddImpl(const std::string& str, bool recover) noexcept {
  // @TODO(wei) treat empty string as NULL for now
  if (str.size() == 0)
    return inline_int_null_value<int32_t>();
  CHECK(str.size() <= MAX_STRLEN);
  int32_t bucket = computeBucket(str, str_ids_, recover);
  if (str_ids_[bucket] == INVALID_STR_ID) {
    if (fillRateIsHigh()) {
      // resize when more than 50% is full
      increaseCapacity();
      bucket = computeBucket(str, str_ids_, recover);
    }
    if (recover) {
      payload_file_off_ += str.size();
    } else {
      appendToStorage(str);
    }
    str_ids_[bucket] = static_cast<int32_t>(str_count_);
    ++str_count_;
    invalidateInvertedIndex();
  }
  return str_ids_[bucket];
}

std::string StringDictionary::getStringChecked(const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!std::get<2>(str_canary));
  return std::string(std::get<0>(str_canary), std::get<1>(str_canary));
}

std::pair<char*, size_t> StringDictionary::getStringBytesChecked(const int string_id) const noexcept {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!std::get<2>(str_canary));
  return std::make_pair(std::get<0>(str_canary), std::get<1>(str_canary));
}

int32_t StringDictionary::computeBucket(const std::string& str,
                                        const std::vector<int32_t>& data,
                                        const bool unique) const noexcept {
  auto bucket = rk_hash(str) & (data.size() - 1);
  while (true) {
    if (data[bucket] == INVALID_STR_ID) {  // In this case it means the slot is available for use
      break;
    }
    // if records are unique I don't need to do this test as I know it will not be the same
    if (!unique) {
      const auto old_str = getStringChecked(data[bucket]);
      if (str.size() == old_str.size() && !memcmp(str.c_str(), old_str.c_str(), str.size())) {
        // found the string
        break;
      }
    }
    // wrap around
    if (++bucket == data.size()) {
      bucket = 0;
    }
  }
  return bucket;
}

void StringDictionary::appendToStorage(const std::string& str) noexcept {
  if (!isTemp_) {
    CHECK_GE(payload_fd_, 0);
    CHECK_GE(offset_fd_, 0);
  }
  // write the payload
  if (payload_file_off_ + str.size() > payload_file_size_) {
    if (!isTemp_) {
      checked_munmap(payload_map_, payload_file_size_);
      addPayloadCapacity();
      CHECK(payload_file_off_ + str.size() <= payload_file_size_);
      payload_map_ = reinterpret_cast<char*>(checked_mmap(payload_fd_, payload_file_size_));
    } else
      addPayloadCapacity();
  }
  memcpy(payload_map_ + payload_file_off_, str.c_str(), str.size());
  // write the offset and length
  size_t offset_file_off = str_count_ * sizeof(StringIdxEntry);
  StringIdxEntry str_meta{static_cast<uint64_t>(payload_file_off_), str.size()};
  payload_file_off_ += str.size();
  if (offset_file_off + sizeof(str_meta) >= offset_file_size_) {
    if (!isTemp_) {
      checked_munmap(offset_map_, offset_file_size_);
      addOffsetCapacity();
      CHECK(offset_file_off + sizeof(str_meta) <= offset_file_size_);
      offset_map_ = reinterpret_cast<StringIdxEntry*>(checked_mmap(offset_fd_, offset_file_size_));
    } else
      addOffsetCapacity();
  }
  memcpy(offset_map_ + str_count_, &str_meta, sizeof(str_meta));
}

std::tuple<char*, size_t, bool> StringDictionary::getStringFromStorage(const int string_id) const noexcept {
  if (!isTemp_) {
    CHECK_GE(payload_fd_, 0);
    CHECK_GE(offset_fd_, 0);
  }
  CHECK_GE(string_id, 0);
  const StringIdxEntry* str_meta = offset_map_ + string_id;
  if (str_meta->size == 0xffff) {
    // hit the canary
    return std::make_tuple(nullptr, 0, true);
  }
  return std::make_tuple(payload_map_ + str_meta->off, str_meta->size, false);
}

void StringDictionary::addPayloadCapacity() noexcept {
  if (!isTemp_)
    payload_file_size_ += addStorageCapacity(payload_fd_);
  else
    payload_map_ = static_cast<char*>(addMemoryCapacity(payload_map_, payload_file_size_));
}

void StringDictionary::addOffsetCapacity() noexcept {
  if (!isTemp_)
    offset_file_size_ += addStorageCapacity(offset_fd_);
  else
    offset_map_ = static_cast<StringIdxEntry*>(addMemoryCapacity(offset_map_, offset_file_size_));
}

size_t StringDictionary::addStorageCapacity(int fd) noexcept {
  static const ssize_t CANARY_BUFF_SIZE = 1024 * PAGE_SIZE;
  if (!CANARY_BUFFER) {
    CANARY_BUFFER = static_cast<char*>(malloc(CANARY_BUFF_SIZE));
    CHECK(CANARY_BUFFER);
    memset(CANARY_BUFFER, 0xff, CANARY_BUFF_SIZE);
  }
  CHECK_NE(lseek(fd, 0, SEEK_END), -1);
  CHECK(write(fd, CANARY_BUFFER, CANARY_BUFF_SIZE) == CANARY_BUFF_SIZE);
  return CANARY_BUFF_SIZE;
}

void* StringDictionary::addMemoryCapacity(void* addr, size_t& mem_size) noexcept {
  static const ssize_t CANARY_BUFF_SIZE = 1024 * PAGE_SIZE;
  if (!CANARY_BUFFER) {
    CANARY_BUFFER = reinterpret_cast<char*>(malloc(CANARY_BUFF_SIZE));
    CHECK(CANARY_BUFFER);
    memset(CANARY_BUFFER, 0xff, CANARY_BUFF_SIZE);
  }
  void* new_addr = realloc(addr, mem_size + CANARY_BUFF_SIZE);
  CHECK(new_addr);
  void* write_addr = reinterpret_cast<void*>(static_cast<char*>(new_addr) + mem_size);
  CHECK(memcpy(write_addr, CANARY_BUFFER, CANARY_BUFF_SIZE));
  mem_size += CANARY_BUFF_SIZE;
  return new_addr;
}

void StringDictionary::invalidateInvertedIndex() noexcept {
  if (!like_cache_.empty()) {
    decltype(like_cache_)().swap(like_cache_);
  }
  if (!regex_cache_.empty()) {
    decltype(regex_cache_)().swap(regex_cache_);
  }
}

char* StringDictionary::CANARY_BUFFER{nullptr};

bool StringDictionary::checkpoint() noexcept {
  if (client_) {
    try {
      return client_->checkpoint();
    } catch (...) {
      return false;
    }
  }
  CHECK(!isTemp_);
  bool ret = true;
  ret = ret && (msync((void*)offset_map_, offset_file_size_, MS_SYNC) == 0);
  ret = ret && (msync((void*)payload_map_, payload_file_size_, MS_SYNC) == 0);
  ret = ret && (fsync(offset_fd_) == 0);
  ret = ret && (fsync(payload_fd_) == 0);
  return ret;
}

void translate_string_ids(std::vector<int32_t>& dest_ids,
                          const LeafHostInfo& dict_server_host,
                          const int32_t dest_dict_id,
                          const std::vector<int32_t>& source_ids,
                          const int32_t source_dict_id,
                          const int32_t dest_generation) {
  StringDictionaryClient string_client(dict_server_host, -1, true);
  string_client.translate_string_ids(dest_ids, dest_dict_id, source_ids, source_dict_id, dest_generation);
}
