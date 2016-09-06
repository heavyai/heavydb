#include "StringDictionary.h"
#include "../Shared/sqltypes.h"
#include "../Utils/StringLike.h"
#include "../Utils/Regexp.h"
#include "Shared/thread_count.h"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <glog/logging.h>
#include <sys/fcntl.h>

#include <thread>

namespace {
const int PAGE_SIZE = getpagesize();

size_t file_size(const int fd) {
  struct stat buf;
  fstat(fd, &buf);
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

StringDictionary::StringDictionary(const std::string& folder, const bool recover, size_t initial_capacity) noexcept
    : str_count_(0),
      str_ids_(initial_capacity, INVALID_STR_ID),
      payload_fd_(-1),
      offset_fd_(-1),
      offset_map_(nullptr),
      payload_map_(nullptr),
      offset_file_size_(0),
      payload_file_size_(0),
      payload_file_off_(0) {
  if (folder.empty()) {
    return;
  }
  // initial capacity must be a power of two for efficient bucket computation
  CHECK_EQ(size_t(0), (initial_capacity & (initial_capacity - 1)));
  boost::filesystem::path storage_path(folder);
  offsets_path_ = (storage_path / boost::filesystem::path("DictOffsets")).string();
  const auto payload_path = (storage_path / boost::filesystem::path("DictPayload")).string();
  payload_fd_ = checked_open(payload_path.c_str(), recover);
  offset_fd_ = checked_open(offsets_path_.c_str(), recover);
  payload_file_size_ = file_size(payload_fd_);
  if (payload_file_size_ == 0) {
    addPayloadCapacity();
  }
  offset_file_size_ = file_size(offset_fd_);
  if (offset_file_size_ == 0) {
    addOffsetCapacity();
  }
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

StringDictionary::~StringDictionary() noexcept {
  if (payload_map_) {
    CHECK(offset_map_);
    checked_munmap(payload_map_, payload_file_size_);
    checked_munmap(offset_map_, offset_file_size_);
    CHECK_GE(payload_fd_, 0);
    close(payload_fd_);
    CHECK_GE(offset_fd_, 0);
    close(offset_fd_);
  }
}

int32_t StringDictionary::getOrAdd(const std::string& str) noexcept {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  return getOrAddImpl(str, false);
}

template <class T>
void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec, T* encoded_vec) noexcept {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  size_t out_idx{0};
  for (const auto& str : string_vec) {
    encoded_vec[out_idx++] = getOrAddImpl(str, false);
  }
}

template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec, int8_t* encoded_vec) noexcept;
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec, int16_t* encoded_vec) noexcept;
template void StringDictionary::getOrAddBulk(const std::vector<std::string>& string_vec, int32_t* encoded_vec) noexcept;

int32_t StringDictionary::getOrAddTransient(const std::string& str) noexcept {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  auto transient_id = getUnlocked(str);
  if (transient_id != INVALID_STR_ID) {
    return transient_id;
  }
  const auto it = transient_str_to_int_.find(str);
  if (it != transient_str_to_int_.end()) {
    return it->second;
  }
  transient_id = -(transient_str_to_int_.size() + 2);  // make sure it's not INVALID_STR_ID
  {
    auto it_ok = transient_str_to_int_.insert(std::make_pair(str, transient_id));
    CHECK(it_ok.second);
  }
  {
    auto it_ok = transient_int_to_str_.insert(std::make_pair(transient_id, str));
    CHECK(it_ok.second);
  }
  return transient_id;
}

int32_t StringDictionary::get(const std::string& str) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return getUnlocked(str);
}

int32_t StringDictionary::getUnlocked(const std::string& str) const noexcept {
  auto str_id = str_ids_[computeBucket(str, str_ids_, false)];
  if (str_id != INVALID_STR_ID || transient_str_to_int_.empty()) {
    return str_id;
  }
  auto it = transient_str_to_int_.find(str);
  return it != transient_str_to_int_.end() ? it->second : INVALID_STR_ID;
}

std::string StringDictionary::getString(int32_t string_id) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return getStringUnlocked(string_id);
}

std::string StringDictionary::getStringUnlocked(int32_t string_id) const noexcept {
  if (string_id >= 0) {
    CHECK_LT(string_id, static_cast<int32_t>(str_count_));
    return getStringChecked(string_id);
  }
  CHECK_NE(INVALID_STR_ID, string_id);
  auto it = transient_int_to_str_.find(string_id);
  CHECK(it != transient_int_to_str_.end());
  return it->second;
}

std::pair<char*, size_t> StringDictionary::getStringBytes(int32_t string_id) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  CHECK_LE(0, string_id);
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringBytesChecked(string_id);
}

size_t StringDictionary::size() const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
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

std::vector<std::string> StringDictionary::getLike(const std::string& pattern,
                                                   const bool icase,
                                                   const bool is_simple,
                                                   const char escape) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  std::vector<std::string> result;
  std::vector<std::thread> workers;
  int worker_count = cpu_threads();
  CHECK_GT(worker_count, 0);
  std::vector<std::vector<std::string>> worker_results(worker_count);
  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&worker_results, &pattern, icase, is_simple, escape, worker_idx, worker_count, this]() {
      for (size_t string_id = worker_idx; string_id < str_count_; string_id += worker_count) {
        const auto str = getStringUnlocked(string_id);
        if (is_like(str, pattern, icase, is_simple, escape)) {
          worker_results[worker_idx].push_back(str);
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
  for (const auto& kv : transient_int_to_str_) {
    const auto str = getStringUnlocked(kv.first);
    if (is_like(str, pattern, icase, is_simple, escape)) {
      result.push_back(str);
    }
  }
  return result;
}

namespace {

bool is_regexp_like(const std::string& str, const std::string& pattern, const char escape) {
  return regexp_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape);
}

}  // namespace

std::vector<std::string> StringDictionary::getRegexpLike(const std::string& pattern, const char escape) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  std::vector<std::string> result;
  std::vector<std::thread> workers;
  int worker_count = cpu_threads();
  CHECK_GT(worker_count, 0);
  std::vector<std::vector<std::string>> worker_results(worker_count);
  for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    workers.emplace_back([&worker_results, &pattern, escape, worker_idx, worker_count, this]() {
      for (size_t string_id = worker_idx; string_id < str_count_; string_id += worker_count) {
        const auto str = getStringUnlocked(string_id);
        if (is_regexp_like(str, pattern, escape)) {
          worker_results[worker_idx].push_back(str);
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
  for (const auto& kv : transient_int_to_str_) {
    const auto str = getStringUnlocked(kv.first);
    if (is_regexp_like(str, pattern, escape)) {
      result.push_back(str);
    }
  }
  return result;
}

void StringDictionary::clearTransient() noexcept {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  decltype(transient_int_to_str_)().swap(transient_int_to_str_);
  decltype(transient_str_to_int_)().swap(transient_str_to_int_);
}

bool StringDictionary::fillRateIsHigh() const noexcept {
  return str_ids_.size() <= str_count_ * 2;
}

void StringDictionary::increaseCapacity() noexcept {
  const size_t MAX_STRCOUNT = 1 << 30;
  CHECK(str_count_ < MAX_STRCOUNT);
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
    return NULL_INT;
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
    if (data[bucket] == INVALID_STR_ID) {
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
  CHECK_GE(payload_fd_, 0);
  CHECK_GE(offset_fd_, 0);
  // write the payload
  if (payload_file_off_ + str.size() > payload_file_size_) {
    checked_munmap(payload_map_, payload_file_size_);
    addPayloadCapacity();
    CHECK(payload_file_off_ + str.size() <= payload_file_size_);
    payload_map_ = reinterpret_cast<char*>(checked_mmap(payload_fd_, payload_file_size_));
  }
  memcpy(payload_map_ + payload_file_off_, str.c_str(), str.size());
  // write the offset and length
  size_t offset_file_off = str_count_ * sizeof(StringIdxEntry);
  StringIdxEntry str_meta{static_cast<uint64_t>(payload_file_off_), str.size()};
  payload_file_off_ += str.size();
  if (offset_file_off + sizeof(str_meta) >= offset_file_size_) {
    checked_munmap(offset_map_, offset_file_size_);
    addOffsetCapacity();
    CHECK(offset_file_off + sizeof(str_meta) <= offset_file_size_);
    offset_map_ = reinterpret_cast<StringIdxEntry*>(checked_mmap(offset_fd_, offset_file_size_));
  }
  memcpy(offset_map_ + str_count_, &str_meta, sizeof(str_meta));
}

std::tuple<char*, size_t, bool> StringDictionary::getStringFromStorage(const int string_id) const noexcept {
  CHECK_GE(payload_fd_, 0);
  CHECK_GE(offset_fd_, 0);
  CHECK_GE(string_id, 0);
  const StringIdxEntry* str_meta = offset_map_ + string_id;
  if (str_meta->size == 0xffff) {
    // hit the canary
    return std::make_tuple(nullptr, 0, true);
  }
  return std::make_tuple(payload_map_ + str_meta->off, str_meta->size, false);
}

void StringDictionary::addPayloadCapacity() noexcept {
  payload_file_size_ += addStorageCapacity(payload_fd_);
}

void StringDictionary::addOffsetCapacity() noexcept {
  offset_file_size_ += addStorageCapacity(offset_fd_);
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

char* StringDictionary::CANARY_BUFFER{nullptr};

bool StringDictionary::checkpoint() noexcept {
  bool ret = true;
  ret = ret && (msync((void*)offset_map_, offset_file_size_, MS_SYNC) == 0);
  ret = ret && (msync((void*)payload_map_, payload_file_size_, MS_SYNC) == 0);
  ret = ret && (fsync(offset_fd_) == 0);
  ret = ret && (fsync(payload_fd_) == 0);
  return ret;
}
