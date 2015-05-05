#include "StringDictionary.h"
#include "../Shared/sqltypes.h"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <glog/logging.h>
#include <sys/fcntl.h>

namespace {
  const int PAGE_SIZE = getpagesize();
  const size_t MAX_STRLEN = (2 << 16) - 1;
  const int32_t INVALID_STR_ID = -1;

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
    return ptr;
  }
}  // namespace

StringDictionary::StringDictionary(
    const std::string& folder,
    const bool recover,
    size_t initial_capacity)
  : str_count_(0)
  , str_ids_(initial_capacity, INVALID_STR_ID)
  , payload_fd_(-1), offset_fd_(-1)
  , offset_map_(nullptr), payload_map_(nullptr)
  , offset_file_size_(0), payload_file_size_(0)
  , payload_file_off_(0) {
  if (folder.empty()) {
    return;
  }
  // initial capacity must be a power of two for efficient bucket computation
  CHECK_EQ(0, (initial_capacity & (initial_capacity - 1)));
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
    unsigned string_id = 0;
    boost::unique_lock<boost::shared_mutex> write_lock(rw_mutex_);
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

StringDictionary::~StringDictionary() {
  CHECK(payload_map_);
  CHECK(offset_map_);
  munmap(payload_map_, payload_file_size_);
  munmap(offset_map_, offset_file_size_);
  close(payload_fd_);
  close(offset_fd_);
}

int32_t StringDictionary::getOrAdd(const std::string& str) {
  boost::unique_lock<boost::shared_mutex> write_lock(rw_mutex_);
  return getOrAddImpl(str, false);
}

void StringDictionary::addBulk(const std::vector<std::string> &stringVec, std::vector<int32_t> &encodedVec) {
  encodedVec.reserve(stringVec.size());
  boost::unique_lock<boost::shared_mutex> write_lock(rw_mutex_);
  for (const auto &str : stringVec) {
    encodedVec.push_back(getOrAddImpl(str, false));
  }
}

int32_t StringDictionary::getOrAddTransient(const std::string& str) {
  boost::unique_lock<boost::shared_mutex> write_lock(rw_mutex_);
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

int32_t StringDictionary::get(const std::string& str) const {
  boost::shared_lock<boost::shared_mutex> read_lock(rw_mutex_);
  return getUnlocked(str);
}

int32_t StringDictionary::getUnlocked(const std::string& str) const {
  auto str_id = str_ids_[computeBucket(str, str_ids_)];
  if (str_id != INVALID_STR_ID || transient_str_to_int_.empty()) {
    return str_id;
  }
  auto it = transient_str_to_int_.find(str);
  return it != transient_str_to_int_.end() ? it->second : INVALID_STR_ID;
}

std::string StringDictionary::getString(int32_t string_id) const {
  boost::shared_lock<boost::shared_mutex> read_lock(rw_mutex_);
  if (string_id >= 0) {
    CHECK_LT(string_id, static_cast<int32_t>(str_count_));
    return getStringChecked(string_id);
  }
  CHECK_NE(INVALID_STR_ID, string_id);
  auto it = transient_int_to_str_.find(string_id);
  CHECK(it != transient_int_to_str_.end());
  return it->second;
}

std::pair<char*, size_t> StringDictionary::getStringBytes(int32_t string_id) const {
  boost::shared_lock<boost::shared_mutex> read_lock(rw_mutex_);
  CHECK_LE(0, string_id);
  CHECK_LT(string_id, static_cast<int32_t>(str_count_));
  return getStringBytesChecked(string_id);
}

void StringDictionary::clearTransient() {
  boost::unique_lock<boost::shared_mutex> write_lock(rw_mutex_);
  decltype(transient_int_to_str_)().swap(transient_int_to_str_);
  decltype(transient_str_to_int_)().swap(transient_str_to_int_);
}

bool StringDictionary::fillRateIsHigh() const {
  return str_ids_.size() <= str_count_ * 2;
}

void StringDictionary::increaseCapacity() {
  const size_t MAX_STRCOUNT = 1 << 30;
  CHECK(str_count_ < MAX_STRCOUNT);
  std::vector<int32_t> new_str_ids(str_ids_.size() * 2, INVALID_STR_ID);
  for (size_t i = 0; i < str_count_; ++i) {
    const auto str = getStringChecked(i);
    int32_t bucket = computeBucket(str, new_str_ids);
    new_str_ids[bucket] = i;
  }
  str_ids_.swap(new_str_ids);
}

int32_t StringDictionary::getOrAddImpl(const std::string& str, bool recover) {
  // @TODO(wei) treat empty string as NULL for now
  if (str.size() == 0)
    return NULL_INT;
  CHECK(str.size() <= MAX_STRLEN);
  int32_t bucket = computeBucket(str, str_ids_);
  if (str_ids_[bucket] == INVALID_STR_ID) {
    if (fillRateIsHigh()) {
      // resize when more than 50% is full
      increaseCapacity();
      bucket = computeBucket(str, str_ids_);
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

std::string StringDictionary::getStringChecked(const int string_id) const {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!std::get<2>(str_canary));
  return std::string(std::get<0>(str_canary), std::get<1>(str_canary));
}

std::pair<char*, size_t> StringDictionary::getStringBytesChecked(const int string_id) const {
  const auto str_canary = getStringFromStorage(string_id);
  CHECK(!std::get<2>(str_canary));
  return std::make_pair(std::get<0>(str_canary), std::get<1>(str_canary));
}

namespace {

size_t rk_hash(const std::string& str) {
  size_t str_hash = 1;
  for (size_t i = 0; i < str.size(); ++i) {
    str_hash = str_hash * 997 + str[i];
  }
  return str_hash;
}

}  // namespace

int32_t StringDictionary::computeBucket(
    const std::string& str,
    const std::vector<int32_t>& data) const {
  auto bucket = rk_hash(str) & (data.size() - 1);
  while (true) {
    if (data[bucket] == INVALID_STR_ID) {
      break;
    }
    const auto old_str = getStringChecked(data[bucket]);
    if (str.size() == old_str.size() &&
        !memcmp(str.c_str(), old_str.c_str(), str.size())) {
      // found the string
      break;
    }
    // wrap around
    if (++bucket == data.size()) {
      bucket = 0;
    }
  }
  return bucket;
}

void StringDictionary::appendToStorage(const std::string& str) {
  CHECK_GE(payload_fd_, 0);
  CHECK_GE(offset_fd_, 0);
  // write the payload
  if (payload_file_off_ + str.size() > payload_file_size_) {
    munmap(payload_map_, payload_file_size_);
    addPayloadCapacity();
    CHECK(payload_file_off_ + str.size() <= payload_file_size_);
    payload_map_ = reinterpret_cast<char*>(checked_mmap(payload_fd_, payload_file_size_));
  }
  memcpy(payload_map_ + payload_file_off_, str.c_str(), str.size());
  // write the offset and length
  size_t offset_file_off = str_count_ * sizeof(StringIdxEntry);
  StringIdxEntry str_meta { static_cast<uint64_t>(payload_file_off_), str.size() };
  payload_file_off_ += str.size();
  if (offset_file_off + sizeof(str_meta) >= offset_file_size_) {
    munmap(offset_map_, offset_file_size_);
    addOffsetCapacity();
    CHECK(offset_file_off + sizeof(str_meta) <= offset_file_size_);
    offset_map_ = reinterpret_cast<StringIdxEntry*>(checked_mmap(offset_fd_, offset_file_size_));
  }
  memcpy(offset_map_ + str_count_, &str_meta, sizeof(str_meta));
}

std::tuple<char*, size_t, bool> StringDictionary::getStringFromStorage(const int string_id) const {
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

void StringDictionary::addPayloadCapacity() {
  payload_file_size_ += addStorageCapacity(payload_fd_);
}

void StringDictionary::addOffsetCapacity() {
  offset_file_size_ += addStorageCapacity(offset_fd_);
}

size_t StringDictionary::addStorageCapacity(int fd) {
  static const ssize_t CANARY_BUFF_SIZE = 1024 * PAGE_SIZE;
  if (!CANARY_BUFFER) {
    CANARY_BUFFER = static_cast<char*>(malloc(CANARY_BUFF_SIZE));
    memset(CANARY_BUFFER, 0xff, CANARY_BUFF_SIZE);
  }
  CHECK_NE(lseek(fd, 0, SEEK_END), -1);
  CHECK(write(fd, CANARY_BUFFER, CANARY_BUFF_SIZE) == CANARY_BUFF_SIZE);
  return CANARY_BUFF_SIZE;
}

char* StringDictionary::CANARY_BUFFER { nullptr };

bool
StringDictionary::checkpoint()
{
  bool ret = true;
  ret = ret && (msync((void*)offset_map_, offset_file_size_, MS_SYNC) == 0);
  ret = ret && (msync((void*)payload_map_, payload_file_size_, MS_SYNC) == 0);
  ret = ret && (fsync(offset_fd_) == 0);
  ret = ret && (fsync(payload_fd_) == 0);
  return ret;
}
