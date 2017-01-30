#ifndef STRINGDICTIONARY_STRINGDICTIONARY_H
#define STRINGDICTIONARY_STRINGDICTIONARY_H

#include "../LeafHostInfo.h"
#include "../Shared/mapd_shared_mutex.h"

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

class StringDictionaryClient;

class StringDictionary {
 public:
  StringDictionary(const std::string& folder, const bool recover = true, size_t initial_capacity = 256) noexcept;
  StringDictionary(const LeafHostInfo& host, const int dict_id);
  ~StringDictionary() noexcept;

  int32_t getOrAdd(const std::string& str) noexcept;
  template <class T>
  void getOrAddBulk(const std::vector<std::string>& string_vec, T* encoded_vec) noexcept;
  int32_t getIdOfString(const std::string& str) const;
  std::string getString(int32_t string_id) const;
  std::pair<char*, size_t> getStringBytes(int32_t string_id) const noexcept;
  size_t storageEntryCount() const noexcept;

  std::vector<std::string> getLike(const std::string& pattern,
                                   const bool icase,
                                   const bool is_simple,
                                   const char escape) const noexcept;

  std::vector<std::string> getRegexpLike(const std::string& pattern, const char escape) const noexcept;

  bool checkpoint() noexcept;

  static const int32_t INVALID_STR_ID;
  static const size_t MAX_STRLEN = (2 << 16) - 1;

 private:
  struct StringIdxEntry {
    uint64_t off : 48;
    uint64_t size : 16;
  };

  bool fillRateIsHigh() const noexcept;
  void increaseCapacity() noexcept;
  int32_t getOrAddImpl(const std::string& str, bool recover) noexcept;
  int32_t getUnlocked(const std::string& str) const noexcept;
  std::string getStringUnlocked(int32_t string_id) const noexcept;
  std::string getStringChecked(const int string_id) const noexcept;
  std::pair<char*, size_t> getStringBytesChecked(const int string_id) const noexcept;
  int32_t computeBucket(const std::string& str, const std::vector<int32_t>& data, const bool unique) const noexcept;
  void appendToStorage(const std::string& str) noexcept;
  std::tuple<char*, size_t, bool> getStringFromStorage(const int string_id) const noexcept;
  void addPayloadCapacity() noexcept;
  void addOffsetCapacity() noexcept;
  size_t addStorageCapacity(int fd) noexcept;
  void invalidateInvertedIndex() noexcept;

  size_t str_count_;
  std::vector<int32_t> str_ids_;
  std::string offsets_path_;
  int payload_fd_;
  int offset_fd_;
  StringIdxEntry* offset_map_;
  char* payload_map_;
  size_t offset_file_size_;
  size_t payload_file_size_;
  size_t payload_file_off_;
  mutable mapd_shared_mutex rw_mutex_;
  mutable std::map<std::tuple<std::string, bool, bool, char>, std::vector<std::string>> like_cache_;
  mutable std::map<std::pair<std::string, char>, std::vector<std::string>> regex_cache_;
  std::unique_ptr<StringDictionaryClient> client_;

  static char* CANARY_BUFFER;
};

#endif  // STRINGDICTIONARY_STRINGDICTIONARY_H
