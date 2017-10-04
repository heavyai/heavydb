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
#include <future>

class StringDictionaryClient;

class StringDictionary {
 public:
  StringDictionary(const std::string& folder,
                   const bool isTemp,
                   const bool recover,
                   size_t initial_capacity = 256) noexcept;
  StringDictionary(const LeafHostInfo& host, const int dict_id);
  ~StringDictionary() noexcept;

  int32_t getOrAdd(const std::string& str) noexcept;
  template <class T>
  void getOrAddBulk(const std::vector<std::string>& string_vec, T* encoded_vec);
  int32_t getIdOfString(const std::string& str) const;
  std::string getString(int32_t string_id) const;
  std::pair<char*, size_t> getStringBytes(int32_t string_id) const noexcept;
  size_t storageEntryCount() const;

  std::vector<int32_t> getLike(const std::string& pattern,
                               const bool icase,
                               const bool is_simple,
                               const char escape,
                               const size_t generation) const;

  std::vector<int32_t> getRegexpLike(const std::string& pattern, const char escape, const size_t generation) const;

  std::shared_ptr<const std::vector<std::string>> copyStrings() const;

  bool checkpoint() noexcept;

  static const int32_t INVALID_STR_ID;
  static const size_t MAX_STRLEN = (1 << 15) - 1;

 private:
  struct StringIdxEntry {
    uint64_t off : 48;
    uint64_t size : 16;
  };

  void processDictionaryFutures(
      std::vector<std::future<std::vector<std::pair<unsigned int, unsigned int>>>>& dictionary_futures);
  bool fillRateIsHigh() const noexcept;
  void increaseCapacity() noexcept;
  int32_t getOrAddImpl(const std::string& str, bool recover) noexcept;
  template <class T>
  void getOrAddBulkRemote(const std::vector<std::string>& string_vec, T* encoded_vec);
  int32_t getUnlocked(const std::string& str) const noexcept;
  std::string getStringUnlocked(int32_t string_id) const noexcept;
  std::string getStringChecked(const int string_id) const noexcept;
  std::pair<char*, size_t> getStringBytesChecked(const int string_id) const noexcept;
  int32_t computeBucket(const std::string& str, const std::vector<int32_t>& data, const bool unique) const noexcept;
  int32_t computeUniqueBucketWithHash(const size_t hash, const std::vector<int32_t>& data) const noexcept;
  void appendToStorage(const std::string& str) noexcept;
  std::tuple<char*, size_t, bool> getStringFromStorage(const int string_id) const noexcept;
  void addPayloadCapacity() noexcept;
  void addOffsetCapacity() noexcept;
  size_t addStorageCapacity(int fd) noexcept;
  void* addMemoryCapacity(void* addr, size_t& mem_size) noexcept;
  void invalidateInvertedIndex() noexcept;

  size_t str_count_;
  std::vector<int32_t> str_ids_;
  bool isTemp_;
  std::string offsets_path_;
  int payload_fd_;
  int offset_fd_;
  StringIdxEntry* offset_map_;
  char* payload_map_;
  size_t offset_file_size_;
  size_t payload_file_size_;
  size_t payload_file_off_;
  mutable mapd_shared_mutex rw_mutex_;
  mutable std::map<std::tuple<std::string, bool, bool, char>, std::vector<int32_t>> like_cache_;
  mutable std::map<std::pair<std::string, char>, std::vector<int32_t>> regex_cache_;
  mutable std::shared_ptr<std::vector<std::string>> strings_cache_;
  std::unique_ptr<StringDictionaryClient> client_;
  std::unique_ptr<StringDictionaryClient> client_no_timeout_;

  static char* CANARY_BUFFER;
};

int32_t truncate_to_generation(const int32_t id, const size_t generation);

void translate_string_ids(std::vector<int32_t>& dest_ids,
                          const LeafHostInfo& dict_server_host,
                          const int32_t dest_dict_id,
                          const std::vector<int32_t>& source_ids,
                          const int32_t source_dict_id,
                          const int32_t dest_generation);

#endif  // STRINGDICTIONARY_STRINGDICTIONARY_H
