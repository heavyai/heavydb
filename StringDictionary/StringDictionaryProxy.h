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

#ifndef STRINGDICTIONARY_STRINGDICTIONARYPROXY_H
#define STRINGDICTIONARY_STRINGDICTIONARYPROXY_H

#include "../Shared/mapd_shared_mutex.h"
#include "StringDictionary.h"

#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// used to access a StringDictionary when transient strings are involved
class StringDictionaryProxy {
  friend bool operator==(const StringDictionaryProxy& sdp1,
                         const StringDictionaryProxy& sdp2);
  friend bool operator!=(const StringDictionaryProxy& sdp1,
                         const StringDictionaryProxy& sdp2);

 public:
  StringDictionaryProxy(std::shared_ptr<StringDictionary> sd,
                        const int32_t string_dict_id,
                        const int64_t generation);

  int32_t getOrAdd(const std::string& str) noexcept;
  StringDictionary* getDictionary() const noexcept;
  int64_t getGeneration() const noexcept;
  std::vector<int32_t> getTransientBulk(const std::vector<std::string>& strings) const;
  int32_t getOrAddTransient(const std::string& str);
  std::vector<int32_t> getOrAddTransientBulk(const std::vector<std::string>& strings);
  int32_t getIdOfString(const std::string& str) const;
  int32_t getIdOfStringNoGeneration(
      const std::string& str) const;  // disregard generation, only used by QueryRenderer
  std::string getString(int32_t string_id) const;
  std::vector<std::string> getStrings(const std::vector<int32_t>& string_ids) const;
  std::pair<const char*, size_t> getStringBytes(int32_t string_id) const noexcept;

  /**
   * @brief Builds a vectorized string_id translation map from this proxy to dest_proxy
   *
   * @param dest_proxy StringDictionaryProxy that we are to map this proxy's string ids to
   *
   * @return std::vector<int32_t> linear dense vector map, where index 0
   * corresponds to the lowest (negative) transient id in this proxy,
   * and with each increasing index corresponding to the next string_id
   * I.e. if there are 3 transient entries in this proxy, and 20 in the underlying
   * string dictionary, there will be 25 total entries, mapping transient id -5
   * (as -1 and -0 are reserved, transients start at -2 (transient_id_ceil)
   * and descend downward). Entries corresponding to -1 and 0 may contain garbage,
   * it is expected that these entries are never accessed. The payload of
   * the vector map are the string ids in the dest_proxy corresponding to the indexed
   * string ids from this proxy
   *
   */
  std::vector<int32_t> buildTranslationMapToOtherProxy(
      const std::shared_ptr<StringDictionaryProxy> dest_proxy) const;
  std::vector<int32_t> buildTranslationMapToOtherProxy(
      const StringDictionaryProxy* dest_proxy) const;

  /**
   * @brief Returns the number of string entries in the underlying string dictionary,
   * at this proxy's generation_ if it is set/valid, otherwise just the current
   * size of the dictionary
   *
   * @return size_t Number of entries in the string dictionary
   * (at this proxy's generation if set)
   *
   */
  size_t storageEntryCount() const;

  /**
   * @brief
   *
   * @return size_t
   */

  /**
   * @brief Returns the number of transient string entries for this proxy,
   *
   * @return size_t Number of transient string entries for this proxy
   *
   */
  size_t transientEntryCount() const;

  /**
   * @brief Returns the number of total string entries for this proxy, both stored
   * in the underlying dictionary and in the transient map. Equal to
   * storageEntryCount() + transientEntryCount()
   *
   * @return size_t Number of total string entries for this proxy
   *
   */

  size_t entryCount() const;

  void updateGeneration(const int64_t generation) noexcept;

  std::vector<int32_t> getLike(const std::string& pattern,
                               const bool icase,
                               const bool is_simple,
                               const char escape) const;

  std::vector<int32_t> getCompare(const std::string& pattern,
                                  const std::string& comp_operator) const;

  std::vector<int32_t> getRegexpLike(const std::string& pattern, const char escape) const;

  const std::map<int32_t, std::string> getTransientMapping() const {
    return transient_int_to_str_;
  }

 private:
  int32_t transientLookupAndAddUnlocked(const std::string& str);
  template <typename String>
  int32_t lookupStringUnlocked(const String& lookup_string) const;
  template <typename String>
  void transientLookupBulk(const std::vector<String>& lookup_strings,
                           int32_t* string_ids) const;
  template <typename String>
  void transientLookupBulkUnlocked(const std::vector<String>& lookup_strings,
                                   int32_t* string_ids) const;
  template <typename String>
  void transientLookupBulkParallelUnlocked(const std::vector<String>& lookup_strings,
                                           int32_t* string_ids) const;
  std::shared_ptr<StringDictionary> string_dict_;
  const int32_t string_dict_id_;
  std::map<int32_t, std::string> transient_int_to_str_;
  std::map<std::string, int32_t> transient_str_to_int_;
  int64_t generation_;
  mutable mapd_shared_mutex rw_mutex_;
};
#endif  // STRINGDICTIONARY_STRINGDICTIONARYPROXY_H
