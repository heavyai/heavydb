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

#ifndef STRINGDICTIONARY_STRINGDICTIONARYPROXY_H
#define STRINGDICTIONARY_STRINGDICTIONARYPROXY_H

#include "StringDictionary.h"

#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// used to access a StringDictionary when transient strings are involved
class StringDictionaryProxy {
 public:
  StringDictionaryProxy(StringDictionaryProxy const&) = delete;
  StringDictionaryProxy const& operator=(StringDictionaryProxy const&) = delete;
  StringDictionaryProxy(std::shared_ptr<StringDictionary> sd,
                        const int32_t string_dict_id,
                        const int64_t generation);

  int32_t getDictId() const noexcept { return string_dict_id_; };

  bool operator==(StringDictionaryProxy const&) const;
  bool operator!=(StringDictionaryProxy const&) const;

  // enum SetOp { kUnion = 0, kIntersection };

  int32_t getOrAdd(const std::string& str) noexcept;
  StringDictionary* getDictionary() const noexcept;
  int64_t getGeneration() const noexcept;

  /**
   * @brief Executes read-only lookup of a vector of strings and returns a vector of their
   integer ids
  *
  * This function, unlike getOrAddTransientBulk, will not add strings to the dictionary.
  * Use this function if strings that don't currently exist in the StringDictionaryProxy
  * should not be added to the proxy as transient entries.
  * This method also has performance advantages over getOrAddTransientBulk for read-only
  * use cases, in that it can:
  * 1) Take a read lock instead of a write lock for the transient lookups
  * 2) Use a tbb::parallel_for implementation of the transient string lookups as
  * we are guaranteed that the underlying map of strings to int ids cannot change

  * @param strings - Vector of strings to perform string id lookups on
  * @return A vector of string_ids of the same length as strings, containing
  * the id of any strings for which were found in the underlying StringDictionary
  * instance or in the proxy's tranient map, otherwise
  * StringDictionary::INVALID_STR_ID for strings not found.
  */

  std::vector<int32_t> getTransientBulk(const std::vector<std::string>& strings) const;
  int32_t getOrAddTransient(const std::string& str);
  // Not currently used
  std::vector<int32_t> getOrAddTransientBulk(const std::vector<std::string>& strings);
  int32_t getIdOfString(const std::string& str) const;
  int32_t getIdOfStringNoGeneration(
      const std::string& str) const;  // disregard generation, only used by QueryRenderer
  std::string getString(int32_t string_id) const;
  std::vector<std::string> getStrings(const std::vector<int32_t>& string_ids) const;
  std::pair<const char*, size_t> getStringBytes(int32_t string_id) const noexcept;

  class IdMap {
    size_t const offset_;
    std::vector<int32_t> vector_map_;

   public:
    // +1 is added to skip string_id=-1 reserved for INVALID_STR_ID. id_map[-1]==-1.
    IdMap(uint32_t const tran_size, uint32_t const dict_size)
        : offset_(tran_size + 1)
        , vector_map_(offset_ + dict_size, StringDictionary::INVALID_STR_ID) {}
    IdMap(IdMap const&) = delete;
    IdMap(IdMap&&) = default;
    bool empty() const { return vector_map_.size() == 1; }
    inline size_t getIndex(int32_t const id) const { return offset_ + id; }
    std::vector<int32_t> const& getVectorMap() const { return vector_map_; }
    size_t numTransients() const { return offset_ - 1; }
    size_t numNonTransients() const { return vector_map_.size() - offset_; }
    int32_t* data() { return vector_map_.data(); }
    int32_t const* data() const { return vector_map_.data(); }
    int32_t domainStart() const { return -static_cast<int32_t>(offset_); }
    int32_t* storageData() { return vector_map_.data() + offset_; }
    int32_t& operator[](int32_t const id) { return vector_map_[getIndex(id)]; }
    int32_t operator[](int32_t const id) const { return vector_map_[getIndex(id)]; }
    friend std::ostream& operator<<(std::ostream&, IdMap const&);
  };

  IdMap initIdMap() const { return IdMap(transient_string_vec_.size(), generation_); }

  /**
   * @brief Builds a vectorized string_id translation map from this proxy to dest_proxy
   *
   * @param dest_proxy StringDictionaryProxy that we are to map this proxy's string ids to
   *
   * @return An IdMap which encapsulates a std::vector<int32_t> of string ids
   * for both transient and non-transient strings, mapping to their translated string_ids.
   * offset_ is defined to be the number of transient entries + 1.
   * The ordering of values in the vector_map_ is:
   *  * the transient ids (there are offset_-1 of these)
   *  * INVALID_STR_ID (=-1)
   *  * the non-transient string ids
   * For example if there are 3 transient entries in this proxy and 20 in the underlying
   * string dictionary, then vector_map_ will be of size() == 24 and offset_=3+1.
   * The formula to translate ids is new_id = vector_map_[offset_ + old_id].
   * It is always the case that vector_map_[offset_-1]==-1 so that INVALID_STR_ID
   * maps to INVALID_STR_ID.
   *
   */
  IdMap buildTranslationMapToOtherProxy(const StringDictionaryProxy* dest_proxy) const;

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

  // The std::string must live in the map, and std::string const* in the vector. As
  // desirable as it might be to have it the other way, string addresses won't change
  // in the std::map when new strings are added, but will change in a std::vector.
  using TransientMap = std::map<std::string, int32_t, std::less<>>;

  const std::vector<std::string const*>& getTransientVector() const {
    return transient_string_vec_;
  }

  // INVALID_STR_ID = -1 is reserved for invalid string_ids.
  // Thus the greatest valid transient string_id is -2.
  static unsigned transientIdToIndex(int32_t const id) {
    constexpr int max_transient_string_id = -2;
    return static_cast<unsigned>(max_transient_string_id - id);
  }

  static int32_t transientIndexToId(unsigned const index) {
    constexpr int max_transient_string_id = -2;
    return static_cast<int32_t>(max_transient_string_id - index);
  }

  // Iterate over transient strings, then non-transients.
  void eachStringSerially(StringDictionary::StringCallback&) const;

  // Union strings from both StringDictionaryProxies into *this as transients.
  // Return map of old string_ids to new string_ids.
  IdMap transientUnion(StringDictionaryProxy const&);

 private:
  size_t transientEntryCountUnlocked() const;
  size_t entryCountUnlocked() const;
  template <typename String>
  int32_t lookupTransientStringUnlocked(const String& lookup_string) const;
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
  TransientMap transient_str_to_int_;
  // Holds pointers into transient_str_to_int_
  std::vector<std::string const*> transient_string_vec_;
  int64_t generation_;
  mutable mapd_shared_mutex rw_mutex_;

  // Return INVALID_STR_ID if not found on string_dict_. Don't lock or check transients.
  template <typename String>
  int32_t getIdOfStringFromClient(String const&) const;
  template <typename String>
  int32_t getOrAddTransientUnlocked(String const&);

  friend class StringLocalCallback;
  friend class StringNetworkCallback;
};
#endif  // STRINGDICTIONARY_STRINGDICTIONARYPROXY_H
