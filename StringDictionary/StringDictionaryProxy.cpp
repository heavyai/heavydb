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

#include "StringDictionary/StringDictionaryProxy.h"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>

#include "Logger/Logger.h"
#include "Shared/sqltypes.h"
#include "Shared/thread_count.h"
#include "StringDictionary/StringDictionary.h"
#include "Utils/Regexp.h"
#include "Utils/StringLike.h"

constexpr int32_t transient_id_ceil{-2};

StringDictionaryProxy::StringDictionaryProxy(std::shared_ptr<StringDictionary> sd,
                                             const int32_t string_dict_id,
                                             const int64_t generation)
    : string_dict_(sd), string_dict_id_(string_dict_id), generation_(generation) {}

int32_t truncate_to_generation(const int32_t id, const size_t generation) {
  if (id == StringDictionary::INVALID_STR_ID) {
    return id;
  }
  CHECK_GE(id, 0);
  return static_cast<size_t>(id) >= generation ? StringDictionary::INVALID_STR_ID : id;
}

std::vector<int32_t> StringDictionaryProxy::getTransientBulk(
    const std::vector<std::string>& strings) const {
  CHECK_GE(generation_, 0);
  const size_t num_strings = strings.size();
  std::vector<int32_t> string_ids(num_strings);
  if (num_strings == 0) {
    return string_ids;
  }
  // Use fast parallel String::Dictionary getBulk method
  // Todo: Evaluate getBulk method that takes callback to do transient lookup
  // to avoid a second rescan of the data
  const size_t num_strings_not_found = string_dict_->getBulk(strings, string_ids.data());
  if (num_strings_not_found > 0) {
    // Dictionary could not find at least 1 target string, now look these up
    // in the transient dictionary
    transientLookupBulk(strings, string_ids.data());
  }
  return string_ids;
}

int32_t StringDictionaryProxy::getOrAddTransient(const std::string& str) {
  CHECK_GE(generation_, 0);
  auto transient_id =
      truncate_to_generation(string_dict_->getIdOfString(str), generation_);
  if (transient_id != StringDictionary::INVALID_STR_ID) {
    return transient_id;
  }
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  return transientLookupAndAddUnlocked(str);
}

std::vector<int32_t> StringDictionaryProxy::getOrAddTransientBulk(
    const std::vector<std::string>& strings) {
  CHECK_GE(generation_, 0);
  const size_t num_strings = strings.size();
  std::vector<int32_t> string_ids(num_strings);
  if (num_strings == 0) {
    return string_ids;
  }
  // Since new strings added to a StringDictionaryProxy are not materialized in the
  // proxy's underlying StringDictionary, we can use the fast parallel
  // StringDictionary::getBulk method to fetch ids from the underlying dictionary (which
  // will return StringDictionary::INVALID_STR_ID for strings that don't exist)

  // Don't need to be under lock here as the string ids for strings in the underlying
  // materialized dictionary are immutable
  const size_t num_strings_not_found = string_dict_->getBulk(strings, string_ids.data());
  if (num_strings_not_found > 0) {
    mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
    for (size_t string_idx = 0; string_idx < num_strings; ++string_idx) {
      const auto transient_id =
          truncate_to_generation(string_ids[string_idx], generation_);
      if (transient_id == StringDictionary::INVALID_STR_ID) {
        string_ids[string_idx] = transientLookupAndAddUnlocked(strings[string_idx]);
      }
    }
  }
  return string_ids;
}

int32_t StringDictionaryProxy::getIdOfString(const std::string& str) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  CHECK_GE(generation_, 0);
  auto str_id = truncate_to_generation(string_dict_->getIdOfString(str), generation_);
  if (str_id != StringDictionary::INVALID_STR_ID || transient_str_to_int_.empty()) {
    return str_id;
  }
  auto it = transient_str_to_int_.find(str);
  return it != transient_str_to_int_.end() ? it->second
                                           : StringDictionary::INVALID_STR_ID;
}

int32_t StringDictionaryProxy::getIdOfStringNoGeneration(const std::string& str) const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  auto str_id = string_dict_->getIdOfString(str);
  if (str_id != StringDictionary::INVALID_STR_ID || transient_str_to_int_.empty()) {
    return str_id;
  }
  auto it = transient_str_to_int_.find(str);
  return it != transient_str_to_int_.end() ? it->second
                                           : StringDictionary::INVALID_STR_ID;
}

std::string StringDictionaryProxy::getString(int32_t string_id) const {
  if (inline_int_null_value<int32_t>() == string_id) {
    return "";
  }
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  if (string_id >= 0 && storageEntryCount() > 0) {
    return string_dict_->getString(string_id);
  }
  CHECK_NE(StringDictionary::INVALID_STR_ID, string_id);
  auto it = transient_int_to_str_.find(string_id);
  CHECK(it != transient_int_to_str_.end());
  return it->second;
}

std::vector<std::string> StringDictionaryProxy::getStrings(
    const std::vector<int32_t>& string_ids) const {
  const size_t num_string_ids = string_ids.size();
  std::vector<std::string> strings;
  if (num_string_ids == size_t(0)) {
    return strings;
  }
  strings.reserve(num_string_ids);
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  for (const auto& string_id : string_ids) {
    if (inline_int_null_value<int32_t>() == string_id) {
      strings.emplace_back("");
      continue;
    }
    if (string_id >= 0) {
      strings.emplace_back(string_dict_->getString(string_id));
      continue;
    }
    auto it = transient_int_to_str_.find(string_id);
    strings.emplace_back(it->second);
  }
  return strings;
}
std::pair<int32_t, int32_t> StringDictionaryProxy::getRange() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return getRangeUnlocked();
}

std::pair<int32_t, int32_t> StringDictionaryProxy::getRangeUnlocked() const {
  const int32_t storage_entry_count = storageEntryCount();
  const int32_t transient_entry_count = transientEntryCountUnlocked();
  const int32_t min_elem =
      transient_entry_count > 0 ? transient_id_ceil + 1 - transient_entry_count : 0;
  const int32_t max_plus_one_elem =
      storage_entry_count > 0 ? storage_entry_count
                              : (transient_entry_count ? transient_id_ceil + 1 : 0);
  return std::make_pair(min_elem, max_plus_one_elem);
}

template <>
int32_t StringDictionaryProxy::lookupTransientStringUnlocked(
    const std::string& lookup_string) const {
  const auto it = transient_str_to_int_.find(lookup_string);
  if (it != transient_str_to_int_.end()) {
    return it->second;
  }
  return StringDictionary::INVALID_STR_ID;
}

template <>
int32_t StringDictionaryProxy::lookupTransientStringUnlocked(
    const std::string_view& lookup_string) const {
  const auto it = transient_str_to_int_.find(std::string(lookup_string));
  if (it != transient_str_to_int_.end()) {
    return it->second;
  }
  return StringDictionary::INVALID_STR_ID;
}

std::shared_ptr<StringDictionaryProxyTranslationMap>
StringDictionaryProxy::buildTranslationMapToOtherProxy(
    const StringDictionaryProxy* dest_proxy) const {
  auto timer = DEBUG_TIMER(__func__);
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  auto str_proxy_translation_map =
      std::make_shared<StringDictionaryProxyTranslationMap>(getRangeUnlocked());
  CHECK(str_proxy_translation_map);

  if (str_proxy_translation_map->isEmpty()) {
    return str_proxy_translation_map;
  }

  // First map transient strings, store at front of vector map
  const size_t num_transient_entries = str_proxy_translation_map->numTransientEntries();
  if (num_transient_entries) {
    std::vector<std::string> transient_lookup_strings(num_transient_entries);

    for (const auto& transient_entry : transient_int_to_str_) {
      const auto transient_id = transient_entry.first;
      CHECK_LE(transient_id, transient_id_ceil);
      CHECK_GT(transient_id,
               transient_id_ceil - static_cast<int32_t>(num_transient_entries));
      const size_t map_idx = transient_entry.first + num_transient_entries + 1;
      transient_lookup_strings[map_idx] = transient_entry.second;
    }
    // This lookup may have a different snapshot of
    // dest_proxy transients and dictionary than what happends under
    // the below dest_proxy_read_lock. We may need an unlocked version of
    // getTransientBulk to ensure consistency (I don't believe
    // current behavior would cause crashes/races, verify this though)

    // Todo(todd): Consider implementing a getTransientBulk call that takes
    // an allocated pointer to avoid extra copy of return vector into
    // the already allocated translation map

    const auto transient_str_to_id_vec_map =
        dest_proxy->getTransientBulk(transient_lookup_strings);
    CHECK_GE(str_proxy_translation_map->size(), transient_str_to_id_vec_map.size());
    std::copy(transient_str_to_id_vec_map.begin(),
              transient_str_to_id_vec_map.end(),
              str_proxy_translation_map->translation_map_.begin());
  }

  // Now map strings in dictionary
  // We start non transient strings after the transient strings
  // if they exist, otherwise at 0
  auto translation_map_stored_entries_ptr =
      str_proxy_translation_map->storageEntriesPtr();

  auto dest_transient_lookup_callback = [dest_proxy, translation_map_stored_entries_ptr](
                                            const std::string_view& source_string,
                                            const int32_t source_string_id) {
    translation_map_stored_entries_ptr[source_string_id] =
        dest_proxy->lookupTransientStringUnlocked(source_string);
    return translation_map_stored_entries_ptr[source_string_id] ==
           StringDictionary::INVALID_STR_ID;
  };

  mapd_lock_guard<mapd_shared_mutex> dest_proxy_read_lock(dest_proxy->rw_mutex_);
  const size_t num_dest_transients = dest_proxy->transientEntryCountUnlocked();

  const size_t num_strings_not_translated =
      string_dict_->buildDictionaryTranslationMap(dest_proxy->string_dict_.get(),
                                                  translation_map_stored_entries_ptr,
                                                  generation_,
                                                  dest_proxy->generation_,
                                                  num_dest_transients > 0UL,
                                                  dest_transient_lookup_callback);
  const size_t num_dest_entries = dest_proxy->entryCountUnlocked();
  const size_t num_total_entries = str_proxy_translation_map->size();
  CHECK_GT(num_total_entries, 0UL);
  CHECK_LE(num_strings_not_translated, str_proxy_translation_map->size());
  const size_t num_entries_translated = num_total_entries - num_strings_not_translated;
  const float match_pct =
      100.0 * static_cast<float>(num_entries_translated) / num_total_entries;
  VLOG(1) << std::fixed << std::setprecision(2) << match_pct << "% ("
          << num_entries_translated << " entries) from dictionary ("
          << string_dict_->getDbId() << ", " << string_dict_->getDictId() << ") with "
          << num_total_entries << " total entries ( " << num_transient_entries
          << " literals)"
          << " translated to dictionary (" << dest_proxy->string_dict_->getDbId() << ", "
          << dest_proxy->string_dict_->getDictId() << ") with " << num_dest_entries
          << " total entries (" << dest_proxy->transientEntryCountUnlocked()
          << " literals).";

  return str_proxy_translation_map;
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

std::vector<int32_t> StringDictionaryProxy::getLike(const std::string& pattern,
                                                    const bool icase,
                                                    const bool is_simple,
                                                    const char escape) const {
  CHECK_GE(generation_, 0);
  auto result = string_dict_->getLike(pattern, icase, is_simple, escape, generation_);
  for (const auto& kv : transient_int_to_str_) {
    const auto str = getString(kv.first);
    if (is_like(str, pattern, icase, is_simple, escape)) {
      result.push_back(kv.first);
    }
  }
  return result;
}

namespace {

bool do_compare(const std::string& str,
                const std::string& pattern,
                const std::string& comp_operator) {
  int res = str.compare(pattern);
  if (comp_operator == "<") {
    return res < 0;
  } else if (comp_operator == "<=") {
    return res <= 0;
  } else if (comp_operator == "=") {
    return res == 0;
  } else if (comp_operator == ">") {
    return res > 0;
  } else if (comp_operator == ">=") {
    return res >= 0;
  } else if (comp_operator == "<>") {
    return res != 0;
  }
  throw std::runtime_error("unsupported string compare operator");
}

}  // namespace

std::vector<int32_t> StringDictionaryProxy::getCompare(
    const std::string& pattern,
    const std::string& comp_operator) const {
  CHECK_GE(generation_, 0);
  auto result = string_dict_->getCompare(pattern, comp_operator, generation_);
  for (const auto& kv : transient_int_to_str_) {
    const auto str = getString(kv.first);
    if (do_compare(str, pattern, comp_operator)) {
      result.push_back(kv.first);
    }
  }
  return result;
}

namespace {

bool is_regexp_like(const std::string& str,
                    const std::string& pattern,
                    const char escape) {
  return regexp_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape);
}

}  // namespace

std::vector<int32_t> StringDictionaryProxy::getRegexpLike(const std::string& pattern,
                                                          const char escape) const {
  CHECK_GE(generation_, 0);
  auto result = string_dict_->getRegexpLike(pattern, escape, generation_);
  for (const auto& kv : transient_int_to_str_) {
    const auto str = getString(kv.first);
    if (is_regexp_like(str, pattern, escape)) {
      result.push_back(kv.first);
    }
  }
  return result;
}

int32_t StringDictionaryProxy::getOrAdd(const std::string& str) noexcept {
  return string_dict_->getOrAdd(str);
}

std::pair<const char*, size_t> StringDictionaryProxy::getStringBytes(
    int32_t string_id) const noexcept {
  if (string_id >= 0) {
    return string_dict_.get()->getStringBytes(string_id);
  }
  CHECK_NE(StringDictionary::INVALID_STR_ID, string_id);
  auto it = transient_int_to_str_.find(string_id);
  CHECK(it != transient_int_to_str_.end());
  return std::make_pair(it->second.c_str(), it->second.size());
}

size_t StringDictionaryProxy::storageEntryCount() const {
  const size_t num_storage_entries{generation_ == -1 ? string_dict_->storageEntryCount()
                                                     : generation_};
  CHECK_LE(num_storage_entries, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  return num_storage_entries;
}

size_t StringDictionaryProxy::transientEntryCountUnlocked() const {
  // CHECK_LE(num_storage_entries,
  // static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  const size_t num_transient_entries{transient_str_to_int_.size()};
  CHECK_LE(num_transient_entries,
           static_cast<size_t>(std::numeric_limits<int32_t>::max()) - 1);
  return num_transient_entries;
}

size_t StringDictionaryProxy::transientEntryCount() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return transientEntryCountUnlocked();
}

size_t StringDictionaryProxy::entryCountUnlocked() const {
  return storageEntryCount() + transientEntryCountUnlocked();
}

size_t StringDictionaryProxy::entryCount() const {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  return entryCountUnlocked();
}

void StringDictionaryProxy::updateGeneration(const int64_t generation) noexcept {
  if (generation == -1) {
    return;
  }
  if (generation_ != -1) {
    CHECK_EQ(generation_, generation);
    return;
  }
  generation_ = generation;
}

int32_t StringDictionaryProxy::transientLookupAndAddUnlocked(const std::string& str) {
  const auto it = transient_str_to_int_.find(str);
  if (it != transient_str_to_int_.end()) {
    return it->second;
  }
  int32_t transient_id =
      -(transient_str_to_int_.size() + 2);  // make sure it's not INVALID_STR_ID
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

template <typename String>
void StringDictionaryProxy::transientLookupBulk(const std::vector<String>& lookup_strings,
                                                int32_t* string_ids) const {
  // std::vector<int32_t>& string_ids) {
  const size_t num_strings = lookup_strings.size();
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);

  if (num_strings == static_cast<size_t>(0) || transient_str_to_int_.empty()) {
    return;
  }
  constexpr size_t tbb_parallel_threshold{20000};
  if (num_strings < tbb_parallel_threshold) {
    transientLookupBulkUnlocked(lookup_strings, string_ids);
  } else {
    transientLookupBulkParallelUnlocked(lookup_strings, string_ids);
  }
}

template void StringDictionaryProxy::transientLookupBulk(
    const std::vector<std::string>& lookup_strings,
    int32_t* string_ids) const;

template void StringDictionaryProxy::transientLookupBulk(
    const std::vector<std::string_view>& lookup_strings,
    int32_t* string_ids) const;

template <typename String>
void StringDictionaryProxy::transientLookupBulkUnlocked(
    const std::vector<String>& lookup_strings,
    int32_t* string_ids) const {
  const size_t num_strings = lookup_strings.size();
  for (size_t string_idx = 0; string_idx < num_strings; ++string_idx) {
    if (string_ids[string_idx] != StringDictionary::INVALID_STR_ID) {
      continue;
    }
    // If we're here it means we need to look up this string as we don't
    // have a valid id for it
    string_ids[string_idx] = lookupTransientStringUnlocked(lookup_strings[string_idx]);
  }
}

template void StringDictionaryProxy::transientLookupBulkUnlocked(
    const std::vector<std::string>& lookup_strings,
    int32_t* string_ids) const;

template void StringDictionaryProxy::transientLookupBulkUnlocked(
    const std::vector<std::string_view>& lookup_strings,
    int32_t* string_ids) const;

template <typename String>
void StringDictionaryProxy::transientLookupBulkParallelUnlocked(
    const std::vector<String>& lookup_strings,
    int32_t* string_ids) const {
  const size_t num_strings = lookup_strings.size();
  const size_t max_thread_count = std::thread::hardware_concurrency();
  const size_t max_inputs_per_thread = 20000;
  const size_t min_grain_size = max_inputs_per_thread / 2;
  const size_t num_threads =
      std::min(max_thread_count,
               ((num_strings + max_inputs_per_thread - 1) / max_inputs_per_thread));

  tbb::task_arena limited_arena(num_threads);
  tbb::task_group tg;
  limited_arena.execute([&] {
    tg.run([&] {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, num_strings, min_grain_size),
          [&](const tbb::blocked_range<size_t>& r) {
            const size_t start_idx = r.begin();
            const size_t end_idx = r.end();
            for (size_t string_idx = start_idx; string_idx < end_idx; ++string_idx) {
              if (string_ids[string_idx] != StringDictionary::INVALID_STR_ID) {
                continue;
              }
              string_ids[string_idx] =
                  lookupTransientStringUnlocked(lookup_strings[string_idx]);
            }
          },
          tbb::simple_partitioner());
    });
  });

  limited_arena.execute([&] { tg.wait(); });
}

template void StringDictionaryProxy::transientLookupBulkParallelUnlocked(
    const std::vector<std::string>& lookup_strings,
    int32_t* string_ids) const;

template void StringDictionaryProxy::transientLookupBulkParallelUnlocked(
    const std::vector<std::string_view>& lookup_strings,
    int32_t* string_ids) const;

StringDictionary* StringDictionaryProxy::getDictionary() const noexcept {
  return string_dict_.get();
}

int64_t StringDictionaryProxy::getGeneration() const noexcept {
  return generation_;
}

bool operator==(const StringDictionaryProxy& sdp1, const StringDictionaryProxy& sdp2) {
  if (sdp1.string_dict_id_ != sdp2.string_dict_id_) {
    return false;
  }
  if (sdp1.transient_int_to_str_.size() != sdp2.transient_int_to_str_.size()) {
    return false;
  }
  return std::equal(sdp1.transient_int_to_str_.begin(),
                    sdp1.transient_int_to_str_.end(),
                    sdp2.transient_int_to_str_.begin());
}

bool operator!=(const StringDictionaryProxy& sdp1, const StringDictionaryProxy& sdp2) {
  return !(sdp1 == sdp2);
}
