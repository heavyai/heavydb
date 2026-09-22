/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "StringDictionary/StringDictionaryProxy.h"

#include "Logger/Logger.h"
#include "Shared/ThreadInfo.h"
#include "Shared/misc.h"
#include "Shared/sqltypes.h"
#include "Shared/thread_count.h"
#include "StringDictionary/StringDictionary.h"
#include "StringOps/StringOps.h"
#include "Utils/Regexp.h"
#include "Utils/StringLike.h"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>

StringDictionaryProxy::StringDictionaryProxy(std::shared_ptr<StringDictionary> sd,
                                             const shared::StringDictKey& string_dict_key,
                                             const int64_t generation)
    : string_dict_(sd), string_dict_key_(string_dict_key), generation_(generation) {}

int32_t truncate_to_generation(const int32_t id, const size_t generation) {
  if (id == StringDictionary::INVALID_STR_ID) {
    return id;
  }
  CHECK_GE(id, 0);
  return static_cast<size_t>(id) >= generation ? StringDictionary::INVALID_STR_ID : id;
}

namespace {

bool is_valid_translated_string_id(const int32_t id) {
  return id != StringDictionary::INVALID_STR_ID && id != inline_int_null_value<int32_t>();
}

struct TranslatedStringIdRange {
  bool has_value{false};
  int32_t min{0};
  int32_t max{0};

  void add(const int32_t id) {
    if (!is_valid_translated_string_id(id)) {
      return;
    }
    if (!has_value) {
      has_value = true;
      min = id;
      max = id;
      return;
    }
    min = std::min(min, id);
    max = std::max(max, id);
  }

  TranslatedStringIdRange merge(const TranslatedStringIdRange& other) const {
    if (!has_value) {
      return other;
    }
    if (!other.has_value) {
      return *this;
    }
    return {true, std::min(min, other.min), std::max(max, other.max)};
  }
};

void tightenStringOpTranslationMapRange(StringDictionaryProxy::IdMap& id_map) {
  // String ops such as SUBSTRING can collapse a large source dictionary into a tiny
  // transient output domain. Use that actual domain for downstream group-by planning.
  auto const& translated_ids = id_map.getVectorMap();
  constexpr size_t min_parallel_range_size = 1'000'000;
  constexpr size_t grain_size = 1'000'000;

  auto scan_range = [&translated_ids](const size_t begin, const size_t end) {
    TranslatedStringIdRange range;
    for (size_t idx = begin; idx < end; ++idx) {
      range.add(translated_ids[idx]);
    }
    return range;
  };

  TranslatedStringIdRange actual_range;
  if (translated_ids.size() >= min_parallel_range_size) {
    actual_range = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, translated_ids.size(), grain_size),
        TranslatedStringIdRange{},
        [&translated_ids](const tbb::blocked_range<size_t>& range,
                          TranslatedStringIdRange local_range) {
          for (size_t idx = range.begin(); idx != range.end(); ++idx) {
            local_range.add(translated_ids[idx]);
          }
          return local_range;
        },
        [](const TranslatedStringIdRange& lhs, const TranslatedStringIdRange& rhs) {
          return lhs.merge(rhs);
        });
  } else {
    actual_range = scan_range(0, translated_ids.size());
  }

  if (!actual_range.has_value) {
    id_map.setRangeStart(0);
    id_map.setRangeEnd(0);
    return;
  }
  CHECK_LT(actual_range.max, std::numeric_limits<int32_t>::max());
  id_map.setRangeStart(actual_range.min);
  id_map.setRangeEnd(actual_range.max + 1);
}

}  // namespace

std::vector<int32_t> StringDictionaryProxy::getTransientBulk(
    const std::vector<std::string>& strings) const {
  CHECK_GE(generation_, 0);
  std::vector<int32_t> string_ids(strings.size());
  getTransientBulkImpl(strings, string_ids.data(), true);
  return string_ids;
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
  const size_t num_strings_not_found =
      string_dict_->getBulk(strings, string_ids.data(), generation_);
  if (num_strings_not_found > 0) {
    std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
    for (size_t string_idx = 0; string_idx < num_strings; ++string_idx) {
      if (string_ids[string_idx] == StringDictionary::INVALID_STR_ID) {
        string_ids[string_idx] = getOrAddTransientUnlocked(strings[string_idx]);
      }
    }
  }
  return string_ids;
}

template <typename String>
int32_t StringDictionaryProxy::getOrAddTransientUnlocked(String const& str) {
  unsigned const new_index = transient_str_to_int_.size();
  auto transient_id = transientIndexToId(new_index);
  auto const emplaced = transient_str_to_int_.emplace(str, transient_id);
  if (emplaced.second) {  // (str, transient_id) was added to transient_str_to_int_.
    transient_string_vec_.push_back(&emplaced.first->first);
  } else {  // str already exists in transient_str_to_int_. Return existing transient_id.
    transient_id = emplaced.first->second;
  }
  return transient_id;
}

template <typename String>
int32_t StringDictionaryProxy::getOrAddTransientImpl(String str) {
  auto const string_id = getIdOfStringFromClient(str);
  if (string_id != StringDictionary::INVALID_STR_ID) {
    return string_id;
  }
  std::lock_guard<std::shared_mutex> write_lock(rw_mutex_);
  return getOrAddTransientUnlocked(str);
}

int32_t StringDictionaryProxy::getOrAddTransient(std::string const& str) {
  return getOrAddTransientImpl<std::string const&>(str);
}

int32_t StringDictionaryProxy::getOrAddTransient(std::string_view const sv) {
  return getOrAddTransientImpl<std::string_view const>(sv);
}

int32_t StringDictionaryProxy::getIdOfString(const std::string& str) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  auto const str_id = getIdOfStringFromClient(str);
  if (str_id != StringDictionary::INVALID_STR_ID || transient_str_to_int_.empty()) {
    return str_id;
  }
  auto it = transient_str_to_int_.find(str);
  return it != transient_str_to_int_.end() ? it->second
                                           : StringDictionary::INVALID_STR_ID;
}

template <typename String>
int32_t StringDictionaryProxy::getIdOfStringFromClient(const String& str) const {
  CHECK_GE(generation_, 0);
  return truncate_to_generation(string_dict_->getIdOfString(str), generation_);
}

int32_t StringDictionaryProxy::getIdOfStringNoGeneration(const std::string& str) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  auto str_id = string_dict_->getIdOfString(str);
  if (str_id != StringDictionary::INVALID_STR_ID || transient_str_to_int_.empty()) {
    return str_id;
  }
  auto it = transient_str_to_int_.find(str);
  return it != transient_str_to_int_.end() ? it->second
                                           : StringDictionary::INVALID_STR_ID;
}

extern "C" DEVICE RUNTIME_EXPORT const char* StringDictionaryProxy_getStringBytes(
    int8_t* proxy_ptr,
    int32_t string_id) {
  CHECK(proxy_ptr != nullptr);
  auto proxy = reinterpret_cast<StringDictionaryProxy*>(proxy_ptr);
  auto [c_str, len] = proxy->getStringBytes(string_id);
  return c_str;
}

extern "C" DEVICE RUNTIME_EXPORT size_t
StringDictionaryProxy_getStringLength(int8_t* proxy_ptr, int32_t string_id) {
  CHECK(proxy_ptr != nullptr);
  auto proxy = reinterpret_cast<StringDictionaryProxy*>(proxy_ptr);
  auto [c_str, len] = proxy->getStringBytes(string_id);
  return len;
}

extern "C" DEVICE RUNTIME_EXPORT int32_t
StringDictionaryProxy_getStringId(int8_t* proxy_ptr, char* c_str_ptr) {
  CHECK(proxy_ptr != nullptr);
  auto proxy = reinterpret_cast<StringDictionaryProxy*>(proxy_ptr);
  std::string str(c_str_ptr);
  return proxy->getOrAddTransient(str);
}

std::string StringDictionaryProxy::getString(int32_t string_id) const {
  if (inline_int_null_value<int32_t>() == string_id) {
    return "";
  }
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  return getStringUnlocked(string_id);
}

bool StringDictionaryProxy::canDecodeStringId(const int32_t string_id) const {
  if (string_id == inline_int_null_value<int32_t>() ||
      string_id == StringDictionary::INVALID_STR_ID) {
    return false;
  }
  if (string_id >= 0) {
    return static_cast<size_t>(string_id) < storageEntryCount();
  }
  const auto string_index = transientIdToIndex(string_id);
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  return string_index < transient_string_vec_.size();
}

std::string StringDictionaryProxy::getStringUnlocked(const int32_t string_id) const {
  if (string_id >= 0) {
    CHECK_LT(static_cast<size_t>(string_id), storageEntryCount());
    return string_dict_->getString(string_id);
  }
  unsigned const string_index = transientIdToIndex(string_id);
  CHECK_LT(string_index, transient_string_vec_.size());
  return *transient_string_vec_[string_index];
}

std::vector<std::string> StringDictionaryProxy::getStrings(
    const std::vector<int32_t>& string_ids) const {
  std::vector<std::string> strings(string_ids.size());
  const auto copy_strings = [&](const size_t begin, const size_t end) {
    for (size_t i = begin; i < end; ++i) {
      const auto string_id = string_ids[i];
      if (string_id >= 0) {
        strings[i] = string_dict_->getString(string_id);
      } else if (inline_int_null_value<int32_t>() == string_id) {
        strings[i].clear();
      } else {
        unsigned const string_index = transientIdToIndex(string_id);
        strings[i] = *transient_string_vec_[string_index];
      }
    }
  };

  constexpr size_t parallel_lookup_threshold{10000};
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  if (string_ids.size() >= parallel_lookup_threshold) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, string_ids.size()),
                      [&](const tbb::blocked_range<size_t>& range) {
                        copy_strings(range.begin(), range.end());
                      });
  } else {
    copy_strings(0, string_ids.size());
  }
  return strings;
}

template <typename String>
int32_t StringDictionaryProxy::lookupTransientStringUnlocked(
    const String& lookup_string) const {
  const auto it = transient_str_to_int_.find(lookup_string);
  return it == transient_str_to_int_.end() ? StringDictionary::INVALID_STR_ID
                                           : it->second;
}

StringDictionaryProxy::TranslationMap<Datum>
StringDictionaryProxy::buildNumericTranslationMap(
    const StringOps_Namespace::StringOps& string_ops) const {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(string_ops.size());
  TranslationMap<Datum> translation_map(transient_string_vec_.size(), generation_);
  if (translation_map.empty()) {
    return translation_map;
  }

  const size_t num_transient_entries = translation_map.numTransients();
  if (num_transient_entries) {
    const int32_t map_domain_start = translation_map.domainStart();
    if (num_transient_entries > 10000UL) {
      tbb::parallel_for(
          tbb::blocked_range<int32_t>(map_domain_start, -1),
          [&](const tbb::blocked_range<int32_t>& r) {
            const int32_t start_idx = r.begin();
            const int32_t end_idx = r.end();
            for (int32_t source_string_id = start_idx; source_string_id < end_idx;
                 ++source_string_id) {
              const auto source_string = getStringUnlocked(source_string_id);
              translation_map[source_string_id] = string_ops.numericEval(source_string);
            }
          });
    } else {
      for (int32_t source_string_id = map_domain_start; source_string_id < -1;
           ++source_string_id) {
        const auto source_string = getStringUnlocked(source_string_id);
        translation_map[source_string_id] = string_ops.numericEval(source_string);
      }
    }
  }

  Datum* translation_map_stored_entries_ptr = translation_map.storageData();
  if (generation_ > 0) {
    string_dict_->buildDictionaryNumericTranslationMap(
        translation_map_stored_entries_ptr, generation_, string_ops);
  }
  translation_map.setNumUntranslatedStrings(0UL);

  // Todo(todd): Set range start/end with scan

  return translation_map;
}

StringDictionaryProxy::IdMap
StringDictionaryProxy::buildIntersectionTranslationMapToOtherProxyUnlocked(
    const StringDictionaryProxy* dest_proxy,
    const StringOps_Namespace::StringOps& string_ops) const {
  auto timer = DEBUG_TIMER(__func__);
  IdMap id_map = initIdMap();

  if (id_map.empty()) {
    return id_map;
  }

  // First map transient strings, store at front of vector map
  const size_t num_transient_entries = id_map.numTransients();
  size_t num_transient_strings_not_translated = 0UL;
  if (num_transient_entries) {
    std::vector<std::string> transient_lookup_strings(num_transient_entries);
    if (string_ops.size()) {
      std::transform(transient_string_vec_.cbegin(),
                     transient_string_vec_.cend(),
                     transient_lookup_strings.rbegin(),
                     [&](std::string const* ptr) { return string_ops(*ptr); });
    } else {
      std::transform(transient_string_vec_.cbegin(),
                     transient_string_vec_.cend(),
                     transient_lookup_strings.rbegin(),
                     [](std::string const* ptr) { return *ptr; });
    }

    // This lookup may have a different snapshot of
    // dest_proxy transients and dictionary than what happens under
    // the below dest_proxy_read_lock. We may need an unlocked version of
    // getTransientBulk to ensure consistency (I don't believe
    // current behavior would cause crashes/races, verify this though)

    // Todo(mattp): Consider variant of getTransientBulkImp that can take
    // a vector of pointer-to-strings so we don't have to materialize
    // transient_string_vec_ into transient_lookup_strings.

    num_transient_strings_not_translated =
        dest_proxy->getTransientBulkImpl(transient_lookup_strings, id_map.data(), false);
  }

  // Now map strings in dictionary
  // We place non-transient strings after the transient strings
  // if they exist, otherwise at index 0
  int32_t* translation_map_stored_entries_ptr = id_map.storageData();

  auto dest_transient_lookup_callback = [dest_proxy, translation_map_stored_entries_ptr](
                                            const std::string_view& source_string,
                                            const int32_t source_string_id) {
    translation_map_stored_entries_ptr[source_string_id] =
        dest_proxy->lookupTransientStringUnlocked(source_string);
    return translation_map_stored_entries_ptr[source_string_id] ==
           StringDictionary::INVALID_STR_ID;
  };

  const size_t num_dest_transients = dest_proxy->transientEntryCountUnlocked();
  const size_t num_persisted_strings_not_translated =
      generation_ > 0 ? string_dict_->buildDictionaryTranslationMap(
                            dest_proxy->string_dict_.get(),
                            translation_map_stored_entries_ptr,
                            generation_,
                            dest_proxy->generation_,
                            num_dest_transients > 0UL,
                            dest_transient_lookup_callback,
                            string_ops)
                      : 0UL;

  const size_t num_dest_entries = dest_proxy->entryCountUnlocked();
  const size_t num_total_entries =
      id_map.getVectorMap().size() - 1UL /* account for skipped entry -1 */;
  CHECK_GT(num_total_entries, 0UL);
  const size_t num_strings_not_translated =
      num_transient_strings_not_translated + num_persisted_strings_not_translated;
  CHECK_LE(num_strings_not_translated, num_total_entries);
  id_map.setNumUntranslatedStrings(num_strings_not_translated);

  // Below is a conservative setting of range based on the size of the destination proxy,
  // but probably not worth a scan over the data (or inline computation as we translate)
  // to compute the actual ranges

  id_map.setRangeStart(
      num_dest_transients > 0 ? -1 - static_cast<int32_t>(num_dest_transients) : 0);
  id_map.setRangeEnd(dest_proxy->storageEntryCount());

  const size_t num_entries_translated = num_total_entries - num_strings_not_translated;
  const float match_pct =
      100.0 * static_cast<float>(num_entries_translated) / num_total_entries;
  VLOG(1) << std::fixed << std::setprecision(2) << match_pct << "% ("
          << num_entries_translated << " entries) from dictionary ("
          << string_dict_->getDictKey() << ") with " << num_total_entries
          << " total entries ( " << num_transient_entries << " literals)"
          << " translated to dictionary (" << dest_proxy->string_dict_->getDictKey()
          << ") with " << num_dest_entries << " total entries ("
          << dest_proxy->transientEntryCountUnlocked() << " literals).";

  return id_map;
}

void order_translation_locks(const shared::StringDictKey& source_dict_key,
                             const shared::StringDictKey& dest_dict_key,
                             std::shared_lock<std::shared_mutex>& source_proxy_read_lock,
                             std::unique_lock<std::shared_mutex>& dest_proxy_write_lock) {
  if (source_dict_key == dest_dict_key) {
    // proxies are same, only take one write lock
    dest_proxy_write_lock.lock();
  } else if (source_dict_key < dest_dict_key) {
    source_proxy_read_lock.lock();
    dest_proxy_write_lock.lock();
  } else {
    dest_proxy_write_lock.lock();
    source_proxy_read_lock.lock();
  }
}

StringDictionaryProxy::IdMap StringDictionaryProxy::buildUnionTranslationMapToOtherProxy(
    StringDictionaryProxy* dest_proxy,
    const StringOps_Namespace::StringOps& string_ops) const {
  auto timer = DEBUG_TIMER(__func__);

  const auto& source_dict_id = getDictKey();
  const auto& dest_dict_id = dest_proxy->getDictKey();
  std::shared_lock<std::shared_mutex> source_proxy_read_lock(rw_mutex_, std::defer_lock);
  std::unique_lock<std::shared_mutex> dest_proxy_write_lock(dest_proxy->rw_mutex_,
                                                            std::defer_lock);
  order_translation_locks(
      source_dict_id, dest_dict_id, source_proxy_read_lock, dest_proxy_write_lock);

  const bool has_string_ops = string_ops.size();
  if (g_enable_lazy_string_dictionary_hash_recovery && has_string_ops &&
      this == dest_proxy && generation_ >= 0 && transientEntryCountUnlocked() <= 1024 &&
      !string_dict_->isHashTableRecovered()) {
    auto id_map = initIdMap();
    if (id_map.empty()) {
      return id_map;
    }

    const size_t initial_transient_count = transientEntryCountUnlocked();
    std::unordered_set<std::string> initial_transient_strings;
    initial_transient_strings.reserve(initial_transient_count);
    std::vector<std::string> transformed_transient_strings;
    transformed_transient_strings.reserve(initial_transient_count);
    for (size_t transient_idx = 0; transient_idx < initial_transient_count;
         ++transient_idx) {
      const auto& transient_string = *transient_string_vec_[transient_idx];
      initial_transient_strings.insert(transient_string);
      transformed_transient_strings.push_back(string_ops(transient_string));
    }

    size_t num_untranslated_strings{0};
    if (string_dict_->tryBuildSelfStringOpUnionTranslationMapWithoutHash(
            id_map.storageData(),
            generation_,
            string_ops,
            [dest_proxy](const std::string_view transformed_string) {
              dest_proxy->getOrAddTransientUnlocked(transformed_string);
            },
            [dest_proxy](const std::string_view transformed_string) {
              return dest_proxy->lookupTransientStringUnlocked(transformed_string);
            },
            num_untranslated_strings)) {
      if (initial_transient_count > 0) {
        std::vector<int32_t> persisted_transient_ids(initial_transient_count);
        string_dict_->getBulk(
            transformed_transient_strings, persisted_transient_ids.data(), generation_);
        for (size_t transient_idx = 0; transient_idx < initial_transient_count;
             ++transient_idx) {
          const auto& transformed_string = transformed_transient_strings[transient_idx];
          auto translated_id = persisted_transient_ids[transient_idx];
          if (translated_id == StringDictionary::INVALID_STR_ID) {
            translated_id = dest_proxy->lookupTransientStringUnlocked(transformed_string);
            if (translated_id == StringDictionary::INVALID_STR_ID) {
              translated_id = dest_proxy->getOrAddTransientUnlocked(transformed_string);
            }
            num_untranslated_strings +=
                !initial_transient_strings.count(transformed_string);
          }
          id_map[transientIndexToId(transient_idx)] = translated_id;
        }
      }
      id_map.setNumUntranslatedStrings(num_untranslated_strings);
      tightenStringOpTranslationMapRange(id_map);
      return id_map;
    }
  }

  auto id_map =
      buildIntersectionTranslationMapToOtherProxyUnlocked(dest_proxy, string_ops);
  if (id_map.empty()) {
    return id_map;
  }
  const auto num_untranslated_strings = id_map.numUntranslatedStrings();
  if (num_untranslated_strings > 0) {
    const size_t total_post_translation_dest_transients =
        num_untranslated_strings + dest_proxy->transientEntryCountUnlocked();
    constexpr size_t max_allowed_transients =
        static_cast<size_t>(std::numeric_limits<int32_t>::max() -
                            2); /* -2 accounts for INVALID_STR_ID and NULL value */
    if (total_post_translation_dest_transients > max_allowed_transients) {
      std::stringstream ss;
      ss << "Union translation to dictionary " << getDictKey() << " would result in "
         << total_post_translation_dest_transients
         << " transient entries, which is more than limit of " << max_allowed_transients
         << " transients.";
      throw std::runtime_error(ss.str());
    }
    const int32_t map_domain_start = id_map.domainStart();
    const int32_t map_domain_end = id_map.domainEnd();

    // Define the masking functor
    auto mask_functor = [&id_map](int32_t id) {
      return id_map[id] == StringDictionary::INVALID_STR_ID;
    };

    // Process transient strings
    {
      auto transient_timer =
          DEBUG_TIMER("UnionTranslationMapToOtherProxy:TransientStrings");
      for (int32_t source_string_id = map_domain_start; source_string_id < -1;
           ++source_string_id) {
        if (id_map[source_string_id] == StringDictionary::INVALID_STR_ID) {
          const auto source_string = getStringUnlocked(source_string_id);
          const auto dest_string_id = dest_proxy->getOrAddTransientUnlocked(
              has_string_ops ? string_ops(source_string) : source_string);
          id_map[source_string_id] = dest_string_id;
        }
      }
    }

    // Process stored strings
    {
      auto stored_timer = DEBUG_TIMER("UnionTranslationMapToOtherProxy:StoredStrings");
      auto* translation_map_stored_entries_ptr = id_map.storageData();

      string_dict_->fillStringOpUnionTranslationMap(
          translation_map_stored_entries_ptr,
          map_domain_end,
          string_ops,
          mask_functor,
          [dest_proxy](std::string_view processed_string) {
            dest_proxy->getOrAddTransientUnlocked(processed_string);
          },
          [dest_proxy](std::string_view processed_string) {
            return dest_proxy->lookupTransientStringUnlocked(processed_string);
          });
    }
  }

  // Update id_map range
  if (has_string_ops) {
    tightenStringOpTranslationMapRange(id_map);
  } else {
    const size_t num_dest_transients = dest_proxy->transientEntryCountUnlocked();
    id_map.setRangeStart(
        num_dest_transients > 0 ? -1 - static_cast<int32_t>(num_dest_transients) : 0);
  }
  return id_map;
}

StringDictionaryProxy::IdMap
StringDictionaryProxy::buildIntersectionTranslationMapToOtherProxy(
    const StringDictionaryProxy* dest_proxy,
    const StringOps_Namespace::StringOps& string_ops) const {
  const auto& source_dict_id = getDictKey();
  const auto& dest_dict_id = dest_proxy->getDictKey();

  std::shared_lock<std::shared_mutex> source_proxy_read_lock(rw_mutex_, std::defer_lock);
  std::unique_lock<std::shared_mutex> dest_proxy_write_lock(dest_proxy->rw_mutex_,
                                                            std::defer_lock);
  order_translation_locks(
      source_dict_id, dest_dict_id, source_proxy_read_lock, dest_proxy_write_lock);
  return buildIntersectionTranslationMapToOtherProxyUnlocked(dest_proxy, string_ops);
}

template <typename T>
std::vector<T> StringDictionaryProxy::getLike(const std::string& pattern,
                                              const bool icase,
                                              const bool is_simple,
                                              const char escape) const {
  return *getLikeShared<T>(pattern, icase, is_simple, escape);
}

template <typename T>
std::shared_ptr<const std::vector<T>> StringDictionaryProxy::getLikeShared(
    const std::string& pattern,
    const bool icase,
    const bool is_simple,
    const char escape) const {
  CHECK_GE(generation_, 0);
  auto persisted_result =
      string_dict_->getLikeShared<T>(pattern, icase, is_simple, escape, generation_);
  if (transient_string_vec_.empty()) {
    return persisted_result;
  }
  auto result = std::make_shared<std::vector<T>>(*persisted_result);
  auto is_like_impl = icase       ? is_simple ? string_ilike_simple : string_ilike
                      : is_simple ? string_like_simple
                                  : string_like;
  for (unsigned index = 0; index < transient_string_vec_.size(); ++index) {
    auto const str = *transient_string_vec_[index];
    if (is_like_impl(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape)) {
      result->push_back(transientIndexToId(index));
    }
  }
  return result;
}

template std::vector<int32_t> StringDictionaryProxy::getLike<int32_t>(
    const std::string& pattern,
    const bool icase,
    const bool is_simple,
    const char escape) const;

template std::vector<int64_t> StringDictionaryProxy::getLike<int64_t>(
    const std::string& pattern,
    const bool icase,
    const bool is_simple,
    const char escape) const;

template std::shared_ptr<const std::vector<int32_t>>
StringDictionaryProxy::getLikeShared<int32_t>(const std::string& pattern,
                                              const bool icase,
                                              const bool is_simple,
                                              const char escape) const;

template std::shared_ptr<const std::vector<int64_t>>
StringDictionaryProxy::getLikeShared<int64_t>(const std::string& pattern,
                                              const bool icase,
                                              const bool is_simple,
                                              const char escape) const;

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
  for (unsigned index = 0; index < transient_string_vec_.size(); ++index) {
    if (do_compare(*transient_string_vec_[index], pattern, comp_operator)) {
      result.push_back(transientIndexToId(index));
    }
  }
  return result;
}

std::vector<std::pair<std::string, int32_t>> extract_and_sort_transient_map(
    const StringDictionaryProxy::TransientMap& transient_str_to_int) {
  std::vector<std::pair<std::string, int32_t>> sorted_elements;
  sorted_elements.reserve(transient_str_to_int.size());

  for (const auto& kv : transient_str_to_int) {
    sorted_elements.emplace_back(kv.first, kv.second);
  }

  std::sort(sorted_elements.begin(),
            sorted_elements.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  return sorted_elements;
}

SortedStringPermutation StringDictionaryProxy::getSortedPermutation(
    const bool should_sort_descending) {
  auto timer = DEBUG_TIMER(__func__);
  const auto transient_strings_to_ids =
      extract_and_sort_transient_map(transient_str_to_int_);
  return string_dict_->getSortedPermutation(transient_strings_to_ids,
                                            should_sort_descending);
}

bool StringDictionaryProxy::isSortedPermutationCacheComplete() const {
  return string_dict_->isSortedPermutationCacheComplete();
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
  for (unsigned index = 0; index < transient_string_vec_.size(); ++index) {
    if (is_regexp_like(*transient_string_vec_[index], pattern, escape)) {
      result.push_back(transientIndexToId(index));
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
  unsigned const string_index = transientIdToIndex(string_id);
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  CHECK_LT(string_index, transient_string_vec_.size());
  std::string const* const str_ptr = transient_string_vec_[string_index];
  return {str_ptr->c_str(), str_ptr->size()};
}

size_t StringDictionaryProxy::storageEntryCount() const {
  const auto dictionary_entry_count = string_dict_->storageEntryCount();
  const size_t num_storage_entries{
      generation_ == -1
          ? dictionary_entry_count
          : std::min(static_cast<size_t>(generation_), dictionary_entry_count)};
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
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  return transientEntryCountUnlocked();
}

size_t StringDictionaryProxy::entryCountUnlocked() const {
  return storageEntryCount() + transientEntryCountUnlocked();
}

size_t StringDictionaryProxy::entryCount() const {
  std::shared_lock<std::shared_mutex> read_lock(rw_mutex_);
  return entryCountUnlocked();
}

// Iterate over transient strings, then non-transients.
void StringDictionaryProxy::eachStringSerially(
    StringDictionary::StringCallback& serial_callback) const {
  constexpr int32_t max_transient_id = -2;
  // Iterate over transient strings.
  for (unsigned index = 0; index < transient_string_vec_.size(); ++index) {
    std::string const& str = *transient_string_vec_[index];
    int32_t const string_id = max_transient_id - index;
    serial_callback(str, string_id);
  }
  // Iterate over non-transient strings.
  string_dict_->eachStringSerially(generation_, serial_callback);
}

// For each (string/_view,old_id) pair passed in:
//  * Get the new_id based on sdp_'s dictionary, or add it as a transient.
//  * The StringDictionary is local, so call the faster getUnlocked() method.
//  * Store the old_id -> new_id translation into the id_map_.
class StringLocalCallback : public StringDictionary::StringCallback {
  StringDictionaryProxy* sdp_;
  StringDictionaryProxy::IdMap& id_map_;

 public:
  StringLocalCallback(StringDictionaryProxy* sdp, StringDictionaryProxy::IdMap& id_map)
      : sdp_(sdp), id_map_(id_map) {
    sdp_->string_dict_->ensureHashTableRecovered();
  }
  void operator()(std::string const& str, int32_t const string_id) override {
    operator()(std::string_view(str), string_id);
  }
  void operator()(std::string_view const sv, int32_t const old_id) override {
    int32_t const new_id = sdp_->string_dict_->getUnlocked(sv);
    id_map_[old_id] = new_id == StringDictionary::INVALID_STR_ID
                          ? sdp_->getOrAddTransientUnlocked(sv)
                          : new_id;
  }
};

// Union strings from both StringDictionaryProxies into *this as transients.
// Return id_map: sdp_rhs:string_id -> this:string_id for each string in sdp_rhs.
StringDictionaryProxy::IdMap StringDictionaryProxy::transientUnion(
    StringDictionaryProxy const& sdp_rhs) {
  IdMap id_map = sdp_rhs.initIdMap();
  // serial_callback cannot be parallelized due to calling getOrAddTransientUnlocked().
  StringLocalCallback serial_callback(this, id_map);
  // Import all non-duplicate strings (transient and non-transient) and add to id_map.
  sdp_rhs.eachStringSerially(serial_callback);
  return id_map;
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

size_t StringDictionaryProxy::getTransientBulkImpl(
    const std::vector<std::string>& strings,
    int32_t* string_ids,
    const bool take_read_lock) const {
  const size_t num_strings = strings.size();
  if (num_strings == 0) {
    return 0UL;
  }
  if (g_enable_lazy_string_dictionary_hash_recovery &&
      !string_dict_->isHashTableRecovered()) {
    std::vector<size_t> persisted_lookup_indices;
    persisted_lookup_indices.reserve(num_strings);
    {
      auto read_lock = take_read_lock ? std::shared_lock<std::shared_mutex>(rw_mutex_)
                                      : std::shared_lock<std::shared_mutex>();
      for (size_t string_idx = 0; string_idx < num_strings; ++string_idx) {
        string_ids[string_idx] = lookupTransientStringUnlocked(strings[string_idx]);
        if (string_ids[string_idx] == StringDictionary::INVALID_STR_ID) {
          persisted_lookup_indices.push_back(string_idx);
        }
      }
    }
    if (persisted_lookup_indices.empty()) {
      return 0UL;
    }

    std::vector<std::string> persisted_lookup_strings;
    persisted_lookup_strings.reserve(persisted_lookup_indices.size());
    for (const auto string_idx : persisted_lookup_indices) {
      persisted_lookup_strings.push_back(strings[string_idx]);
    }
    std::vector<int32_t> persisted_string_ids(persisted_lookup_strings.size());
    const auto num_strings_not_found = string_dict_->getBulk(
        persisted_lookup_strings, persisted_string_ids.data(), generation_);
    for (size_t lookup_idx = 0; lookup_idx < persisted_lookup_indices.size();
         ++lookup_idx) {
      string_ids[persisted_lookup_indices[lookup_idx]] = persisted_string_ids[lookup_idx];
    }
    return num_strings_not_found;
  }
  // StringDictionary::getBulk returns the number of strings not found
  if (string_dict_->getBulk(strings, string_ids, generation_) == 0UL) {
    return 0UL;
  }

  // If here, dictionary could not find at least 1 target string,
  // now look these up in the transient dictionary
  // transientLookupBulk returns the number of strings not found
  return transientLookupBulk(strings, string_ids, take_read_lock);
}

template <typename String>
size_t StringDictionaryProxy::transientLookupBulk(
    const std::vector<String>& lookup_strings,
    int32_t* string_ids,
    const bool take_read_lock) const {
  const size_t num_strings = lookup_strings.size();
  auto read_lock = take_read_lock ? std::shared_lock<std::shared_mutex>(rw_mutex_)
                                  : std::shared_lock<std::shared_mutex>();

  if (num_strings == static_cast<size_t>(0) || transient_str_to_int_.empty()) {
    return 0UL;
  }
  constexpr size_t tbb_parallel_threshold{20000};
  if (num_strings < tbb_parallel_threshold) {
    return transientLookupBulkUnlocked(lookup_strings, string_ids);
  } else {
    return transientLookupBulkParallelUnlocked(lookup_strings, string_ids);
  }
}

template <typename String>
size_t StringDictionaryProxy::transientLookupBulkUnlocked(
    const std::vector<String>& lookup_strings,
    int32_t* string_ids) const {
  const size_t num_strings = lookup_strings.size();
  size_t num_strings_not_found = 0;
  for (size_t string_idx = 0; string_idx < num_strings; ++string_idx) {
    if (string_ids[string_idx] != StringDictionary::INVALID_STR_ID) {
      continue;
    }
    // If we're here it means we need to look up this string as we don't
    // have a valid id for it
    string_ids[string_idx] = lookupTransientStringUnlocked(lookup_strings[string_idx]);
    if (string_ids[string_idx] == StringDictionary::INVALID_STR_ID) {
      num_strings_not_found++;
    }
  }
  return num_strings_not_found;
}

template <typename String>
size_t StringDictionaryProxy::transientLookupBulkParallelUnlocked(
    const std::vector<String>& lookup_strings,
    int32_t* string_ids) const {
  const size_t num_lookup_strings = lookup_strings.size();
  const size_t target_inputs_per_thread = 20000L;
  ThreadInfo thread_info(
      std::thread::hardware_concurrency(), num_lookup_strings, target_inputs_per_thread);
  CHECK_GE(thread_info.num_threads, 1L);
  CHECK_GE(thread_info.num_elems_per_thread, 1L);

  std::vector<size_t> num_strings_not_found_per_thread(thread_info.num_threads, 0UL);

  tbb::task_arena limited_arena(thread_info.num_threads);
  limited_arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(
            0, num_lookup_strings, thread_info.num_elems_per_thread /* tbb grain_size */),
        [&](const tbb::blocked_range<size_t>& r) {
          const size_t start_idx = r.begin();
          const size_t end_idx = r.end();
          size_t num_local_strings_not_found = 0;
          for (size_t string_idx = start_idx; string_idx < end_idx; ++string_idx) {
            if (string_ids[string_idx] != StringDictionary::INVALID_STR_ID) {
              continue;
            }
            string_ids[string_idx] =
                lookupTransientStringUnlocked(lookup_strings[string_idx]);
            if (string_ids[string_idx] == StringDictionary::INVALID_STR_ID) {
              num_local_strings_not_found++;
            }
          }
          const size_t tbb_thread_idx = tbb::this_task_arena::current_thread_index();
          num_strings_not_found_per_thread[tbb_thread_idx] = num_local_strings_not_found;
        },
        tbb::simple_partitioner());
  });
  size_t num_strings_not_found = 0;
  for (int64_t thread_idx = 0; thread_idx < thread_info.num_threads; ++thread_idx) {
    num_strings_not_found += num_strings_not_found_per_thread[thread_idx];
  }
  return num_strings_not_found;
}

StringDictionary* StringDictionaryProxy::getDictionary() const noexcept {
  return string_dict_.get();
}

int64_t StringDictionaryProxy::getGeneration() const noexcept {
  return generation_;
}

bool StringDictionaryProxy::operator==(StringDictionaryProxy const& rhs) const {
  return string_dict_key_ == rhs.string_dict_key_ &&
         transient_str_to_int_ == rhs.transient_str_to_int_;
}

bool StringDictionaryProxy::operator!=(StringDictionaryProxy const& rhs) const {
  return !operator==(rhs);
}
