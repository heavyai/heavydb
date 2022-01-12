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

#include "TestHelpers.h"

#include "../StringDictionary/StringDictionary.h"
#include "../StringDictionary/StringDictionaryProxy.h"

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

#ifndef BASE_PATH1
#define BASE_PATH1 "./tmp/dict1"
#endif

#ifndef BASE_PATH2
#define BASE_PATH2 "./tmp/dict2"
#endif

extern bool g_cache_string_hash;

TEST(StringDictionary, AddAndGet) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, g_cache_string_hash);
  auto id1 = string_dict.getOrAdd("foo bar");
  auto id2 = string_dict.getOrAdd("foo bar");
  ASSERT_EQ(id1, id2);
  ASSERT_EQ(0, id1);
  auto id3 = string_dict.getOrAdd("baz");
  ASSERT_EQ(1, id3);
  auto id4 = string_dict.getIdOfString("foo bar");
  ASSERT_EQ(id1, id4);
  ASSERT_EQ("foo bar", string_dict.getString(id4));
}

TEST(StringDictionary, Recover) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, true, g_cache_string_hash);
  auto id1 = string_dict.getOrAdd("baz");
  ASSERT_EQ(1, id1);
  auto id2 = string_dict.getOrAdd("baz");
  ASSERT_EQ(1, id2);
  auto id3 = string_dict.getOrAdd("foo bar");
  ASSERT_EQ(0, id3);
  auto id4 = string_dict.getOrAdd("fizzbuzz");
  ASSERT_EQ(2, id4);
  ASSERT_EQ("baz", string_dict.getString(id2));
  ASSERT_EQ("foo bar", string_dict.getString(id3));
  ASSERT_EQ("fizzbuzz", string_dict.getString(id4));
}

TEST(StringDictionary, HandleEmpty) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, g_cache_string_hash);
  auto id1 = string_dict.getOrAdd("");
  auto id2 = string_dict.getOrAdd("");
  ASSERT_EQ(id1, id2);
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), id1);
}

TEST(StringDictionary, RecoverZero) {
  const DictRef dict_ref(-1, 1);
  {
    StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, g_cache_string_hash);
    size_t num_strings = string_dict.storageEntryCount();
    ASSERT_EQ(static_cast<size_t>(0), num_strings);
  }
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, true, g_cache_string_hash);
  size_t num_strings = string_dict.storageEntryCount();
  ASSERT_EQ(static_cast<size_t>(0), num_strings);
}

const int g_op_count{250000};

TEST(StringDictionary, ManyAddsAndGets) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, g_cache_string_hash);
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(i, string_dict.getOrAdd(std::to_string(i)));
  }
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(i, string_dict.getOrAdd(std::to_string(i)));
  }
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(std::to_string(i), string_dict.getString(i));
  }
}

TEST(StringDictionary, RecoverMany) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, true, g_cache_string_hash);
  size_t num_strings = string_dict.storageEntryCount();
  ASSERT_EQ(static_cast<size_t>(g_op_count), num_strings);
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(i, string_dict.getOrAdd(std::to_string(i)));
  }
  for (int i = 0; i < g_op_count; ++i) {
    CHECK_EQ(std::to_string(i), string_dict.getString(i));
  }
}

TEST(StringDictionary, GetStringViews) {
  const DictRef dict_ref(-1, 1);
  std::shared_ptr<StringDictionary> string_dict = std::make_shared<StringDictionary>(
      dict_ref, BASE_PATH1, false, false, g_cache_string_hash);
  std::vector<std::string> strings;
  std::vector<int32_t> string_ids(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    strings.emplace_back(std::to_string(i));
  }
  string_dict->getOrAddBulk(strings, string_ids.data());

  const auto string_views = string_dict->getStringViews();
  ASSERT_EQ(string_views.size(), static_cast<size_t>(g_op_count));
  for (int i = 0; i < g_op_count; ++i) {
    ASSERT_EQ(strings[i], std::string(string_views[i]));
  }
}

TEST(StringDictionary, GetOrAddBulk) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, g_cache_string_hash);
  {
    // First insert all new strings
    std::vector<int32_t> string_ids(g_op_count);
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }
    string_dict.getOrAddBulk(strings, string_ids.data());
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }

  {
    // Now insert half new, half existing strings
    // Even idxs are existing, odd are new
    std::vector<int32_t> string_ids(g_op_count);
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(g_op_count + i / 2));
      }
    }
    string_dict.getOrAddBulk(strings, string_ids.data());
    for (int i = 0; i < g_op_count; ++i) {
      // Even though the if and else branches result in the same test, both are
      // left help diagnose the source of test failures
      // (existing strings for the if branch and new strings for the else branch)
      if (i % 2 == 0) {
        // Existing strings
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        // New strings
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      }
    }
  }
}

TEST(StringDictionary, GetBulk) {
  const DictRef dict_ref(-1, 1);
  // Use existing dictionary from GetOrAddBulk
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, true, g_cache_string_hash);
  {
    // First iteration is identical to first of GetOrAddBulk, and results should be the
    // same
    std::vector<int32_t> string_ids(g_op_count);
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }
    string_dict.getBulk(strings, string_ids.data());
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }
  {
    // Second iteration - even idxs exist, odd do not and should return INVALID_STR_ID
    std::vector<int32_t> string_ids(g_op_count);
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        // Won't exist in dictionary
        strings.emplace_back(std::to_string(g_op_count * 2 + i));
      }
    }
    string_dict.getBulk(strings, string_ids.data());
    const std::string invalid_str_id_as_string =
        std::to_string(StringDictionary::INVALID_STR_ID);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(string_ids[i], StringDictionary::INVALID_STR_ID);
      }
    }
  }
}

TEST(StringDictionary, BuildTranslationMap) {
  const DictRef dict_ref1(-1, 1);
  const DictRef dict_ref2(-1, 2);
  std::shared_ptr<StringDictionary> source_string_dict =
      std::make_shared<StringDictionary>(
          dict_ref1, BASE_PATH1, false, false, g_cache_string_hash);
  std::shared_ptr<StringDictionary> dest_string_dict = std::make_shared<StringDictionary>(
      dict_ref2, BASE_PATH2, false, false, g_cache_string_hash);

  // Prep: insert g_op_count strings into source_string_dict
  std::vector<int32_t> string_ids(g_op_count);
  std::vector<std::string> strings;
  strings.reserve(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    strings.emplace_back(std::to_string(i));
  }
  source_string_dict->getOrAddBulk(strings, string_ids.data());
  auto dummy_callback = [](const std::string_view& source_string,
                           const int32_t source_string_id) { return false; };

  {
    // First try to translate to empty dictionary.
    // Should get back all INVALID_STR_IDs

    const auto translated_ids = source_string_dict->buildDictionaryTranslationMap(
        dest_string_dict, dummy_callback);
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids, source_string_dict->storageEntryCount());
    for (size_t idx = 0; idx < num_ids; ++idx) {
      ASSERT_EQ(translated_ids[idx], StringDictionary::INVALID_STR_ID);
    }
  }

  {
    // Now insert the same strings inserted into source_string_dict,
    // but in reversed order and missing the first 2 strings in the source
    // dictionary. Make sure they translate correctly, and that the missing
    // strings translate to INVALID_STR_ID
    std::vector<std::string> reversed_strings;
    constexpr int32_t num_missing_strings{2};
    ASSERT_LE(num_missing_strings, g_op_count);
    std::vector<int32_t> reversed_string_ids(g_op_count - num_missing_strings);
    for (int i = 0; i < g_op_count - num_missing_strings; ++i) {
      reversed_strings.emplace_back(std::to_string(g_op_count - i - 1));
    }
    dest_string_dict->getOrAddBulk(reversed_strings, reversed_string_ids.data());
    ASSERT_EQ(dest_string_dict->storageEntryCount(), reversed_strings.size());
    const auto translated_ids = source_string_dict->buildDictionaryTranslationMap(
        dest_string_dict, dummy_callback);
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids, static_cast<size_t>(g_op_count));
    ASSERT_EQ(num_ids,
              reversed_strings.size() + static_cast<size_t>(num_missing_strings));
    for (int32_t idx = 0; idx < static_cast<int32_t>(num_ids); ++idx) {
      if (idx < num_missing_strings) {
        ASSERT_EQ(translated_ids[idx], StringDictionary::INVALID_STR_ID);
      } else {
        ASSERT_EQ(translated_ids[idx], g_op_count - idx - 1);
      }
    }
  }
}

TEST(StringDictionaryProxy, GetOrAddTransient) {
  const DictRef dict_ref(-1, 1);
  std::shared_ptr<StringDictionary> string_dict = std::make_shared<StringDictionary>(
      dict_ref, BASE_PATH1, false, false, g_cache_string_hash);

  // Prep underlying dictionary data
  std::vector<int32_t> string_ids(g_op_count);
  std::vector<std::string> strings;
  strings.reserve(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    if (i % 2 == 0) {
      strings.emplace_back(std::to_string(i));
    } else {
      strings.emplace_back(std::to_string(i * -1));
    }
  }
  string_dict->getOrAddBulk(strings, string_ids.data());

  // Now make proxy from dictionary

  StringDictionaryProxy string_dict_proxy(
      string_dict, 1 /* string_dict_id */, string_dict->storageEntryCount());

  {
    // First iteration is identical to first of the StringDictionary GetOrAddBulk test,
    // and results should be the same
    std::vector<std::string> strings;
    std::vector<int32_t> string_ids(g_op_count);
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }

    for (size_t i = 0; i < g_op_count; ++i) {
      string_ids[i] = string_dict_proxy.getOrAddTransient(strings[i]);
    }

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }

  {
    // Now make the odd idx strings new, should get back transient string ids for those
    std::vector<std::string> strings;
    std::vector<int32_t> string_ids(g_op_count);
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(g_op_count * 4 + i));
      }
    }

    for (size_t i = 0; i < g_op_count; ++i) {
      string_ids[i] = string_dict_proxy.getOrAddTransient(strings[i]);
    }

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(string_ids[i], -2 - (i / 2));
      }
    }
  }
}

TEST(StringDictionaryProxy, GetOrAddTransientBulk) {
  // Use existing dictionary from GetBulk
  const DictRef dict_ref(-1, 1);
  std::shared_ptr<StringDictionary> string_dict = std::make_shared<StringDictionary>(
      dict_ref, BASE_PATH1, false, true, g_cache_string_hash);
  StringDictionaryProxy string_dict_proxy(
      string_dict, 1 /* string_dict_id */, string_dict->storageEntryCount());
  {
    // First iteration is identical to first of the StringDictionary GetOrAddBulk test,
    // and results should be the same
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }

    const auto string_ids = string_dict_proxy.getOrAddTransientBulk(strings);

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }

  {
    // Now make the odd idx strings new, should get back transient string ids for those
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(g_op_count * 6 + i));
      }
    }

    const auto string_ids = string_dict_proxy.getOrAddTransientBulk(strings);

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(string_ids[i], -2 - (i / 2));
      }
    }
  }
}

TEST(StringDictionaryProxy, GetTransientBulk) {
  // Use existing dictionary from GetBulk
  const DictRef dict_ref(-1, 1);
  std::shared_ptr<StringDictionary> string_dict = std::make_shared<StringDictionary>(
      dict_ref, BASE_PATH1, false, true, g_cache_string_hash);
  StringDictionaryProxy string_dict_proxy(
      string_dict, 1 /* string_dict_id */, string_dict->storageEntryCount());
  {
    // First iteration is identical to first of the StryingDictionaryProxy
    // GetOrAddTransientBulk test, and results should be the same
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(i * -1));
      }
    }

    const auto string_ids = string_dict_proxy.getTransientBulk(strings);

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(std::to_string(string_ids[i] * -1), strings[i]);
      }
    }
  }
  {
    // Second iteration - even idxs exist, odd do not and should return INVALID_STR_ID
    std::vector<std::string> strings;
    strings.reserve(g_op_count);
    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        strings.emplace_back(std::to_string(i));
      } else {
        strings.emplace_back(std::to_string(g_op_count * 8 + i));
      }
    }

    const auto string_ids = string_dict_proxy.getTransientBulk(strings);

    for (int i = 0; i < g_op_count; ++i) {
      if (i % 2 == 0) {
        ASSERT_EQ(std::to_string(string_ids[i]), strings[i]);
      } else {
        ASSERT_EQ(string_ids[i], StringDictionary::INVALID_STR_ID);
      }
    }
  }
}

TEST(StringDictionaryProxy, BuildTranslationMapToOtherProxy) {
  // Use existing dictionary from GetBulk
  const DictRef dict_ref1(-1, 1);
  const DictRef dict_ref2(-1, 2);
  std::shared_ptr<StringDictionary> source_string_dict =
      std::make_shared<StringDictionary>(
          dict_ref1, BASE_PATH1, false, false, g_cache_string_hash);
  std::shared_ptr<StringDictionary> dest_string_dict = std::make_shared<StringDictionary>(
      dict_ref2, BASE_PATH2, false, false, g_cache_string_hash);

  // Prep: insert g_op_count strings into source_string_dict
  std::vector<int32_t> string_ids(g_op_count);
  std::vector<std::string> strings;
  strings.reserve(g_op_count);
  for (int i = 0; i < g_op_count; ++i) {
    strings.emplace_back(std::to_string(i));
  }
  source_string_dict->getOrAddBulk(strings, string_ids.data());

  {
    // First try to translate to empty dictionary.
    // Should get back all INVALID_STR_IDs
    std::shared_ptr<StringDictionaryProxy> source_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(source_string_dict,
                                                1 /* string_dict_id */,
                                                source_string_dict->storageEntryCount());
    std::shared_ptr<StringDictionaryProxy> dest_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(dest_string_dict,
                                                2 /* string_dict_id */,
                                                dest_string_dict->storageEntryCount());
    const auto str_proxy_translation_map =
        source_string_dict_proxy->buildTranslationMapToOtherProxy(
            dest_string_dict_proxy.get());
    CHECK(str_proxy_translation_map);
    ASSERT_FALSE(str_proxy_translation_map->isEmpty());
    const auto& translated_ids = str_proxy_translation_map->translation_map_;
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids, source_string_dict_proxy->entryCount());
    ASSERT_EQ(num_ids, str_proxy_translation_map->size());
    for (size_t idx = 0; idx < num_ids; ++idx) {
      ASSERT_EQ(translated_ids[idx], StringDictionary::INVALID_STR_ID);
    }
  }

  {
    // Add transient entries to source proxy
    constexpr int32_t num_missing_ids{10};
    ASSERT_LE(num_missing_ids, g_op_count);
    const int32_t num_dest_strings{g_op_count - num_missing_ids};
    std::vector<int32_t> dest_string_ids(num_dest_strings);
    std::vector<std::string> dest_strings;
    dest_strings.reserve(num_dest_strings);
    for (int i = 0; i < num_dest_strings; ++i) {
      dest_strings.emplace_back(std::to_string(i));
    }
    dest_string_dict->getOrAddBulk(dest_strings, dest_string_ids.data());

    std::shared_ptr<StringDictionaryProxy> source_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(source_string_dict,
                                                1 /* string_dict_id */,
                                                source_string_dict->storageEntryCount());
    std::shared_ptr<StringDictionaryProxy> dest_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(dest_string_dict,
                                                2 /* string_dict_id */,
                                                dest_string_dict->storageEntryCount());

    constexpr int32_t num_source_proxy_transient_ids{64};
    constexpr int32_t num_dest_proxy_transient_ids{32};
    constexpr int32_t proxy_transient_start{g_op_count * 2};
    for (int i = proxy_transient_start;
         i < proxy_transient_start + num_source_proxy_transient_ids;
         ++i) {
      source_string_dict_proxy->getOrAddTransient(std::to_string(i));
    }
    for (int i = proxy_transient_start;
         i < proxy_transient_start + num_dest_proxy_transient_ids;
         ++i) {
      dest_string_dict_proxy->getOrAddTransient(std::to_string(i));
    }

    const auto str_proxy_translation_map =
        source_string_dict_proxy->buildTranslationMapToOtherProxy(
            dest_string_dict_proxy.get());
    CHECK(str_proxy_translation_map);
    ASSERT_FALSE(str_proxy_translation_map->isEmpty());
    const auto& translated_ids = str_proxy_translation_map->translation_map_;
    const size_t num_ids = translated_ids.size();
    // Todo (todd): Expose transient start offset instead of having to use magic number
    ASSERT_EQ(num_ids, source_string_dict_proxy->entryCount() + 1);
    ASSERT_EQ(num_ids, str_proxy_translation_map->size());
    const int32_t start_source_id =
        source_string_dict_proxy->transientEntryCount() > 0
            ? (source_string_dict_proxy->transientEntryCount() + 1) * -1
            : 0;
    for (int32_t idx = 0; idx < static_cast<int32_t>(num_ids); ++idx) {
      const int32_t source_id = idx + start_source_id;
      if (source_id == -1 || source_id == 0) {
        continue;
      }
      const std::string source_string = source_string_dict_proxy->getString(source_id);
      const int32_t source_string_as_int = std::stoi(source_string);
      const bool should_be_missing =
          !((source_string_as_int >= 0 && source_string_as_int < num_dest_strings) ||
            (source_string_as_int >= (g_op_count * 2) &&
             source_string_as_int < (g_op_count * 2 + num_dest_proxy_transient_ids)));
      if (translated_ids[idx] != StringDictionary::INVALID_STR_ID) {
        ASSERT_FALSE(should_be_missing);
        const std::string dest_string =
            dest_string_dict_proxy->getString(translated_ids[idx]);
        ASSERT_EQ(source_string, dest_string);

      } else {
        ASSERT_TRUE(should_be_missing);
      }
    }
  }
}

void create_directory_if_not_exists(const std::string& path) {
  // Up to caller to catch any exceptions
  std::filesystem::path fs_path(path);
  if (!(std::filesystem::exists(path))) {
    std::filesystem::create_directory(path);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;
  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all test");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()(
      "enable-string-dict-hash-cache",
      po::value<bool>(&g_cache_string_hash)
          ->default_value(g_cache_string_hash)
          ->implicit_value(true),
      "Cache string hash values in the string dictionary server during import.");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  try {
    create_directory_if_not_exists(BASE_PATH1);
    create_directory_if_not_exists(BASE_PATH2);
  } catch (std::exception& error) {
    LOG(FATAL) << "Could not create string dictionary directories.";
  }

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
