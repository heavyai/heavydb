/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "Shared/funcannotations.h"
#include "StringDictionary/StringDictionaryProxy.h"

#include <boost/lexical_cast.hpp>

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>

#ifndef BASE_PATH1
#define BASE_PATH1 "./tmp/dict1"
#endif

#ifndef BASE_PATH2
#define BASE_PATH2 "./tmp/dict2"
#endif

extern bool g_cache_string_hash;

class StringDictionaryTest : public TestHelpers::TsanTbbPrivateServerKiller {};
class StringDictionaryProxyTest : public TestHelpers::TsanTbbPrivateServerKiller {};

TEST_F(StringDictionaryTest, AddAndGet) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, g_cache_string_hash);
  auto id1 = string_dict.getOrAdd("foo bar");
  auto id2 = string_dict.getOrAdd("foo bar");
  ASSERT_EQ(id1, id2);
  ASSERT_EQ(0, id1);
  auto id3 = string_dict.getOrAdd("baz");
  ASSERT_EQ(1, id3);
  auto id4 = string_dict.getIdOfString(std::string_view("foo bar"));
  ASSERT_EQ(id1, id4);
  ASSERT_EQ("foo bar", string_dict.getString(id4));
  auto id5 = string_dict.getIdOfString(std::string("foo bar"));
  ASSERT_EQ(id1, id5);
  ASSERT_EQ("foo bar", string_dict.getString(id5));
}

TEST_F(StringDictionaryTest, Recover) {
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

TEST_F(StringDictionaryTest, HandleEmpty) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, g_cache_string_hash);
  auto id1 = string_dict.getOrAdd("");
  auto id2 = string_dict.getOrAdd("");
  ASSERT_EQ(id1, id2);
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), id1);
}

TEST_F(StringDictionaryTest, RecoverZero) {
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

TEST_F(StringDictionaryTest, ManyAddsAndGets) {
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

TEST_F(StringDictionaryTest, RecoverMany) {
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

TEST_F(StringDictionaryTest, GetStringViews) {
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

TEST_F(StringDictionaryTest, GetOrAddBulk) {
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

TEST_F(StringDictionaryTest, GetBulk) {
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

TEST_F(StringDictionaryTest, BuildTranslationMap) {
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

TEST_F(StringDictionaryProxyTest, GetOrAddTransient) {
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

TEST_F(StringDictionaryProxyTest, GetOrAddTransientBulk) {
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

TEST_F(StringDictionaryProxyTest, GetTransientBulk) {
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

TEST_F(StringDictionaryProxyTest, BuildIntersectionTranslationMapToOtherProxy) {
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
        source_string_dict_proxy->buildIntersectionTranslationMapToOtherProxy(
            dest_string_dict_proxy.get(), {});
    ASSERT_FALSE(str_proxy_translation_map.empty());
    const auto& translated_ids = str_proxy_translation_map.getVectorMap();
    const size_t num_ids = translated_ids.size() - 1;  // Subtract 1 for INVALID_STR_ID
    ASSERT_EQ(num_ids, source_string_dict_proxy->entryCount());
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
        source_string_dict_proxy->buildIntersectionTranslationMapToOtherProxy(
            dest_string_dict_proxy.get(), {});
    ASSERT_FALSE(str_proxy_translation_map.empty());
    const auto& translated_ids = str_proxy_translation_map.getVectorMap();
    const size_t num_ids = translated_ids.size() - 1;  // Subtract 1 for INVALID_STR_ID
    ASSERT_EQ(num_ids, source_string_dict_proxy->entryCount());
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

TEST_F(StringDictionaryProxyTest, BuildUnionTranslationMapToEmptyProxy) {
  // Todo(todd): Migrate this and intersection translation tests to use
  // approach and methods in BuildUnionTranslationMapToPartialOverlapProxy
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
    // First try to union translate to empty dictionary.
    // All strings should end up as transient entries in
    // destination proxy
    std::shared_ptr<StringDictionaryProxy> source_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(source_string_dict,
                                                1 /* string_dict_id */,
                                                source_string_dict->storageEntryCount());
    std::shared_ptr<StringDictionaryProxy> dest_string_dict_proxy =
        std::make_shared<StringDictionaryProxy>(dest_string_dict,
                                                2 /* string_dict_id */,
                                                dest_string_dict->storageEntryCount());
    const auto str_proxy_translation_map =
        source_string_dict_proxy->buildUnionTranslationMapToOtherProxy(
            dest_string_dict_proxy.get(), {});
    ASSERT_FALSE(str_proxy_translation_map.empty());
    ASSERT_EQ(str_proxy_translation_map.numUntranslatedStrings(), strings.size());
    const auto& translated_ids = str_proxy_translation_map.getVectorMap();
    const size_t num_ids = translated_ids.size();
    ASSERT_EQ(num_ids - 1UL /* for INVALID_STR_ID */,
              source_string_dict_proxy->entryCount());
    const auto domain_start = str_proxy_translation_map.domainStart();
    for (size_t idx = 0; idx < num_ids; ++idx) {
      const auto string_id = static_cast<int32_t>(idx) + domain_start;
      if (string_id == -1) {
        ASSERT_EQ(translated_ids[idx], StringDictionary::INVALID_STR_ID);
      } else {
        ASSERT_LT(translated_ids[idx], StringDictionary::INVALID_STR_ID);
        ASSERT_EQ(dest_string_dict_proxy->getString(translated_ids[idx]),
                  strings[string_id]);
      }
    }
  }
}

std::vector<std::string> add_strings_numeric_range(std::shared_ptr<StringDictionary> sd,
                                                   const size_t num_vals,
                                                   const int32_t start_val) {
  CHECK_GE(start_val, 0);
  const auto end_val = start_val + static_cast<int32_t>(num_vals);
  std::vector<std::string> strings(num_vals);
  for (int32_t int_val = start_val; int_val < end_val; ++int_val) {
    strings[int_val - start_val] = std::to_string(int_val);
  }
  std::vector<int32_t> string_ids(num_vals);
  sd->getOrAddBulk(strings, string_ids.data());
  sd->checkpoint();
  return strings;
}

std::vector<std::string> add_strings_numeric_range(StringDictionaryProxy& sdp,
                                                   const size_t num_vals,
                                                   const int32_t start_val) {
  CHECK_LE(start_val, -2);
  std::vector<std::string> strings(num_vals);
  const auto end_val = start_val - static_cast<int32_t>(num_vals);
  for (int32_t int_val = start_val; int_val > end_val; --int_val) {
    strings[start_val - int_val] = std::to_string(int_val);
  }
  sdp.getOrAddTransientBulk(strings);
  return strings;
}

size_t calc_expected_untranslated_strings(const size_t num_source_entries,
                                          const size_t num_dest_entries,
                                          const int32_t source_start_val,
                                          const int32_t dest_start_val) {
  const int32_t source_abs_min = std::abs(source_start_val);  // 0
  const int32_t source_abs_max =
      std::abs(source_start_val) + static_cast<int32_t>(num_source_entries);  // 10
  const int32_t dest_abs_min = std::abs(dest_start_val);                      // 7
  const int32_t dest_abs_max =
      std::abs(dest_start_val) + static_cast<int32_t>(num_dest_entries);  // 12
  return static_cast<size_t>(std::max(dest_abs_min - source_abs_min, 0) +
                             std::max(source_abs_max - dest_abs_max, 0));
}

void verify_translation(const StringDictionaryProxy& source_proxy,
                        const StringDictionaryProxy& dest_proxy,
                        const StringDictionaryProxy::IdMap& id_map,
                        const std::vector<std::string>& persisted_source_strings,
                        const std::vector<std::string>& transient_source_strings,
                        const std::vector<std::string>& persisted_dest_strings,
                        const std::vector<std::string>& transient_dest_strings,
                        const bool is_union) {
  std::unordered_set<std::string> dest_string_set;
  for (const auto& str : persisted_dest_strings) {
    ASSERT_TRUE(dest_string_set.insert(str).second);
  }
  for (const auto& str : transient_dest_strings) {
    ASSERT_TRUE(dest_string_set.insert(str).second);
  }
  for (const auto& persisted_source_str : persisted_source_strings) {
    const int32_t source_str_id = source_proxy.getIdOfString(persisted_source_str);
    ASSERT_GE(source_str_id, 0);
    const bool str_was_in_dest_proxy =
        dest_string_set.find(persisted_source_str) != dest_string_set.end();
    const auto translated_id = id_map[source_str_id];
    if (str_was_in_dest_proxy) {
      ASSERT_GE(translated_id, 0);
      const auto dest_str = dest_proxy.getString(translated_id);
      ASSERT_EQ(persisted_source_str, dest_str);
    } else {
      if (is_union) {
        ASSERT_LE(translated_id, -2);
        const auto dest_str = dest_proxy.getString(translated_id);
        ASSERT_EQ(persisted_source_str, dest_str);
      } else {
        ASSERT_EQ(translated_id, StringDictionary::INVALID_STR_ID);
      }
    }
  }
}

TEST_F(StringDictionaryProxyTest, BuildUnionTranslationMapToPartialOverlapProxy) {
  const DictRef dict_ref1(-1, 1);
  const DictRef dict_ref2(-1, 2);
  std::shared_ptr<StringDictionary> source_sd = std::make_shared<StringDictionary>(
      dict_ref1, BASE_PATH1, false, false, g_cache_string_hash);
  std::shared_ptr<StringDictionary> dest_sd = std::make_shared<StringDictionary>(
      dict_ref2, BASE_PATH2, false, false, g_cache_string_hash);

  constexpr size_t num_source_persisted_entries{10};
  constexpr size_t num_dest_persisted_entries{5};
  constexpr int32_t source_persisted_start_val{0};
  constexpr int32_t dest_persisted_start_val{7};

  constexpr size_t num_source_transient_entries{10};
  constexpr size_t num_dest_transient_entries{5};
  constexpr int32_t source_transient_start_val{-2};
  constexpr int32_t dest_transient_start_val{-9};
  const auto persisted_source_strings = add_strings_numeric_range(
      source_sd, num_source_persisted_entries, source_persisted_start_val);
  ASSERT_EQ(source_sd->storageEntryCount(), num_source_persisted_entries);

  const auto persisted_dest_strings = add_strings_numeric_range(
      dest_sd, num_dest_persisted_entries, dest_persisted_start_val);
  ASSERT_EQ(dest_sd->storageEntryCount(), num_dest_persisted_entries);

  StringDictionaryProxy source_sdp(
      source_sd, 1 /* string_dict_id */, source_sd->storageEntryCount());
  StringDictionaryProxy dest_sdp(
      dest_sd, 2 /* string_dict_id */, dest_sd->storageEntryCount());
  const auto transient_source_strings = add_strings_numeric_range(
      source_sdp, num_source_transient_entries, source_transient_start_val);
  ASSERT_EQ(source_sdp.getDictId(), 1);
  ASSERT_EQ(source_sdp.storageEntryCount(), num_source_persisted_entries);
  ASSERT_EQ(source_sdp.transientEntryCount(), num_source_transient_entries);

  const auto transient_dest_strings = add_strings_numeric_range(
      dest_sdp, num_dest_transient_entries, dest_transient_start_val);
  ASSERT_EQ(dest_sdp.getDictId(), 2);
  ASSERT_EQ(dest_sdp.storageEntryCount(), num_dest_persisted_entries);
  ASSERT_EQ(dest_sdp.transientEntryCount(), num_dest_transient_entries);

  const auto id_map = source_sdp.buildUnionTranslationMapToOtherProxy(&dest_sdp, {});
  const size_t expected_num_untranslated_strings =
      calc_expected_untranslated_strings(num_source_persisted_entries,
                                         num_dest_persisted_entries,
                                         source_persisted_start_val,
                                         dest_persisted_start_val) +
      calc_expected_untranslated_strings(num_source_transient_entries,
                                         num_dest_transient_entries,
                                         source_transient_start_val,
                                         dest_transient_start_val);
  ASSERT_EQ(id_map.size(),
            num_source_persisted_entries + num_source_transient_entries +
                1UL /* slot for INVALID_STR_ID */);
  ASSERT_EQ(id_map.numNonTransients(), num_source_persisted_entries);
  ASSERT_EQ(id_map.numTransients(), num_source_transient_entries);
  ASSERT_EQ(id_map.numUntranslatedStrings(), expected_num_untranslated_strings);

  verify_translation(source_sdp,
                     dest_sdp,
                     id_map,
                     persisted_source_strings,
                     transient_source_strings,
                     persisted_dest_strings,
                     transient_dest_strings,
                     true);
}

TEST_F(StringDictionaryTest, TransientUnion) {
  using namespace std::literals;  // for sv suffix

  std::string const sd_lhs_path = std::string(BASE_PATH) + "/sd_lhs";
  std::string const sd_rhs_path = std::string(BASE_PATH) + "/sd_rhs";
#ifdef _WIN32
  mkdir(sd_lhs_path.c_str());
  mkdir(sd_rhs_path.c_str());
#else
  mkdir(sd_lhs_path.c_str(), 0755);
  mkdir(sd_rhs_path.c_str(), 0755);
#endif

  dict_ref_t const dict_ref_lhs(100, 10);
  auto sd_lhs = std::make_shared<StringDictionary>(
      dict_ref_lhs, sd_lhs_path, false, true, g_cache_string_hash);
  sd_lhs->getOrAdd("a");  // id = 0
  sd_lhs->getOrAdd("b");  // id = 1
  sd_lhs->getOrAdd("c");  // id = 2
  sd_lhs->getOrAdd("d");  // id = 3
  sd_lhs->getOrAdd("e");  // id = 4

  dict_ref_t const dict_ref_rhs(100, 20);
  auto sd_rhs = std::make_shared<StringDictionary>(
      dict_ref_rhs, sd_rhs_path, false, true, g_cache_string_hash);
  sd_rhs->getOrAdd("c");  // id = 0
  sd_rhs->getOrAdd("d");  // id = 1
  sd_rhs->getOrAdd("e");  // id = 2
  sd_rhs->getOrAdd("f");  // id = 3
  sd_rhs->getOrAdd("g");  // id = 4
  sd_rhs->getOrAdd("h");  // id = 5
  sd_rhs->getOrAdd("i");  // id = 6

  {
    // TODO cleanup redundancy of setting SD id here.
    StringDictionaryProxy sdp_lhs(
        sd_lhs, dict_ref_lhs.dictId, sd_lhs->storageEntryCount());
    sdp_lhs.getOrAddTransient("t0"sv);  // id = -2
    StringDictionaryProxy sdp_rhs(
        sd_rhs, dict_ref_rhs.dictId, sd_rhs->storageEntryCount());
    sdp_rhs.getOrAddTransient("t0"sv);  // id = -2
    sdp_rhs.getOrAddTransient("t1"sv);  // id = -3
    auto const id_map = sdp_lhs.transientUnion(sdp_rhs);
    // Expected output:
    // source_domain_min_ = -3 = -1 - number of transients in sdp_rhs
    // t1 -3 (1st transient added to lhs when calling transientUnion())
    // t0 -2 (exising transient id in sd_lhs)
    //    -1 (There is always at least one -1 so that id=-1 maps to -1)
    // c   2 (existing id in sd_lhs)
    // d   3 (existing id in sd_lhs)
    // e   4 (existing id in sd_lhs)
    // f  -4 (2nd transient added to lhs when calling transientUnion())
    // g  -5 (3rd transient added to lhs when calling transientUnion())
    // h  -6 (4th transient added to lhs when calling transientUnion())
    // i  -7 (5th transient added to lhs when calling transientUnion())
    //           rhs id(-3 -2 -1 0 1 2  3  4  5  6)
    // translation_map_(-3 -2 -1 2 3 4 -4 -5 -6 -7)
    ASSERT_EQ("IdMap(offset_(3) vector_map_(-3 -2 -1 2 3 4 -4 -5 -6 -7))",
              boost::lexical_cast<std::string>(id_map));
    ASSERT_EQ(0, sdp_lhs.getIdOfString("a"));
    ASSERT_EQ(1, sdp_lhs.getIdOfString("b"));
    ASSERT_EQ(2, sdp_lhs.getIdOfString("c"));
    ASSERT_EQ(3, sdp_lhs.getIdOfString("d"));
    ASSERT_EQ(4, sdp_lhs.getIdOfString("e"));
    ASSERT_EQ(-2, sdp_lhs.getIdOfString("t0"));
    ASSERT_EQ(-3, sdp_lhs.getIdOfString("t1"));
    ASSERT_EQ(-4, sdp_lhs.getIdOfString("f"));
    ASSERT_EQ(-5, sdp_lhs.getIdOfString("g"));
    ASSERT_EQ(-6, sdp_lhs.getIdOfString("h"));
    ASSERT_EQ(-7, sdp_lhs.getIdOfString("i"));
  }

  {  // Swap sd_lhs <-> sd_rhs
    StringDictionaryProxy sdp_lhs(
        sd_rhs, dict_ref_lhs.dictId, sd_rhs->storageEntryCount());
    sdp_lhs.getOrAddTransient("t0"sv);
    StringDictionaryProxy sdp_rhs(
        sd_lhs, dict_ref_rhs.dictId, sd_lhs->storageEntryCount());
    sdp_rhs.getOrAddTransient("t0"sv);
    sdp_rhs.getOrAddTransient("t1"sv);
    auto const id_map = sdp_lhs.transientUnion(sdp_rhs);
    // Expected output:
    // source_domain_min_ = -3 = -1 - number of transients in sdp_rhs
    // t1 -3 (1st transient added to lhs when calling transientUnion())
    // t0 -2 (existing transient id in sd_lhs)
    //    -1 (There is always at least one -1 so that id=-1 maps to -1)
    // a  -4 (2nd transient added to lhs when calling transientUnion())
    // b  -5 (3rd transient added to lhs when calling transientUnion())
    // c   0 (existing id in sd_rhs)
    // d   1 (existing id in sd_rhs)
    // e   2 (existing id in sd_rhs)
    // translation_map_(-3 -2 -1 -4 -5 0 1 2)
    ASSERT_EQ("IdMap(offset_(3) vector_map_(-3 -2 -1 -4 -5 0 1 2))",
              boost::lexical_cast<std::string>(id_map));
    ASSERT_EQ(-2, sdp_lhs.getIdOfString("t0"));
    ASSERT_EQ(-3, sdp_lhs.getIdOfString("t1"));
    ASSERT_EQ(-4, sdp_lhs.getIdOfString("a"));
    ASSERT_EQ(-5, sdp_lhs.getIdOfString("b"));
    ASSERT_EQ(0, sdp_lhs.getIdOfString("c"));
    ASSERT_EQ(1, sdp_lhs.getIdOfString("d"));
    ASSERT_EQ(2, sdp_lhs.getIdOfString("e"));
    ASSERT_EQ(3, sdp_lhs.getIdOfString("f"));
    ASSERT_EQ(4, sdp_lhs.getIdOfString("g"));
    ASSERT_EQ(5, sdp_lhs.getIdOfString("h"));
    ASSERT_EQ(6, sdp_lhs.getIdOfString("i"));
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
