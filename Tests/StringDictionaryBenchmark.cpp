/*
 * Copyright 2022 OmniSci, Inc.
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
#include "TestHelpers.h"

#include <benchmark/benchmark.h>

#ifndef BASE_PATH1
#define BASE_PATH1 "./tmp/dict1"
#endif

#ifndef BASE_PATH2
#define BASE_PATH2 "./tmp/dict2"
#endif

#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <vector>

std::string generate_random_str(std::mt19937& generator, const int64_t str_len) {
  constexpr char alphanum_lookup_table[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  constexpr size_t char_mod = sizeof(alphanum_lookup_table) - 1;
  std::uniform_int_distribution<int32_t> rand_distribution(0, char_mod);

  std::string tmp_s;
  tmp_s.reserve(str_len);
  for (int i = 0; i < str_len; ++i) {
    tmp_s += alphanum_lookup_table[rand_distribution(generator)];
  }
  return tmp_s;
}

std::vector<std::string> generate_random_strs(const size_t num_strings,
                                              const size_t num_unique_strings,
                                              const size_t str_len,
                                              const uint64_t seed = 42) {
  std::mt19937 rand_generator(seed);
  std::vector<std::string> unique_strings(num_unique_strings);
  for (size_t string_idx = 0; string_idx < num_unique_strings; ++string_idx) {
    unique_strings[string_idx] = generate_random_str(rand_generator, str_len);
  }
  std::vector<std::string> strings(num_strings);
  for (size_t string_idx = 0; string_idx < num_strings; ++string_idx) {
    strings[string_idx] = unique_strings[string_idx % num_unique_strings];
  }
  return strings;
}

void create_directory_if_not_exists(const std::string& path) {
  // Up to caller to catch any exceptions
  std::filesystem::path fs_path(path);
  if (!(std::filesystem::exists(path))) {
    std::filesystem::create_directory(path);
  }
}

std::vector<std::string> append_strings_10M_100K_10;
std::vector<std::string> append_strings_10M_1M_10;
std::vector<std::string> append_strings_10M_1M_10_randomized;
std::vector<std::string> append_strings_10M_10M_10;
std::vector<std::string> append_strings_10M_10M_10_randomized;

std::once_flag setup_flag;
void global_setup() {
  TestHelpers::init_logger_stderr_only();
  create_directory_if_not_exists(BASE_PATH1);
  create_directory_if_not_exists(BASE_PATH2);
  append_strings_10M_100K_10 = generate_random_strs(10000000, 100000, 10, 1);
  append_strings_10M_1M_10 = generate_random_strs(10000000, 1000000, 10, 2);
  append_strings_10M_10M_10 = generate_random_strs(10000000, 10000000, 10, 3);
  append_strings_10M_1M_10_randomized = append_strings_10M_1M_10;
  std::random_shuffle(append_strings_10M_1M_10_randomized.begin(),
                      append_strings_10M_1M_10_randomized.end());
  append_strings_10M_10M_10_randomized = append_strings_10M_10M_10;
  std::random_shuffle(append_strings_10M_10M_10_randomized.begin(),
                      append_strings_10M_10M_10_randomized.end());
}

class StringDictionaryFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);
  }
  void TearDown(const ::benchmark::State& state) override {
    // noop
  }
};

std::shared_ptr<StringDictionary> create_str_dict(const int32_t dict_id,
                                                  const std::string& path,
                                                  const bool is_temp,
                                                  const bool recover,
                                                  const bool materialize_hashes) {
  const DictRef dict_ref(-1, dict_id);
  std::shared_ptr<StringDictionary> string_dict = std::make_shared<StringDictionary>(
      dict_ref, path, is_temp, recover, materialize_hashes);
  return string_dict;
}

std::shared_ptr<StringDictionary> create_and_populate_str_dict(
    const int32_t dict_id,
    const std::string& path,
    const bool is_temp,
    const bool recover,
    const bool materialize_hashes,
    const std::vector<std::string>& strings) {
  auto string_dict = create_str_dict(dict_id, path, is_temp, recover, materialize_hashes);
  std::vector<int32_t> string_ids(strings.size());
  string_dict->getOrAddBulk(strings, string_ids.data());
  string_dict->checkpoint();
  return string_dict;
}

std::shared_ptr<StringDictionaryProxy> create_str_proxy(const int32_t dict_id,
                                                        const std::string& path,
                                                        const bool is_temp,
                                                        const bool recover,
                                                        const bool materialize_hashes) {
  auto string_dict = create_str_dict(dict_id, path, is_temp, recover, materialize_hashes);
  std::shared_ptr<StringDictionaryProxy> string_proxy =
      std::make_shared<StringDictionaryProxy>(string_dict,
                                              dict_id, /* string_dict_id */
                                              string_dict->storageEntryCount());
  return string_proxy;
}

std::shared_ptr<StringDictionaryProxy> create_and_populate_str_proxy(
    const int32_t dict_id,
    const std::string& path,
    const bool is_temp,
    const bool recover,
    const bool materialize_hashes,
    const std::vector<std::string>& strings) {
  auto string_dict = create_and_populate_str_dict(
      dict_id, path, is_temp, recover, materialize_hashes, strings);
  std::shared_ptr<StringDictionaryProxy> string_proxy =
      std::make_shared<StringDictionaryProxy>(string_dict,
                                              dict_id, /* string_dict_id */
                                              string_dict->storageEntryCount());
  return string_proxy;
}

BENCHMARK_DEFINE_F(StringDictionaryFixture, Create)
(benchmark::State& state) {
  for (auto _ : state) {
    const DictRef dict_ref(-1, 1);
    StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, true);
  }
}

// Todo(todd): Template BulkAppend* and other tests where we just vary
// the sizes/inputs to avoid repetition and allow for more granular
// testing of input parameters

BENCHMARK_DEFINE_F(StringDictionaryFixture, BulkAppend_100K_Unique)
(benchmark::State& state) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, true);
  std::vector<int32_t> string_ids(append_strings_10M_100K_10.size());
  for (auto _ : state) {
    string_dict.getOrAddBulk(append_strings_10M_100K_10, string_ids.data());
  }
}

BENCHMARK_DEFINE_F(StringDictionaryFixture, BulkAppend_1M_Unique)
(benchmark::State& state) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, true);
  std::vector<int32_t> string_ids(append_strings_10M_1M_10.size());
  for (auto _ : state) {
    string_dict.getOrAddBulk(append_strings_10M_1M_10, string_ids.data());
  }
}

BENCHMARK_DEFINE_F(StringDictionaryFixture, BulkAppend_10M_Unique)
(benchmark::State& state) {
  const DictRef dict_ref(-1, 1);
  StringDictionary string_dict(dict_ref, BASE_PATH1, false, false, true);
  std::vector<int32_t> string_ids(append_strings_10M_10M_10.size());
  for (auto _ : state) {
    string_dict.getOrAddBulk(append_strings_10M_10M_10, string_ids.data());
  }
}

BENCHMARK_DEFINE_F(StringDictionaryFixture, Reload)
(benchmark::State& state) {
  for (auto _ : state) {
    const DictRef dict_ref(-1, 1);
    StringDictionary string_dict(dict_ref, BASE_PATH1, false, true, true);
    CHECK_EQ(string_dict.storageEntryCount(), 10000000UL);
  }
}

BENCHMARK_DEFINE_F(StringDictionaryFixture, BulkAppend_10M_Dups)
(benchmark::State& state) {
  const auto string_dict = create_and_populate_str_dict(
      1, BASE_PATH1, false, false, false, append_strings_10M_10M_10);
  std::vector<int32_t> string_ids(append_strings_10M_10M_10_randomized.size());
  for (auto _ : state) {
    string_dict->getOrAddBulk(append_strings_10M_10M_10_randomized, string_ids.data());
  }
}

BENCHMARK_DEFINE_F(StringDictionaryFixture, BulkGet_10M_Unique)
(benchmark::State& state) {
  const auto string_dict = create_and_populate_str_dict(
      1, BASE_PATH1, false, false, false, append_strings_10M_10M_10);
  std::vector<int32_t> string_ids(append_strings_10M_10M_10_randomized.size());
  for (auto _ : state) {
    string_dict->getBulk(append_strings_10M_10M_10_randomized, string_ids.data());
  }
}

BENCHMARK_DEFINE_F(StringDictionaryFixture, BulkTranslation_10M_Unique)
(benchmark::State& state) {
  const auto source_string_dict = create_and_populate_str_dict(
      1, BASE_PATH1, false, false, true, append_strings_10M_10M_10);
  const auto dest_string_dict = create_and_populate_str_dict(
      2, BASE_PATH2, false, false, true, append_strings_10M_10M_10_randomized);

  const size_t num_source_strings = source_string_dict->storageEntryCount();
  std::vector<int32_t> string_ids(num_source_strings);
  const size_t num_dest_strings = dest_string_dict->storageEntryCount();
  auto dummy_callback = [](const std::string_view& source_string,
                           const int32_t source_string_id) { return true; };
  for (auto _ : state) {
    source_string_dict->buildDictionaryTranslationMap(dest_string_dict.get(),
                                                      string_ids.data(),
                                                      num_source_strings,
                                                      num_dest_strings,
                                                      false,
                                                      dummy_callback);
  }
}

BENCHMARK_REGISTER_F(StringDictionaryFixture, Create)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryFixture, BulkAppend_100K_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryFixture, BulkAppend_1M_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryFixture, BulkAppend_10M_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryFixture, Reload)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryFixture, BulkAppend_10M_Dups)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryFixture, BulkGet_10M_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryFixture, BulkTranslation_10M_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

class StringDictionaryProxyFixture : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);
  }
  void TearDown(const ::benchmark::State& state) override {
    // noop
  }
};

BENCHMARK_DEFINE_F(StringDictionaryProxyFixture, GetOrAddTransientBulk_1M_Unique)
(benchmark::State& state) {
  const auto string_dict_proxy = create_str_proxy(1, BASE_PATH1, false, false, true);
  for (auto _ : state) {
    const auto string_ids =
        string_dict_proxy->getOrAddTransientBulk(append_strings_10M_1M_10);
  }
}

BENCHMARK_DEFINE_F(StringDictionaryProxyFixture, GetOrAddTransient_1M_Unique)
(benchmark::State& state) {
  const auto string_dict_proxy = create_str_proxy(1, BASE_PATH1, false, false, true);

  const size_t num_strings = append_strings_10M_1M_10.size();
  std::vector<int32_t> string_ids(num_strings);
  for (auto _ : state) {
    for (size_t string_idx = 0; string_idx < num_strings; ++string_idx) {
      string_ids[string_idx] =
          string_dict_proxy->getOrAddTransient(append_strings_10M_1M_10[string_idx]);
    }
  }
}

BENCHMARK_DEFINE_F(StringDictionaryProxyFixture,
                   GetOrAddTransientBulk_AllPersisted_1M_Unique)
(benchmark::State& state) {
  const auto string_dict_proxy = create_and_populate_str_proxy(
      1, BASE_PATH1, false, false, true, append_strings_10M_1M_10);
  for (auto _ : state) {
    const auto proxy_string_ids =
        string_dict_proxy->getOrAddTransientBulk(append_strings_10M_1M_10_randomized);
  }
}

BENCHMARK_DEFINE_F(StringDictionaryProxyFixture, GetOrAddTransient_AllPersisted_1M_Unique)
(benchmark::State& state) {
  const auto string_proxy = create_and_populate_str_proxy(
      1, BASE_PATH1, false, false, true, append_strings_10M_1M_10);
  const size_t num_strings = append_strings_10M_1M_10.size();
  std::vector<int32_t> proxy_string_ids(num_strings);
  for (auto _ : state) {
    for (size_t string_idx = 0; string_idx < num_strings; ++string_idx) {
      proxy_string_ids[string_idx] = string_proxy->getOrAddTransient(
          append_strings_10M_1M_10_randomized[string_idx]);
    }
  }
}

BENCHMARK_DEFINE_F(StringDictionaryProxyFixture,
                   BuildIntersectionTranlsatationMapToOtherProxy_10M_Unique_Full_Overlap)
(benchmark::State& state) {
  const auto source_proxy = create_and_populate_str_proxy(
      1, BASE_PATH1, false, false, true, append_strings_10M_10M_10);
  const auto dest_proxy = create_and_populate_str_proxy(
      2, BASE_PATH2, false, false, true, append_strings_10M_10M_10_randomized);
  for (auto _ : state) {
    auto id_map =
        source_proxy->buildIntersectionTranslationMapToOtherProxy(dest_proxy.get());
    CHECK_EQ(id_map.numUntranslatedStrings(), 0UL);
  }
}

BENCHMARK_DEFINE_F(
    StringDictionaryProxyFixture,
    BuildIntersectionTranlsatationMapToOtherProxy_10M_Unique_8M_Overlap_100_Transients)
(benchmark::State& state) {
  const auto source_proxy = create_and_populate_str_proxy(
      1, BASE_PATH1, false, false, true, append_strings_10M_10M_10);
  const size_t num_elems = 10000000UL;
  const size_t first_n_elems = 8000000UL;
  const size_t last_n_elems = 100;
  std::vector<std::string> append_strings_10M_10M_10_randomized_truncated_8M(
      first_n_elems);
  std::vector<std::string> append_strings_10M_10M_10_randomized_truncated_100(100);
  std::copy(append_strings_10M_10M_10_randomized.begin(),
            append_strings_10M_10M_10_randomized.begin() + first_n_elems,
            append_strings_10M_10M_10_randomized_truncated_8M.begin());
  std::copy(append_strings_10M_10M_10_randomized.begin() + (num_elems - last_n_elems),
            append_strings_10M_10M_10_randomized.end(),
            append_strings_10M_10M_10_randomized_truncated_100.begin());
  auto dest_proxy =
      create_and_populate_str_proxy(2,
                                    BASE_PATH2,
                                    false,
                                    false,
                                    true,
                                    append_strings_10M_10M_10_randomized_truncated_8M);
  dest_proxy->getOrAddTransientBulk(append_strings_10M_10M_10_randomized_truncated_100);
  for (auto _ : state) {
    auto id_map =
        source_proxy->buildIntersectionTranslationMapToOtherProxy(dest_proxy.get());
    const size_t num_expected_untranslated_strings =
        num_elems - first_n_elems - last_n_elems;
    CHECK_EQ(id_map.numUntranslatedStrings(), num_expected_untranslated_strings);
  }
}

BENCHMARK_DEFINE_F(
    StringDictionaryProxyFixture,
    BuildUnionTranlsatationMapToOtherProxy_10M_Unique_8M_Overlap_100_Transients)
(benchmark::State& state) {
  const auto source_proxy = create_and_populate_str_proxy(
      1, BASE_PATH1, false, false, true, append_strings_10M_10M_10);
  const size_t num_elems = 10000000UL;
  const size_t first_n_elems = 8000000UL;
  const size_t last_n_elems = 100;
  std::vector<std::string> append_strings_10M_10M_10_randomized_truncated_8M(
      first_n_elems);
  std::vector<std::string> append_strings_10M_10M_10_randomized_truncated_100(100);
  std::copy(append_strings_10M_10M_10_randomized.begin(),
            append_strings_10M_10M_10_randomized.begin() + first_n_elems,
            append_strings_10M_10M_10_randomized_truncated_8M.begin());
  std::copy(append_strings_10M_10M_10_randomized.begin() + (num_elems - last_n_elems),
            append_strings_10M_10M_10_randomized.end(),
            append_strings_10M_10M_10_randomized_truncated_100.begin());
  auto dest_proxy =
      create_and_populate_str_proxy(2,
                                    BASE_PATH2,
                                    false,
                                    false,
                                    true,
                                    append_strings_10M_10M_10_randomized_truncated_8M);
  dest_proxy->getOrAddTransientBulk(append_strings_10M_10M_10_randomized_truncated_100);
  for (auto _ : state) {
    auto id_map = source_proxy->buildUnionTranslationMapToOtherProxy(dest_proxy.get());
    const size_t num_expected_untranslated_strings =
        num_elems - first_n_elems - last_n_elems;
    CHECK_EQ(id_map.numUntranslatedStrings(), num_expected_untranslated_strings);
  }
}

BENCHMARK_DEFINE_F(StringDictionaryProxyFixture,
                   TransientUnion_10M_Unique_8M_Overlap_100_Transients)
(benchmark::State& state) {
  const auto source_proxy = create_and_populate_str_proxy(
      1, BASE_PATH1, false, false, true, append_strings_10M_10M_10);
  const size_t num_elems = 10000000UL;
  const size_t first_n_elems = 8000000UL;
  const size_t last_n_elems = 100;
  std::vector<std::string> append_strings_10M_10M_10_randomized_truncated_8M(
      first_n_elems);
  std::vector<std::string> append_strings_10M_10M_10_randomized_truncated_100(100);
  std::copy(append_strings_10M_10M_10_randomized.begin(),
            append_strings_10M_10M_10_randomized.begin() + first_n_elems,
            append_strings_10M_10M_10_randomized_truncated_8M.begin());
  std::copy(append_strings_10M_10M_10_randomized.begin() + (num_elems - last_n_elems),
            append_strings_10M_10M_10_randomized.end(),
            append_strings_10M_10M_10_randomized_truncated_100.begin());
  auto dest_proxy =
      create_and_populate_str_proxy(2,
                                    BASE_PATH2,
                                    false,
                                    false,
                                    true,
                                    append_strings_10M_10M_10_randomized_truncated_8M);
  dest_proxy->getOrAddTransientBulk(append_strings_10M_10M_10_randomized_truncated_100);
  for (auto _ : state) {
    auto id_map = dest_proxy->transientUnion(*(source_proxy.get()));
  }
}

BENCHMARK_REGISTER_F(StringDictionaryProxyFixture, GetOrAddTransientBulk_1M_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryProxyFixture, GetOrAddTransient_1M_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryProxyFixture,
                     GetOrAddTransientBulk_AllPersisted_1M_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryProxyFixture,
                     GetOrAddTransient_AllPersisted_1M_Unique)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(
    StringDictionaryProxyFixture,
    BuildIntersectionTranlsatationMapToOtherProxy_10M_Unique_Full_Overlap)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(
    StringDictionaryProxyFixture,
    BuildIntersectionTranlsatationMapToOtherProxy_10M_Unique_8M_Overlap_100_Transients)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(
    StringDictionaryProxyFixture,
    BuildUnionTranlsatationMapToOtherProxy_10M_Unique_8M_Overlap_100_Transients)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(StringDictionaryProxyFixture,
                     TransientUnion_10M_Unique_8M_Overlap_100_Transients)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
