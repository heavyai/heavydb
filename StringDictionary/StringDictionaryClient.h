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

#ifndef STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H
#define STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H

#include "DictRef.h"
#include "LeafHostInfo.h"

#include <memory>
#include <mutex>

class StringDictionaryClient {
 public:
  StringDictionaryClient(const LeafHostInfo& server_host,
                         const DictRef dict_id,
                         const bool with_timeout) {
    CHECK(false);
  };

  void create(const DictRef dict_ref, const bool is_temp) { CHECK(false); };

  void drop(const DictRef dict_ref) { CHECK(false); };

  int32_t get(const std::string& str) {
    CHECK(false);
    return 0;
  };

  void get_string(std::string& _return, const int32_t string_id) { CHECK(false); };

  int64_t storage_entry_count() {
    CHECK(false);
    return 0;
  };

  std::vector<int32_t> get_like(const std::string& pattern,
                                const bool icase,
                                const bool is_simple,
                                const char escape,
                                const int64_t generation) {
    CHECK(false);
    return std::vector<int32_t>{};
  };

  std::vector<int32_t> get_compare(const std::string& pattern,
                                   const std::string& comp_operator,
                                   const int64_t generation) {
    CHECK(false);
    return std::vector<int32_t>{};
  };

  std::vector<int32_t> get_regexp_like(const std::string& pattern,
                                       const char escape,
                                       const int64_t generation) {
    CHECK(false);
    return std::vector<int32_t>{};
  };

  template <class String>
  void get_bulk(std::vector<int32_t>& string_ids, const std::vector<String>& strings);

  template <class String>
  void get_or_add_bulk(std::vector<int32_t>& string_ids,
                       const std::vector<String>& strings) {
    CHECK(false);
  };

  template <class String>
  void get_or_add_bulk_array(std::vector<std::vector<int32_t>>& string_ids_array,
                             const std::vector<std::vector<String>>& strings_array) {
    CHECK(false);
  }

  void populate_string_ids(std::vector<int32_t>& dest_ids,
                           const DictRef dest_dict_ref,
                           const std::vector<int32_t>& source_ids,
                           const DictRef source_dict_ref);
  void populate_string_array_ids(
      std::vector<std::vector<int32_t>>& dest_array_ids,
      const DictRef dest_dict_ref,
      const std::vector<std::vector<int32_t>>& source_array_ids,
      const DictRef source_dict_ref);
  void translate_string_ids(std::vector<int32_t>& dest_ids,
                            const DictRef dest_dict_ref,
                            const std::vector<int32_t>& source_ids,
                            const DictRef source_dict_ref,
                            const int32_t dest_generation) {
    CHECK(false);
  };

  bool checkpoint() {
    CHECK(false);
    return false;
  };
};

#endif  // STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H
