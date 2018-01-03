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

#ifndef STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H
#define STRINGDICTIONARY_STRINGDICTIONARYCLIENT_H

#include "DictRef.h"
#include "LeafHostInfo.h"

#include <memory>
#include <mutex>
#include <glog/logging.h>

class StringDictionaryClient {
 public:
  StringDictionaryClient(const LeafHostInfo& server_host, const DictRef dict_id, const bool with_timeout) { CHECK(false); };

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

  std::vector<int32_t> get_regexp_like(const std::string& pattern, const char escape, const int64_t generation) {
    CHECK(false);
    return std::vector<int32_t>{};
  };

  void get_or_add_bulk(std::vector<int32_t>& string_ids, const std::vector<std::string>& strings) { CHECK(false); };

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
