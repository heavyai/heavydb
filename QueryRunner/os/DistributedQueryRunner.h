/*
 * Copyright 2019 OmniSci, Inc.
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

#include "../QueryRunner.h"

class DistributedQueryRunner : public QueryRunner {
 public:
  static std::unique_ptr<QueryRunner> init(
      const char* db_path,
      const std::string& user,
      const std::string& pass,
      const std::string& db_name,
      const std::vector<LeafHostInfo>& string_servers,
      const std::vector<LeafHostInfo>& leaf_servers,
      bool uses_gpus,
      const size_t reserved_gpu_mem,
      const bool create_user,
      const bool create_db) {
    static_assert(
        "Distributed Query Runner is only supported in distributed capable "
        "installations.");
  }
};
