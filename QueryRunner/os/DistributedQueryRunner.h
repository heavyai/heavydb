/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
