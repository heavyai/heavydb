/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <string>

class ForeignServer;

namespace foreign_storage {
struct UserMappingType {
  static constexpr char const* USER = "USER";
  static constexpr char const* PUBLIC = "PUBLIC";
};

struct UserMapping {
  int32_t id;
  int32_t user_id;
  int32_t foreign_server_id;
  std::string type;
  std::string options;

  UserMapping() {}

  UserMapping(const int32_t id,
              const int32_t user_id,
              const int32_t foreign_server_id,
              const std::string type,
              const std::string options)
      : id(id)
      , user_id(user_id)
      , foreign_server_id(foreign_server_id)
      , type(type)
      , options(options) {}

  std::map<std::string, std::string, std::less<>> getUnencryptedOptions() const {
    return {};
  }

  void validate(const ForeignServer* foreign_server) {}
};
}  // namespace foreign_storage
