/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace QueryRunner {

inline int test_port_from_environment(const char* environment_variable,
                                      const int default_port) {
  const auto* value = std::getenv(environment_variable);
  if (!value || !*value) {
    return default_port;
  }

  char* end = nullptr;
  errno = 0;
  const auto port = std::strtol(value, &end, 10);
  if (errno || end == value || *end != '\0' || port < 1 || port > 65535) {
    throw std::runtime_error("Invalid TCP port in " + std::string(environment_variable) +
                             ": " + value);
  }
  return static_cast<int>(port);
}

inline int calcite_port() {
  return test_port_from_environment("HEAVYDB_TEST_CALCITE_PORT", 3279);
}

inline int db_handler_calcite_port() {
  return test_port_from_environment("HEAVYDB_TEST_DB_HANDLER_CALCITE_PORT", 3280);
}

}  // namespace QueryRunner
