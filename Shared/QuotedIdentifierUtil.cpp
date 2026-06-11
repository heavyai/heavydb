/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QuotedIdentifierUtil.h"

#include <iomanip>

#include "Logger/Logger.h"

namespace {

std::vector<std::string> tokenize(const std::string& str, char quote, char delimiter) {
  std::vector<std::string> tokens;

  bool within_quote = false;
  size_t last_start_index = 0;
  for (size_t i = 0; i < str.size(); ++i) {
    char c = str.at(i);
    if (c == quote) {
      within_quote = within_quote ? false : true;
    }
    if (!within_quote && c == delimiter) {
      tokens.push_back(str.substr(last_start_index, i - last_start_index));
      last_start_index = i + 1;
    }
  }

  tokens.push_back(str.substr(last_start_index, str.size() - last_start_index));
  return tokens;
}

}  // namespace

namespace shared {
std::string concatenate_identifiers(const std::vector<std::string>& identifiers,
                                    const char delimiter) {
  std::string output;
  for (const auto& identifier : identifiers) {
    if (!output.empty()) {
      output += delimiter;
    }
    output += identifier;
  }
  return output;
}

std::vector<std::string> split_identifiers(const std::string& composite_identifier,
                                           const char delimiter,
                                           const char quote) {
  return tokenize(composite_identifier, quote, delimiter);
}

}  // namespace shared
