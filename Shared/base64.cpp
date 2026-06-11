/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/base64.h"

namespace shared {

std::string decode_base64_uri(const std::string& data, bool trim_nulls) {
  // Allocate a string large enough to hold exta '=' as padding
  std::string uri_dec;
  size_t data_len = data.length();
  // base64_uri encoding removes '=' padding at the end of the string.
  size_t padding = 4 - (data_len % 4);
  if (padding == 4) {
    padding = 0;
  }
  uri_dec.resize(data_len + padding);

  // base64_uri encoding replaces all '+' and '/' with '-' and '_' respectively.
  std::transform(
      data.begin(), data.end(), uri_dec.begin(), [](unsigned char c) -> unsigned char {
        switch (c) {
          case '-':
            return '+';
          case '_':
            return '/';
          default:
            return c;
        }
      });
  if (padding != 0) {
    uri_dec.replace(uri_dec.begin() + data_len, uri_dec.end(), padding, '=');
  }
  // in the case of a signature from a JWT trim_nulls should be false
  return decode_base64(uri_dec, trim_nulls);
}

}  // namespace shared
