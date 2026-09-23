/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "misc.h"
#include "sqltypes.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <iomanip>

namespace shared {

size_t compute_hash(int32_t item_1, int32_t item_2) {
  static_assert(sizeof(item_1) + sizeof(item_2) <= sizeof(size_t));
  return (static_cast<size_t>(item_1) << (8 * sizeof(item_2))) |
         (static_cast<size_t>(item_2));
}

// Escape and quote contents of filename as a json string and output to os.
// Q: Why not just return the file contents as a string?
// A: Constructing a string may unnecessarily contribute to memory fragmentation,
//    and is probably less performant due to the extra heap allocations.
void FileContentsEscaper::quoteAndPrint(std::ostream& os) const {
  std::ifstream file(filename);
  if (!file.is_open()) {
    os << "\"Unable to open " << filename << '"';
    return;
  }
  char ch;
  std::ios orig_os_state(nullptr);
  orig_os_state.copyfmt(os);
  os << '"';
  while (file.get(ch)) {
    if (ch == '"') {
      os << "\\\"";
    } else if (ch == '\\') {
      os << "\\\\";
    } else if (std::isprint(ch) || ch == ' ') {
      os << ch;
    } else {
      switch (ch) {
        // clang-format off
        case '\b': os << "\\b"; break;
        case '\f': os << "\\f"; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        // clang-format on
        default:
          os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<unsigned>(static_cast<unsigned char>(ch));
          break;
      }
    }
  }
  os << '"';
  os.copyfmt(orig_os_state);
}

std::ostream& operator<<(std::ostream& os, FileContentsEscaper const& fce) {
  fce.quoteAndPrint(os);
  return os;
}

}  // namespace shared
