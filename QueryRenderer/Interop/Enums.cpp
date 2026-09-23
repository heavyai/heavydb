/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Interop/Enums.h"

namespace QueryRenderer {

std::string to_string(QueryBufferType type) {
  switch (type) {
    case QueryBufferType::kVertex:
      return "Vertex";
    case QueryBufferType::kIndex:
      return "Index";
    case QueryBufferType::kStorage:
      return "Storage";
    case QueryBufferType::kIndirectVertex:
      return "IndirectVertex";
    case QueryBufferType::kIndirectIndex:
      return "IndirectIndex";
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, const QueryBufferType& type) {
  switch (type) {
    case QueryBufferType::kVertex:
      os << "Vertex";
      break;
    case QueryBufferType::kIndex:
      os << "Index";
      break;
    case QueryBufferType::kStorage:
      os << "Storage";
      break;
    case QueryBufferType::kIndirectVertex:
      os << "IndirectVertex";
      break;
    case QueryBufferType::kIndirectIndex:
      os << "IndirectIndex";
      break;
  }
  return os;
}

}  // namespace QueryRenderer
