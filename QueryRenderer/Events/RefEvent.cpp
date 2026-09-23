/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Events/RefEvent.h"

namespace QueryRenderer {

std::string to_string(const RefEventType ref_event_type) {
  switch (ref_event_type) {
    case RefEventType::kUpdate:
      return "UPDATE";
    case RefEventType::kRemove:
      return "REMOVE";
    case RefEventType::kReplace:
      return "REPLACE";
    case RefEventType::kAll:
      return "ALL";
  }
  UNREACHABLE();
  return "";
}

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os,
                         const QueryRenderer::RefEventType ref_event_type) {
  os << QueryRenderer::to_string(ref_event_type);
  return os;
}
