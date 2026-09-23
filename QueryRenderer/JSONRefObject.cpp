/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/JSONRefObject.h"

namespace QueryRenderer {

rapidjson::Value JSONRefObject::toJSON(
    rapidjson::Document::AllocatorType* allocator) const {
  std::unique_ptr<rapidjson::Document> d;
  if (!allocator) {
    d = std::make_unique<rapidjson::Document>();
    allocator = &d->GetAllocator();
  }
  rapidjson::Value obj(rapidjson::kObjectType);
  rapidjson::Value namev(name_.c_str(), name_.length(), *allocator);
  obj.AddMember("name", namev, *allocator);
  toJSONInternal(obj, *allocator);
  return obj;
}

std::string to_string(const RefType ref_type) {
  switch (ref_type) {
    case RefType::kData:
      return "DATA";
    case RefType::kScale:
      return "SCALE";
    case RefType::kProjection:
      return "PROJECTION";
  }
  UNREACHABLE();
  return "";
}

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const QueryRenderer::RefType ref_type) {
  os << QueryRenderer::to_string(ref_type);
  return os;
}
