/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <rapidjson/pointer.h>
#include <rapidjson/writer.h>

#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/JSONRefErrorLogger.h"

namespace QueryRenderer {

enum class RefType { kData = 0, kProjection, kScale };
class JSONRefObject {
 public:
  virtual ~JSONRefObject() {}

  RefType getRefType() const { return ref_type_; }
  std::string getName() const { return name_; }
  const std::string& getNameRef() const { return name_; }
  const rapidjson::Pointer& getJsonPathRef() const { return json_path_; }
  const QueryRendererContext& getQueryRendererContext() const { return ctx_; }

  rapidjson::Value toJSON(rapidjson::Document::AllocatorType* allocator = nullptr) const;

 protected:
  JSONRefObject(QueryRendererContext& ctx,
                const RefType ref_type,
                const std::string& name,
                const rapidjson::Pointer& json_path)
      : ctx_{ctx}, name_{name}, json_path_{json_path}, ref_type_{ref_type} {}

  inline JSONRefErrorLogger createJSONRefError(std::string&& err_str) const {
    return JSONRefErrorLogger(*this, std::move(err_str));
  }

  QueryRendererContext& ctx_;
  std::string name_;
  rapidjson::Pointer json_path_;

 private:
  RefType ref_type_;

  virtual void toJSONInternal(rapidjson::Value& obj,
                              rapidjson::Document::AllocatorType& allocator) const = 0;
};
using RefObjWkPtr = std::shared_ptr<JSONRefObject>;
using RefObjShPtr = std::shared_ptr<JSONRefObject>;

std::string to_string(const RefType ref_type);

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const ::QueryRenderer::RefType ref_type);
