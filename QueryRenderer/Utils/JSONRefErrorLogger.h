/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "GfxDriver/RenderError.h"

namespace QueryRenderer {

struct RenderSessionKey;
class QueryRendererContext;
class JSONRefObject;
class JSONLocation;

struct JSONRefErrorLogger : public gfx::RenderErrorLogger {
  const RenderSessionKey& session;
  const rapidjson::Pointer& json_path;
  const std::string err_str;

  explicit JSONRefErrorLogger(const JSONRefObject& obj, std::string&& in_err_str);
  explicit JSONRefErrorLogger(const JSONLocation& loc, std::string&& in_err_str);
  explicit JSONRefErrorLogger(const RenderSessionKey& in_session,
                              const rapidjson::Pointer& in_json_path,
                              std::string&& in_err_str);

  std::string getLogMsg() const final;
  std::string getExceptionMsg() const final;
};

}  // namespace QueryRenderer
