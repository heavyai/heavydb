/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Utils/JSONRefErrorLogger.h"
#include "QueryRenderer/Interface/RenderSessionKey.h"
#include "QueryRenderer/JSONRefObject.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

namespace {
constexpr bool enable_logging = true;
constexpr const char json_err_str_prefix[] = "JSON parse error";
}  // namespace

JSONRefErrorLogger::JSONRefErrorLogger(const JSONRefObject& obj, std::string&& in_err_str)
    : gfx::RenderErrorLogger(enable_logging)
    , session(obj.getQueryRendererContext().getRenderSessionKey())
    , json_path(obj.getJsonPathRef())
    , err_str(std::move(in_err_str)) {}

JSONRefErrorLogger::JSONRefErrorLogger(const JSONLocation& loc, std::string&& in_err_str)
    : gfx::RenderErrorLogger(enable_logging)
    , session(loc.getRenderSessionRef())
    , json_path(loc.getPathRef())
    , err_str(std::move(in_err_str)) {}

JSONRefErrorLogger::JSONRefErrorLogger(const RenderSessionKey& in_session,
                                       const rapidjson::Pointer& in_json_path,
                                       std::string&& in_err_str)
    : gfx::RenderErrorLogger(enable_logging)
    , session(in_session)
    , json_path(in_json_path)
    , err_str(std::move(in_err_str)) {}

std::string JSONRefErrorLogger::getLogMsg() const {
  auto exc_msg = getExceptionMsg();
  return std::string(session) + ", " + exc_msg;
};

std::string JSONRefErrorLogger::getExceptionMsg() const {
  return std::string(json_err_str_prefix) + " (" +
         RapidJSONUtils::getPointerPath(json_path) + "): " + err_str;
}

}  // namespace QueryRenderer
