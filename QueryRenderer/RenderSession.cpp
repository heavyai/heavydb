/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/RenderSession.h"

#include <memory>

#include "QueryRenderer/Interface/RenderSessionKey.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/TimeUtils.h"

namespace QueryRenderer {

RenderSession::RenderSession(std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
                             const WidgetId widget_id,
                             std::string&& vega_json)
    : key_{RenderSessionKey(session_info, widget_id)}
    , render_context_{nullptr}
    , last_render_time_{getCurrentTimeMS()}
    , vega_json_{std::move(vega_json)} {}

const RenderSessionKey& RenderSession::getKey() const {
  return key_;
}

SessionId RenderSession::getSessionId() const {
  return key_.getSessionId();
}

WidgetId RenderSession::getWidgetId() const {
  return key_.getWidgetId();
}

QueryRendererContext& RenderSession::getRenderContext() const {
  CHECK(render_context_);
  return *render_context_;
}

std::chrono::milliseconds RenderSession::getLastRenderTime() const {
  return last_render_time_;
}

const std::string& RenderSession::getVegaJSON() const {
  return vega_json_;
}

}  // namespace QueryRenderer
