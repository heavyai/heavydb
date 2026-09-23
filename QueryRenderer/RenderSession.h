/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <memory>

#include "QueryRenderer/QueryRendererContext.h"

namespace QueryRenderer {

class RenderSession {
 public:
  RenderSession(std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
                const WidgetId widget_id,
                std::string&& vega_json);

  const RenderSessionKey& getKey() const;
  SessionId getSessionId() const;
  WidgetId getWidgetId() const;
  QueryRendererContext& getRenderContext() const;
  std::chrono::milliseconds getLastRenderTime() const;
  const std::string& getVegaJSON() const;

 private:
  const RenderSessionKey key_;
  QueryRendererContextUqPtr render_context_;
  std::chrono::milliseconds last_render_time_;
  std::string vega_json_;

  friend class RenderSessionMgr;
};

}  // namespace QueryRenderer
