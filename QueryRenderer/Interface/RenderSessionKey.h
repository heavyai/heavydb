/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace Catalog_Namespace {
class SessionInfo;
}

namespace QueryRenderer {

using SessionId = std::string;
using WidgetId = int64_t;

struct RenderSessionKey {
  const std::shared_ptr<Catalog_Namespace::SessionInfo const> session_info;
  const WidgetId widget_id;

  explicit RenderSessionKey(
      const std::shared_ptr<Catalog_Namespace::SessionInfo const> in_session_info,
      const WidgetId in_widget_id)
      : session_info(in_session_info), widget_id(in_widget_id) {}

  const Catalog_Namespace::SessionInfo& getSessionInfo() const;
  std::shared_ptr<Catalog_Namespace::SessionInfo const> getSessionInfoPtr() const;
  SessionId getSessionId() const;
  WidgetId getWidgetId() const;
  bool operator==(const RenderSessionKey& other) const;
  bool operator!=(const RenderSessionKey& other) const;
  operator std::string() const;
};

size_t hash_value(const RenderSessionKey& render_session_key);

std::ostream& operator<<(std::ostream& os, const RenderSessionKey& render_session_key);

}  // namespace QueryRenderer
