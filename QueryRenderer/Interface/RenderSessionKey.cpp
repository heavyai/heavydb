/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "RenderSessionKey.h"

#include <Catalog/Catalog.h>
#include <boost/functional/hash.hpp>

namespace QueryRenderer {

const Catalog_Namespace::SessionInfo& RenderSessionKey::getSessionInfo() const {
  CHECK(session_info);
  return *session_info;
}

std::shared_ptr<Catalog_Namespace::SessionInfo const>
RenderSessionKey::getSessionInfoPtr() const {
  return session_info;
}

SessionId RenderSessionKey::getSessionId() const {
  return (session_info ? session_info->get_session_id() : "");
}
WidgetId RenderSessionKey::getWidgetId() const {
  return widget_id;
}

bool RenderSessionKey::operator==(const RenderSessionKey& other) const {
  return getSessionId() == other.getSessionId() && widget_id == other.widget_id;
}

bool RenderSessionKey::operator!=(const RenderSessionKey& other) const {
  return !operator==(other);
}

RenderSessionKey::operator std::string() const {
  CHECK(session_info);
  return "[" + std::string(*session_info) + ", " + std::to_string(widget_id) + "]";
}

size_t hash_value(const RenderSessionKey& render_session_key) {
  size_t seed{0};
  ::boost::hash_combine(seed, render_session_key.session_info->get_session_id());
  ::boost::hash_combine(seed, render_session_key.widget_id);
  return seed;
}

std::ostream& operator<<(std::ostream& os, const RenderSessionKey& render_session_key) {
  os << std::string(render_session_key);
  return os;
}
}  // namespace QueryRenderer
