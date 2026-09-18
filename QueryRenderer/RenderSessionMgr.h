/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <functional>
#include <memory>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index_container.hpp>

#include "QueryRenderer/Interface/RenderSessionKey.h"
#include "QueryRenderer/RenderSession.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {
class RenderSessionMgr {
 public:
  RenderSessionMgr() = delete;
  explicit RenderSessionMgr(size_t render_cache_limit);

  const RenderSession& addRenderSession(
      std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
      const WidgetId widget_id,
      std::string&& vega_json);

  bool hasRenderSession(const RenderSessionKey& render_session_key) const;

  const RenderSession& getRenderSession(const RenderSessionKey& render_session_key) const;

  void iterateRenderSessions(std::function<void(const RenderSession&)> callback);

  void updateLastRenderTime(const RenderSessionKey& render_session_key) const;
  void updateQueryRendererContext(const RenderSessionKey& render_session_key,
                                  QueryRendererContextUqPtr&& context) const;
  void updateVegaJSONAndLastRenderTime(const RenderSessionKey& render_session_key,
                                       std::string&& vega_json) const;

  void removeSessionId(const SessionId& session_id);
  void clear();
  void purgeUnusedRenderSessions() const;

 private:
  struct SessionIdTag {};
  struct RenderSessionTag {};
  struct LastRenderTimeTag {};

  using RendererMap = ::boost::multi_index_container<
      RenderSession,
      ::boost::multi_index::indexed_by<
          ::boost::multi_index::hashed_unique<
              ::boost::multi_index::tag<RenderSessionTag>,
              ::boost::multi_index::const_mem_fun<RenderSession,
                                                  const RenderSessionKey&,
                                                  &RenderSession::getKey>>,

          ::boost::multi_index::ordered_non_unique<
              ::boost::multi_index::tag<SessionIdTag>,
              ::boost::multi_index::
                  const_mem_fun<RenderSession, SessionId, &RenderSession::getSessionId>>,

          ::boost::multi_index::sequenced<::boost::multi_index::tag<LastRenderTimeTag>>>>;

  using RendererMap_by_UserId = RendererMap::index<SessionIdTag>::type;
  using RendererMap_by_LastRenderTime = RendererMap::index<LastRenderTimeTag>::type;

  mutable RendererMap renderer_map_;
  const size_t render_cache_limit_;
  static const std::chrono::milliseconds max_widget_idle_time_;

  using ModifyCallback = std::function<void(RenderSession&)>;
  auto modifySession(const RenderSessionKey& key, const ModifyCallback modify_cb) const;
};

}  // namespace QueryRenderer
