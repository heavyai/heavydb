/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/RenderSessionMgr.h"

#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "Logger/Logger.h"
#include "QueryRenderer/Utils/TimeUtils.h"

namespace QueryRenderer {

const std::chrono::milliseconds RenderSessionMgr::max_widget_idle_time_ =
    std::chrono::milliseconds(300000);  // 5 minutes, in ms

RenderSessionMgr::RenderSessionMgr(size_t render_cache_limit)
    : renderer_map_{}, render_cache_limit_{render_cache_limit} {
  RUNTIME_EX_ASSERT(render_cache_limit_ > 0,
                    "max concurrent render session limit must be > 0")
}

const RenderSession& RenderSessionMgr::getRenderSession(
    const RenderSessionKey& render_session_key) const {
  auto itr = renderer_map_.find(render_session_key);
  CHECK(itr != renderer_map_.end());
  auto const* rsi = &(*itr);
  CHECK(rsi->render_context_);
  return *rsi;
}

bool RenderSessionMgr::hasRenderSession(
    const RenderSessionKey& render_session_key) const {
  return (renderer_map_.find(render_session_key) != renderer_map_.end());
}

const RenderSession& RenderSessionMgr::addRenderSession(
    std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
    const WidgetId widget_id,
    std::string&& vega_json) {
  // Check if the current num of connections is maxed out, NOTE: can only add 1 at a
  // time here
  if (renderer_map_.size() == render_cache_limit_) {
    auto& last_render_time_list = renderer_map_.get<LastRenderTimeTag>();
    auto itr = last_render_time_list.begin();
    LOG(INFO) << "QueryRenderManager render session limit reached. Removing longest-idle "
                 "session "
              << itr->key_;
    last_render_time_list.erase(itr);
  }

  auto [itr, emplaced] =
      renderer_map_.emplace(session_info, widget_id, std::move(vega_json));
  CHECK(emplaced) << "RenderSession already exists";

  return *itr;
}

auto RenderSessionMgr::modifySession(const RenderSessionKey& render_session_key,
                                     const ModifyCallback modify_cb) const {
  auto itr = renderer_map_.find(render_session_key);
  CHECK(itr != renderer_map_.end());
  renderer_map_.modify(itr, modify_cb);
  return itr;
}

void RenderSessionMgr::updateLastRenderTime(
    const RenderSessionKey& render_session_key) const {
  auto update_last_render_time = [&](RenderSession& render_session) {
    render_session.last_render_time_ = getCurrentTimeMS();
  };
  auto itr = modifySession(render_session_key, update_last_render_time);

  auto& last_render_time_list = renderer_map_.get<LastRenderTimeTag>();
  auto last_render_time_itr = renderer_map_.project<LastRenderTimeTag>(itr);
  last_render_time_list.relocate(last_render_time_list.end(), last_render_time_itr);
}

void RenderSessionMgr::updateQueryRendererContext(
    const RenderSessionKey& render_session_key,
    QueryRendererContextUqPtr&& context) const {
  auto set_renderer_context = [&](RenderSession& render_session) {
    render_session.render_context_ = std::move(context);
  };
  modifySession(render_session_key, set_renderer_context);
}

void RenderSessionMgr::updateVegaJSONAndLastRenderTime(
    const RenderSessionKey& render_session_key,
    std::string&& vega_json) const {
  auto update_vega_json = [&](RenderSession& render_session) {
    render_session.vega_json_ = std::move(vega_json);
    render_session.last_render_time_ = getCurrentTimeMS();
  };
  auto itr = modifySession(render_session_key, update_vega_json);

  // duplicated from above because factoring it out into a private method
  // would require knowing the actual type of "itr" and that would double
  // the size of the header file!
  auto& last_render_time_list = renderer_map_.get<LastRenderTimeTag>();
  auto last_render_time_itr = renderer_map_.project<LastRenderTimeTag>(itr);
  last_render_time_list.relocate(last_render_time_list.end(), last_render_time_itr);
}

void RenderSessionMgr::removeSessionId(const SessionId& session_id) {
  auto& session_id_map = renderer_map_.get<SessionIdTag>();

  auto range_itr = session_id_map.equal_range(session_id);
  if (range_itr.first == session_id_map.end()) {
    return;
  }

  session_id_map.erase(range_itr.first, range_itr.second);
}

void RenderSessionMgr::clear() {
  renderer_map_.clear();
}

void RenderSessionMgr::iterateRenderSessions(
    std::function<void(const RenderSession&)> callback) {
  for (auto& session : renderer_map_) {
    callback(session);
  }
}

void RenderSessionMgr::purgeUnusedRenderSessions() const {
  RENDER_LOG_SCOPE();
  std::chrono::milliseconds cutoff_time = getCurrentTimeMS() - max_widget_idle_time_;

  auto& last_render_time_list = renderer_map_.get<LastRenderTimeTag>();

  int cnt = 0;
  auto itr = last_render_time_list.begin();
  while (itr != last_render_time_list.end() && itr->last_render_time_ < cutoff_time) {
    cnt++;
    itr++;
  }

  LOG_IF(INFO, cnt > 0) << "QueryRenderManager - purging " << cnt
                        << " idle render sessions.";
  RENDER_LOG() << "purging " << cnt << " idle render sessions";

  last_render_time_list.erase(last_render_time_list.begin(), itr);
}

}  // namespace QueryRenderer
