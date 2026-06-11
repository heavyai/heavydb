/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file    RequestInfo.h
 * @description A struct for parsing TSessionInfo that contains a request_id.
 */

#pragma once

#include "Logger/Logger.h"

namespace heavyai {

class RequestInfo {
  std::string session_id_;
  logger::RequestId request_id_;

 public:
  // If session_id_or_json starts with a '{' then it is assumed a json object.
  // Otherwise interpret as a session_id and set request_id_=0;
  RequestInfo(std::string const& session_id_or_json);
  RequestInfo(std::string session_id, logger::RequestId const request_id)
      : session_id_(std::move(session_id)), request_id_(request_id) {}
  std::string json() const;
  logger::RequestId requestId() const { return request_id_; }
  std::string const& sessionId() const { return session_id_; }
  // Set request_id_ to current g_thread_local_ids.request_id_.
  void setRequestId(logger::RequestId const request_id) { request_id_ = request_id; }
};

}  // namespace heavyai
