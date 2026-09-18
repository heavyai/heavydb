/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <string>
#include <unordered_map>

#include "QueryRenderer/Events/RefEvent.h"
#include "QueryRenderer/Events/Types.h"

namespace QueryRenderer {

//
// RefEventCallbacksMap
//
class RefEventCallbacksMap {
 public:
  RefEventCallbacksMap();

  RefCallbackId subscribe(const RefEventType event_type,
                          const RefObjShPtr& event_obj,
                          RefEventCallback callback);

  void unsubscribe(const RefEventType event_type,
                   const RefObjShPtr& event_obj,
                   const RefCallbackId callback_id);

  void notify(const RefEventType event_type, const RefObjShPtr& event_obj);

  void clear();

 private:
  using CallbackMap = std::unordered_map<RefCallbackId, RefEventCallback>;
  using CallbacksArray = std::array<CallbackMap, static_cast<size_t>(RefEventType::kAll)>;
  using CallbacksByNameMap = std::unordered_map<std::string, CallbacksArray>;

  RefCallbackId curr_callback_id_;

  using CallbacksMap = std::unordered_map<int, CallbacksByNameMap>;
  CallbacksMap callbacks_map_;
};

}  // namespace QueryRenderer
