/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <string>
#include <vector>

#include "QueryRenderer/Events/RefEvent.h"

namespace QueryRenderer {

template <typename ObservedType>
using NotifyRefEvents =
    std::array<std::vector<ObservedType>, static_cast<size_t>(RefEventType::kAll)>;

template <typename ObservedType>
using NotifyRefEventCB = std::function<void(RefEventType, const ObservedType&)>;

template <typename ObservedType>
class NotifyRefEventQueue {
 public:
  explicit NotifyRefEventQueue(NotifyRefEventCB<ObservedType> notify_event_cb)
      : notify_callback_{notify_event_cb} {}

  NotifyRefEvents<ObservedType> events;
  void push_back(const ObservedType& target, RefEventType event_type) {
    events[static_cast<size_t>(event_type)].push_back(target);
  }

  void notify() {
    for (size_t i = 0; i < events.size(); ++i) {
      for (auto& target : events[i]) {
        notify_callback_(static_cast<RefEventType>(i), target);
      }
    }
  }

  bool contains(const std::string& name) {
    for (const auto& event_type : events) {
      if (std::find_if(event_type.begin(), event_type.end(), [&name](const auto& t) {
            return t->getName() == name;
          }) != event_type.end()) {
        return true;
        break;
      }
    }
    return false;
  }

 private:
  NotifyRefEventCB<ObservedType> notify_callback_;
};

}  // namespace QueryRenderer
