/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <unordered_map>

#include "QueryRenderer/Events/Types.h"

namespace QueryRenderer {

template <typename T>
class EventedProperty {
  using TParam = typename std::conditional_t<std::is_fundamental_v<T>, T, T&>;
  using CallbackFunc = std::function<void(const TParam, const TParam)>;

 public:
  EventedProperty() : curr_value_() {}
  EventedProperty(const TParam prop_default) : curr_value_{prop_default} {}

  EventedProperty& operator=(const TParam new_value) {
    if (new_value != curr_value_) {
      auto old_value = curr_value_;
      curr_value_ = new_value;
      for (auto& itr : callbacks_) {
        itr.second(curr_value_, old_value);
      }
    }
    return *this;
  }

  const T& getDataRef() const { return curr_value_; }

  operator T() const { return curr_value_; }

  void addCallback(const PropUpdateCallbackId callback_id, CallbackFunc callback_func) {
    CHECK(callbacks_.find(callback_id) == callbacks_.end());
    callbacks_.insert({callback_id, callback_func});
  }

  bool removeCallback(const PropUpdateCallbackId callback_id) {
    return callbacks_.erase(callback_id) > 0;
  }

 private:
  T curr_value_;
  std::unordered_map<PropUpdateCallbackId, CallbackFunc> callbacks_;
};

}  // namespace QueryRenderer
