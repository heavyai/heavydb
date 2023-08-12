/*
 * Copyright 2023 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <limits>
#include <map>
#include <type_traits>
#include <vector>

/**
 * Template class for managing notification callbacks
 *
 * IDType must be an intergral type
 * CallbackType must be a callable type: lambda, functor, etc
 * ContextType can be any basic type, struct, or class
 *
 * Notification broadcast is in registration order
 * IDs DO NOT wrap around, and a free-list is not used, so IDType must
 * not be exhaustible
 *
 * Unregistering a callback during notification handling will defer
 * unregistration until notification completes
 *
 * Example:
 *
 * struct MyContextType {
 *   int value;
 *   ...
 * };
 * using MyCallbackType = std::function<void(const MyContextType&)>;
 * using MyCallbackID = uint32_t;
 *
 * class SomeClass {
 *   CallbackManager<MyCallbackID, MyCallbackType, MyContextType> callbacks;
 *
 *   SomeClass() {
 *     callbacks.registerCallback([this](const MyContextType& context) {
 *         handleCallback(context);
 *       }
 *     );
 *   }
 *
 *   void handleCallback(const MyContextType& context) {
 *     doStuff(context.value);
 *   }
 * };
 * */
template <typename IDType, typename CallbackType, typename ContextType>
class CallbackManager {
  static_assert(std::is_integral_v<IDType>, "IDType must be and integral type");
  static_assert(
      std::is_invocable_v<CallbackType, const ContextType&>,
      "CallbackType must be callable with \'const ContextType&\' as the argument");

 public:
  IDType registerCallback(CallbackType callback) {
    CHECK_LT(next_id_, std::numeric_limits<IDType>::max());
    callbacks_.emplace(next_id_++, callback);
    return next_id_ - 1;
  }

  void unregisterCallback(IDType id) {
    CHECK(callbacks_.count(id)) << "Callback id not found";
    if (is_notifying_) {
      // Defer unregistration until notification loop is complete
      deferred_unregister_ids_.push_back(id);
    } else {
      callbacks_.erase(id);
    }
  }

  void notify(const ContextType& context) {
    is_notifying_ = true;
    for (auto callback_pair : callbacks_) {
      callback_pair.second(context);
    }
    is_notifying_ = false;

    // Unregister any callbacks that tried to unregister during notification
    if (!deferred_unregister_ids_.empty()) {
      for (auto id : deferred_unregister_ids_) {
        unregisterCallback(id);
      }
      deferred_unregister_ids_.clear();
    }
  }

  bool isEmpty() const { return callbacks_.empty(); }

 private:
  // Use std::map to ensure callback iteration is in registration order
  std::map<IDType, CallbackType> callbacks_;
  IDType next_id_;
  bool is_notifying_;
  std::vector<IDType> deferred_unregister_ids_;
};
