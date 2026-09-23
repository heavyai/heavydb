/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Events/RefEventCallbacksMap.h"
#include "QueryRenderer/Events/Types.h"
#include "QueryRenderer/JSONRefObject.h"

namespace QueryRenderer {

RefEventCallbacksMap::RefEventCallbacksMap() : curr_callback_id_{0} {}

RefCallbackId RefEventCallbacksMap::subscribe(const RefEventType event_type,
                                              const RefObjShPtr& event_obj,
                                              RefEventCallback callback) {
  auto ref_type = event_obj->getRefType();
  const std::string& event_obj_name = event_obj->getNameRef();

  auto ref_type_conv = static_cast<int>(ref_type);
  auto id = ++curr_callback_id_;

  CallbacksByNameMap::iterator name_itr;
  CallbacksMap::iterator type_itr;

  if ((type_itr = callbacks_map_.find(ref_type_conv)) == callbacks_map_.end()) {
    CallbacksByNameMap name_map = {{event_obj_name, CallbacksArray()}};
    type_itr = callbacks_map_.insert(type_itr,
                                     std::make_pair(ref_type_conv, std::move(name_map)));
    name_itr = type_itr->second.begin();
  } else if ((name_itr = type_itr->second.find(event_obj_name)) ==
             type_itr->second.end()) {
    name_itr = type_itr->second.insert(name_itr,
                                       std::make_pair(event_obj_name, CallbacksArray()));
  }

  size_t idx = static_cast<size_t>(event_type);
  if (event_type == RefEventType::kAll) {
    for (size_t i = 0; i < idx; ++i) {
      CHECK(name_itr->second[i].emplace(id, callback).second);
    }
  } else {
    CHECK(name_itr->second[idx].emplace(id, callback).second);
  }

  return id;
}

void RefEventCallbacksMap::unsubscribe(const RefEventType event_type,
                                       const RefObjShPtr& event_obj,
                                       const RefCallbackId callback_id) {
  CallbacksByNameMap::iterator name_itr;
  CallbacksMap::iterator mitr;
  CallbackMap::iterator sitr;

  const std::string& event_obj_name = event_obj->getNameRef();
  auto ref_type = static_cast<int>(event_obj->getRefType());

  if ((mitr = callbacks_map_.find(ref_type)) != callbacks_map_.end() &&
      (name_itr = mitr->second.find(event_obj_name)) != mitr->second.end()) {
    size_t idx = static_cast<size_t>(event_type);

    if (event_type == RefEventType::kAll) {
      for (size_t i = 0; i < idx; ++i) {
        if ((sitr = name_itr->second[i].find(callback_id)) != name_itr->second[i].end()) {
          name_itr->second[i].erase(sitr);
        }
      }
    } else {
      if ((sitr = name_itr->second[idx].find(callback_id)) !=
          name_itr->second[idx].end()) {
        name_itr->second[idx].erase(sitr);
      }
    }

    if (!name_itr->second[0].size() && !name_itr->second[1].size() &&
        !name_itr->second[2].size()) {
      mitr->second.erase(event_obj_name);
    }
    if (!mitr->second.size()) {
      callbacks_map_.erase(ref_type);
    }
  }

  // TODO(croot): throw an error or warning?
}

void RefEventCallbacksMap::notify(const RefEventType event_type,
                                  const RefObjShPtr& event_obj) {
  CHECK(event_type != RefEventType::kAll);

  CallbacksByNameMap::iterator name_itr;
  CallbacksMap::iterator mitr;

  auto ref_type = static_cast<int>(event_obj->getRefType());
  auto& event_obj_name = event_obj->getNameRef();

  if ((mitr = callbacks_map_.find(ref_type)) != callbacks_map_.end() &&
      (name_itr = mitr->second.find(event_obj_name)) != mitr->second.end()) {
    size_t idx = static_cast<size_t>(event_type);

    std::vector<RefEventCallback> callbacks_to_call(name_itr->second[idx].size());

    int i = 0;
    for (auto& cb : name_itr->second[idx]) {
      // callbacks have the ability to subscribe and unsubscribe from the events
      // so we can't just call them here while looping through the data structure
      // that holds the callbacks as that data structure can be modified mid-stream.
      // So we'll store an additional data structure for all callbacks that need
      // calling and call them.
      callbacks_to_call[i++] = cb.second;
    }

    for (auto& cb : callbacks_to_call) {
      cb(event_type, event_obj);
    }
  }

  // TODO(croot): throw an error or warning if eventObj not found in map?
}

void RefEventCallbacksMap::clear() {
  callbacks_map_.clear();
}

}  // namespace QueryRenderer
