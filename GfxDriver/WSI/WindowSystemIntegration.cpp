/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/WSI/WindowSystemIntegration.h"

#include <limits>
#include <map>
#include <vector>

#include "Logger/Logger.h"
#include "Shared/CallbackManager.h"

namespace gfx {

class WindowSystemIntegration::Impl {
 public:
  WSIEventHandlerID registerEventHandler(WSIEventHandler handler);
  void unregisterEventHandler(WSIEventHandlerID id);
  void notifyEvent(const WSIEvent& context);
  bool hasEventHandlers() const;

 private:
  CallbackManager<WSIEventHandlerID, WSIEventHandler, WSIEvent> event_handlers_;
};

WSIEventHandlerID WindowSystemIntegration::Impl::registerEventHandler(
    WSIEventHandler handler) {
  return event_handlers_.registerCallback(handler);
}

void WindowSystemIntegration::Impl::unregisterEventHandler(WSIEventHandlerID id) {
  event_handlers_.unregisterCallback(id);
}

void WindowSystemIntegration::Impl::notifyEvent(const WSIEvent& context) {
  event_handlers_.notify(context);
}

bool WindowSystemIntegration::Impl::hasEventHandlers() const {
  return !event_handlers_.isEmpty();
}

WindowSystemIntegration::WindowSystemIntegration()
    : impl_{std::make_unique<WindowSystemIntegration::Impl>()} {}

WindowSystemIntegration::~WindowSystemIntegration() {}

WSIEventHandlerID WindowSystemIntegration::registerEventHandler(WSIEventHandler handler) {
  return impl_->registerEventHandler(handler);
}

void WindowSystemIntegration::unregisterEventHandler(WSIEventHandlerID id) {
  impl_->unregisterEventHandler(id);
}

void WindowSystemIntegration::notifyEvent(const WSIEvent& event) {
  impl_->notifyEvent(event);
}

bool WindowSystemIntegration::hasEventHandlers() const {
  return impl_->hasEventHandlers();
}

}  // namespace gfx
