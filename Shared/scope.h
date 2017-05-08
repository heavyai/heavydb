/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef SHARED_SCOPE_H
#define SHARED_SCOPE_H

#include <functional>

class ScopeGuard {
 public:
  template <class Callable>
  ScopeGuard(Callable&& at_exit) : at_exit_(std::forward<Callable>(at_exit)) {}

  // make it non-copyable
  ScopeGuard(const ScopeGuard&) = delete;
  void operator=(const ScopeGuard&) = delete;

  ScopeGuard(ScopeGuard&& other) : at_exit_(std::move(other.at_exit_)) { other.at_exit_ = nullptr; }

  ~ScopeGuard() {
    if (at_exit_) {
      // note that at_exit_ must not throw
      at_exit_();
    }
  }

 private:
  std::function<void()> at_exit_;
};

#endif  // SHARED_SCOPE_H
