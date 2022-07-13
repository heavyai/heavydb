/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#include "L0Exception.h"

namespace l0 {
L0Exception::L0Exception(L0result status) : status_(status) {}

const char* L0Exception::what() const noexcept {
  // avoid clang unused private member warning
  // marking status_ directly triggers a gcc attribute warning
  [[maybe_unused]] int foo = status_;
  return "L0 is not enabled";
}
}  // namespace l0