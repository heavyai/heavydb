/*
 * Copyright 2019 OmniSci, Inc.
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

#ifndef GENERICTYPEUTILITIES_H
#define GENERICTYPEUTILITIES_H

template <typename... TYPE_ARGS, typename T>
bool is_pointer_castable_to(T* ptr) {
  bool cast_succeeded = false;
  [[gnu::unused]] bool discard[] = {
      (cast_succeeded |= (dynamic_cast<TYPE_ARGS const*>(ptr)) != nullptr)...};
  return cast_succeeded;
}

#endif
