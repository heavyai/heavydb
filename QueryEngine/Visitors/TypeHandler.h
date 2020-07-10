/*
 * Copyright 2020 OmniSci, Inc.
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

/*
 * @description Sortable utility struct for visitor patterns.
 */

#pragma once

#include <typeindex>

template <typename T, typename U>
struct TypeHandler {
  std::type_index type_index;
  void (T::*handler)(U const*);
};

template <typename T, typename U>
bool operator<(TypeHandler<T, U> const& lhs, TypeHandler<T, U> const& rhs) {
  return lhs.type_index < rhs.type_index;
}

template <typename T, typename U>
bool operator<(TypeHandler<T, U> const& lhs, std::type_index const& rhs) {
  return lhs.type_index < rhs;
}

template <typename T, typename U>
bool operator<(std::type_index const& lhs, TypeHandler<T, U> const& rhs) {
  return lhs < rhs.type_index;
}
