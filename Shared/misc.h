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

#ifndef SHARED_MISC_H
#define SHARED_MISC_H

#include <deque>
#include <list>
#include <set>
#include <unordered_set>
#include <vector>

namespace shared {

// source is destructively appended to the back of destination.
// source.empty() is true after call. Return number of elements appended.
template <typename T>
size_t appendMove(std::vector<T>& destination, std::vector<T>& source) {
  if (source.empty()) {
    return 0;
  } else if (destination.empty()) {
    destination = std::move(source);
    return destination.size();
  } else {
    size_t const source_size = source.size();
    destination.reserve(destination.size() + source_size);
    std::move(std::begin(source), std::end(source), std::back_inserter(destination));
    source.clear();
    return source_size;
  }
}

template <typename... Ts, typename T>
bool dynamic_castable_to_any(T const* ptr) {
  return (... || dynamic_cast<Ts const*>(ptr));
}

// Helper to print out contents of simple containers (e.g. vector, list, deque)
// including nested containers, e.g. 2d vectors, list of vectors, etc.
// Base value_type must be a std::is_scalar_v type, though you can add custom
// objects below with a new `else if constexpr` block.
// Example: VLOG(1) << "container=" << shared::printContainer(container);
template <typename CONTAINER>
struct PrintContainer {
  CONTAINER& container;
};

template <typename CONTAINER>
PrintContainer<CONTAINER> printContainer(CONTAINER& container) {
  return {container};
}

template <typename CONTAINER>
struct is_std_container : std::false_type {};
template <typename T, typename A>
struct is_std_container<std::deque<T, A> > : std::true_type {};
template <typename T, typename A>
struct is_std_container<std::list<T, A> > : std::true_type {};
template <typename T, typename A>
struct is_std_container<std::set<T, A> > : std::true_type {};
template <typename T, typename A>
struct is_std_container<std::unordered_set<T, A> > : std::true_type {};
template <typename T, typename A>
struct is_std_container<std::vector<T, A> > : std::true_type {};

template <typename OSTREAM, typename CONTAINER>
OSTREAM& operator<<(OSTREAM& os, PrintContainer<CONTAINER> pc) {
  if (pc.container.empty()) {
    return os << "()";
  } else {
    if constexpr (is_std_container<typename CONTAINER::value_type>::value) {
      os << '(';
      for (auto& container : pc.container) {
        os << printContainer(container);
      }
    } else {
      for (auto itr = pc.container.begin(); itr != pc.container.end(); ++itr) {
        if constexpr (std::is_pointer_v<typename CONTAINER::value_type>) {
          os << (itr == pc.container.begin() ? '(' : ' ') << (void const*)*itr;
        } else {
          os << (itr == pc.container.begin() ? '(' : ' ') << *itr;
        }
      }
    }
    return os << ')';
  }
}

}  // namespace shared

#endif  // SHARED_MISC_H
