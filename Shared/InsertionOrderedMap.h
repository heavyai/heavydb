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

// InsertionOrderedMap: A hash map that remembers the insertion order for its values.

// Simply a std::unordered_map plus a std::vector plus some helper functions to keep the
// two data structures in sync. This is one of the recommended approaches on Stack
// Overflow.

// The class was thrown together quickly for a narrow purpose so is probably lacking some
// obvious features for general-purpose use.

// An alternative to the unordered_map+vector design might be Boost's multi_index
// container, although we only want one index (the insertion ordering) so do we even need
// "multi"?

#pragma once

#include <unordered_map>
#include <vector>

struct InsertionOrderedMap {
  std::unordered_map<llvm::Value*, llvm::Value*> m_;
  std::vector<llvm::Value*> v_;

  auto& operator[](llvm::Value* key) {
    auto m_it = m_.find(key);
    if (m_it == m_.end()) {
      v_.push_back(key);
      return m_[key];
    }
    return m_it->second;
  }

  void replace(llvm::Value* key1, llvm::Value* key2) {
    if (m_.count(key2)) {
      return;
    }
    auto m_it = m_.find(key1);
    if (m_it == m_.end()) {
      return;
    }
    auto v_it = std::find(v_.begin(), v_.end(), key1);
    if (v_it == v_.end()) {
      return;
    }
    *v_it = key2;
    m_[key2] = m_it->second;
    m_.erase(m_it);
  }

  struct Iterator {
    InsertionOrderedMap* that_;
    std::vector<llvm::Value*>::iterator v_it_;

    Iterator(InsertionOrderedMap* that, std::vector<llvm::Value*>::iterator v_it)
        : that_(that), v_it_(v_it) {}
    auto& operator++() { return ++v_it_; }
    auto operator++(int) { return v_it_++; }
    auto& operator*() {
      CHECK(that_);
      CHECK(v_it_ != that_->v_.end());
      CHECK(that_->m_.find(*v_it_) != that_->m_.end());
      return *(that_->m_.find(*v_it_));
    }
    auto operator->() {
      CHECK(that_);
      CHECK(v_it_ != that_->v_.end());
      CHECK(that_->m_.find(*v_it_) != that_->m_.end());
      return &*(that_->m_.find(*v_it_));
    }
    bool operator==(const Iterator& peer) { return (v_it_ == peer.v_it_); }
    bool operator!=(const Iterator& peer) { return (v_it_ != peer.v_it_); }
  };

  auto begin() { return Iterator{this, v_.begin()}; }
  auto end() { return Iterator{this, v_.end()}; }

  auto find(llvm::Value* key) {
    if (m_.count(key)) {
      return Iterator{this, std::find(v_.begin(), v_.end(), key)};
    } else {
      return Iterator{this, v_.end()};
    }
  }

  std::pair<Iterator, bool> emplace(llvm::Value* key, llvm::Value* val) {
    auto it = find(key);
    if (it != end()) {
      return std::pair(it, false);
    } else {
      m_.emplace(key, val);
      v_.push_back(key);
      return std::pair(Iterator{this, v_.end() - 1}, true);
    }
  }
};  // struct InsertionOrderedMap
