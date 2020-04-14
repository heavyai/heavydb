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

#pragma once

#include "ParserNode.h"

namespace Parser {

/**
 * Class used to ensure ownership of string or Node objects that are dynamically
 * allocated within the parser.
 *
 * @tparam T - std::string or Node type
 */
template <typename T>
class TrackedPtr {
  static_assert(std::is_same<std::string, T>::value || std::is_same<Node, T>::value);

 public:
  TrackedPtr(const TrackedPtr&) = delete;

  TrackedPtr& operator=(const TrackedPtr&) = delete;

  /**
   * Releases ownership of contained string or Node value and returns a pointer to
   * contained value.
   */
  T* release() {
    if (is_empty_) {
      return nullptr;
    }
    CHECK(!is_released_);
    is_released_ = true;
    return value_.release();
  }

  /**
   * Returns a pointer to contained string or Node value.
   */
  T* get() {
    if (is_empty_) {
      return nullptr;
    }
    return value_.get();
  }

  /**
   * Creates an instance of a TrackedPtr and returns a pointer to created instance.
   *
   * @param tracked_ptrs - TrackedPtr unique pointer vector, which takes ownership of
   * dynamically allocated objects.
   * @param args - Arguments to be used when creating contained string or Node values.
   * @return - pointer to created TrackedPtr
   */
  template <typename... Args>
  static TrackedPtr<T>* make(std::vector<std::unique_ptr<TrackedPtr<T>>>& tracked_ptrs,
                             Args&&... args) {
    std::unique_ptr<TrackedPtr<T>> tracked_ptr{
        new TrackedPtr<T>(std::forward<Args>(args)...)};
    const auto& ptr = tracked_ptrs.emplace_back(std::move(tracked_ptr));
    return ptr.get();
  }

  /**
   * Returns a pointer to a TrackedPtr instance that represents an empty TrackedPtr.
   */
  static TrackedPtr<T>* makeEmpty() {
    static std::unique_ptr<TrackedPtr<T>> empty_tracked_ptr_ =
        std::unique_ptr<TrackedPtr<T>>(new TrackedPtr<T>());
    return empty_tracked_ptr_.get();
  }

 private:
  std::unique_ptr<T> value_;
  bool is_empty_;
  bool is_released_;

  TrackedPtr() : is_empty_(true), is_released_(false){};

  TrackedPtr(Node* node)
      : value_(std::unique_ptr<Node>(node)), is_empty_(false), is_released_(false) {}

  TrackedPtr(const std::string& str)
      : value_(std::make_unique<std::string>(str))
      , is_empty_(false)
      , is_released_(false) {}

  TrackedPtr(const char* str, size_t len)
      : value_(std::make_unique<std::string>(str, len))
      , is_empty_(false)
      , is_released_(false) {}
};

/**
 * Class used to ensure ownership of dynamically allocated lists to string or Node objects
 * that are dynamically allocated within the parser.
 *
 * @tparam T - std::string or Node type
 */
template <typename T>
class TrackedListPtr {
  static_assert(std::is_same<std::string, T>::value || std::is_same<Node, T>::value);

 public:
  TrackedListPtr(const TrackedListPtr&) = delete;

  TrackedListPtr& operator=(const TrackedListPtr&) = delete;

  /**
   * Releases ownership of contained string or Node pointer list and returns a pointer to
   * the list.
   */
  std::list<T*>* release() {
    if (is_empty_) {
      return nullptr;
    }
    CHECK(!is_released_);
    is_released_ = true;

    auto result = new std::list<T*>();
    for (auto& ptr : *value_) {
      result->emplace_back(ptr->release());
    }
    return result;
  }

  /**
   * Proxy method for adding a TrackedPtr object to the end of contained string or Node
   * pointer list.
   *
   * @param item - TrackedPtr object to be added to list
   */
  void push_back(TrackedPtr<T>* item) { value_->emplace_back(item); }

  /**
   * Proxy method for adding a string or Node pointer to the end of contained string or
   * Node pointer list. This method accepts a string or Node pointer, which is wrapped in
   * a TrackedPtr object that is then added to the list.
   *
   * @param item - string or Node pointer to be added to list
   */
  void push_back(T* item) {
    value_->emplace_back(TrackedPtr<T>::make(owned_ptrs_, item));
  }

  /**
   * Creates an instance of a TrackedListPtr and returns a pointer to created instance.
   *
   * @param tracked_ptrs - TrackedListPtr unique pointer vector, which takes ownership of
   * dynamically allocated objects.
   * @param args - Arguments to be used when creating contained string or Node pointer
   * lists.
   * @return - pointer to created TrackedListPtr
   */
  template <typename... Args>
  static TrackedListPtr<T>* make(
      std::vector<std::unique_ptr<TrackedListPtr<T>>>& tracked_ptrs,
      Args&&... args) {
    std::unique_ptr<TrackedListPtr<T>> tracked_ptr{
        new TrackedListPtr<T>(std::forward<Args>(args)...)};
    const auto& ptr = tracked_ptrs.emplace_back(std::move(tracked_ptr));
    return ptr.get();
  }

  /**
   * Returns a pointer to a TrackedListPtr instance that represents an empty
   * TrackedListPtr.
   */
  static TrackedListPtr<T>* makeEmpty() {
    static std::unique_ptr<TrackedListPtr<T>> empty_tracked_ptr_ =
        std::unique_ptr<TrackedListPtr<T>>(new TrackedListPtr<T>());
    return empty_tracked_ptr_.get();
  }

 private:
  std::unique_ptr<std::list<TrackedPtr<T>*>> value_;
  std::vector<std::unique_ptr<TrackedPtr<T>>> owned_ptrs_{};
  bool is_empty_;
  bool is_released_;

  TrackedListPtr() : is_empty_(true), is_released_(false){};

  TrackedListPtr(size_t n)
      : value_(std::make_unique<std::list<TrackedPtr<T>*>>(n))
      , is_empty_(false)
      , is_released_(false) {}

  TrackedListPtr(size_t n, TrackedPtr<T>* val)
      : value_(std::make_unique<std::list<TrackedPtr<T>*>>(n, val))
      , is_empty_(false)
      , is_released_(false) {}

  TrackedListPtr(size_t n, T* val)
      : value_(std::make_unique<std::list<TrackedPtr<T>*>>(
            n,
            TrackedPtr<T>::make(owned_ptrs_, (T*)val)))
      , is_empty_(false)
      , is_released_(false) {}
};
}  // namespace Parser
