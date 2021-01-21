/*
 * Copyright (c) 2020 OmniSci, Inc.
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

// Used for creating iterators that can be used to sort two vectors.
// In other words, double_sort::Iterator<> is a writable zip iterator.

#pragma once

#include "funcannotations.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#ifndef __CUDACC__
#include <ostream>
#endif

// Overriding swap() isn't sufficient, since std:sort() uses insertion sort
// when the number of elements is 16 or less which bypasses swap.

// Iterator sent to sort() sorts two containers simultaneously.
namespace double_sort {

template <typename T>
union Variant {
  T* ptr_;
  T value_;
};

// Named Value insofar as it is the return value of Iterator::operator*().
// Default and copy/move constructors initial to a value (ref_=false).
template <typename T0, typename T1>
struct Value {
  Variant<T0> v0_;
  Variant<T1> v1_;
  bool const ref_;  // Use ptr_ if true, else value_.
  DEVICE Value(T0* ptr0, T1* ptr1) : v0_{ptr0}, v1_{ptr1}, ref_(true) {}
  // thrust::sort() copies Values, std::sort() moves Values.
#ifdef __CUDACC__
  DEVICE Value() : ref_(false) {}
  DEVICE Value(Value const& b) : ref_(false) {
    v0_.value_ = b.ref_ ? *b.v0_.ptr_ : b.v0_.value_;
    v1_.value_ = b.ref_ ? *b.v1_.ptr_ : b.v1_.value_;
  }
  DEVICE Value& operator=(Value const& b) {
    // Both branches are used by thrust::sort().
    if (ref_) {
      *v0_.ptr_ = b.ref_ ? *b.v0_.ptr_ : b.v0_.value_;
      *v1_.ptr_ = b.ref_ ? *b.v1_.ptr_ : b.v1_.value_;
    } else {
      v0_.value_ = b.ref_ ? *b.v0_.ptr_ : b.v0_.value_;
      v1_.value_ = b.ref_ ? *b.v1_.ptr_ : b.v1_.value_;
    }
    return *this;
  }
#else
  Value(Value&& b) : ref_(false) {
    v0_.value_ = b.ref_ ? std::move(*b.v0_.ptr_) : std::move(b.v0_.value_);
    v1_.value_ = b.ref_ ? std::move(*b.v1_.ptr_) : std::move(b.v1_.value_);
  }
  Value& operator=(Value&& b) {
    if (ref_) {
      *v0_.ptr_ = b.ref_ ? std::move(*b.v0_.ptr_) : std::move(b.v0_.value_);
      *v1_.ptr_ = b.ref_ ? std::move(*b.v1_.ptr_) : std::move(b.v1_.value_);
    } else {
      v0_.value_ = b.ref_ ? std::move(*b.v0_.ptr_) : std::move(b.v0_.value_);
      v1_.value_ = b.ref_ ? std::move(*b.v1_.ptr_) : std::move(b.v1_.value_);
    }
    return *this;
  }
#endif
  DEVICE T0 value0() const { return ref_ ? *v0_.ptr_ : v0_.value_; }
  DEVICE T1 value1() const { return ref_ ? *v1_.ptr_ : v1_.value_; }
};

#ifndef __CUDACC__
template <typename T0, typename T1>
std::ostream& operator<<(std::ostream& out, Value<T0, T1> const& ds) {
  return out << "ref_(" << ds.ref_ << ") v0_.value_(" << ds.v0_.value_ << ") v1_.value_("
             << ds.v1_.value_ << ')' << std::endl;
}
#endif

template <typename T0, typename T1>
struct Iterator : public std::iterator<std::input_iterator_tag, Value<T0, T1>> {
  Value<T0, T1> this_;  // this_ is always a reference object. I.e. this_.ref_ == true.
  DEVICE Iterator(T0* ptr0, T1* ptr1) : this_(ptr0, ptr1) {}
  DEVICE Iterator(Iterator const& b) : this_(b.this_.v0_.ptr_, b.this_.v1_.ptr_) {}
  DEVICE Iterator(Iterator&& b) : this_(b.this_.v0_.ptr_, b.this_.v1_.ptr_) {}
  DEVICE Iterator& operator=(Iterator const& b) {
    this_.v0_.ptr_ = b.this_.v0_.ptr_;
    this_.v1_.ptr_ = b.this_.v1_.ptr_;
    return *this;
  }
  DEVICE Iterator& operator=(Iterator&& b) {
    this_.v0_.ptr_ = b.this_.v0_.ptr_;
    this_.v1_.ptr_ = b.this_.v1_.ptr_;
    return *this;
  }
  // Returns a reference object by reference
  DEVICE Value<T0, T1>& operator*() { return this_; }
  // Required by thrust::sort().
  // Returns a reference object by value
  DEVICE Value<T0, T1> operator[](int i) const { return operator+(i).this_; }
  DEVICE Iterator& operator++() {
    ++this_.v0_.ptr_;
    ++this_.v1_.ptr_;
    return *this;
  }
  // Required by thrust::sort().
  DEVICE Iterator& operator+=(int i) {
    this_.v0_.ptr_ += i;
    this_.v1_.ptr_ += i;
    return *this;
  }
  DEVICE Iterator& operator--() {
    --this_.v0_.ptr_;
    --this_.v1_.ptr_;
    return *this;
  }
  DEVICE
  auto operator-(Iterator const& b) const { return this_.v0_.ptr_ - b.this_.v0_.ptr_; }
  DEVICE
  Iterator operator+(int i) const { return {this_.v0_.ptr_ + i, this_.v1_.ptr_ + i}; }
  DEVICE
  Iterator operator-(int i) const { return {this_.v0_.ptr_ - i, this_.v1_.ptr_ - i}; }
  DEVICE
  bool operator==(Iterator const& b) const { return this_.v0_.ptr_ == b.this_.v0_.ptr_; }
  DEVICE
  bool operator!=(Iterator const& b) const { return this_.v0_.ptr_ != b.this_.v0_.ptr_; }
  DEVICE
  bool operator<(Iterator const& b) const { return this_.v0_.ptr_ < b.this_.v0_.ptr_; }
  // Required by MacOS /usr/local/opt/llvm/include/c++/v1/algorithm:4036
  DEVICE
  bool operator>(Iterator const& b) const { return this_.v0_.ptr_ > b.this_.v0_.ptr_; }
  // Required by MacOS /usr/local/opt/llvm/include/c++/v1/algorithm:4000
  DEVICE
  bool operator>=(Iterator const& b) const { return this_.v0_.ptr_ >= b.this_.v0_.ptr_; }
};

}  // namespace double_sort
