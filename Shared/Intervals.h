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

/*
 * @file   Intervals.h
 * @author Matt Pulver <matt.pulver@omnisci.com>
 * @description Divide up indexes (A, A+1, A+2, ..., B-2, B-1) among
 *              N workers as evenly as possible in a range-based for loop:
 *              for (auto const& interval : makeIntervals(A, B, N)) {}
 *              where interval is a 2-member struct of (begin,end) values.
 *              This just does the arithmetic to divide up work into chunks;
 *              asynchronous thread management is done separately.
 *
 * This is a common pattern when dividing work among CPU threads.
 * This is NOT appropriate for GPU threads which are better done with strides.
 * 2 guarantees:
 *  - Larger intervals always precede smaller intervals.
 *  - The difference in work between any two intervals never exceeds 1.
 *
 * Example Usage. Divide up this loop:
 *
 *   for (int i=0 ; i<100 ; ++i)
 *     work(i);
 *
 * into 3 parallel workers:
 *
 *   for (auto const& interval : makeIntervals(0, 100, 3))
 *     spawn_thread(work, interval);
 *
 * where, for example, spawn_thread spawns threads running:
 *
 *   for (int j=interval.begin ; j<interval.end ; ++j)
 *     work(j);
 *
 * This would be equivalent to running the following 3 loops in parallel:
 *
 *   for (int j= 0 ; j< 34 ; ++j) work(j);
 *   for (int j=34 ; j< 67 ; ++j) work(j);
 *   for (int j=67 ; j<100 ; ++j) work(j);
 *
 * The number of iterations in this example is 34, 33, 33, respectively.
 * In general the larger iterations always precede the smaller, and the
 * maximum difference between the largest and smallest never exceeds 1.
 */

#pragma once

#include <iterator>
#include <limits>
#include <type_traits>

template <typename T, typename U = typename std::make_unsigned<T>::type>
struct Interval {
  T const begin;
  T const end;
  U const index;  // [0, n_workers)
  U size() const { return U(end) - U(begin); }
};

template <typename T, typename U = typename std::make_unsigned<T>::type>
class Intervals {
  T const begin_;
  U const total_size_;
  U const quot_;
  U const rem_;

  Intervals(T begin, T end, U n_workers)
      : begin_(begin)
      , total_size_(begin < end && n_workers ? end - begin : 0)
      , quot_(n_workers ? total_size_ / n_workers : 0)
      , rem_(n_workers ? total_size_ % n_workers : 0) {
    static_assert(std::is_integral<T>::value);
  }

 public:
  class Iterator : public std::iterator<std::input_iterator_tag, Interval<T>> {
    T begin_;
    U const quot_;
    U rem_;
    U index{0};

   public:
    Iterator(T begin, U quot, U rem) : begin_(begin), quot_(quot), rem_(rem) {}
    Interval<T> operator*() const {
      return {begin_, T(begin_ + quot_ + bool(rem_)), index};
    }
    void operator++() {
      begin_ += quot_ + (rem_ && rem_--);
      ++index;
    }
    bool operator==(Iterator const& rhs) const { return begin_ == rhs.begin_; }
    bool operator!=(Iterator const& rhs) const { return begin_ != rhs.begin_; }
  };

  bool empty() const { return !total_size_; }
  Iterator begin() const { return {begin_, quot_, rem_}; }
  Iterator end() const { return {static_cast<T>(begin_ + total_size_), quot_, 0}; }
  template <typename T1, typename U1>
  friend Intervals<T1> makeIntervals(T1 begin, T1 end, std::size_t n_workers);
};

template <typename T, typename U = typename std::make_unsigned<T>::type>
Intervals<T> makeIntervals(T begin, T end, std::size_t n_workers) {
  if constexpr (sizeof(U) < sizeof(std::size_t)) {
    if (std::numeric_limits<U>::max() < n_workers) {
      n_workers = std::numeric_limits<U>::max();
    }
  }
  return {begin, end, static_cast<U>(n_workers)};
}
