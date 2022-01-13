/*
    Copyright (c) 2005-2020 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>

template <typename KIter, typename VIter, typename Comp>
void insertion_sort_by_key(KIter keys, VIter values, size_t size, Comp comp) {
  for (size_t i = 1; i < size; ++i) {
    auto k = keys[i];
    auto v = values[i];
    int64_t j = i - 1;
    for (; j >= 0 && comp(k, keys[j]); --j) {
      keys[j + 1] = keys[j];
      values[j + 1] = values[j];
    }
    keys[j + 1] = k;
    values[j + 1] = v;
  }
}

template <typename KIter, typename VIter, typename Compare>
class quick_sort_range {
  inline size_t median_of_three(const KIter& keys, size_t l, size_t m, size_t r) const {
    return comp(keys[l], keys[m])
               ? (comp(keys[m], keys[r]) ? m : (comp(keys[l], keys[r]) ? r : l))
               : (comp(keys[r], keys[m]) ? m : (comp(keys[r], keys[l]) ? r : l));
  }

  inline size_t pseudo_median_of_nine(const KIter& keys,
                                      const quick_sort_range& range) const {
    size_t offset = range.size / 8u;
    return median_of_three(keys,
                           median_of_three(keys, 0, offset, offset * 2),
                           median_of_three(keys, offset * 3, offset * 4, offset * 5),
                           median_of_three(keys, offset * 6, offset * 7, range.size - 1));
  }

  size_t split_range(quick_sort_range& range) {
    using std::iter_swap;
    KIter keys = range.k_begin;
    VIter values = range.v_begin;
    KIter key0 = range.k_begin;
    VIter key0_val = range.v_begin;

    size_t m = pseudo_median_of_nine(keys, range);
    if (m) {
      iter_swap(keys, keys + m);
      iter_swap(values, values + m);
    }

    size_t i = 0;
    size_t j = range.size;
    // Partition interval [i+1,j-1] with key *key0.
    for (;;) {
      assert(i < j);
      // Loop must terminate since keys[l]==*key0.
      do {
        --j;
        assert(i <= j);
      } while (comp(*key0, keys[j]));
      do {
        assert(i <= j);
        if (i == j)
          goto partition;
        ++i;
      } while (comp(keys[i], *key0));
      if (i == j)
        goto partition;
      iter_swap(keys + i, keys + j);
      iter_swap(values + i, values + j);
    }
  partition:
    // Put the partition key were it belongs
    iter_swap(keys + j, key0);
    iter_swap(values + j, key0_val);
    // keys[l..j) is less or equal to key.
    // keys(j..r) is greater or equal to key.
    // keys[j] is equal to key
    i = j + 1;
    size_t new_range_size = range.size - i;
    range.size = j;
    return new_range_size;
  }

 public:
  static const size_t grainsize = 500;
  const Compare& comp;
  size_t size;
  KIter k_begin;
  VIter v_begin;

  quick_sort_range(KIter k_begin_, VIter v_begin_, size_t size_, const Compare& comp_)
      : comp(comp_), size(size_), k_begin(k_begin_), v_begin(v_begin_) {}

  bool empty() const { return size == 0; }
  bool is_divisible() const { return size >= grainsize; }

  quick_sort_range(quick_sort_range& range, tbb::split)
      : comp(range.comp)
      , size(split_range(range))
      // +1 accounts for the pivot element, which is at its correct place
      // already and, therefore, is not included into subranges.
      , k_begin(range.k_begin + range.size + 1)
      , v_begin(range.v_begin + range.size + 1) {}

  ~quick_sort_range() {}
};

//! Body class used to sort elements in a range that is smaller than the grainsize.
/** @ingroup algorithms */
template <typename KIter, typename VIter, typename Compare>
struct quick_sort_body {
  void operator()(const quick_sort_range<KIter, VIter, Compare>& range) const {
    // SerialQuickSort( range.begin, range.size, range.comp );
    insertion_sort_by_key(range.k_begin, range.v_begin, range.size, range.comp);
  }
};

template <typename KIter, typename VIter, typename Compare>
void parallel_sort_by_key_impl(KIter keys,
                               VIter values,
                               size_t size,
                               const Compare& comp) {
  tbb::parallel_for(quick_sort_range<KIter, VIter, Compare>(keys, values, size, comp),
                    quick_sort_body<KIter, VIter, Compare>(),
                    tbb::auto_partitioner());
}

template <typename KIter, typename VIter, typename Compare>
void parallel_sort_by_key(KIter keys, VIter values, size_t size, const Compare& comp) {
  const int min_parallel_size = 500;
  if (size > 1) {
    if (size < min_parallel_size) {
      insertion_sort_by_key(keys, values, size, comp);
    } else {
      parallel_sort_by_key_impl(keys, values, size, comp);
    }
  }
}
