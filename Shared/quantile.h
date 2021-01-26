/*
 * Copyright (c) 2021 OmniSci, Inc.
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
 * @file    quantile.h
 * @author  Matt Pulver <matt.pulver@omnisci.com>
 * @description Calculate approximate median and general quantiles, based on
 *   "Computing Extremely Accurate Quantiles Using t-Digests" by T. Dunning et al.
 *   https://arxiv.org/abs/1902.04023
 *
 */

#pragma once

#include "DoubleSort.h"
#include "SimpleAllocator.h"
#include "VectorView.h"
#include "gpu_enabled.h"

#ifndef __CUDACC__
#include <iomanip>
#include <ostream>
#endif

#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>

namespace quantile {

namespace detail {

template <typename RealType, typename IndexType>
struct Centroid {
  RealType sum_;
  IndexType count_;
};

template <typename RealType, typename IndexType>
struct Centroids {
  IndexType curr_idx_;  // used during mergeCentroids algorithm
  IndexType next_idx_;  // used during mergeCentroids algorithm
  int inc_;             // 1 or -1 : forward or reverse iteration
  VectorView<RealType> sums_;
  VectorView<IndexType> counts_;
  static constexpr RealType infinity = std::numeric_limits<RealType>::infinity();
  static constexpr RealType nan = std::numeric_limits<RealType>::quiet_NaN();
  RealType max_{-infinity};
  RealType min_{infinity};

  Centroids() = default;

  DEVICE Centroids(RealType* sums, IndexType* counts, IndexType const size)
      : sums_(sums, size, size), counts_(counts, size, size) {}

  DEVICE Centroids(VectorView<RealType> sums, VectorView<IndexType> counts)
      : sums_(sums), counts_(counts) {}

  DEVICE void appendAndSortCurrent(Centroids& buff);

  DEVICE IndexType capacity() const { return sums_.capacity(); }

  DEVICE void clear() {
    sums_.clear();
    counts_.clear();
    max_ = -infinity;
    min_ = infinity;
  }

  DEVICE IndexType currCount() const { return counts_[curr_idx_]; }

  DEVICE RealType currMean() const { return mean(curr_idx_); }

  DEVICE bool hasCurr() const { return curr_idx_ < size(); }

  DEVICE bool hasNext() const { return next_idx_ < size(); }

  DEVICE RealType mean(IndexType const i) const { return sums_[i] / counts_[i]; }

  // Return true if centroid was merged, false if not.
  DEVICE bool mergeIfFits(Centroids& centroid, IndexType const max_count);

  DEVICE void moveNextToCurrent();

  DEVICE IndexType nextCount() const { return counts_[next_idx_]; }

  DEVICE RealType nextSum() const { return sums_[next_idx_]; }

  // Assumes this->hasNext() and b.hasNext().
  // Order by (nextMean(), -nextCount()).
  DEVICE bool operator<(Centroids const& b) const {
    // lhs < rhs is same as nextMean() < b.nextMean() without division
    RealType const lhs = nextSum() * b.nextCount();
    RealType const rhs = b.nextSum() * nextCount();
    return lhs < rhs || (lhs == rhs && b.nextCount() < nextCount());
  }

  DEVICE void push_back(RealType const value, RealType const count) {
    sums_.push_back(value);
    counts_.push_back(count);
  }

  DEVICE void resetIndices(bool const forward);

  DEVICE size_t size() const { return sums_.size(); }

  DEVICE IndexType totalWeight() const {
    return gpu_enabled::accumulate(counts_.begin(), counts_.end(), IndexType(0));
  }

#ifndef __CUDACC__
  template <typename RealType2, typename IndexType2>
  friend std::ostream& operator<<(std::ostream&, Centroids<RealType2, IndexType2> const&);
#endif
};

template <typename RealType, typename IndexType>
class CentroidsMerger {
  Centroids<RealType, IndexType>* buf_;  // incoming buffer
  Centroids<RealType, IndexType>* centroids_;
  Centroids<RealType, IndexType>* curr_centroid_;
  IndexType prefix_sum_{0};  // Prefix-sum of centroid counts.
  IndexType const total_weight_;
  bool const forward_;

  DEVICE void mergeMinMax();
  DEVICE void setCurrCentroid();

 public:
  DEVICE CentroidsMerger(Centroids<RealType, IndexType>* buf,
                         Centroids<RealType, IndexType>* centroids,
                         bool const forward);

  // nullptr if !hasNext()
  DEVICE Centroids<RealType, IndexType>* getNextCentroid() const;

  DEVICE bool hasNext() const { return buf_->hasNext() || centroids_->hasNext(); }

  // Merge as many centroids as possible for count <= max_count.
  DEVICE void merge(IndexType const max_count);

  // Assume curr_centroid_ is fully merged.
  DEVICE void next();

  DEVICE IndexType prefixSum() const { return prefix_sum_; }

  DEVICE IndexType totalWeight() const { return total_weight_; }
};

// Instantiated on host for reserving memory for Centroids.
template <typename RealType, typename IndexType>
class CentroidsMemory {
  // TODO CUDA memory containers
  std::vector<RealType> sums_;
  std::vector<IndexType> counts_;

 public:
  explicit CentroidsMemory(size_t const size) : sums_(size), counts_(size) {}
  size_t nbytes() const { return sums_.size() * (sizeof(RealType) + sizeof(IndexType)); }
  size_t size() const { return sums_.size(); }
  VectorView<RealType> sums() { return {sums_.data(), sums_.size(), sums_.size()}; }
  VectorView<IndexType> counts() {
    return {counts_.data(), counts_.size(), counts_.size()};
  }
};

template <typename RealType, typename IndexType = size_t>
class TDigest {
  Centroids<RealType, IndexType> buf_;  // incoming buffer
  Centroids<RealType, IndexType> centroids_;
  bool forward_{true};  // alternate direction on each call to mergeCentroids().

  // simple_allocator_, buf_allocate_, centroids_allocate_ are used only by allocate().
  SimpleAllocator* const simple_allocator_{nullptr};
  IndexType const buf_allocate_{0};
  IndexType const centroids_allocate_{0};

  DEVICE RealType max() const { return centroids_.max_; }
  DEVICE RealType min() const { return centroids_.min_; }

  DEVICE IndexType maxCardinality(IndexType const sum, IndexType const total_weight);

  // Require: centroids are sorted by (mean, -count)
  DEVICE void mergeCentroids(Centroids<RealType, IndexType>&);

  DEVICE RealType firstCentroid(RealType const x);
  DEVICE RealType interiorCentroid(RealType const x,
                                   IndexType const idx1,
                                   IndexType const prefix_sum);
  DEVICE RealType lastCentroid(RealType const x, IndexType const N);
  DEVICE RealType oneCentroid(RealType const x);

  DEVICE RealType slope(IndexType const idx1, IndexType const idx2);

 public:
  using Memory = CentroidsMemory<RealType, IndexType>;

  TDigest() = default;

  DEVICE TDigest(Memory& mem)
      : centroids_(mem.sums().data(), mem.counts().data(), mem.size()) {
    centroids_.clear();
  }

  DEVICE TDigest(SimpleAllocator* simple_allocator,
                 IndexType buf_allocate,
                 IndexType centroids_allocate)
      : simple_allocator_(simple_allocator)
      , buf_allocate_(buf_allocate)
      , centroids_allocate_(centroids_allocate) {}

  DEVICE Centroids<RealType, IndexType>& centroids() { return centroids_; }

  // Store value to buf_, and merge when full.
  DEVICE void add(RealType value);

  // Allocate memory if needed by simple_allocator_.
  DEVICE void allocate();

  DEVICE void mergeBuffer();

  // Import from sorted memory range. Ok to change values during merge.
  DEVICE void mergeSorted(RealType* sums, IndexType* counts, IndexType size);

  DEVICE void mergeTDigest(TDigest& t_digest) {
    mergeBuffer();
    t_digest.mergeBuffer();
    mergeCentroids(t_digest.centroids_);
  }

  // Uses buf_ as scratch space.
  DEVICE RealType quantile(RealType const q) {
    assert(centroids_.size() <= buf_.capacity());
    return quantile(buf_.counts_.data(), q);
  }

  // Uses buf as scratch space.
  DEVICE RealType quantile(IndexType* buf, RealType const q);

  // Assumes mem is externally managed.
  DEVICE void setBuffer(Memory& mem) {
    buf_ = Centroids<RealType, IndexType>(
        mem.sums().data(), mem.counts().data(), mem.size());
    buf_.clear();
  }

  // Assumes mem is externally managed.
  DEVICE void setCentroids(Memory& mem) {
    centroids_ = Centroids<RealType, IndexType>(
        mem.sums().data(), mem.counts().data(), mem.size());
    centroids_.clear();
  }

  DEVICE void setCentroids(VectorView<RealType> const sums,
                           VectorView<IndexType> const counts) {
    centroids_ = Centroids<RealType, IndexType>(sums, counts);
    centroids_.clear();
  }

  // Total number of data points in centroids_.
  DEVICE IndexType totalWeight() const { return centroids_.totalWeight(); }
};

// Class template member definitions

// Class Centroids<>

namespace {

// Order by (mean, -count)
template <typename RealType, typename IndexType>
struct OrderByMeanAscCountDesc {
  using Value = double_sort::Value<RealType, IndexType>;
  // Assume value1() is positive and use multiplication instead of division.
  DEVICE bool operator()(Value const& a, Value const& b) const {
    auto const lhs = a.value0() * b.value1();
    auto const rhs = b.value0() * a.value1();
    return lhs < rhs || (lhs == rhs && b.value1() < a.value1());
  }
};

}  // namespace

template <typename RealType, typename IndexType>
DEVICE void Centroids<RealType, IndexType>::appendAndSortCurrent(Centroids& buff) {
  if (inc_ == -1 && curr_idx_ != 0) {
    // Shift data to the left left by curr_idx_.
    // Prefer to copy, but thrust::copy doesn't support overlapping ranges.
    // Reverse instead, which gets sorted below.
    // gpu_enabled::copy(sums_.begin() + curr_idx_, sums_.end(), sums_.begin());
    // gpu_enabled::copy(counts_.begin() + curr_idx_, counts_.end(), counts_.begin());
    gpu_enabled::reverse(sums_.begin(), sums_.end());
    gpu_enabled::reverse(counts_.begin(), counts_.end());
  }
  // Shift VectorViews to the right by buff.curr_idx_.
  IndexType const offset = inc_ == 1 ? 0 : buff.curr_idx_;
  IndexType const buff_size =
      inc_ == 1 ? buff.curr_idx_ + 1 : buff.size() - buff.curr_idx_;
  VectorView<RealType> buff_sums(buff.sums_.begin() + offset, buff_size, buff_size);
  VectorView<IndexType> buff_counts(buff.counts_.begin() + offset, buff_size, buff_size);
  // Copy buff into sums_ and counts_.
  IndexType const curr_size = inc_ == 1 ? curr_idx_ + 1 : size() - curr_idx_;
  IndexType const total_size = curr_size + buff_sums.size();
  assert(total_size <= sums_.capacity());  // TODO proof of inequality
  sums_.resize(total_size);
  gpu_enabled::copy(buff_sums.begin(), buff_sums.end(), sums_.begin() + curr_size);
  assert(total_size <= counts_.capacity());
  counts_.resize(total_size);
  gpu_enabled::copy(buff_counts.begin(), buff_counts.end(), counts_.begin() + curr_size);

  // Sort sums_ and counts_ by (mean, -count).
  double_sort::Iterator<RealType, IndexType> const begin(sums_.begin(), counts_.begin());
  double_sort::Iterator<RealType, IndexType> const end(sums_.end(), counts_.end());
  gpu_enabled::sort(begin, end, OrderByMeanAscCountDesc<RealType, IndexType>());
}

// Return true if centroid was merged, false if not.
template <typename RealType, typename IndexType>
DEVICE bool Centroids<RealType, IndexType>::mergeIfFits(Centroids& centroid,
                                                        IndexType const max_count) {
  if (counts_[curr_idx_] + centroid.nextCount() <= max_count) {
    sums_[curr_idx_] += centroid.nextSum();
    counts_[curr_idx_] += centroid.nextCount();
    centroid.next_idx_ += centroid.inc_;
    return true;
  }
  return false;
}

template <typename RealType, typename IndexType>
DEVICE void Centroids<RealType, IndexType>::moveNextToCurrent() {
  curr_idx_ += inc_;
  if (curr_idx_ != next_idx_) {
    sums_[curr_idx_] = sums_[next_idx_];
    counts_[curr_idx_] = counts_[next_idx_];
  }
  next_idx_ += inc_;
}

template <typename RealType, typename IndexType>
DEVICE void Centroids<RealType, IndexType>::resetIndices(bool const forward) {
  if (forward) {
    inc_ = 1;
    curr_idx_ = ~IndexType(0);
  } else {
    inc_ = -1;
    curr_idx_ = size();
  }
  static_assert(std::is_unsigned<IndexType>::value,
                "IndexType must be an unsigned type.");
  next_idx_ = curr_idx_ + inc_;
}

#ifndef __CUDACC__
template <typename RealType, typename IndexType>
std::ostream& operator<<(std::ostream& out,
                         Centroids<RealType, IndexType> const& centroids) {
  out << "Centroids<" << typeid(RealType).name() << ',' << typeid(IndexType).name()
      << ">(size(" << centroids.size() << ") curr_idx_(" << centroids.curr_idx_
      << ") next_idx_(" << centroids.next_idx_ << ") sums_(";
  for (IndexType i = 0; i < centroids.sums_.size(); ++i) {
    out << (i ? " " : "") << std::setprecision(20) << centroids.sums_[i];
  }
  out << ") counts_(";
  for (IndexType i = 0; i < centroids.counts_.size(); ++i) {
    out << (i ? " " : "") << centroids.counts_[i];
  }
  return out << "))";
}
#endif

// Class CentroidsMerger<>

template <typename RealType, typename IndexType>
DEVICE CentroidsMerger<RealType, IndexType>::CentroidsMerger(
    Centroids<RealType, IndexType>* buf,
    Centroids<RealType, IndexType>* centroids,
    bool const forward)
    : buf_(buf)
    , centroids_(centroids)
    , total_weight_(centroids->totalWeight() + buf->totalWeight())
    , forward_(forward) {
  buf_->resetIndices(forward_);
  centroids_->resetIndices(forward_);
  mergeMinMax();
  setCurrCentroid();
}

template <typename RealType, typename IndexType>
DEVICE Centroids<RealType, IndexType>*
CentroidsMerger<RealType, IndexType>::getNextCentroid() const {
  if (buf_->hasNext() && centroids_->hasNext()) {
    return (*buf_ < *centroids_) == forward_ ? buf_ : centroids_;
  } else if (buf_->hasNext()) {
    return buf_;
  } else if (centroids_->hasNext()) {
    return centroids_;
  } else {
    return nullptr;  // hasNext() is false
  }
}

namespace {

// Helper struct for mergeCentroids() for tracking skipped centroids.
template <typename RealType, typename IndexType>
class Skipped {
  struct Data {
    Centroids<RealType, IndexType>* centroid_{nullptr};
    IndexType start_{};
    IndexType count_merged_{};
    IndexType count_skipped_{};
  } data_[2];
  Centroid<RealType, IndexType> mean_;

  DEVICE static void shiftCentroids(Data& data) {
    if (data.count_merged_) {
      shiftRange(data.centroid_->sums_.begin() + data.start_,
                 data.count_skipped_,
                 data.count_merged_,
                 data.centroid_->inc_);
      shiftRange(data.centroid_->counts_.begin() + data.start_,
                 data.count_skipped_,
                 data.count_merged_,
                 data.centroid_->inc_);
      data.start_ += data.centroid_->inc_ * data.count_merged_;
    }
  }
  template <typename T>
  DEVICE static void shiftRange(T* const begin,
                                IndexType skipped,
                                IndexType const merged,
                                int const inc) {
#ifdef __CUDACC__
    T* src = begin + inc * (skipped - 1);
    T* dst = src + inc * merged;
    for (; skipped; --skipped, src -= inc, dst -= inc) {
      *dst = *src;
    }
#else
    if (inc == 1) {
      std::copy_backward(begin, begin + skipped, begin + skipped + merged);
    } else {
      std::copy(begin + 1 - skipped, begin + 1, begin + 1 - skipped - merged);
    }
#endif
  }

 public:
  DEVICE bool index(Centroids<RealType, IndexType>* centroid) const {
    return data_[0].centroid_ != centroid;
  }
  DEVICE bool isDifferentMean(Centroids<RealType, IndexType>* next_centroid) const {
    return mean_.sum_ * next_centroid->nextCount() !=
           next_centroid->nextSum() * mean_.count_;
  }
  DEVICE void merged(Centroids<RealType, IndexType>* next_centroid) {
    IndexType const idx = index(next_centroid);
    if (idx == 1 && data_[1].centroid_ == nullptr) {
      data_[1] = {next_centroid, next_centroid->next_idx_ + next_centroid->inc_, 0, 0};
    } else if (data_[idx].count_skipped_) {
      ++data_[idx].count_merged_;
    } else {
      assert(idx == 1);
      data_[1].start_ += next_centroid->inc_;
    }
  }
  DEVICE operator bool() const { return data_[0].centroid_; }
  // Shift skipped centroids over merged centroids, and rewind next_idx_.
  DEVICE void shiftCentroidsAndSetNext() {
    shiftCentroids(data_[0]);
    data_[0].centroid_->next_idx_ = data_[0].start_;
    if (data_[1].centroid_) {
      shiftCentroids(data_[1]);
      data_[1].centroid_->next_idx_ = data_[1].start_;
    }
  }
  DEVICE void skipFirst(Centroids<RealType, IndexType>* next_centroid) {
    mean_ = Centroid<RealType, IndexType>{next_centroid->nextSum(),
                                          next_centroid->nextCount()};
    data_[0] = {next_centroid, next_centroid->next_idx_, 0, 1};
  }
  DEVICE void skipSubsequent(Centroids<RealType, IndexType>* next_centroid) {
    IndexType const idx = index(next_centroid);
    if (idx == 1 && data_[1].centroid_ == nullptr) {
      data_[1] = {next_centroid, next_centroid->next_idx_, 0, 1};
    } else {
      if (data_[idx].count_merged_) {
        shiftCentroids(data_[idx]);
        data_[idx].count_merged_ = 0;
      }
      ++data_[idx].count_skipped_;
    }
    next_centroid->next_idx_ += next_centroid->inc_;
  }
};

}  // namespace

// Merge as many centroids as possible for count <= max_count.
template <typename RealType, typename IndexType>
DEVICE void CentroidsMerger<RealType, IndexType>::merge(IndexType const max_count) {
  Skipped<RealType, IndexType> skipped;
  while (auto* next_centroid = getNextCentroid()) {
    if (skipped) {
      if (skipped.isDifferentMean(next_centroid)) {
        break;
      } else if (curr_centroid_->mergeIfFits(*next_centroid, max_count)) {
        skipped.merged(next_centroid);
      } else {
        skipped.skipSubsequent(next_centroid);
      }
    } else if (!curr_centroid_->mergeIfFits(*next_centroid, max_count)) {
      skipped.skipFirst(next_centroid);
    }
  }
  if (skipped) {
    skipped.shiftCentroidsAndSetNext();
  }
}

// Track min/max without assuming the data points exist in any particular centroids.
// Otherwise to assume and track their existence in the min/max centroids introduces
// significant complexity. For example, the min centroid may not remain the min
// centroid if it merges into it the 3rd smallest centroid, skipping the 2nd due to
// the scaling function constraint.
// When the quantile() is calculated, the min/max data points are assumed to exist
// in the min/max centroids for the purposes of calculating the quantile within those
// centroids.
template <typename RealType, typename IndexType>
DEVICE void CentroidsMerger<RealType, IndexType>::mergeMinMax() {
  if (centroids_->max_ < buf_->max_) {
    centroids_->max_ = buf_->max_;
  }
  if (buf_->min_ < centroids_->min_) {
    centroids_->min_ = buf_->min_;
  }
}

// Assume curr_centroid_ is fully merged.
template <typename RealType, typename IndexType>
DEVICE void CentroidsMerger<RealType, IndexType>::next() {
  prefix_sum_ += curr_centroid_->currCount();
  setCurrCentroid();
}

template <typename RealType, typename IndexType>
DEVICE void CentroidsMerger<RealType, IndexType>::setCurrCentroid() {
  if ((curr_centroid_ = getNextCentroid())) {
    curr_centroid_->moveNextToCurrent();
  }
}

// class TDigest<>

template <typename RealType, typename IndexType>
DEVICE void TDigest<RealType, IndexType>::add(RealType value) {
  if (buf_.sums_.full()) {
    mergeBuffer();
  }
  buf_.sums_.push_back(value);
  buf_.counts_.push_back(1);
}

// Assumes buf_ is allocated iff centroids_ is allocated.
template <typename RealType, typename IndexType>
DEVICE void TDigest<RealType, IndexType>::allocate() {
  if (buf_.capacity() == 0) {
    auto* p0 = simple_allocator_->allocate(buf_allocate_ * sizeof(RealType));
    auto* p1 = simple_allocator_->allocate(buf_allocate_ * sizeof(IndexType));
    buf_ = Centroids<RealType, IndexType>(
        VectorView<RealType>(reinterpret_cast<RealType*>(p0), 0, buf_allocate_),
        VectorView<IndexType>(reinterpret_cast<IndexType*>(p1), 0, buf_allocate_));
    p0 = simple_allocator_->allocate(centroids_allocate_ * sizeof(RealType));
    p1 = simple_allocator_->allocate(centroids_allocate_ * sizeof(IndexType));
    centroids_ = Centroids<RealType, IndexType>(
        VectorView<RealType>(reinterpret_cast<RealType*>(p0), 0, centroids_allocate_),
        VectorView<IndexType>(reinterpret_cast<IndexType*>(p1), 0, centroids_allocate_));
  }
}

template <typename RealType, typename IndexType>
DEVICE IndexType
TDigest<RealType, IndexType>::maxCardinality(IndexType const sum,
                                             IndexType const total_weight) {
  IndexType const max_bins = centroids_.capacity();
  return max_bins < total_weight ? 2 * total_weight / max_bins : 0;
}

// Assumes buf_ consists only of singletons.
template <typename RealType, typename IndexType>
DEVICE void TDigest<RealType, IndexType>::mergeBuffer() {
  if (buf_.size()) {
    gpu_enabled::sort(buf_.sums_.begin(), buf_.sums_.end());
    buf_.min_ = buf_.sums_.front();
    buf_.max_ = buf_.sums_.back();
    mergeCentroids(buf_);
  }
}

template <typename RealType, typename IndexType>
DEVICE void TDigest<RealType, IndexType>::mergeSorted(RealType* sums,
                                                      IndexType* counts,
                                                      IndexType size) {
  if (size) {
    if (buf_.capacity() == 0) {
      buf_ = Centroids<RealType, IndexType>(sums, counts, size);  // Set capacity and size
    } else {
      buf_.sums_.set(sums, size);  // Does not change capacity
      buf_.counts_.set(counts, size);
    }
    gpu_enabled::fill(buf_.counts_.begin(), buf_.counts_.end(), IndexType(1));
    buf_.min_ = buf_.sums_.front();
    buf_.max_ = buf_.sums_.back();
    mergeCentroids(buf_);
  }
}

// Require:
//  * buf centroids are not empty and sorted by (mean, -count)
//  * buf.min_ and buf.max_ are correctly set.
// During filling stage, buf=buf_.
// During reduction, buf_=centroids_[i]
template <typename RealType, typename IndexType>
DEVICE void TDigest<RealType, IndexType>::mergeCentroids(
    Centroids<RealType, IndexType>& buf) {
  // Loop over sorted sequence of buf and centroids_.
  // Some latter centroids may be merged into the current centroid, so the number
  // of iterations is only at most equal to the number of initial centroids.
  using CM = CentroidsMerger<RealType, IndexType>;
  for (CM cm(&buf, &centroids_, forward_); cm.hasNext(); cm.next()) {
    // cm.prefixSum() == 0 on first iteration.
    // Max cardinality for current centroid to be fully merged based on scaling function.
    IndexType const max_cardinality = maxCardinality(cm.prefixSum(), cm.totalWeight());
    cm.merge(max_cardinality);
  }
  // Combine sorted centroids buf[0..curr_idx_] + centroids_[0..curr_idx_] if forward
  centroids_.appendAndSortCurrent(buf);
  buf.clear();
  forward_ ^= true;  // alternate merge direction on each call
}

namespace {
template <typename CountsIterator>
DEVICE bool isSingleton(CountsIterator itr) {
  return *itr == 1;
}
}  // namespace

// Assumes x < centroids_.counts_.front().
template <typename RealType, typename IndexType>
DEVICE RealType TDigest<RealType, IndexType>::firstCentroid(RealType const x) {
  if (x < 1) {
    return min();
  } else if (centroids_.size() == 1) {
    return oneCentroid(x);
  } else if (centroids_.counts_.front() == 2) {
    RealType const sum = centroids_.sums_.front();
    return x == 1 ? 0.5 * sum : sum - min();
  } else {
    RealType const count = centroids_.counts_.front();
    RealType const dx = x - RealType(0.5) * (1 + count);
    RealType const mean = (centroids_.sums_.front() - min()) / (count - 1);
    return mean + slope(0, 0 < dx) * dx;
  }
}

// x is between first and last centroids.
template <typename RealType, typename IndexType>
DEVICE RealType
TDigest<RealType, IndexType>::interiorCentroid(RealType const x,
                                               IndexType const idx1,
                                               IndexType const prefix_sum) {
  if (isSingleton(centroids_.counts_.begin() + idx1)) {
    RealType const sum1 = centroids_.sums_[idx1];
    if (x == prefix_sum - centroids_.counts_[idx1]) {
      if (isSingleton(centroids_.counts_.begin() + idx1 - 1)) {
        return 0.5 * (centroids_.sums_[idx1 - 1] + sum1);
      } else if (idx1 == 1 && centroids_.counts_[0] == 2) {
        return 0.5 * (centroids_.sums_[idx1 - 1] - min() + sum1);
      }
    }
    return sum1;
  } else {
    RealType const dx = x + RealType(0.5) * centroids_.counts_[idx1] - prefix_sum;
    IndexType const idx2 = idx1 + 2 * (0 < dx) - 1;
    return centroids_.mean(idx1) + slope(idx1, idx2) * dx;
  }
}

// Assumes N - centroids_.counts_.back() <= x < N, and there is more than 1 centroid.
template <typename RealType, typename IndexType>
DEVICE RealType TDigest<RealType, IndexType>::lastCentroid(RealType const x,
                                                           IndexType const N) {
  if (N - 1 < x) {
    return max();
  }
  IndexType const idx1 = centroids_.size() - 1;
  RealType const sum1 = centroids_.sums_[idx1];
  IndexType const count1 = centroids_.counts_[idx1];
  if (count1 == 1) {  // => x == N - 1
    if (isSingleton(centroids_.counts_.begin() + (idx1 - 1))) {
      return 0.5 * (centroids_.sums_[idx1 - 1] + sum1);
    } else if (idx1 == 1 && centroids_.counts_[0] == 2) {
      return 0.5 * (centroids_.sums_[idx1 - 1] - min() + sum1);
    } else {
      return sum1;
    }
  } else if (count1 == 2) {  // => 3 <= N
    if (x == N - 1) {
      return 0.5 * sum1;
    } else if (x == N - 2) {
      RealType const sum2 = centroids_.sums_[idx1 - 1];
      if (isSingleton(centroids_.counts_.begin() + (idx1 - 1))) {
        return 0.5 * (sum2 + sum1 - max());
      } else if (idx1 == 1 && centroids_.counts_[0] == 2) {
        return 0.5 * (sum2 - min() + sum1 - max());
      }
    }
    return sum1 - max();
  } else {  // => 3 <= count1
    RealType const dx = x + RealType(0.5) * (count1 + 1) - N;
    RealType const mean = (sum1 - max()) / (count1 - 1);
    return mean + slope(idx1, idx1 - (dx < 0)) * dx;
  }
}

// Assumes there is only 1 centroid, and 1 <= x < centroids_.counts_.front().
template <typename RealType, typename IndexType>
DEVICE RealType TDigest<RealType, IndexType>::oneCentroid(RealType const x) {
  IndexType const N = centroids_.counts_.front();
  if (N - 1 < x) {  // includes case N == 1
    return max();
  } else if (N == 2) {  // x == 1
    return 0.5 * centroids_.sums_.front();
  } else if (N == 3) {  // 1 <= x <= 2
    if (x == 2) {
      return 0.5 * (centroids_.sums_.front() - min());
    } else {
      RealType const s = centroids_.sums_.front() - max();
      return x == 1 ? 0.5 * s : s - min();
    }
  } else {  // 3 < N
    RealType const dx = x - RealType(0.5) * N;
    RealType const mean = (centroids_.sums_.front() - (min() + max())) / (N - 2);
    RealType const slope = 2 * (0 < dx ? max() - mean : mean - min()) / (N - 2);
    return mean + slope * dx;
  }
}

// No need to calculate entire partial_sum unless multiple calls to quantile() are made.
template <typename RealType, typename IndexType>
DEVICE RealType TDigest<RealType, IndexType>::quantile(IndexType* buf, RealType const q) {
  if (centroids_.size()) {
    VectorView<IndexType> partial_sum(buf, centroids_.size(), centroids_.size());
    gpu_enabled::partial_sum(
        centroids_.counts_.begin(), centroids_.counts_.end(), partial_sum.begin());
    IndexType const N = partial_sum.back();
    RealType const x = q * N;
    auto const it1 = gpu_enabled::upper_bound(partial_sum.begin(), partial_sum.end(), x);
    if (it1 == partial_sum.begin()) {
      return firstCentroid(x);
    } else if (it1 == partial_sum.end()) {  // <==> 1 <= q
      return max();
    } else if (it1 + 1 == partial_sum.end()) {
      return lastCentroid(x, N);
    } else {
      return interiorCentroid(x, it1 - partial_sum.begin(), *it1);
    }
  } else {
    return centroids_.nan;
  }
}

// Requirement: 1 < M and 0 <= idx1, idx2 < M where M = Number of centroids.
// Return slope of line segment connecting idx1 and idx2.
// If equal then assume it is the section contained within an extrema.
// Centroid for idx1 is not a singleton, but idx2 may be.
template <typename RealType, typename IndexType>
DEVICE RealType TDigest<RealType, IndexType>::slope(IndexType idx1, IndexType idx2) {
  IndexType const M = centroids_.size();
  if (idx1 == idx2) {  // Line segment is contained in either the first or last centroid.
    RealType const n = static_cast<RealType>(centroids_.counts_[idx1]);
    RealType const s = centroids_.sums_[idx1];
    return idx1 == 0 ? 2 * (s - n * min()) / ((n - 1) * (n - 1))
                     : 2 * (n * max() - s) / ((n - 1) * (n - 1));
  } else {
    bool const min1 = idx1 == 0;      // idx1 is the min centroid
    bool const max1 = idx1 == M - 1;  // idx1 is the max centroid
    bool const min2 = idx2 == 0;      // idx2 is the min centroid
    bool const max2 = idx2 == M - 1;  // idx2 is the max centroid
    RealType const n1 = static_cast<RealType>(centroids_.counts_[idx1] - min1 - max1);
    RealType const s1 = centroids_.sums_[idx1] - (min1 ? min() : max1 ? max() : 0);
    RealType const s2 = centroids_.sums_[idx2] - (min2 ? min() : max2 ? max() : 0);
    if (isSingleton(centroids_.counts_.begin() + idx2)) {
      return (idx1 < idx2 ? 2 : -2) * (n1 * s2 - s1) / (n1 * n1);
    } else {
      RealType const n2 = static_cast<RealType>(centroids_.counts_[idx2] - min2 - max2);
      return (idx1 < idx2 ? 2 : -2) * (n1 * s2 - n2 * s1) / (n1 * n2 * (n1 + n2));
    }
  }
}

}  // namespace detail

using TDigest = detail::TDigest<double, size_t>;

}  // namespace quantile
