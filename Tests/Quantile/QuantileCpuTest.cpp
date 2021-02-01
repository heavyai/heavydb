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

#include "Shared/Intervals.h"
#include "Shared/quantile.h"
#include "Tests/TestHelpers.h"

#include <gtest/gtest.h>

#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#include <tbb/tbb.h>

#include <chrono>
#include <iostream>
#include <numeric>  // iota
#include <thread>   // hardware_concurrency - in tbb?

using Real = double;
using Index = size_t;
using TDigest = quantile::detail::TDigest<Real, Index>;

// Requirement: sorted is sorted, 0 <= q <= 1.
template <typename Vector>
Real exact_quantile(Vector const& sorted, Real q) {
  if (sorted.empty()) {
    return Real(0) / Real(0);
  } else if (q == 1) {
    return sorted.back();
  }
  Real index;
  Real frac = std::modf(q * sorted.size(), &index);
  size_t const i = static_cast<size_t>(index);
  return frac == 0 && i != 0 ? 0.5 * (sorted[i] + sorted[i - 1]) : sorted[i];
}

double msSince(std::chrono::steady_clock::time_point const start) {
  using namespace std::chrono;
  steady_clock::time_point const stop = steady_clock::now();
  return duration_cast<microseconds>(stop - start).count() * 1e-3;
}

TEST(Quantile, Basic) {
  // Set data
  // size_t const N = 1e9;             // number of data points
  size_t const N = 1e8;             // number of data points
  size_t const M = 300;             // Max number of centroids per digest
  size_t const buf_size = 10000;    // Input buffer size
  size_t const mixer = 2654435761;  // closest prime to 2^32 / phi
  Real const q = 0.5;

  // Number of partitions = number of independent digests
  size_t const ndigests = std::thread::hardware_concurrency();
  std::cout << "ndigests = " << ndigests << std::endl;

  std::cout << "\nAllocating test data of size " << N << "... " << std::flush;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::unique_ptr<Real[]> data_ptr = std::make_unique<Real[]>(N);
  VectorView<Real> data(data_ptr.get(), N, N);
  std::cout << msSince(start) << "ms" << std::endl;

  std::cout << "\nSetting ordered data... " << std::flush;
  start = std::chrono::steady_clock::now();
  tbb::parallel_for_each(makeIntervals<size_t>(0, N, ndigests), [&](auto interval) {
    std::iota(data.begin() + interval.begin,
              data.begin() + interval.end,
              Real(interval.begin + 1));
  });
  std::cout << msSince(start) << "ms" << std::endl;

  // Actual quantile
  Real const exact = exact_quantile(data, q);
  std::cout.setf(std::ios::fixed);
  std::cout << "exact_quantile(data, " << q << ") = " << exact << std::endl;

  std::cout << "\nSetting shuffled data... " << std::flush;
  start = std::chrono::steady_clock::now();
  // Fill data with a pseudo-random permutation of 1, 2, 3, ..., N
  tbb::parallel_for_each(makeIntervals<size_t>(0, N, ndigests), [&](auto interval) {
    for (size_t i = interval.begin; i != interval.end; ++i) {
      data[i] = i * mixer % N + 1;
    }
  });
  std::cout << msSince(start) << "ms" << std::endl;

  std::cout << "\nReserving centroids memory... " << std::flush;
  start = std::chrono::steady_clock::now();
  std::vector<TDigest> digests(ndigests);
  TDigest::Memory centroids_memory(ndigests * M);
  // Doesn't have to be parallel, but useful template for GPU.
  tbb::parallel_for_each(
      makeIntervals<size_t>(0, centroids_memory.size(), ndigests), [&](auto interval) {
        size_t const size = interval.end - interval.begin;
        VectorView<Real> sums(
            centroids_memory.sums().data() + interval.begin, size, size);
        VectorView<Index> counts(
            centroids_memory.counts().data() + interval.begin, size, size);
        digests[interval.index].setCentroids(sums, counts);
      });
  std::cout << " with " << centroids_memory.nbytes() * 1e-6
            << "MB of centroids memory in " << msSince(start) << "ms" << std::endl;

  // Incoming buffer.
  std::cout << "\nReserving buffer memory... " << std::flush;
  TDigest::Memory buffer(buf_size);
  std::cout << " with " << buffer.nbytes() * 1e-6 << "MB of centroids memory in "
            << msSince(start) << "ms" << std::endl;

  std::cout << "\nFilling digests... " << std::flush;
  start = std::chrono::steady_clock::now();
  for (auto src_begin = data.cbegin(); src_begin != data.cend();) {
    size_t const n = std::min(size_t(data.cend() - src_begin), buffer.sums().size());
    // Faster way to copy?
    std::copy(src_begin, src_begin + n, buffer.sums().begin());
    tbb::parallel_sort(buffer.sums().begin(), buffer.sums().begin() + n);
    // Process ndigests partitions in parallel
    tbb::parallel_for_each(makeIntervals<size_t>(0, n, ndigests), [&](auto interval) {
      digests[interval.index].mergeSorted(buffer.sums().data() + interval.begin,
                                          buffer.counts().data() + interval.begin,
                                          interval.end - interval.begin);
      /*
      auto const counts =
          std::accumulate(digests[interval.index].centroids().counts().begin(),
                          digests[interval.index].centroids().counts().end(),
                          size_t(0));
      std::cout << counts << ' ';
      */
    });
    src_begin += n;
  }
  std::cout << msSince(start) << "ms" << std::endl;

  std::cout << "\nReducing digests... " << std::flush;
  start = std::chrono::steady_clock::now();
  // Reduce into digests.front(). ceil(log_2(ndigests)) serial iterations
  for (size_t i = 1; i < ndigests; i <<= 1) {
    tbb::parallel_for(size_t(0), ndigests, i << 1, [&](size_t j) {
      if ((i ^ j) < ndigests) {
        digests[j].mergeTDigest(digests[i ^ j]);
      }
    });
  }
  std::cout << msSince(start) << "ms" << std::endl;

  std::cout << "\nCalculating quantile estimate... " << std::flush;
  start = std::chrono::steady_clock::now();
  Real const estimated = digests.front().quantile(buffer.counts().data(), q);
  std::cout << msSince(start) << "ms" << std::endl;
  std::cout.setf(std::ios::fixed);
  std::cout << "digests.front().quantile(" << q << ") = " << estimated << std::endl;
  EXPECT_NEAR(exact, estimated, 0.001 * exact);
}

TEST(Quantile, TwoValues) {
  // Set data
  constexpr size_t N = 1e8 + 1;         // number of data points
  constexpr size_t M = 300;             // Max number of centroids per digest
  constexpr size_t buf_size = 10000;    // Input buffer size
  constexpr size_t mixer = 2654435761;  // closest prime to 2^32 / phi
  constexpr Real q = 0.5;

  // Number of partitions = number of independent digests
  size_t const ndigests = std::thread::hardware_concurrency();
  std::cout << "ndigests = " << ndigests << std::endl;

  std::cout << "\nAllocating test data of size " << N << "... " << std::flush;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::unique_ptr<Real[]> data_ptr = std::make_unique<Real[]>(N);
  VectorView<Real> data(data_ptr.get(), N, N);
  std::cout << msSince(start) << "ms" << std::endl;

  std::cout << "\nSetting ordered data... " << std::flush;
  start = std::chrono::steady_clock::now();
  tbb::parallel_for_each(makeIntervals<size_t>(0, N, ndigests), [&](auto interval) {
    for (size_t i = interval.begin; i != interval.end; ++i) {
      data[i] = N / 2 < i;
    }
  });
  std::cout << msSince(start) << "ms" << std::endl;

  // Actual quantile
  Real const exact = exact_quantile(data, q);
  std::cout.setf(std::ios::fixed);
  std::cout << "exact_quantile(data, " << q << ") = " << exact << std::endl;

  std::cout << "\nSetting shuffled data... " << std::flush;
  start = std::chrono::steady_clock::now();
  // Fill data with a pseudo-random permutation of 0, 0, 0, ..., 1, 1
  tbb::parallel_for_each(makeIntervals<size_t>(0, N, ndigests), [&](auto interval) {
    for (size_t i = interval.begin; i != interval.end; ++i) {
      data[i] = N / 2 < i * mixer % N;
    }
  });
  std::cout << msSince(start) << "ms" << std::endl;

  std::cout << "\nReserving centroids memory... " << std::flush;
  start = std::chrono::steady_clock::now();
  std::vector<TDigest> digests(ndigests);
  TDigest::Memory centroids_memory(ndigests * M);
  // Doesn't have to be parallel, but useful template for GPU.
  tbb::parallel_for_each(
      makeIntervals<size_t>(0, centroids_memory.size(), ndigests), [&](auto interval) {
        size_t const size = interval.end - interval.begin;
        VectorView<Real> sums(
            centroids_memory.sums().data() + interval.begin, size, size);
        VectorView<Index> counts(
            centroids_memory.counts().data() + interval.begin, size, size);
        digests[interval.index].setCentroids(sums, counts);
      });
  std::cout << " with " << centroids_memory.nbytes() * 1e-6
            << "MB of centroids memory in " << msSince(start) << "ms" << std::endl;

  // Incoming buffer.
  std::cout << "\nReserving buffer memory... " << std::flush;
  TDigest::Memory buffer(buf_size);
  std::cout << " with " << buffer.nbytes() * 1e-6 << "MB of centroids memory in "
            << msSince(start) << "ms" << std::endl;

  std::cout << "\nFilling digests... " << std::flush;
  start = std::chrono::steady_clock::now();
  for (auto src_begin = data.cbegin(); src_begin != data.cend();) {
    size_t const n = std::min(size_t(data.cend() - src_begin), buffer.sums().size());
    // Faster way to copy?
    std::copy(src_begin, src_begin + n, buffer.sums().begin());
    tbb::parallel_sort(buffer.sums().begin(), buffer.sums().begin() + n);
    // Process ndigests partitions in parallel
    tbb::parallel_for_each(makeIntervals<size_t>(0, n, ndigests), [&](auto interval) {
      digests[interval.index].mergeSorted(buffer.sums().data() + interval.begin,
                                          buffer.counts().data() + interval.begin,
                                          interval.end - interval.begin);
      /*
      auto const counts =
          std::accumulate(digests[interval.index].centroids().counts().begin(),
                          digests[interval.index].centroids().counts().end(),
                          size_t(0));
      std::cout << counts << ' ';
      */
    });
    src_begin += n;
  }
  std::cout << msSince(start) << "ms" << std::endl;

  std::cout << "\nReducing digests... " << std::flush;
  start = std::chrono::steady_clock::now();
  // Reduce into digests.front(). ceil(log_2(ndigests)) serial iterations
  for (size_t i = 1; i < ndigests; i <<= 1) {
    tbb::parallel_for(size_t(0), ndigests, i << 1, [&](size_t j) {
      if ((i ^ j) < ndigests) {
        digests[j].mergeTDigest(digests[i ^ j]);
      }
    });
  }
  std::cout << msSince(start) << "ms" << std::endl;

  std::cout << "\nCalculating quantile estimate... " << std::flush;
  start = std::chrono::steady_clock::now();
  Real const estimated = digests.front().quantile(buffer.counts().data(), q);
  std::cout << msSince(start) << "ms" << std::endl;
  std::cout.setf(std::ios::fixed);
  std::cout << "digests.front().quantile(" << q << ") = " << estimated << std::endl;
  EXPECT_NEAR(exact, estimated, 10.0);
}

TEST(Quantile, EmptyDataSets) {
  TDigest::Memory memory0(3);
  TDigest t_digest0(memory0);
  EXPECT_TRUE(std::isnan(t_digest0.quantile(0.5)));

  TDigest::Memory memory1(3);
  TDigest t_digest1(memory1);
  t_digest1.mergeTDigest(t_digest0);
  EXPECT_TRUE(std::isnan(t_digest1.quantile(0.5)));
}

TEST(Quantile, Singletons) {
  TDigest::Memory memory0(3), buffer(3);
  TDigest t_digest0(memory0);
  t_digest0.setBuffer(buffer);

  constexpr double x = 10.0;
  t_digest0.add(x);
  t_digest0.mergeBuffer();
  EXPECT_EQ(x, t_digest0.quantile(0.0));
  EXPECT_EQ(x, t_digest0.quantile(0.5));
  EXPECT_EQ(x, t_digest0.quantile(1.0));

  TDigest::Memory memory1(3);
  TDigest t_digest1(memory1);
  t_digest1.mergeTDigest(t_digest0);
  t_digest1.setBuffer(buffer);
  EXPECT_EQ(x, t_digest1.quantile(0.0));
  EXPECT_EQ(x, t_digest1.quantile(0.5));
  EXPECT_EQ(x, t_digest1.quantile(1.0));
}

TEST(Quantile, Pairs) {
  TDigest::Memory memory0(3), buffer(3);
  TDigest t_digest0(memory0);
  t_digest0.setBuffer(buffer);

  t_digest0.add(10);
  t_digest0.add(20);
  t_digest0.mergeBuffer();
  EXPECT_EQ(10, t_digest0.quantile(0.0));
  EXPECT_EQ(10, t_digest0.quantile(0.0001));
  EXPECT_EQ(10, t_digest0.quantile(0.4999));
  EXPECT_EQ(15, t_digest0.quantile(0.5));
  EXPECT_EQ(20, t_digest0.quantile(0.5001));
  EXPECT_EQ(20, t_digest0.quantile(0.9999));
  EXPECT_EQ(20, t_digest0.quantile(1.0));

  TDigest::Memory memory1(3);
  TDigest t_digest1(memory1);
  t_digest1.mergeTDigest(t_digest0);
  t_digest1.setBuffer(buffer);
  EXPECT_EQ(10, t_digest1.quantile(0.0));
  EXPECT_EQ(10, t_digest1.quantile(0.0001));
  EXPECT_EQ(10, t_digest1.quantile(0.4999));
  EXPECT_EQ(15, t_digest1.quantile(0.5));
  EXPECT_EQ(20, t_digest1.quantile(0.5001));
  EXPECT_EQ(20, t_digest1.quantile(0.9999));
  EXPECT_EQ(20, t_digest1.quantile(1.0));
}

TEST(Quantile, SmallDataSetsAndMemory) {
  for (unsigned mem = 1; mem <= 6; ++mem) {
    for (unsigned N = 1; N <= 3; ++N) {
      TDigest::Memory memory0(mem), buffer(mem);
      TDigest t_digest0(memory0);
      t_digest0.setBuffer(buffer);

      for (unsigned i = 0; i < N; ++i) {
        t_digest0.add(10 * (i + 1));
      }
      t_digest0.mergeBuffer();
      EXPECT_EQ(N, t_digest0.totalWeight());

      EXPECT_EQ(10, t_digest0.quantile(-0.00001));
      EXPECT_EQ(10, t_digest0.quantile(0.0));
      EXPECT_EQ(10, t_digest0.quantile(0.333333));
      if (N == 1) {
        EXPECT_EQ(10, t_digest0.quantile(0.5));
        EXPECT_EQ(10, t_digest0.quantile(1.0));
        EXPECT_EQ(10, t_digest0.quantile(1.1));
      } else if (N == 2) {
        EXPECT_EQ(10, t_digest0.quantile(0.4999));
        EXPECT_EQ(15, t_digest0.quantile(0.5));
        EXPECT_EQ(20, t_digest0.quantile(0.5001));
        EXPECT_EQ(20, t_digest0.quantile(0.9999));
        EXPECT_EQ(20, t_digest0.quantile(1.0));
        EXPECT_EQ(20, t_digest0.quantile(1.1));
      } else if (N == 3) {
        EXPECT_EQ(15, t_digest0.quantile(1.0 / 3));
        EXPECT_EQ(20, t_digest0.quantile(0.3334));
        EXPECT_EQ(20, t_digest0.quantile(0.5));
        EXPECT_EQ(20, t_digest0.quantile(0.6666));
        EXPECT_EQ(25, t_digest0.quantile(2.0 / 3));
        EXPECT_EQ(30, t_digest0.quantile(0.6667));
        EXPECT_EQ(30, t_digest0.quantile(0.9999));
        EXPECT_EQ(30, t_digest0.quantile(1.0));
        EXPECT_EQ(30, t_digest0.quantile(1.0001));
      }
    }
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
