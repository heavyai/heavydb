/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VEGARENDERTESTCPUSETUP_H
#define VEGARENDERTESTCPUSETUP_H

#ifndef HAVE_CUDA

#include <stdint.h>
#include <memory>

#include <random>

namespace QueryRenderer {
class QueryDataLayout;
}

template <class T>
struct UniformRandomNumberGenerator {
  std::default_random_engine generator;
  std::uniform_real_distribution<T>* distribution;

  UniformRandomNumberGenerator(int min, int max) {
    distribution = new std::uniform_real_distribution<T>(min, max);
  }
  ~UniformRandomNumberGenerator() { delete distribution; }

  T operator()() { return (*this->distribution)(generator); }
};

struct SimplePointRow {
  int64_t key;
  double x;
  double y;
  double val;
  int64_t party;
  int64_t rowid;

  static std::shared_ptr<::QueryRenderer::QueryDataLayout> getQueryDataLayout();
  static void get_random_data(UniformRandomNumberGenerator<double>& rand,
                              SimplePointRow* row,
                              int numPts,
                              const std::array<double, 2>& xExtents,
                              const std::array<double, 2>& yExtents,
                              const std::array<double, 2>& valExtents,
                              const std::array<int64_t, 2>& partyExtents);
};

#endif  // HAVE_CUDA

#endif  // VEGARENDERTESTCPUSETUP_H
