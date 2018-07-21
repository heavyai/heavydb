/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef TEST_HELPERS_H_
#define TEST_HELPERS_H_

#include "../QueryEngine/TargetValue.h"

#include <gtest/gtest.h>
#include <boost/variant.hpp>

namespace TestHelpers {

template <class T>
void compare_array(const TargetValue& r,
                   const std::vector<T>& arr,
                   const double tol = -1.) {
  auto scalar_tv_vector = boost::get<std::vector<ScalarTargetValue>>(&r);
  CHECK(scalar_tv_vector);
  ASSERT_EQ(scalar_tv_vector->size(), arr.size());
  size_t ctr = 0;
  for (const ScalarTargetValue scalar_tv : *scalar_tv_vector) {
    auto p = boost::get<T>(&scalar_tv);
    if (tol < 0.) {
      ASSERT_EQ(*p, arr[ctr++]);
    } else {
      ASSERT_NEAR(*p, arr[ctr++], tol);
    }
  }
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}
}  // namespace TestHelpers

#endif  // TEST_HELPERS_H_
