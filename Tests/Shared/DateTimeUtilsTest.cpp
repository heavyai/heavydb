/*
 * Copyright 2018 OmniSci, Inc.
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

#include "../../Shared/DateConversions.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace {

template <class T>
void compare_epoch(const std::vector<T>& expected, const std::vector<T>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < actual.size(); i++) {
    ASSERT_EQ(expected[i], actual[i]);
  }
}

}  // namespace

struct SampleDateEpochs {
  const std::vector<int64_t> epoch{-31496400, -31536000, 31579200, 31536000};
  const std::vector<int64_t> actual_days{-365, -365, 365, 365};
  const std::vector<int64_t> expected_seconds{-31536000, -31536000, 31536000, 31536000};
};

TEST(DATE, EpochSecondsToDaysTest) {
  const auto sample = SampleDateEpochs();
  std::vector<int64_t> computed;
  for (const auto ep : sample.epoch) {
    computed.push_back(DateConverters::get_epoch_days_from_seconds(ep));
  }
  compare_epoch(computed, sample.actual_days);
}

TEST(DATE, EpochDaysToSecondsTest) {
  const auto sample = SampleDateEpochs();
  std::vector<int64_t> computed;
  for (const auto ep : sample.actual_days) {
    computed.push_back(DateConverters::get_epoch_seconds_from_days(ep));
  }
  compare_epoch(computed, sample.expected_seconds);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
