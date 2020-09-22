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

#include "Shared/DateConverters.h"
#include "Shared/DateTimeParser.h"
#include "Tests/TestHelpers.h"

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

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

TEST(TIME, LegalParseTimeString) {
  using namespace std::string_literals;
  static const std::unordered_map<std::string, int64_t> values = {
      {"22:28:48"s, 80928},
      {"22:28:48.876"s, 80928},
      {"T22:28:48"s, 80928},
      {"222848"s, 80928},
      {"22:28:48-05:00"s, 98928},
      {"22:28:48+05:00"s, 62928},
      {"22:28"s, 80880}};

  for (const auto& [time_str, expected_epoch] : values) {
    ASSERT_EQ(expected_epoch, dateTimeParse<kTIME>(time_str, 0)) << time_str;
  }
}

TEST(TIME, IllegalParseTimeString) {
  using namespace std::string_literals;
  static const std::unordered_set<std::string> values = {
      "22-28-48"s, "2228.48"s, "22.28.48"s, "22"s};

  for (const auto& val : values) {
    if (val == "2228.48") {
      ASSERT_NO_THROW(dateTimeParse<kTIME>(val, 0));  // Parsed as 22:02:08.48.
    } else {
      ASSERT_THROW(dateTimeParse<kTIME>(val, 0), std::runtime_error) << val;
    }
  }
}

TEST(TIMESTAMPS, OverflowUnderflow) {
  using namespace std::string_literals;
  static const std::unordered_set<std::string> values = {
      "2273-01-01 23:12:12"s,
      "2263-01-01 00:00:00"s,
      "09/21/1676 00:12:43.145224193"s,
      "09/21/1677 00:00:43.145224193"s};
  for (const auto& value : values) {
    ASSERT_NO_THROW(dateTimeParse<kTIMESTAMP>(value, 0));
    ASSERT_NO_THROW(dateTimeParse<kTIMESTAMP>(value, 3));
    ASSERT_NO_THROW(dateTimeParse<kTIMESTAMP>(value, 6));
    ASSERT_NO_THROW(dateTimeParse<kTIMESTAMP>(value, 9));
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
