/*
 * Copyright 2019 OmniSci, Inc.
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

/**
 * @file    uuid.h
 * @author  Steve Blackmon <steve.blackmon@omnisci.com>
 * @brief   No-frills UUID type class to allow easy containerization and comparison of
 *          device UUIDs from different APIs.
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <iomanip>
#include <iterator>
#include <sstream>

namespace omnisci {

class UUID {
  using value_type = uint8_t;

 public:
  // Constructors
  constexpr UUID() noexcept : data_({}) {}

  // copy from value_type[16] (OpenGL / Vulkan)
  explicit UUID(const value_type (&arr)[16]) noexcept {
    std::copy(std::cbegin(arr), std::cend(arr), std::begin(data_));
  }

  // copy from char[16] (Cuda)
  explicit UUID(char (&arr)[16]) noexcept {
    std::transform(
        std::cbegin(arr), std::cend(arr), std::begin(data_), [](char c) -> value_type {
          return static_cast<value_type>(c);
        });
  }

  void swap(UUID& other) noexcept { data_.swap(other.data_); }

  // Access for underlying array data
  constexpr value_type* data() noexcept { return data_.data(); }
  constexpr const value_type* data() const noexcept { return data_.data(); }

  // clang-format off
  friend std::ostream& operator<< (std::ostream& s, const UUID& id) {
    auto fill = s.fill();
    auto ff = s.flags();
    s << std::hex << std::setfill('0')
      << std::setw(2) << +id.data_[0]
      << std::setw(2) << +id.data_[1]
      << std::setw(2) << +id.data_[2]
      << std::setw(2) << +id.data_[3]
      << '-'
      << std::setw(2) << +id.data_[4]
      << std::setw(2) << +id.data_[5]
      << '-'
      << std::setw(2) << +id.data_[6]
      << std::setw(2) << +id.data_[7]
      << '-'
      << std::setw(2) << +id.data_[8]
      << std::setw(2) << +id.data_[9]
      << '-'
      << std::setw(2) << +id.data_[10]
      << std::setw(2) << +id.data_[11]
      << std::setw(2) << +id.data_[12]
      << std::setw(2) << +id.data_[13]
      << std::setw(2) << +id.data_[14]
      << std::setw(2) << +id.data_[15]
      << std::setfill(fill);
    s.flags(ff);
    return s;
  }
  // clang-format on

 private:
  std::array<value_type, 16> data_;

  friend bool operator==(const UUID& lhs, const UUID& rhs) noexcept;
  friend bool operator<(const UUID& lhs, const UUID& rhs) noexcept;
};

inline bool operator==(const UUID& lhs, const UUID& rhs) noexcept {
  return lhs.data_ == rhs.data_;
}

inline bool operator!=(const UUID& lhs, const UUID& rhs) noexcept {
  return !(lhs == rhs);
}

inline bool operator<(const UUID& lhs, const UUID& rhs) noexcept {
  return lhs.data_ < rhs.data_;
}

inline std::string to_string(const UUID& uuid) {
  std::stringstream ss;
  ss << uuid;
  return ss.str();
}

constexpr UUID empty_uuid{};

}  // namespace omnisci
