/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "Logger/Logger.h"

namespace gfx {

//
// StatsAccumulator
//
// Stores an instances of StatsType and a count (N)
// Operator += is used to add new StatsType and increment the count so custom StatsType
// objects must implement +=
template <typename StatsType>
class StatsAccumulator {
 public:
  explicit StatsAccumulator(const std::string& name) : stats_{}, name_{name}, N_{0} {}

  void operator+=(const StatsType& new_stats) {
    stats_ += new_stats;
    N_++;
  }

  std::ostream& print(std::ostream& os, size_t max_name_length) const {
    // TODO(scb): custom formatter
    os << std::left << std::setw(max_name_length + 2) << name_;
    os << std::left << std::setw(8) << std::string("  N=" + std::to_string(N_));
    // Handle single value integral stats
    if constexpr (std::is_arithmetic_v<StatsType>) {
      os << std::left << std::setw(15) << std::string("  time=" + std::to_string(stats_));
    } else {
      // Custom stats must implement operator<<(ostream&);
      os << stats_;
    }
    return os;
  }

  StatsType& getStats() const { return stats_; }
  uint32_t getN() const { return N_; }

 private:
  mutable StatsType stats_;
  std::string name_;
  uint32_t N_;
};

//
// StatsMap
//
// Map a string (name) to a StatsAccumulator
// Can fill a vector with average values, then sort it
// Can generate a report of sorted average values
//
// StatType must support these operators:
//   += for accumulation
//   /= (uint32_t) for averaging
//   >  for sorting
template <typename StatsType>
struct StatsMap {
  void accumulate(const std::string& name, const StatsType& stats) {
    // Get current accumulator or create a new one
    auto result = stats_map_.emplace(name, StatsAccumulator<StatsType>(name));
    CHECK(result.first != stats_map_.end());
    result.first->second += stats;  // add to accumulator
  }

  // Copy stats into vector, average values in place, sort vector
  // min_N can be used to filter Stats with a low N value (e.g. one off compiles)
  std::pair<std::vector<StatsAccumulator<StatsType>>, size_t> averageAndSort(
      uint32_t min_N) const {
    std::vector<StatsAccumulator<StatsType>> sorted_stats;
    size_t max_name_length{};
    for (auto [name, src_accumulator] : stats_map_) {
      if (src_accumulator.getN() >= min_N) {
        auto& accumulator_avg = sorted_stats.emplace_back(src_accumulator);
        accumulator_avg.getStats() /= accumulator_avg.getN();
        max_name_length = std::max(name.length(), max_name_length);
      }
    }

    // sort stats
    std::sort(sorted_stats.begin(),
              sorted_stats.end(),
              [](StatsAccumulator<StatsType>& a, StatsAccumulator<StatsType>& b) {
                return a.getStats() > b.getStats();
              });

    return {std::move(sorted_stats), max_name_length};
  }

  // Create temp vector of sorted average values, then
  // write a formatted report to ostream
  void generateReport(std::ostream& os, const std::string& label, uint32_t min_N) const {
    // Copy stats into vector, average values, and sort
    auto [sorted_stats, max_name_length] = averageAndSort(min_N);

    // Write report
    os << "------------------------------\n";
    os << label << "\n";
    os << "------------------------------\n";
    for (auto const& stats_accum : sorted_stats) {
      stats_accum.print(os, max_name_length) << "\n";
    }
  }

  bool is_empty() const { return stats_map_.empty(); }

 private:
  std::map<std::string, StatsAccumulator<StatsType>> stats_map_;
};

}  // namespace gfx
